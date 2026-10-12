"""Model Reroute: switch the active LoRA branch at a chosen sampling step.

A single KSampler run normally keeps one ModelPatcher for its whole schedule.
This node takes two LoRA variants of the same base model (they share the base
weights via ModelPatcher.clone) and, the first time the sampler's sigma drops
below the configured threshold, hot-swaps the branch-specific weight patches,
object patches and injections on the live patcher: ``unpatch_model`` -> replace
only the keys that differ between the two branches -> ``patch_model``. Every
later step of the run then denoises with the second LoRA.

The swap is deliberately narrow:

* Only keys that actually differ between ``model_a`` and ``model_b`` are
  replaced, so patches added downstream of this node, or present on both
  branches (shared LoRA, attention overrides, ...), stay active on both sides.
* Nothing in ``model_options`` / ``transformer_options`` is touched, so
  transformer_options-level integrations (Sol-Attn's attention override,
  Spectrum's wrappers and runtime keys, ...) are unaffected.

The switch is one-way (A -> B). An ON_CLEANUP callback restores branch A at
the end of every run, and the wrapper re-asserts A at run start, so reusing
the same MODEL output across multiple KSamplers stays deterministic. Regular
and dynamic (--fast) ModelPatchers are both supported; dynamic patchers take
the managed unpatch -> patch(load_weights=False) -> load(dirty) sequence.
A per-pass reset wrapper makes the switch fire again for each full schedule
re-run on the same guider (e.g. Spectrum's offline smoothing replay).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import torch

from comfy.patcher_extension import CallbacksMP, WrappersMP
from comfy_api.latest import io

LOG = logging.getLogger("model_reroute")
WRAPPER_KEY = "model_reroute"
STATE_ATTR = "_model_reroute_state"
LIVE_ATTR = "_model_reroute_live"


def _snapshot(patcher: Any) -> dict[str, dict[str, Any]]:
    """Deep-ish copies of the three branch-carrying state containers."""
    return {
        "patches": {k: list(v) for k, v in patcher.patches.items()},
        "object_patches": dict(patcher.object_patches),
        "injections": {k: list(v) for k, v in patcher.injections.items()},
    }


def _diff_keys(snap_a: dict[str, Any], snap_b: dict[str, Any]) -> dict[str, set]:
    """Keys whose content differs between the two branches, per dimension.

    Identical keys (shared LoRA, patches applied to both branches) are left
    alone, which is what keeps post-reroute additions alive on both sides.
    """
    diff = {}
    for dim in ("patches", "object_patches", "injections"):
        keys = set(snap_a[dim]) | set(snap_b[dim])
        diff[dim] = {k for k in keys if snap_a[dim].get(k) != snap_b[dim].get(k)}
    return diff


def _swap_diff(patcher: Any, config: "RerouteConfig", source: dict[str, Any]) -> None:
    for dim in ("patches", "object_patches", "injections"):
        target = getattr(patcher, dim)
        for key in config.diff[dim]:
            if key in source[dim]:
                target[key] = source[dim][key]
            else:
                target.pop(key, None)


def _apply_branch(patcher: Any, config: "RerouteConfig", branch: str) -> None:
    """Make ``branch``'s LoRA state live on the shared base model.

    unpatch_model ejects the old injections (and, for regular patchers,
    restores the base weights); patch_model applies the swapped object
    patches and re-injects. Only the diff keys are replaced, so keys added
    after this node survive the round-trip.

    Dynamic patchers (--fast) forbid ``patch_model(load_weights=True)``, so
    they take the managed sequence mirroring ModelPatcherDynamic.partially_load:
    unpatch without weight restore, re-apply patches without loading, then
    ``load(dirty=True)`` re-merges the weight patches from the swapped dict.
    """
    source = config.snap_b if branch == "B" else config.snap_a
    if patcher.is_dynamic():
        patcher.unpatch_model(patcher.offload_device, unpatch_weights=False)
        _swap_diff(patcher, config, source)
        patcher.patches_uuid = uuid.uuid4()
        patcher.patch_model(load_weights=False)
        patcher.load(patcher.load_device, dirty=True)
    else:
        patcher.unpatch_model()
        _swap_diff(patcher, config, source)
        patcher.patches_uuid = uuid.uuid4()
        patcher.patch_model()


def _sigma_value(timestep: Any) -> float:
    if torch.is_tensor(timestep):
        timestep = timestep.flatten()[0]
    return float(timestep)


def _current_step(timestep: Any, model_options: dict[str, Any] | None) -> int | None:
    sigmas = (model_options or {}).get("transformer_options", {}).get("sample_sigmas")
    if not torch.is_tensor(sigmas) or sigmas.numel() == 0:
        return None
    return int((sigmas > _sigma_value(timestep)).sum().item())


def _should_switch(state: dict[str, Any], timestep: Any, model_options: dict[str, Any] | None) -> bool:
    if state["trigger"] == "step":
        step = _current_step(timestep, model_options)
        return step is not None and step >= state["switch_step"]
    return _sigma_value(timestep) < state["sigma_threshold"]


class RerouteConfig:
    """Per-node configuration captured at execute time; shared by the wrappers."""

    def __init__(self, snap_a, snap_b, diff, trigger, sigma_threshold, switch_step):
        self.snap_a = snap_a
        self.snap_b = snap_b
        self.diff = diff
        self.trigger = trigger
        self.sigma_threshold = sigma_threshold
        self.switch_step = switch_step


def _make_outer_sample_wrapper():
    """Reset per-pass switch state at the start of every sampling pass.

    Some integrations re-run the whole schedule within one guider (e.g.
    Spectrum's offline smoothing replay runs a second full pass), so the
    switch must fire again per pass, not once per guider.
    """

    def wrapper(executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        guider = executor.class_obj
        if hasattr(guider, STATE_ATTR):
            delattr(guider, STATE_ATTR)
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    return wrapper


def _make_predict_noise_wrapper(config: RerouteConfig):
    def wrapper(executor, x, timestep, model_options=None, seed=None):
        guider = executor.class_obj
        state = getattr(guider, STATE_ATTR, None)
        if state is None:
            patcher = getattr(guider, "model_patcher", None)
            state = {
                "enabled": patcher is not None,
                "switched": False,
                "logged": False,
                "trigger": config.trigger,
                "sigma_threshold": config.sigma_threshold,
                "switch_step": config.switch_step,
            }
            setattr(guider, STATE_ATTR, state)
            if patcher is not None and getattr(patcher, LIVE_ATTR, "A") != "A":
                # A previous run on this patcher ended mid-switch; re-assert A
                # before this run's first model call.
                _apply_branch(patcher, config, "A")
                setattr(patcher, LIVE_ATTR, "A")
        if state["enabled"] and not state["switched"]:
            if "multigpu_clones" in (model_options or {}):
                state["enabled"] = False
                if not state["logged"]:
                    state["logged"] = True
                    LOG.warning(
                        "Model Reroute: multi-GPU parallel sampling detected; "
                        "step switching disabled for this run"
                    )
            elif _should_switch(state, timestep, model_options):
                patcher = getattr(guider, "model_patcher", None)
                _apply_branch(patcher, config, "B")
                setattr(patcher, LIVE_ATTR, "B")
                state["switched"] = True
                LOG.info(
                    "Model Reroute: switched to branch B at sigma=%.6g",
                    _sigma_value(timestep),
                )
        return executor(x, timestep, model_options, seed)

    return wrapper


def _make_cleanup(config: RerouteConfig):
    def cleanup(patcher):
        if getattr(patcher, LIVE_ATTR, "A") != "A":
            _apply_branch(patcher, config, "A")
            setattr(patcher, LIVE_ATTR, "A")

    return cleanup


class ModelReroute(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ModelReroute",
            display_name="Model Reroute (Step LoRA Switch)",
            category="model/reroute",
            description=(
                "在同一底模的两条 LoRA 分支间，于采样到达指定步数/σ 阈值时热切换。"
                "model_a 与 model_b 必须是同一底模（同一 checkpoint）加载的不同 LoRA 组合；"
                "切换只替换两分支差异的权重补丁，不触碰 transformer_options，"
                "因此与 Sol-Attn、Spectrum 等基于 model_options 的节点兼容。"
                "默认 switch_percent=0.2 表示前 20% 采样步用分支 A，之后用分支 B。"
            ),
            inputs=[
                io.Model.Input(
                    "model_a",
                    tooltip="分支 A：底模 + 该阶段要用的 LoRA。",
                ),
                io.Model.Input(
                    "model_b",
                    tooltip="分支 B：同一底模 + 切换后要用的 LoRA。必须与 model_a 共享同一底模对象。",
                ),
                io.Combo.Input(
                    "trigger",
                    options=["percent", "step"],
                    default="percent",
                    tooltip="percent：按 σ 阈值（switch_percent）切换；step：按采样步数（switch_step）切换。",
                ),
                io.Float.Input(
                    "switch_percent",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="触发阈值（percent 模式）：噪声水平首次低于该百分比对应 σ 时切换。"
                            "0.2 = 前 20% 采样步用 A；0 = 第 1 步即切换；1 = 永不切换。",
                ),
                io.Int.Input(
                    "switch_step",
                    default=-1,
                    min=-1,
                    max=100000,
                    tooltip="触发步数（step 模式，>=0 生效）：第 N 步起切换到分支 B。",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model_a,
        model_b,
        trigger="percent",
        switch_percent=0.2,
        switch_step=-1,
    ) -> io.NodeOutput:
        if getattr(model_a, "model", None) is not getattr(model_b, "model", None):
            raise ValueError(
                "Model Reroute requires model_a and model_b to share the same "
                "base model (same checkpoint). Connect both LoRA chains to a "
                "common base MODEL."
            )
        if trigger == "step" and switch_step < 0:
            raise ValueError("switch_step must be >= 0 when trigger is 'step'.")

        m = model_a.clone()
        snap_a = _snapshot(m)
        snap_b = _snapshot(model_b)
        diff = _diff_keys(snap_a, snap_b)
        if not any(diff.values()):
            LOG.warning(
                "Model Reroute: model_a and model_b carry identical patches; "
                "the switch will change nothing"
            )

        model_sampling = m.get_model_object("model_sampling")
        sigma_threshold = float(model_sampling.percent_to_sigma(switch_percent))
        config = RerouteConfig(snap_a, snap_b, diff, trigger, sigma_threshold, switch_step)

        m.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, WRAPPER_KEY)
        m.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, _make_outer_sample_wrapper())
        m.remove_wrappers_with_key(WrappersMP.PREDICT_NOISE, WRAPPER_KEY)
        m.add_wrapper_with_key(WrappersMP.PREDICT_NOISE, WRAPPER_KEY, _make_predict_noise_wrapper(config))
        m.remove_callbacks_with_key(CallbacksMP.ON_CLEANUP, WRAPPER_KEY)
        m.add_callback_with_key(CallbacksMP.ON_CLEANUP, WRAPPER_KEY, _make_cleanup(config))
        return io.NodeOutput(m)
