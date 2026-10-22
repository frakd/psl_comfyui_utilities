"""LoRA Step Schedule (Bypass) — per-step LoRA strength without merging weights.

The usual LoRA loader turns a LoRA into weight patches with one fixed strength.
This node keeps the base weights untouched and injects the LoRA branch into the
module forward pass instead:

    output = base_forward(x) + schedule[step] * lora_branch(x)

A ``PREDICT_NOISE`` wrapper resolves the actual sampling step from
``sample_sigmas`` on every step and updates the multiplier on all injected
modules before the diffusion forward runs.  If the sampler has more steps than
the configured list the last strength is held; if it has fewer steps only the
leading values are used.

Only adapters that can be evaluated without the base weight are supported
(standard LoRA, LoHa, LoKr; no DoRA / reshape / diff / set patches).  Fused
projection LoRAs whose key map target is a tuple with an output-row offset
(e.g. MOSS-Image q/k/v, Qwen gate_up, Flux3 linear1_qkv) are supported by
adding each adapter delta to the corresponding output slice.  The remaining
variants cannot be expressed by a pure forward branch, and silently producing
wrong output would be worse than an error.

The node is model-only.  Its panel lives in ``ui/lora_step_schedule.js``.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
import uuid
import weakref
from collections import OrderedDict, namedtuple
from typing import Any

import torch
import torch.nn as nn

import folder_paths
import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.utils
from comfy.patcher_extension import CallbacksMP, WrappersMP
from comfy.weight_adapter import (
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
    WeightAdapterBase,
)
from comfy_api.latest import io

LOG = logging.getLogger("psl.lora_step_schedule")

CURVE_NAMES = ("linear", "ease_in", "ease_out", "ease_in_out", "cosine")
PRECISION_TO_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_MAX_LIST_POINTS = 2000
_LORA_CACHE_MAX = 2
_CHAIN_ATTACHMENT_KEY = "psl_lora_step_schedule_chain"
_lora_cache: "OrderedDict[tuple[str, int, int], dict]" = OrderedDict()
_lora_cache_lock = threading.Lock()

_MODEL_KEY_MAP_CACHE_MAX = 4
_model_key_map_cache: "OrderedDict[int, tuple[weakref.ref, dict]]" = OrderedDict()
_model_key_map_lock = threading.Lock()

_ModuleInfo = namedtuple(
    "_ModuleInfo",
    "is_conv conv_dim kernel_size in_channels out_channels kw_dict",
)


def _curve_value(curve: str, t: float) -> float:
    """Monotonic 0 -> 1 curve used by both the backend and the JS panel."""
    t = min(1.0, max(0.0, float(t)))
    if curve == "linear":
        return t
    if curve == "ease_in":
        return t * t
    if curve == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if curve == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    if curve == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    raise ValueError(f"未知曲线类型: {curve!r}（可选: {', '.join(CURVE_NAMES)}）")


def _generate_strengths(
    start_strength: float,
    end_strength: float,
    curve: str,
    points: int,
) -> list[float]:
    points = int(points)
    if points < 2:
        raise ValueError("points 至少为 2")
    if curve not in CURVE_NAMES:
        raise ValueError(f"未知曲线类型: {curve!r}（可选: {', '.join(CURVE_NAMES)}）")
    start = float(start_strength)
    end = float(end_strength)
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("开始/结束强度必须是有限数值")

    span = points - 1
    values = []
    for i in range(points):
        t = i / span
        values.append(start + (end - start) * _curve_value(curve, t))
    return values


_LIST_SEPARATORS = re.compile(r"[,;\s，、]+")


def _parse_strength_list(text: Any) -> list[float]:
    """Parse a manual list; empty text means "generate from the curve"."""
    text = "" if text is None else str(text).strip()
    if not text:
        return []
    if text[0] in "[(":
        text = text[1:]
    if text and text[-1] in "])":
        text = text[:-1]

    values = []
    for token in _LIST_SEPARATORS.split(text):
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"强度列表含非数字项 {token!r}；请用逗号/空格分隔数字"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"强度列表含非有限数值: {token!r}")
        values.append(value)

    if len(values) > _MAX_LIST_POINTS:
        raise ValueError(f"强度列表过长（{len(values)} 项，上限 {_MAX_LIST_POINTS}）")
    return values


def _resolve_schedule(
    start_strength: float,
    end_strength: float,
    curve: str,
    points: int,
    strength_list: Any,
) -> tuple[list[float], bool]:
    manual = _parse_strength_list(strength_list)
    if manual:
        return manual, True
    return _generate_strengths(start_strength, end_strength, curve, points), False


def _strength_for_step(schedule: list[float], step: int) -> float:
    if step <= 0:
        return schedule[0]
    if step >= len(schedule):
        return schedule[-1]
    return schedule[step]


def _parse_patch_key(key: Any) -> tuple[str, tuple[int, int, int] | None]:
    """Parse a ``load_lora`` target key.

    ComfyUI's fused-projection key maps use tuples like
    ``("...proj.qkv.weight", (0, start, length))``: the adapter is trained for
    output rows ``start:start+length`` of the fused weight.  Only output-row
    slices are produced by the core key maps; input/kernel slices are rejected
    rather than silently applied to the wrong axis.
    """
    function = None
    if isinstance(key, str):
        target = key
        offset = None
    elif isinstance(key, (tuple, list)) and key:
        target = key[0]
        offset = key[1] if len(key) > 1 else None
        function = key[2] if len(key) > 2 else None
    else:
        raise ValueError(f"无法解析的 LoRA 键类型: {type(key).__name__}")

    if not isinstance(target, str) or not target.endswith(".weight"):
        raise ValueError("目标不是字符串 weight 键")
    if function is not None:
        raise ValueError("该键带 function 变换，暂不支持按前向分支调度")
    if offset is None:
        return target, None

    if not isinstance(offset, (tuple, list)) or len(offset) != 3:
        raise ValueError(f"无法解析 offset: {offset!r}")
    dim, start, length = (int(value) for value in offset)
    if dim != 0:
        raise ValueError(f"只支持输出维度 dim=0 的切片，收到 dim={dim}")
    if start < 0 or length <= 0:
        raise ValueError(f"非法切片 offset={offset!r}")
    return target, (dim, start, length)


def _sigma_value(timestep: Any) -> float | None:
    if timestep is None:
        return None
    if torch.is_tensor(timestep):
        if timestep.numel() == 0:
            return None
        timestep = timestep.detach().flatten()[0]
    try:
        value = float(timestep)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _normalize_tuple(value: Any, dim: int, name: str) -> Any:
    if isinstance(value, int):
        return (value,) * dim
    if isinstance(value, str):
        return value
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError(f"无法解析模块的 {name}: {value!r}") from exc
    return result


def _module_info(module: nn.Module) -> _ModuleInfo | None:
    """Detect layer type without touching ``module.weight``.

    On MOSS-Image's int8 dynamic patcher every weight access can stage or
    dequantize that layer.  Reading 178 weights just to configure forward hooks
    cost ~10 seconds; the adapter's ``h()`` only needs layer type / conv attrs.
    """
    if isinstance(module, nn.Conv1d):
        is_conv, conv_dim = True, 1
    elif isinstance(module, nn.Conv2d):
        is_conv, conv_dim = True, 2
    elif isinstance(module, nn.Conv3d):
        is_conv, conv_dim = True, 3
    elif isinstance(module, nn.Linear):
        is_conv, conv_dim = False, 0
    else:
        class_name = type(module).__name__.lower()
        if "conv3d" in class_name:
            is_conv, conv_dim = True, 3
        elif "conv2d" in class_name:
            is_conv, conv_dim = True, 2
        elif "conv1d" in class_name:
            is_conv, conv_dim = True, 1
        elif "conv" in class_name:
            is_conv, conv_dim = True, 2
        elif "linear" in class_name:
            is_conv, conv_dim = False, 0
        elif hasattr(module, "kernel_size") and hasattr(module, "in_channels"):
            is_conv = True
            kernel = getattr(module, "kernel_size", None)
            conv_dim = len(kernel) if isinstance(kernel, (tuple, list)) else 2
        elif hasattr(module, "in_features") and hasattr(module, "out_features"):
            is_conv, conv_dim = False, 0
        else:
            return None

    if not is_conv:
        return _ModuleInfo(
            False,
            0,
            (1,),
            None,
            getattr(module, "out_features", None),
            {},
        )

    kernel_size = getattr(module, "kernel_size", None)
    if kernel_size is None:
        return None
    kernel_size = _normalize_tuple(kernel_size, conv_dim, "kernel_size")
    if not isinstance(kernel_size, tuple) or len(kernel_size) != conv_dim:
        return None

    groups = int(getattr(module, "groups", 1) or 1)
    in_channels = getattr(module, "in_channels", None)
    if in_channels is None:
        # conv LoRA needs this to reshape a 2-D down matrix; do not read weight.
        return None
    out_channels = getattr(module, "out_channels", None)

    kw_dict = {
        "stride": _normalize_tuple(getattr(module, "stride", 1), conv_dim, "stride"),
        "padding": _normalize_tuple(getattr(module, "padding", 0), conv_dim, "padding"),
        "dilation": _normalize_tuple(
            getattr(module, "dilation", 1), conv_dim, "dilation"
        ),
        "groups": groups,
    }
    return _ModuleInfo(
        True,
        conv_dim,
        kernel_size,
        int(in_channels),
        None if out_channels is None else int(out_channels),
        kw_dict,
    )


_DORA_INDEX = {
    LoRAAdapter: 4,
    LoHaAdapter: 7,
    LoKrAdapter: 8,
}


def _adapter_problem(adapter: WeightAdapterBase) -> str | None:
    """Return a human-readable reason when the adapter cannot be bypassed."""
    if not isinstance(adapter, (LoRAAdapter, LoHaAdapter, LoKrAdapter)):
        return f"不支持的适配器类型 {type(adapter).__name__}"
    cls = type(adapter)
    if cls.bypass_forward is not WeightAdapterBase.bypass_forward:
        return "该适配器需要自定义 bypass_forward"
    if cls.g is not WeightAdapterBase.g:
        return "该适配器带输出变换 g()"
    if cls.h is WeightAdapterBase.h:
        return "该适配器没有 h() 前向分支"

    weights = getattr(adapter, "weights", None)
    if not isinstance(weights, (tuple, list)):
        return "适配器权重格式未知"

    dora_index = next(
        (index for (adapter_cls, index) in _DORA_INDEX.items() if isinstance(adapter, adapter_cls)),
        None,
    )
    if dora_index is not None and len(weights) > dora_index and weights[dora_index] is not None:
        return "DoRA (dora_scale) 不能按前向分支调度"
    if isinstance(adapter, LoRAAdapter) and len(weights) > 5 and weights[5] is not None:
        return "reshape_weight 不能按前向分支调度"
    return None


def _validate_lora_shape(
    adapter: WeightAdapterBase,
    info: _ModuleInfo,
    module: nn.Module,
    offset: tuple[int, int, int] | None,
    label: str,
) -> None:
    """Validate shapes from module attrs; never read ``module.weight``."""
    if not isinstance(adapter, LoRAAdapter):
        return
    up, down = adapter.weights[0], adapter.weights[1]
    if not (isinstance(up, torch.Tensor) and isinstance(down, torch.Tensor)):
        return
    if up.dim() != 2 or down.dim() != 2:
        return

    if info.is_conv:
        if info.in_channels is None or not info.kernel_size or down.dim() != 2:
            return
        in_dim = (
            int(info.in_channels)
            // max(1, int(info.kw_dict.get("groups", 1)))
            * int(math.prod(info.kernel_size))
        )
        out_dim = info.out_channels
    else:
        in_dim = getattr(module, "in_features", None)
        out_dim = getattr(module, "out_features", None)

    if offset is not None and out_dim is not None:
        _dim, start, length = offset
        if start + length > int(out_dim):
            raise RuntimeError(
                f"LoRA 切片 {label} 超出模块输出维度 {int(out_dim)}"
            )
    if in_dim is None or out_dim is None:
        return

    target_out = int(out_dim) if offset is None else int(offset[2])
    if int(up.shape[0]) != target_out or int(down.shape[1]) != int(in_dim):
        raise RuntimeError(
            f"LoRA 适配器 {label} 形状不匹配："
            f"up={tuple(up.shape)}, down={tuple(down.shape)}, "
            f"目标 out={target_out}, in={int(in_dim)}。"
            "请确认 LoRA 与底模是同一架构。"
        )


def _map_tensors(value: Any, fn) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, fn) for v in value)
    if isinstance(value, list):
        return [_map_tensors(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    return value


def _cast_adapter(
    adapter: WeightAdapterBase, device: torch.device, dtype: torch.dtype
) -> None:
    weights = getattr(adapter, "weights", None)
    if weights is None:
        return

    def move(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point():
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    adapter.weights = _map_tensors(weights, move)


def _cast_adapters(
    adapters: list[WeightAdapterBase], device: torch.device, dtype: torch.dtype
) -> None:
    seen = set()
    for adapter in adapters:
        if id(adapter) in seen:
            continue
        seen.add(id(adapter))
        _cast_adapter(adapter, device, dtype)


def _load_lora_file(path: str) -> dict:
    path = os.path.abspath(path)
    try:
        stat = os.stat(path)
    except OSError as exc:
        raise RuntimeError(f"无法读取 LoRA 文件: {path}") from exc
    key = (path, stat.st_mtime_ns, stat.st_size)

    with _lora_cache_lock:
        cached = _lora_cache.get(key)
        if cached is not None:
            _lora_cache.move_to_end(key)
            return cached

    LOG.info("加载 LoRA 文件: %s", path)
    data = comfy.utils.load_torch_file(path, safe_load=True)

    with _lora_cache_lock:
        _lora_cache[key] = data
        _lora_cache.move_to_end(key)
        while len(_lora_cache) > _LORA_CACHE_MAX:
            _lora_cache.popitem(last=False)
    return data


def _model_lora_key_map(model) -> dict:
    """Build (and cache) the LoRA key map for one base model object.

    ``model_lora_keys_unet`` walks the model state dict.  On MOSS-Image int8
    dynamic patchers that walk is the expensive part of an otherwise cheap
    loader, and the architecture does not change between executions, so keep
    a small weakref-keyed cache.
    """
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("输入的 MODEL 无效：缺少底层模型对象")

    # Attribute cache survives a custom-node hot reload, where the module-level
    # weakref cache below would otherwise be recreated.
    cached = getattr(base, "_psl_lora_key_map", None)
    if isinstance(cached, dict):
        return cached

    cache_key = id(base)
    with _model_key_map_lock:
        entry = _model_key_map_cache.get(cache_key)
        if entry is not None:
            ref, key_map = entry
            if ref() is base:
                _model_key_map_cache.move_to_end(cache_key)
                LOG.debug("LoRA key map 缓存命中")
                return key_map
            _model_key_map_cache.pop(cache_key, None)

    key_map = comfy.lora.model_lora_keys_unet(base, {})
    try:
        setattr(base, "_psl_lora_key_map", key_map)
    except Exception:
        LOG.debug("无法在模型对象上缓存 LoRA key map", exc_info=True)

    try:
        ref = weakref.ref(base)
    except TypeError:
        return key_map

    with _model_key_map_lock:
        _model_key_map_cache[cache_key] = (ref, key_map)
        _model_key_map_cache.move_to_end(cache_key)
        while len(_model_key_map_cache) > _MODEL_KEY_MAP_CACHE_MAX:
            _model_key_map_cache.popitem(last=False)
    return key_map


def _compute_lora_delta(
    adapter: WeightAdapterBase,
    x: torch.Tensor,
    output: torch.Tensor,
    offset: tuple[int, int, int] | None,
    label: str,
    out_dim: int,
    compute_dtype: torch.dtype,
    strength: float,
) -> torch.Tensor | None:
    """Evaluate one adapter branch and return a delta ready for ``output``."""
    if strength == 0.0:
        return None
    x_compute = x if x.dtype == compute_dtype else x.to(dtype=compute_dtype)
    try:
        delta = adapter.h(x_compute, output)
    except Exception:
        LOG.exception("LoRA 前向分支计算失败：%s", label)
        raise
    if not isinstance(delta, torch.Tensor):
        raise RuntimeError(f"LoRA 适配器 {label} 的 h() 未返回张量")

    if offset is None:
        expected = tuple(output.shape)
    else:
        _dim, _start, length = offset
        expected_shape = list(output.shape)
        expected_shape[out_dim] = length
        expected = tuple(expected_shape)
    if tuple(delta.shape) != expected:
        raise RuntimeError(
            "LoRA 前向分支输出形状与目标切片不匹配："
            f"{tuple(delta.shape)} vs {expected}（{label}）。"
            "请确认 LoRA 与底模架构/层类型一致。"
        )

    # Scale in the selected compute dtype, then cast for the addition.
    if strength != 1.0:
        delta = delta * strength
    if delta.dtype != output.dtype:
        delta = delta.to(dtype=output.dtype)
    return delta


class _ModuleLoRAHook:
    """Forward branch for one module, with optional fused-projection slices.

    ``specs`` is a list of ``(adapter, offset, label)``.  A tuple key from
    ``model_lora_keys_unet`` such as
    ``("...proj.qkv.weight", (0, start, length))`` means the adapter was trained
    for output rows ``start:start+length`` of the fused weight, so its delta is
    added to that slice of the module output.
    """

    def __init__(
        self,
        module_name: str,
        specs: list[tuple[WeightAdapterBase, tuple[int, int, int] | None, str]],
        compute_dtype: torch.dtype,
    ):
        self.module_name = module_name
        self.specs = list(specs)
        self.compute_dtype = compute_dtype
        self.strength = 0.0
        self._module: nn.Module | None = None
        self._handle = None
        self._skip_warned = False
        self._is_conv = False

    def _labels(self) -> str:
        return ", ".join(label for _, _, label in self.specs)

    def attach(self, patcher) -> None:
        if self._handle is not None:
            return
        try:
            module = patcher.get_model_object(self.module_name)
        except Exception as exc:
            raise RuntimeError(
                f"LoRA 模块 {self.module_name} 在当前模型上不存在"
            ) from exc

        info = _module_info(module)
        if info is None:
            raise RuntimeError(
                f"LoRA 模块 {self.module_name}（{type(module).__name__}）不支持前向注入"
            )

        for adapter, offset, label in self.specs:
            _validate_lora_shape(adapter, info, module, offset, label)

        self._is_conv = info.is_conv
        for adapter, _offset, _label in self.specs:
            adapter.multiplier = 1.0
            adapter.is_conv = info.is_conv
            adapter.conv_dim = info.conv_dim
            adapter.kernel_size = info.kernel_size
            adapter.in_channels = info.in_channels
            adapter.out_channels = info.out_channels
            adapter.kw_dict = info.kw_dict

        self._module = module
        self._handle = module.register_forward_hook(self._forward_hook)
        LOG.debug(
            "挂载 LoRA 前向分支: %s（%d 个适配器） -> %s",
            self.module_name,
            len(self.specs),
            type(module).__name__,
        )

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._module = None

    def _warn_skip(self, reason: str) -> None:
        if self._skip_warned:
            return
        self._skip_warned = True
        LOG.warning(
            "LoRA 前向分支跳过 %s：%s。该层不会应用 LoRA。",
            self._labels(),
            reason,
        )

    def _forward_hook(self, module: nn.Module, args, output):
        if self.strength == 0.0:
            return None
        if not args or not isinstance(args[0], torch.Tensor):
            self._warn_skip("输入不是单个张量")
            return None
        if not isinstance(output, torch.Tensor):
            self._warn_skip("模块输出不是单个张量")
            return None
        x = args[0]
        if not x.is_floating_point():
            self._warn_skip("输入不是浮点张量")
            return None

        compute_dtype = self.compute_dtype
        if x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)

        out_dim = 1 if self._is_conv else -1
        deltas = []
        for adapter, offset, label in self.specs:
            delta = _compute_lora_delta(
                adapter,
                x,
                output,
                offset,
                label,
                out_dim,
                compute_dtype,
                self.strength,
            )
            if delta is not None:
                deltas.append((offset, delta))

        if not deltas:
            return None
        if len(deltas) == 1 and deltas[0][0] is None:
            return output + deltas[0][1]

        # Fused q/k/v adapters share one module; clone once and add all slices.
        result = output.clone()
        for offset, delta in deltas:
            if offset is None:
                result += delta
            else:
                _dim, start, length = offset
                result.narrow(out_dim, start, length).add_(delta)
        return result


class _MergedModuleHook:
    """One forward hook for all dynamic LoRAs that target the same module."""

    def __init__(
        self,
        module_name: str,
        specs: list[tuple[Any, WeightAdapterBase, tuple[int, int, int] | None, str]],
    ):
        self.module_name = module_name
        self.specs = list(specs)
        self._module: nn.Module | None = None
        self._handle = None
        self._skip_warned = False
        self._is_conv = False

    def _labels(self) -> str:
        return ", ".join(label for _, _, _, label in self.specs)

    def attach(self, patcher) -> None:
        if self._handle is not None:
            return
        try:
            module = patcher.get_model_object(self.module_name)
        except Exception as exc:
            raise RuntimeError(
                f"LoRA 模块 {self.module_name} 在当前模型上不存在"
            ) from exc

        info = _module_info(module)
        if info is None:
            raise RuntimeError(
                f"LoRA 模块 {self.module_name}（{type(module).__name__}）不支持前向注入"
            )

        self._is_conv = info.is_conv
        for _state, adapter, offset, label in self.specs:
            adapter.multiplier = 1.0
            adapter.is_conv = info.is_conv
            adapter.conv_dim = info.conv_dim
            adapter.kernel_size = info.kernel_size
            adapter.in_channels = info.in_channels
            adapter.out_channels = info.out_channels
            adapter.kw_dict = info.kw_dict
            _validate_lora_shape(adapter, info, module, offset, label)

        self._module = module
        self._handle = module.register_forward_hook(self._forward_hook)
        LOG.debug(
            "合并挂载 LoRA 前向分支: %s（%d 个适配器） -> %s",
            self.module_name,
            len(self.specs),
            type(module).__name__,
        )

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._module = None

    def _warn_skip(self, reason: str) -> None:
        if self._skip_warned:
            return
        self._skip_warned = True
        LOG.warning(
            "合并 LoRA 前向分支跳过 %s：%s。该层不会应用这些 LoRA。",
            self._labels(),
            reason,
        )

    def _forward_hook(self, module: nn.Module, args, output):
        if not args or not isinstance(args[0], torch.Tensor):
            self._warn_skip("输入不是单个张量")
            return None
        if not isinstance(output, torch.Tensor):
            self._warn_skip("模块输出不是单个张量")
            return None
        x = args[0]
        if not x.is_floating_point():
            self._warn_skip("输入不是浮点张量")
            return None

        out_dim = 1 if self._is_conv else -1
        deltas = []
        for state, adapter, offset, label in self.specs:
            delta = _compute_lora_delta(
                adapter,
                x,
                output,
                offset,
                label,
                out_dim,
                state.compute_dtype,
                float(state.current_strength),
            )
            if delta is not None:
                deltas.append((offset, delta))

        if not deltas:
            return None
        if len(deltas) == 1 and deltas[0][0] is None:
            return output + deltas[0][1]

        # One clone for all LoRAs on this module, including fused q/k/v slices.
        result = output.clone()
        for offset, delta in deltas:
            if offset is None:
                result += delta
            else:
                _dim, start, length = offset
                result.narrow(out_dim, start, length).add_(delta)
        return result


class _ChainCoordinator:
    """Merge all dynamic LoRAs in one MODEL chain onto shared module hooks."""

    def __init__(self, states: list[Any]):
        unique = []
        seen = set()
        for state in states:
            if id(state) in seen:
                continue
            seen.add(id(state))
            unique.append(state)
        self.states = unique
        self.hooks: list[_MergedModuleHook] = []
        self.active = False
        self._attached_model = None

    def _grouped_specs(self) -> dict[str, list[tuple[Any, WeightAdapterBase, Any, str]]]:
        groups: dict[str, list[tuple[Any, WeightAdapterBase, Any, str]]] = {}
        for state in self.states:
            for hook in state.hooks:
                for adapter, offset, label in hook.specs:
                    groups.setdefault(hook.module_name, []).append(
                        (state, adapter, offset, label)
                    )
        return groups

    def attach(self, patcher) -> None:
        model = getattr(patcher, "model", None)
        if self.active:
            if model is self._attached_model:
                return
            self._detach_hooks()

        # A state may have been active alone before this chain was formed.
        for state in self.states:
            state._deactivate(restore_adapters=False)
            state.current_strength = float(state.schedule[0])

        device = getattr(patcher, "load_device", None)
        if not isinstance(device, torch.device):
            device = comfy.model_management.get_torch_device()

        hooks: list[_MergedModuleHook] = []
        try:
            for state in self.states:
                _cast_adapters(state.adapters, device, state.compute_dtype)
            for module_name, specs in self._grouped_specs().items():
                hook = _MergedModuleHook(module_name, specs)
                hook.attach(patcher)
                hooks.append(hook)
        except Exception:
            for hook in hooks:
                hook.detach()
            for state in self.states:
                state._restore_adapters()
            raise

        self.hooks = hooks
        self.active = True
        self._attached_model = model
        LOG.info(
            "合并 %d 个动态 LoRA：%d 个模块，%d 个适配器",
            len(self.states),
            len(hooks),
            sum(len(hook.specs) for hook in hooks),
        )

    def _detach_hooks(self) -> None:
        for hook in self.hooks:
            hook.detach()
        self.hooks = []
        self.active = False
        self._attached_model = None

    def cleanup(self, patcher) -> None:
        if not self.active:
            return
        if getattr(patcher, "model", None) is not self._attached_model:
            return
        self._detach_hooks()

    def detach(self, patcher, unpatch_all=True) -> None:
        if self.active:
            self._detach_hooks()
        for state in self.states:
            state._restore_adapters()


def _chain_states(patcher) -> list[Any]:
    try:
        chain = patcher.get_attachment(_CHAIN_ATTACHMENT_KEY)
    except Exception:
        return []
    if not isinstance(chain, dict):
        return []
    states = chain.get("states")
    return list(states) if isinstance(states, list) else []


def _new_chain(states: list[Any]) -> dict:
    return {"states": list(states), "coordinator": None}


def _chain_coordinator(patcher, create=False):
    try:
        chain = patcher.get_attachment(_CHAIN_ATTACHMENT_KEY)
    except Exception:
        return None
    if not isinstance(chain, dict):
        return None
    states = chain.get("states")
    if not isinstance(states, list) or len(states) <= 1:
        return None
    coordinator = chain.get("coordinator")
    if coordinator is None and create:
        coordinator = _ChainCoordinator(states)
        chain["coordinator"] = coordinator
    return coordinator


class _ScheduleState:
    """Per-execution schedule, forward hooks and sampler-step wrapper."""

    def __init__(
        self,
        hooks: list[_ModuleLoRAHook],
        adapters: list[WeightAdapterBase],
        schedule: list[float],
        compute_dtype: torch.dtype,
        manual: bool,
    ):
        self.hooks = hooks
        self.adapters = adapters
        self.schedule = tuple(schedule)
        self.compute_dtype = compute_dtype
        self.manual = manual
        self.current_strength = float(self.schedule[0])

        self._active = False
        self._attached_model = None
        self._no_patcher_warned = False
        self._logged = False
        self._sigmas_ref = None
        self._sigmas_cpu: torch.Tensor | None = None
        self._fallback_step = 0
        self._last_sigma: float | None = None
        # Keep the loader's CPU tensors so detach can drop the GPU copies
        # without making a second CPU copy.
        self._originals = [
            (adapter, getattr(adapter, "weights", None)) for adapter in adapters
        ]

    # ---- lazy attach / cleanup ------------------------------------------------
    def _ensure_active(self, patcher) -> None:
        model = getattr(patcher, "model", None)
        if self._active:
            if model is self._attached_model:
                return
            self._deactivate(restore_adapters=True)
        try:
            attach_started = time.perf_counter()
            for hook in self.hooks:
                hook.attach(patcher)
            attach_done = time.perf_counter()
            device = getattr(patcher, "load_device", None)
            if not isinstance(device, torch.device):
                device = comfy.model_management.get_torch_device()
            _cast_adapters(self.adapters, device, self.compute_dtype)
            cast_done = time.perf_counter()
            for hook in self.hooks:
                hook.strength = self.current_strength
        except Exception:
            for hook in self.hooks:
                hook.detach()
            self._restore_adapters()
            raise
        self._active = True
        self._attached_model = model
        log = LOG.info if cast_done - attach_started > 0.25 else LOG.debug
        log(
            "LoRA Step Schedule 挂载耗时：attach %.3fs，cast %.3fs，总计 %.3fs",
            attach_done - attach_started,
            cast_done - attach_done,
            cast_done - attach_started,
        )

    def _deactivate(self, restore_adapters: bool) -> None:
        if self._active:
            for hook in self.hooks:
                hook.detach()
            self._active = False
            self._attached_model = None
        if restore_adapters:
            self._restore_adapters()

    def _restore_adapters(self) -> None:
        for adapter, weights in self._originals:
            adapter.weights = weights

    def on_pre_run(self, patcher) -> None:
        """Called on the patcher about to sample, before the first step."""
        # A path that never reaches predict_noise should at least use the
        # first schedule value instead of the previous run's last value.
        self.current_strength = float(self.schedule[0])
        coordinator = _chain_coordinator(patcher, create=True)
        if coordinator is not None:
            coordinator.attach(patcher)
        else:
            self._ensure_active(patcher)

    def on_cleanup(self, patcher) -> None:
        """Sampling finished: detach hooks, keep adapters resident for re-use."""
        coordinator = _chain_coordinator(patcher)
        if coordinator is not None:
            coordinator.cleanup(patcher)
            return
        if not self._active:
            return
        if getattr(patcher, "model", None) is not self._attached_model:
            return
        self._deactivate(restore_adapters=False)

    def on_detach(self, patcher, unpatch_all=True) -> None:
        """Model is being unloaded: release the GPU copies of the adapters."""
        coordinator = _chain_coordinator(patcher)
        if coordinator is not None:
            coordinator.detach(patcher, unpatch_all)
            return
        if self._active and getattr(patcher, "model", None) is self._attached_model:
            self._deactivate(restore_adapters=True)
        else:
            self._restore_adapters()

    # ---- per-step scheduling -------------------------------------------------
    def predict_noise_wrapper(self, executor, *args, **kwargs):
        guider = getattr(executor, "class_obj", None)
        patcher = getattr(guider, "model_patcher", None)
        if patcher is not None:
            coordinator = _chain_coordinator(patcher, create=True)
            if coordinator is not None:
                coordinator.attach(patcher)
            else:
                self._ensure_active(patcher)
        elif not self._active and not self._no_patcher_warned:
            self._no_patcher_warned = True
            LOG.warning(
                "PREDICT_NOISE wrapper 未拿到 guider.model_patcher，"
                "无法挂载 LoRA；请确认采样器使用 CFGGuider 路径。"
            )
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        model_options = args[2] if len(args) > 2 else kwargs.get("model_options")
        self._update(timestep, model_options)
        return executor(*args, **kwargs)

    def _set_strength(self, strength: float) -> None:
        self.current_strength = float(strength)
        for hook in self.hooks:
            hook.strength = self.current_strength

    def _sigmas_for(self, model_options: Any) -> torch.Tensor | None:
        if not isinstance(model_options, dict):
            return None
        transformer_options = model_options.get("transformer_options")
        if not isinstance(transformer_options, dict):
            return None
        sigmas = transformer_options.get("sample_sigmas")
        if sigmas is None:
            return None

        if torch.is_tensor(sigmas):
            if sigmas is not self._sigmas_ref:
                self._sigmas_ref = sigmas
                self._sigmas_cpu = (
                    sigmas.detach().to(device="cpu", dtype=torch.float32).flatten()
                )
            return self._sigmas_cpu
        if isinstance(sigmas, (list, tuple)):
            self._sigmas_ref = None
            self._sigmas_cpu = torch.tensor(
                [float(value) for value in sigmas], dtype=torch.float32
            )
            return self._sigmas_cpu
        return None

    def _step_index(self, sigma: float | None, model_options: Any) -> int:
        sigmas = self._sigmas_for(model_options)
        if sigmas is not None and sigmas.numel() > 0 and sigma is not None:
            # sample_sigmas is descending; count the values above the current
            # sigma.  A model call at sigmas[i] therefore maps to step i.
            # The final sigma (usually 0) would map one past the last real
            # step; clamp so the schedule never consumes more than the actual
            # step count.
            actual_steps = max(1, int(sigmas.numel()) - 1)
            index = int((sigmas > sigma).sum().item())
            return min(index, actual_steps - 1)

        if sigma is None:
            index = self._fallback_step
            self._fallback_step += 1
            return index

        if self._last_sigma is None or sigma > self._last_sigma + 1e-9:
            self._fallback_step = 0
        elif abs(sigma - self._last_sigma) > 1e-9:
            self._fallback_step += 1
        self._last_sigma = sigma
        return self._fallback_step

    def _update(self, timestep: Any, model_options: Any) -> None:
        sigma = _sigma_value(timestep)
        step = self._step_index(sigma, model_options)
        self._set_strength(_strength_for_step(list(self.schedule), step))

        if not self._logged:
            self._logged = True
            self._log_first_run(model_options)

    def _log_first_run(self, model_options: Any) -> None:
        sigmas = self._sigmas_for(model_options)
        if isinstance(model_options, dict) and "multigpu_clones" in model_options:
            LOG.warning(
                "检测到多 GPU 并行采样：按步强度只挂到主模型上，"
                "其余 GPU 分片可能仍按初始强度计算。"
            )
        if sigmas is None or sigmas.numel() == 0:
            LOG.warning(
                "没有找到 sample_sigmas，改按模型调用次数递增步号；"
                "自定义采样器可能不准确。"
            )
            return
        actual_steps = max(0, int(sigmas.numel()) - 1)
        points = len(self.schedule)
        if actual_steps > points:
            detail = f"实际 {actual_steps} 步 > 列表 {points} 点，末尾保持"
        elif actual_steps < points:
            detail = f"实际 {actual_steps} 步 < 列表 {points} 点，取前 {actual_steps} 点"
        else:
            detail = f"实际 {actual_steps} 步 = 列表 {points} 点"
        LOG.info(
            "LoRA 按步调度：%s，%s；强度 %.4f -> %.4f",
            "手动列表" if self.manual else "曲线生成",
            detail,
            self.schedule[0],
            self.schedule[-1],
        )


class PSLLoRAStepSchedule(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return io.Schema(
            node_id="PSLLoRAStepSchedule",
            display_name="LoRA Step Schedule (Bypass)",
            category="loaders",
            search_aliases=[
                "lora",
                "step",
                "schedule",
                "curve",
                "bypass",
                "强度",
                "曲线",
                "按步",
                "逐步",
            ],
            is_experimental=True,
            description=(
                "按采样步给 LoRA 施加不同强度：不合并权重，而是把 LoRA 分支注入模块前向，"
                "每一步更新一次强度。曲线生成或手动列表二选一；实际步数多于列表时末尾保持，"
                "少于列表时取前缀。默认用 bf16 计算 LoRA 分支。"
                "支持标准 LoRA / LoHa / LoKr 及 fused qkv 输出切片键（MOSS-Image 等）；"
                "DoRA、reshape、diff/set 补丁会明确报错。"
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    display_name="MODEL",
                    tooltip="要挂载 LoRA 的扩散模型。底模权重不会被修改。",
                ),
                io.Combo.Input(
                    "lora_name",
                    display_name="LoRA",
                    options=lora_list,
                    default="None",
                    tooltip="选择 LoRA 文件；None = 原样输出 MODEL。",
                ),
                io.Float.Input(
                    "start_strength",
                    display_name="开始强度",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="曲线第一个点的强度（手动列表非空时仅用于参考曲线）。",
                ),
                io.Float.Input(
                    "end_strength",
                    display_name="结束强度",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="曲线最后一个点的强度（手动列表非空时仅用于参考曲线）。",
                ),
                io.Combo.Input(
                    "curve",
                    display_name="曲线",
                    options=list(CURVE_NAMES),
                    default="linear",
                    tooltip=(
                        "单调曲线：linear 线性；ease_in 二次加速；ease_out 二次减速；"
                        "ease_in_out smoothstep；cosine 余弦。"
                        "全部从开始强度平滑过渡到结束强度，开始强度可以大于结束强度。"
                    ),
                ),
                io.Int.Input(
                    "points",
                    display_name="曲线点数",
                    default=20,
                    min=2,
                    max=200,
                    step=1,
                    tooltip=(
                        "曲线生成多少个强度点，也就是自动列表的长度。"
                        "手动列表非空时它只影响面板里的虚线参考曲线。"
                    ),
                ),
                io.String.Input(
                    "strength_list",
                    display_name="强度列表",
                    default="",
                    multiline=False,
                    placeholder="留空=按曲线生成；也可填 1.0, 0.8, 0.5",
                    tooltip=(
                        "手动强度列表，逗号/空格/换行分隔；非空时优先于开始/结束强度和曲线。"
                        "节点下方面板拖动圆点会实时写回这里；点“清空”恢复曲线生成。"
                    ),
                ),
                io.Combo.Input(
                    "precision",
                    display_name="LoRA 精度",
                    options=list(PRECISION_TO_DTYPE.keys()),
                    default="bf16",
                    tooltip=(
                        "LoRA 分支（down/up 矩阵乘加）的计算精度，不影响底模精度。"
                        "bf16 省显存且动态范围大；fp16 更快但极端值可能溢出；fp32 最稳但更慢、更占显存。"
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    "model",
                    display_name="MODEL",
                    tooltip="挂载了按步 LoRA 前向分支的模型。",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        lora_name="None",
        start_strength=1.0,
        end_strength=1.0,
        curve="linear",
        points=20,
        strength_list="",
        precision="bf16",
    ) -> io.NodeOutput:
        started_at = time.perf_counter()
        schedule, manual = _resolve_schedule(
            start_strength, end_strength, curve, points, strength_list
        )
        precision = str(precision).lower()
        if precision not in PRECISION_TO_DTYPE:
            raise ValueError(
                f"未知精度 {precision!r}；可选: {', '.join(PRECISION_TO_DTYPE)}"
            )

        if lora_name in (None, "", "None"):
            LOG.info("LoRA Step Schedule：未选择 LoRA，原样输出 MODEL")
            return io.NodeOutput(model)

        compute_dtype = PRECISION_TO_DTYPE[precision]
        if precision == "bf16" and torch.cuda.is_available():
            device = comfy.model_management.get_torch_device()
            if device.type == "cuda" and not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "当前 GPU 不支持 bf16。请把 LoRA 精度改为 fp16 或 fp32。"
                )

        if getattr(model, "model", None) is None:
            raise RuntimeError("输入的 MODEL 无效：缺少底层模型对象")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise RuntimeError(f"找不到 LoRA 文件: {lora_name}")

        lora_data = _load_lora_file(lora_path)
        lora_data = comfy.lora_convert.convert_lora(lora_data)
        file_ready_at = time.perf_counter()
        key_map = _model_lora_key_map(model)
        key_map_ready_at = time.perf_counter()
        loaded = comfy.lora.load_lora(lora_data, key_map, log_missing=False)
        adapters_ready_at = time.perf_counter()
        if not loaded:
            raise RuntimeError(
                f"LoRA {lora_name!r} 没有任何权重匹配当前模型架构。"
                "请确认 LoRA 与底模属于同一模型系列。"
            )

        module_specs: dict[
            str, list[tuple[WeightAdapterBase, tuple[int, int, int] | None, str]]
        ] = {}
        configs: dict[str, tuple] = {}
        unsupported: list[str] = []
        for key, patch in loaded.items():
            if isinstance(patch, WeightAdapterBase):
                problem = _adapter_problem(patch)
                if problem is not None:
                    unsupported.append(f"{key!r}（{type(patch).__name__}: {problem}）")
                    continue
                try:
                    target, offset = _parse_patch_key(key)
                except ValueError as exc:
                    unsupported.append(f"{key!r}（{exc}）")
                    continue
                module_name = target[: -len(".weight")]
                if offset is None:
                    label = target
                else:
                    label = f"{target}[{offset[1]}:{offset[1] + offset[2]}]"
                module_specs.setdefault(module_name, []).append(
                    (patch, offset, label)
                )
            elif isinstance(patch, tuple) and len(patch) == 2 and patch[0] == "config":
                configs[key] = patch
            else:
                unsupported.append(f"{key!r}（{type(patch).__name__} 补丁）")

        if unsupported:
            preview = "；".join(unsupported[:8])
            if len(unsupported) > 8:
                preview += f"；… 共 {len(unsupported)} 项"
            raise RuntimeError(
                "该 LoRA 含有无法按步调度的补丁，已拒绝加载以免结果不一致："
                f"{preview}。请改用核心 Load LoRA 节点，或使用纯 LoRA 文件。"
            )
        if not module_specs and not configs:
            raise RuntimeError(f"LoRA {lora_name!r} 没有可用的权重。")
        classified_at = time.perf_counter()

        all_adapters: list[WeightAdapterBase] = [
            spec[0] for specs in module_specs.values() for spec in specs
        ]
        upstream_states = _chain_states(model)

        new_model = model.clone()
        clone_at = time.perf_counter()
        if configs:
            new_model.add_patches(configs, 1.0)

        hooks = [
            _ModuleLoRAHook(module_name, specs, compute_dtype)
            for module_name, specs in module_specs.items()
        ]
        state = _ScheduleState(
            hooks,
            all_adapters,
            schedule,
            compute_dtype,
            manual,
        )

        tag = uuid.uuid4().hex
        new_model.add_wrapper_with_key(
            WrappersMP.PREDICT_NOISE,
            f"psl_lora_step_schedule_wrap_{tag}",
            state.predict_noise_wrapper,
        )
        new_model.add_callback_with_key(
            CallbacksMP.ON_PRE_RUN,
            f"psl_lora_step_schedule_pre_{tag}",
            state.on_pre_run,
        )
        new_model.add_callback_with_key(
            CallbacksMP.ON_CLEANUP,
            f"psl_lora_step_schedule_clean_{tag}",
            state.on_cleanup,
        )
        new_model.add_callback_with_key(
            CallbacksMP.ON_DETACH,
            f"psl_lora_step_schedule_detach_{tag}",
            state.on_detach,
        )
        chain = _new_chain([*upstream_states, state])
        new_model.set_attachments(_CHAIN_ATTACHMENT_KEY, chain)
        registered_at = time.perf_counter()

        LOG.info(
            "LoRA Step Schedule 链：本节点为第 %d 个动态 LoRA（上游 %d 个）",
            len(chain["states"]),
            len(upstream_states),
        )
        LOG.info(
            "LoRA Step Schedule 耗时：文件/转换 %.3fs，key_map %.3fs，load_lora %.3fs，"
            "分组 %.3fs，clone %.3fs，注册 %.3fs，总计 %.3fs",
            file_ready_at - started_at,
            key_map_ready_at - file_ready_at,
            adapters_ready_at - key_map_ready_at,
            classified_at - adapters_ready_at,
            clone_at - classified_at,
            registered_at - clone_at,
            registered_at - started_at,
        )
        LOG.info(
            "LoRA Step Schedule：%s，%d 个模块 / %d 个适配器，%d 点%s列表，精度 %s",
            lora_name,
            len(hooks),
            len(all_adapters),
            len(schedule),
            "手动" if manual else "自动",
            precision,
        )
        return io.NodeOutput(new_model)
