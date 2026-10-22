"""Unit tests for PSLLoRAStepSchedule, runnable without a ComfyUI server.

    cd ComfyUI
    ../python/python.exe custom_nodes/psl_comfyui_utilities/tests/test_lora_step_schedule.py
"""

import importlib.util
import math
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

MODULE_PATH = REPO_ROOT / "nodes" / "lora_step_schedule.py"
spec = importlib.util.spec_from_file_location("psl_lora_step_schedule", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

Node = module.PSLLoRAStepSchedule


def _close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def test_schema_defaults():
    schema = Node.GET_SCHEMA()
    assert schema.node_id == "PSLLoRAStepSchedule"
    assert [output.id for output in schema.outputs] == ["model"]
    inputs = {item.id: item.as_dict() for item in schema.inputs}
    assert inputs["precision"]["default"] == "bf16"
    assert inputs["precision"]["options"] == ["bf16", "fp16", "fp32"]
    assert inputs["curve"]["default"] == "linear"
    assert inputs["points"]["default"] == 20
    assert inputs["strength_list"]["default"] == ""


def test_curve_endpoints_and_monotonicity():
    for curve in module.CURVE_NAMES:
        assert _close(module._curve_value(curve, 0.0), 0.0)
        assert _close(module._curve_value(curve, 1.0), 1.0)
        previous = -math.inf
        for i in range(101):
            value = module._curve_value(curve, i / 100)
            assert value >= previous - 1e-12, (curve, i, value, previous)
            previous = value


def test_generate_schedule_uses_endpoints_and_point_count():
    values = module._generate_strengths(1.0, 0.0, "linear", 5)
    assert values == [1.0, 0.75, 0.5, 0.25, 0.0]

    values = module._generate_strengths(-0.5, 1.5, "cosine", 17)
    assert len(values) == 17
    assert _close(values[0], -0.5)
    assert _close(values[-1], 1.5)

    values = module._generate_strengths(2.0, 2.0, "ease_in_out", 7)
    assert all(_close(v, 2.0) for v in values)


def test_parse_strength_list_separators_and_brackets():
    assert module._parse_strength_list("") == []
    assert module._parse_strength_list("   ") == []
    assert module._parse_strength_list("1, 0.5; -2 3\n4") == [1.0, 0.5, -2.0, 3.0, 4.0]
    assert module._parse_strength_list("[1, 2，3、4]") == [1.0, 2.0, 3.0, 4.0]
    assert module._parse_strength_list("(0.2 0.8)") == [0.2, 0.8]


def test_parse_strength_list_rejects_bad_values():
    for text in ("1, foo", "1, nan", "1, inf", "1, 2 3x"):
        try:
            module._parse_strength_list(text)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad list accepted: {text!r}")


def test_parse_patch_key_supports_output_slices():
    target, offset = module._parse_patch_key("diffusion_model.x.weight")
    assert target == "diffusion_model.x.weight"
    assert offset is None

    target, offset = module._parse_patch_key(
        ("diffusion_model.qkv.weight", (0, 3072, 3072))
    )
    assert target == "diffusion_model.qkv.weight"
    assert offset == (0, 3072, 3072)

    for bad in (
        "diffusion_model.x",  # not a weight key
        ("diffusion_model.qkv.weight", (1, 0, 3072)),  # unsupported axis
        ("diffusion_model.qkv.weight", (0, 0, 3072), object()),  # function
        ("diffusion_model.qkv.weight", (0, 0)),  # bad offset tuple
    ):
        try:
            module._parse_patch_key(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad patch key accepted: {bad!r}")


def test_resolve_schedule_manual_wins():
    manual, is_manual = module._resolve_schedule(1.0, 0.0, "linear", 6, "0.2, 0.4")
    assert is_manual is True
    assert manual == [0.2, 0.4]

    auto, is_manual = module._resolve_schedule(1.0, 0.0, "linear", 6, "  ")
    assert is_manual is False
    assert len(auto) == 6


def test_strength_for_step_truncates_and_holds_tail():
    schedule = [0.1, 0.2, 0.3]
    assert _close(module._strength_for_step(schedule, -5), 0.1)
    assert _close(module._strength_for_step(schedule, 0), 0.1)
    assert _close(module._strength_for_step(schedule, 2), 0.3)
    assert _close(module._strength_for_step(schedule, 99), 0.3)


class DummyHook:
    def __init__(self):
        self.strength = 0.0


def _make_state(schedule):
    return module._ScheduleState(
        [DummyHook()], [], list(schedule), torch.float32, False
    )


def test_step_index_maps_actual_sigmas_and_holds_tail():
    state = _make_state([0.1, 0.2, 0.3])
    options = {
        "transformer_options": {
            "sample_sigmas": torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        }
    }
    for sigma, expected in ((1.0, 0.1), (0.75, 0.2), (0.5, 0.3), (0.25, 0.3)):
        state._update(torch.tensor(sigma), options)
        assert _close(state.current_strength, expected), (sigma, state.current_strength)
    assert _close(state.hooks[0].strength, state.current_strength)


def test_step_index_truncates_to_prefix_when_fewer_steps():
    state = _make_state([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    options = {
        "transformer_options": {
            "sample_sigmas": torch.tensor([1.0, 0.5, 0.0])
        }
    }
    state._update(torch.tensor(1.0), options)
    assert _close(state.current_strength, 0.1)
    state._update(torch.tensor(0.5), options)
    assert _close(state.current_strength, 0.2)
    # sigma 0 is the final boundary, not an extra schedule point
    state._update(torch.tensor(0.0), options)
    assert _close(state.current_strength, 0.2)


def test_step_index_falls_back_to_call_counter():
    state = _make_state([0.1, 0.2, 0.3])
    state._update(torch.tensor(1.0), {})
    assert _close(state.current_strength, 0.1)
    state._update(torch.tensor(0.5), {})
    assert _close(state.current_strength, 0.2)
    state._update(torch.tensor(0.5), {})
    assert _close(state.current_strength, 0.2)
    state._update(torch.tensor(0.2), {})
    assert _close(state.current_strength, 0.3)


def test_module_info_detects_linear_and_conv():
    linear = nn.Linear(4, 3, bias=False)
    info = module._module_info(linear)
    assert info is not None
    assert info.is_conv is False
    assert info.out_channels == 3

    conv = nn.Conv2d(4, 6, 3, padding=1)
    info = module._module_info(conv)
    assert info is not None
    assert info.is_conv is True
    assert info.kernel_size == (3, 3)
    assert info.in_channels == 4
    assert info.out_channels == 6
    assert info.kw_dict["padding"] == (1, 1)


class FakeAdapter:
    def __init__(self, delta_weight):
        self.weights = (torch.ones_like(delta_weight),)
        self.delta_weight = delta_weight
        self.multiplier = 1.0
        self.seen_dtypes = []

    def h(self, x, base_out):
        self.seen_dtypes.append(x.dtype)
        return torch.nn.functional.linear(x, self.delta_weight.to(x.dtype))


class FakePatcher:
    def __init__(self, module):
        self._module = module

    def get_model_object(self, name):
        assert name == "lin"
        return self._module


def test_forward_hook_adds_scaled_delta_and_detaches():
    torch.manual_seed(0)
    linear = nn.Linear(4, 3, bias=False)
    adapter = FakeAdapter(torch.full((3, 4), 0.5))
    hook = module._ModuleLoRAHook("lin", [(adapter, None, "lin.weight")], torch.float32)

    hook.attach(FakePatcher(linear))
    x = torch.randn(5, 4)
    base = linear(x).detach()
    hook.strength = 0.5
    with torch.no_grad():
        out = linear(x)
    expected = base + 0.5 * torch.nn.functional.linear(x, adapter.delta_weight)
    assert torch.allclose(out, expected, atol=1e-6), (out, expected)

    hook.detach()
    with torch.no_grad():
        after = linear(x)
    assert torch.allclose(after, base, atol=1e-6)


def test_forward_hook_uses_requested_compute_dtype():
    linear = nn.Linear(4, 3, bias=False)
    adapter = FakeAdapter(torch.zeros(3, 4))
    hook = module._ModuleLoRAHook("lin", [(adapter, None, "lin.weight")], torch.bfloat16)
    hook.attach(FakePatcher(linear))
    hook.strength = 1.0

    x = torch.randn(2, 4, dtype=torch.float32)
    with torch.no_grad():
        out = linear(x)
    assert out.dtype == torch.float32
    assert adapter.seen_dtypes[-1] == torch.bfloat16
    hook.detach()


def test_forward_hook_skips_zero_strength():
    linear = nn.Linear(4, 3, bias=False)
    adapter = FakeAdapter(torch.ones(3, 4))
    hook = module._ModuleLoRAHook("lin", [(adapter, None, "lin.weight")], torch.float32)
    hook.attach(FakePatcher(linear))
    hook.strength = 0.0
    x = torch.randn(2, 4)
    with torch.no_grad():
        out = linear(x)
    assert adapter.seen_dtypes == []
    hook.detach()


def test_module_hook_does_not_touch_weight():
    class WeightlessLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_features = 4
            self.out_features = 3

        @property
        def weight(self):
            raise AssertionError("attach must not access module.weight")

        def forward(self, x):
            raise AssertionError("forward should not be called")

    layer = WeightlessLinear()
    adapter = module.LoRAAdapter(
        set(),
        (
            torch.zeros(3, 2),
            torch.zeros(2, 4),
            1.0,
            None,
            None,
            None,
        ),
    )
    hook = module._ModuleLoRAHook(
        "lin", [(adapter, None, "lin.weight")], torch.float32
    )
    hook.attach(FakePatcher(layer))
    assert hook._handle is not None
    hook.detach()


def test_module_hook_adds_fused_output_slices():
    torch.manual_seed(6)
    linear = nn.Linear(4, 9, bias=False)
    deltas = [torch.randn(3, 4) for _ in range(3)]
    specs = [
        (FakeAdapter(deltas[i]), (0, i * 3, 3), f"qkv[{i * 3}:{i * 3 + 3}]")
        for i in range(3)
    ]
    hook = module._ModuleLoRAHook("qkv", specs, torch.float32)

    class FakeQKVPatcher:
        def get_model_object(self, name):
            assert name == "qkv"
            return linear

    hook.attach(FakeQKVPatcher())
    x = torch.randn(2, 4)
    base = linear(x).detach()
    hook.strength = 0.5
    with torch.no_grad():
        out = linear(x)

    expected = base.clone()
    for i, delta in enumerate(deltas):
        expected[:, i * 3 : (i + 1) * 3] += 0.5 * torch.nn.functional.linear(x, delta)
    assert torch.allclose(out, expected, atol=1e-6), (out, expected)
    hook.detach()


def test_adapter_problem_rejects_dora_and_reshape():
    up = torch.zeros(3, 4)
    down = torch.zeros(2, 4)
    plain = module.LoRAAdapter(set(), (up, down, 1.0, None, None, None))
    assert module._adapter_problem(plain) is None

    dora = module.LoRAAdapter(set(), (up, down, 1.0, None, torch.ones(3), None))
    assert "DoRA" in module._adapter_problem(dora)

    reshape = module.LoRAAdapter(set(), (up, down, 1.0, None, None, (3, 4)))
    assert "reshape" in module._adapter_problem(reshape)


def test_lazy_activation_cleanup_and_detach_restores_weights():
    linear = nn.Linear(4, 3, bias=False)
    adapter = FakeAdapter(torch.zeros(3, 4))
    hook = module._ModuleLoRAHook("lin", [(adapter, None, "lin.weight")], torch.float32)
    state = module._ScheduleState(
        [hook], [adapter], [0.25, 0.5], torch.float32, False
    )
    original = adapter.weights
    patcher = FakePatcher(linear)

    old_get_device = module.comfy.model_management.get_torch_device
    module.comfy.model_management.get_torch_device = lambda: torch.device("cpu")
    try:
        state.on_pre_run(patcher)
        assert hook._handle is not None
        assert adapter.weights is not original  # cast copy

        state.on_cleanup(patcher)
        assert hook._handle is None
        assert adapter.weights is not original  # kept resident for re-use

        state.on_pre_run(patcher)
        assert hook._handle is not None

        state.on_detach(patcher, False)
        assert hook._handle is None
        assert adapter.weights is original
    finally:
        module.comfy.model_management.get_torch_device = old_get_device


class FakeBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = nn.Linear(4, 3, bias=False)
        self.model_config = types.SimpleNamespace(unet_config={})


class FakeQKVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.layers = nn.ModuleList([nn.Module()])
        self.diffusion_model.layers[0].proj = nn.Module()
        self.diffusion_model.layers[0].proj.qkv = nn.Linear(4, 9, bias=False)
        self.model_config = types.SimpleNamespace(unet_config={})


def test_execute_end_to_end_with_real_patcher():
    from safetensors.torch import save_file
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrappersMP

    torch.manual_seed(3)
    base_model = FakeBaseModel()
    patcher = ModelPatcher(
        base_model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    base_weight = base_model.diffusion_model.weight.detach().clone()

    down = torch.randn(2, 4)
    up = torch.randn(3, 2)
    alpha = 1.5
    rank = down.shape[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        lora_path = Path(temp_dir) / "fake_lora.safetensors"
        save_file(
            {
                "diffusion_model.lora_down.weight": down,
                "diffusion_model.lora_up.weight": up,
                "diffusion_model.alpha": torch.tensor(alpha),
            },
            str(lora_path),
        )

        old_get_full_path = module.folder_paths.get_full_path
        old_get_device = module.comfy.model_management.get_torch_device
        module.folder_paths.get_full_path = (
            lambda folder, name: str(lora_path)
            if name == "fake_lora.safetensors"
            else old_get_full_path(folder, name)
        )
        module.comfy.model_management.get_torch_device = lambda: torch.device("cpu")
        try:
            out = Node.execute(
                model=patcher,
                lora_name="fake_lora.safetensors",
                start_strength=1.0,
                end_strength=0.0,
                curve="linear",
                points=2,
                strength_list="",
                precision="fp32",
            ).result[0]
            assert out is not patcher

            x = torch.randn(2, 4)
            base = torch.nn.functional.linear(x, base_weight)
            lora = torch.nn.functional.linear(
                torch.nn.functional.linear(x, down), up
            )
            delta = lora * (alpha / rank)

            out.pre_run()
            x = torch.randn(2, 4)
            base = torch.nn.functional.linear(x, base_weight)
            lora = torch.nn.functional.linear(
                torch.nn.functional.linear(x, down), up
            )
            delta = lora * (alpha / rank)

            with torch.no_grad():
                y = out.model.diffusion_model(x)
            assert torch.allclose(y, base + delta, atol=1e-6), (y, base + delta)

            wrappers = out.get_all_wrappers(WrappersMP.PREDICT_NOISE)
            assert len(wrappers) == 1
            options = {
                "transformer_options": {
                    "sample_sigmas": torch.tensor([1.0, 0.5, 0.0])
                }
            }

            class Executor:
                class_obj = types.SimpleNamespace(model_patcher=out)

                def __call__(self, *args, **kwargs):
                    return None

            wrappers[0](Executor(), x, torch.tensor(0.5), options, 0)
            with torch.no_grad():
                y_zero = out.model.diffusion_model(x)
            assert torch.allclose(y_zero, base, atol=1e-6), (y_zero, base)

            out.cleanup()
            with torch.no_grad():
                y_after = out.model.diffusion_model(x)
            assert torch.allclose(y_after, base, atol=1e-6)
        finally:
            module.folder_paths.get_full_path = old_get_full_path
            module.comfy.model_management.get_torch_device = old_get_device


def test_execute_end_to_end_with_fused_qkv_slices():
    from safetensors.torch import save_file
    from comfy.model_patcher import ModelPatcher

    torch.manual_seed(7)
    base_model = FakeQKVModel()
    patcher = ModelPatcher(
        base_model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    qkv = base_model.diffusion_model.layers[0].proj.qkv
    base_weight = qkv.weight.detach().clone()

    downs = [torch.randn(2, 4) for _ in range(3)]
    ups = [torch.randn(3, 2) for _ in range(3)]

    with tempfile.TemporaryDirectory() as temp_dir:
        lora_path = Path(temp_dir) / "fused_qkv.safetensors"
        tensors = {}
        for i, proj in enumerate(("q", "k", "v")):
            tensors[f"diffusion_model.layers.0.proj.{proj}.lora_A.weight"] = downs[i]
            tensors[f"diffusion_model.layers.0.proj.{proj}.lora_B.weight"] = ups[i]
        save_file(tensors, str(lora_path))

        old_get_full_path = module.folder_paths.get_full_path
        old_key_map = module.comfy.lora.model_lora_keys_unet
        module.folder_paths.get_full_path = (
            lambda folder, name: str(lora_path)
            if name == "fused_qkv.safetensors"
            else old_get_full_path(folder, name)
        )
        module.comfy.lora.model_lora_keys_unet = lambda model, key_map: {
            "diffusion_model.layers.0.proj.q": (
                "diffusion_model.layers.0.proj.qkv.weight",
                (0, 0, 3),
            ),
            "diffusion_model.layers.0.proj.k": (
                "diffusion_model.layers.0.proj.qkv.weight",
                (0, 3, 3),
            ),
            "diffusion_model.layers.0.proj.v": (
                "diffusion_model.layers.0.proj.qkv.weight",
                (0, 6, 3),
            ),
        }
        try:
            out = Node.execute(
                model=patcher,
                lora_name="fused_qkv.safetensors",
                start_strength=1.0,
                end_strength=1.0,
                curve="linear",
                points=2,
                strength_list="",
                precision="fp32",
            ).result[0]
            out.pre_run()
            x = torch.randn(2, 4)
            base = torch.nn.functional.linear(x, base_weight)
            expected = base.clone()
            for i in range(3):
                delta = torch.nn.functional.linear(
                    torch.nn.functional.linear(x, downs[i]), ups[i]
                )
                expected[:, i * 3 : (i + 1) * 3] += delta

            with torch.no_grad():
                y = out.model.diffusion_model.layers[0].proj.qkv(x)
            assert torch.allclose(y, expected, atol=1e-6), (y, expected)
            out.cleanup()
        finally:
            module.folder_paths.get_full_path = old_get_full_path
            module.comfy.lora.model_lora_keys_unet = old_key_map


def test_stacked_chain_merges_and_keeps_per_state_schedules():
    from safetensors.torch import save_file
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrappersMP

    torch.manual_seed(11)
    base_model = FakeBaseModel()
    patcher = ModelPatcher(
        base_model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    base_weight = base_model.diffusion_model.weight.detach().clone()

    down_a = torch.randn(2, 4)
    up_a = torch.randn(3, 2)
    down_b = torch.randn(2, 4)
    up_b = torch.randn(3, 2)

    with tempfile.TemporaryDirectory() as temp_dir:
        path_a = Path(temp_dir) / "chain_a.safetensors"
        path_b = Path(temp_dir) / "chain_b.safetensors"
        for path, down, up in ((path_a, down_a, up_a), (path_b, down_b, up_b)):
            save_file(
                {
                    "diffusion_model.lora_A.weight": down,
                    "diffusion_model.lora_B.weight": up,
                },
                str(path),
            )

        old_get_full_path = module.folder_paths.get_full_path
        paths = {"chain_a.safetensors": path_a, "chain_b.safetensors": path_b}
        module.folder_paths.get_full_path = (
            lambda folder, name: str(paths[name]) if name in paths else old_get_full_path(folder, name)
        )
        try:
            out_a = Node.execute(
                model=patcher,
                lora_name="chain_a.safetensors",
                start_strength=1.0,
                end_strength=0.0,
                curve="linear",
                points=2,
                strength_list="",
                precision="fp32",
            ).result[0]
            out_b = Node.execute(
                model=out_a,
                lora_name="chain_b.safetensors",
                start_strength=0.5,
                end_strength=0.5,
                curve="linear",
                points=2,
                strength_list="",
                precision="fp32",
            ).result[0]
            out_sibling = Node.execute(
                model=patcher,
                lora_name="chain_b.safetensors",
                start_strength=0.5,
                end_strength=0.5,
                curve="linear",
                points=2,
                strength_list="",
                precision="fp32",
            ).result[0]

            key = module._CHAIN_ATTACHMENT_KEY
            assert len(out_a.get_attachment(key)["states"]) == 1
            assert len(out_b.get_attachment(key)["states"]) == 2
            # Sibling branches must not inherit the other branch's LoRA.
            assert len(out_sibling.get_attachment(key)["states"]) == 1

            out_b.pre_run()
            coordinator = out_b.get_attachment(key)["coordinator"]
            assert coordinator is not None
            assert len(coordinator.hooks) == 1
            assert len(coordinator.hooks[0].specs) == 2

            x = torch.randn(2, 4)
            base = torch.nn.functional.linear(x, base_weight)
            delta_a = torch.nn.functional.linear(
                torch.nn.functional.linear(x, down_a), up_a
            )
            delta_b = torch.nn.functional.linear(
                torch.nn.functional.linear(x, down_b), up_b
            )

            with torch.no_grad():
                y = out_b.model.diffusion_model(x)
            assert torch.allclose(y, base + delta_a + delta_b * 0.5, atol=1e-6), y

            wrappers = out_b.get_all_wrappers(WrappersMP.PREDICT_NOISE)
            assert len(wrappers) == 2
            options = {
                "transformer_options": {
                    "sample_sigmas": torch.tensor([1.0, 0.5, 0.0])
                }
            }

            class Executor:
                class_obj = types.SimpleNamespace(model_patcher=out_b)

                def __call__(self, *args, **kwargs):
                    return None

            executor = Executor()
            for wrapper in wrappers:
                wrapper(executor, x, torch.tensor(0.5), options, 0)
            with torch.no_grad():
                y_step1 = out_b.model.diffusion_model(x)
            assert torch.allclose(y_step1, base + delta_b * 0.5, atol=1e-6), y_step1

            out_b.cleanup()
            with torch.no_grad():
                y_after = out_b.model.diffusion_model(x)
            assert torch.allclose(y_after, base, atol=1e-6)

            # A third state must join the same chain and still use one hook.
            out_c = Node.execute(
                model=out_b,
                lora_name="chain_a.safetensors",
                start_strength=1.0,
                end_strength=1.0,
                curve="linear",
                points=2,
                strength_list="",
                precision="fp32",
            ).result[0]
            assert len(out_c.get_attachment(key)["states"]) == 3
            out_c.pre_run()
            coordinator_c = out_c.get_attachment(key)["coordinator"]
            assert coordinator_c is not None
            assert len(coordinator_c.hooks) == 1
            assert len(coordinator_c.hooks[0].specs) == 3
            with torch.no_grad():
                y_c = out_c.model.diffusion_model(x)
            assert torch.allclose(
                y_c, base + delta_a * 2.0 + delta_b * 0.5, atol=1e-6
            ), y_c
            out_c.cleanup()
        finally:
            module.folder_paths.get_full_path = old_get_full_path


def test_stacked_chain_merges_fused_qkv_slices():
    from safetensors.torch import save_file
    from comfy.model_patcher import ModelPatcher

    torch.manual_seed(12)
    base_model = FakeQKVModel()
    patcher = ModelPatcher(
        base_model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    qkv = base_model.diffusion_model.layers[0].proj.qkv
    base_weight = qkv.weight.detach().clone()

    tensors_a = {}
    tensors_b = {}
    expected_down_a = []
    expected_up_a = []
    expected_down_b = []
    expected_up_b = []
    for i, proj in enumerate(("q", "k", "v")):
        down_a = torch.randn(2, 4)
        up_a = torch.randn(3, 2)
        down_b = torch.randn(2, 4)
        up_b = torch.randn(3, 2)
        tensors_a[f"diffusion_model.layers.0.proj.{proj}.lora_A.weight"] = down_a
        tensors_a[f"diffusion_model.layers.0.proj.{proj}.lora_B.weight"] = up_a
        tensors_b[f"diffusion_model.layers.0.proj.{proj}.lora_A.weight"] = down_b
        tensors_b[f"diffusion_model.layers.0.proj.{proj}.lora_B.weight"] = up_b
        expected_down_a.append(down_a)
        expected_up_a.append(up_a)
        expected_down_b.append(down_b)
        expected_up_b.append(up_b)

    with tempfile.TemporaryDirectory() as temp_dir:
        path_a = Path(temp_dir) / "qkv_a.safetensors"
        path_b = Path(temp_dir) / "qkv_b.safetensors"
        save_file(tensors_a, str(path_a))
        save_file(tensors_b, str(path_b))

        old_get_full_path = module.folder_paths.get_full_path
        old_key_map = module.comfy.lora.model_lora_keys_unet
        module.folder_paths.get_full_path = (
            lambda folder, name: str(path_a)
            if name == "qkv_a.safetensors"
            else str(path_b)
            if name == "qkv_b.safetensors"
            else old_get_full_path(folder, name)
        )
        module.comfy.lora.model_lora_keys_unet = lambda model, key_map: {
            f"diffusion_model.layers.0.proj.{proj}": (
                "diffusion_model.layers.0.proj.qkv.weight",
                (0, i * 3, 3),
            )
            for i, proj in enumerate(("q", "k", "v"))
        }
        try:
            out_a = Node.execute(
                model=patcher,
                lora_name="qkv_a.safetensors",
                start_strength=1.0,
                end_strength=1.0,
                curve="linear",
                points=2,
                strength_list="",
                precision="fp32",
            ).result[0]
            out_b = Node.execute(
                model=out_a,
                lora_name="qkv_b.safetensors",
                start_strength=0.5,
                end_strength=0.5,
                curve="linear",
                points=2,
                strength_list="",
                precision="fp32",
            ).result[0]

            out_b.pre_run()
            coordinator = out_b.get_attachment(module._CHAIN_ATTACHMENT_KEY)["coordinator"]
            assert coordinator is not None
            assert len(coordinator.hooks) == 1
            assert len(coordinator.hooks[0].specs) == 6

            x = torch.randn(2, 4)
            expected = torch.nn.functional.linear(x, base_weight).clone()
            for i in range(3):
                delta_a = torch.nn.functional.linear(
                    torch.nn.functional.linear(x, expected_down_a[i]),
                    expected_up_a[i],
                )
                delta_b = torch.nn.functional.linear(
                    torch.nn.functional.linear(x, expected_down_b[i]),
                    expected_up_b[i],
                )
                expected[:, i * 3 : (i + 1) * 3] += delta_a + delta_b * 0.5

            with torch.no_grad():
                y = out_b.model.diffusion_model.layers[0].proj.qkv(x)
            assert torch.allclose(y, expected, atol=1e-6), y
            out_b.cleanup()
        finally:
            module.folder_paths.get_full_path = old_get_full_path
            module.comfy.lora.model_lora_keys_unet = old_key_map


class FakeModel:
    def __init__(self):
        self.model = object()

    def clone(self):
        raise AssertionError("None LoRA must not clone the model")


def test_execute_none_lora_is_passthrough():
    model = FakeModel()
    out = Node.execute(model=model, lora_name="None").result[0]
    assert out is model


def test_execute_rejects_bad_strength_list_even_without_lora():
    try:
        Node.execute(model=FakeModel(), lora_name="None", strength_list="1, x")
    except ValueError as error:
        assert "非数字" in str(error)
    else:
        raise AssertionError("invalid strength list was accepted")


if __name__ == "__main__":
    test_schema_defaults()
    test_curve_endpoints_and_monotonicity()
    test_generate_schedule_uses_endpoints_and_point_count()
    test_parse_strength_list_separators_and_brackets()
    test_parse_strength_list_rejects_bad_values()
    test_parse_patch_key_supports_output_slices()
    test_resolve_schedule_manual_wins()
    test_strength_for_step_truncates_and_holds_tail()
    test_step_index_maps_actual_sigmas_and_holds_tail()
    test_step_index_truncates_to_prefix_when_fewer_steps()
    test_step_index_falls_back_to_call_counter()
    test_module_info_detects_linear_and_conv()
    test_forward_hook_adds_scaled_delta_and_detaches()
    test_forward_hook_uses_requested_compute_dtype()
    test_forward_hook_skips_zero_strength()
    test_module_hook_does_not_touch_weight()
    test_module_hook_adds_fused_output_slices()
    test_adapter_problem_rejects_dora_and_reshape()
    test_lazy_activation_cleanup_and_detach_restores_weights()
    test_execute_end_to_end_with_real_patcher()
    test_execute_end_to_end_with_fused_qkv_slices()
    test_stacked_chain_merges_and_keeps_per_state_schedules()
    test_stacked_chain_merges_fused_qkv_slices()
    test_execute_none_lora_is_passthrough()
    test_execute_rejects_bad_strength_list_even_without_lora()
    print("lora step schedule tests passed")
