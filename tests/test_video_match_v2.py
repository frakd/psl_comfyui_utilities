"""Unit tests for PSLAdvancedVideoMatchV2, runnable without a ComfyUI server.

    cd ComfyUI
    python custom_nodes/psl_comfyui_utilities/tests/test_video_match_v2.py
"""

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

MODULE_PATH = REPO_ROOT / "nodes" / "video_match_v2.py"
spec = importlib.util.spec_from_file_location("psl_video_match_v2", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

Node = module.PSLAdvancedVideoMatchV2

_WORKFLOW_DEFAULTS = {
    "window_size": 3,
    "fade_out": 20,
    "match_intensity": 1.0,
    "highlight_protect": 0.85,
    "ref_anchor_frames": 8,
    "stat_smoothing": 0.85,
    "stat_mode": "dual_anchor",
    "color_space": "oklab",
    "match_mode": "affine_cdf",
    "cdf_weight": 0.50,
    "cross_channel": 0.10,
    "cdf_max_shift": 0.08,
    "cdf_chroma_guard": 0.90,
}


def _run(reference, target, **overrides):
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        return Node.execute(
            reference_video=reference,
            target_video=target,
            **{**_WORKFLOW_DEFAULTS, **overrides},
        ).result[0]
    finally:
        torch.cuda.is_available = original


_TO_SPACE, _, _ = Node._space_fns("oklab", "cpu")
_H = _W = 24
_YS, _XS = torch.meshgrid(torch.linspace(0, 1, _H), torch.linspace(0, 1, _W), indexing="ij")
_SCENE = torch.stack([
    0.20 + 0.60 * _XS,
    0.25 + 0.50 * _YS,
    0.20 + 0.40 * (1 - _XS) * _YS,
], dim=-1)
_SCENE[6:14, 4:18] = torch.tensor([0.75, 0.20, 0.35])


def _video(frames):
    noise = torch.randn(frames, _H, _W, 3) * 0.005
    return (_SCENE.unsqueeze(0) + noise).clamp(0, 1)


def _cast(video):
    matrix = torch.tensor([
        [1.12, -0.10, 0.04],
        [0.08, 0.90, -0.06],
        [0.02, 0.10, 0.96],
    ])
    offset = torch.tensor([-0.04, 0.06, 0.02])
    return (video.reshape(-1, 3) @ matrix.t() + offset).reshape(video.shape).clamp(0, 1)


def _stats(video):
    rows = []
    for frame in video:
        sm = _TO_SPACE(frame).reshape(-1, 3)
        rows.append(torch.stack([sm[:, 0].mean(), torch.sqrt(sm[:, 1] ** 2 + sm[:, 2] ** 2).mean()]))
    return torch.stack(rows)


def test_schema_defaults_match_workflow():
    schema = Node.GET_SCHEMA()

    assert schema.node_id == "PSLAdvancedVideoMatchV2"
    assert [output.id for output in schema.outputs] == ["image"]
    defaults = {item.id: item.as_dict().get("default") for item in schema.inputs}
    for name, expected in _WORKFLOW_DEFAULTS.items():
        assert defaults[name] == expected, (name, defaults[name], expected)
    modes = {item.id: item.as_dict() for item in schema.inputs}["stat_mode"]
    assert "stable_anchor" in modes["options"]


def test_execute_preserves_reference_prefix():
    torch.manual_seed(0)
    reference = torch.rand(4, 16, 20, 3)
    target = torch.rand(8, 16, 20, 3)

    out = _run(reference, target)

    assert out.shape == (12, 16, 20, 3), out.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert torch.equal(out[:4], reference)


def test_empty_target_returns_reference():
    reference = torch.rand(3, 8, 8, 3)
    out = _run(reference, torch.empty(0, 8, 8, 3))

    assert torch.equal(out, reference)


def test_mismatched_size_is_reported():
    reference = torch.rand(3, 8, 8, 3)
    try:
        _run(reference, torch.rand(4, 16, 16, 3))
    except ValueError as error:
        assert "share H/W" in str(error)
    else:
        raise AssertionError("mismatched H/W was silently accepted")


def test_stable_anchor_preserves_exposure_ramp():
    torch.manual_seed(1)
    reference = _video(10)
    ramp = torch.linspace(0.75, 1.15, 20).view(-1, 1, 1, 1)
    target = (_cast(_video(20)) * ramp).clamp(0, 1)

    out = _run(reference, target, stat_mode="stable_anchor")[-20:]
    input_stats = _stats(target)
    output_stats = _stats(out)

    assert float((output_stats[:, 0] - input_stats[:, 0]).abs().max()) < 0.05
    input_span = float(input_stats[-1, 0] - input_stats[0, 0])
    output_span = float(output_stats[-1, 0] - output_stats[0, 0])
    assert abs(output_span - input_span) < 0.02, (input_span, output_span)


def test_stable_anchor_ignores_faded_tail():
    torch.manual_seed(2)
    reference = _video(10)
    target = _cast(_video(20))
    target[-8:] = target[-8:] * torch.linspace(1.0, 0.08, 8).view(-1, 1, 1, 1)

    out = _run(reference, target, stat_mode="stable_anchor")[-20:]
    input_luma = _stats(target)[:, 0]
    output_luma = _stats(out)[:, 0]

    assert float((output_luma - input_luma).abs().max()) < 0.05


def test_stable_anchor_suppresses_dissolve_saturation_spike():
    torch.manual_seed(3)
    reference = _video(10)
    target = _cast(_video(20))
    for i in (1, 2, 3):
        mean = target[i].mean(dim=(0, 1), keepdim=True)
        target[i] = mean + (target[i] - mean) * 0.08

    out = _run(reference, target, stat_mode="stable_anchor")[-20:]
    input_chroma = _stats(target)[1:4, 1]
    output_chroma = _stats(out)[1:4, 1]

    assert float((output_chroma - input_chroma).abs().max()) < 0.005


def test_stable_anchor_cross_channel_has_effect():
    torch.manual_seed(4)
    reference = _video(10)
    target = _cast(_video(20))

    low = _run(reference, target, stat_mode="stable_anchor", cross_channel=0.0)
    high = _run(reference, target, stat_mode="stable_anchor", cross_channel=1.0)

    assert not torch.equal(low, high)


if __name__ == "__main__":
    test_schema_defaults_match_workflow()
    test_execute_preserves_reference_prefix()
    test_empty_target_returns_reference()
    test_mismatched_size_is_reported()
    test_stable_anchor_preserves_exposure_ramp()
    test_stable_anchor_ignores_faded_tail()
    test_stable_anchor_suppresses_dissolve_saturation_spike()
    test_stable_anchor_cross_channel_has_effect()
    print("video match v2 tests passed")
