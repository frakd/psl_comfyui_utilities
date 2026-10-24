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


def test_stable_anchor_spreads_transient_overexposure():
    torch.manual_seed(5)
    reference = _video(12)
    target = reference.clone()
    target[2:4] = (target[2:4] * 1.20).clamp(0, 1)

    out = _run(reference, target, stat_mode="stable_anchor")[-12:]
    input_luma = _stats(target)[:, 0]
    output_luma = _stats(out)[:, 0]
    neighbors = [0, 1, 4, 5]

    input_excess = float(input_luma[2:4].max() - input_luma[neighbors].mean())
    output_excess = float(output_luma[2:4].max() - output_luma[neighbors].mean())

    assert input_excess > 0.03
    assert output_excess < input_excess * 0.65, (input_excess, output_excess)
    # 峰的能量被摊到相邻帧, 而不是只把峰值压暗
    assert output_luma[1] > input_luma[1] + 0.003
    assert output_luma[4] > input_luma[4] + 0.003


def _luma_std(video):
    rows = []
    for frame in video:
        rows.append(_TO_SPACE(frame).reshape(-1, 3)[:, 0].std())
    return torch.stack(rows)


def test_scene_cut_cap_applies_to_all_modes():
    torch.manual_seed(12)
    reference = _video(10)
    first_scene = _cast(_video(6))
    second_scene = torch.flip(_cast(_video(10)), dims=[2])
    target = torch.cat([first_scene, second_scene], dim=0)

    for mode in ("dual_anchor", "stable_anchor", "frozen", "rolling"):
        out = _run(
            reference, target, stat_mode=mode,
            window_size=1, fade_out=20, ref_anchor_frames=1,
        )[-16:]
        post_cut = _stats(out)[6:, 0] - _stats(target)[6:, 0]
        assert float(post_cut.abs().max()) < 0.002, (mode, float(post_cut.abs().max()))


def test_scene_cut_anchors_all_modes_use_first_scene():
    torch.manual_seed(13)
    reference = _video(10)
    first_scene = _cast(_video(6))
    second_scene = torch.flip(_cast(_video(10)), dims=[2])
    target = torch.cat([first_scene, second_scene], dim=0)

    for mode in ("dual_anchor", "stable_anchor", "frozen", "rolling"):
        kwargs = dict(
            stat_mode=mode, window_size=1, fade_out=20, ref_anchor_frames=1,
        )
        full = _run(reference, target, **kwargs)[-16:][:6]
        head = _run(reference, target[:6], **kwargs)[-6:]
        diff = _stats(full)[:, 0] - _stats(head)[:, 0]
        assert float(diff.abs().max()) < 0.02, (mode, float(diff.abs().max()))


def test_scene_cut_detection_uses_content():
    torch.manual_seed(10)
    first_scene = _video(6)
    second_scene = torch.flip(_video(10), dims=[2])
    sequence = torch.cat([first_scene, second_scene], dim=0)
    cuts = Node._scene_cuts(sequence, "cpu")

    assert bool(cuts[5])

    brightness_only = torch.cat([first_scene, (first_scene[:6] * 0.60).clamp(0, 1)], dim=0)
    assert not bool(Node._scene_cuts(brightness_only, "cpu").any())


def test_scene_cut_limits_target_anchor_and_correction():
    torch.manual_seed(11)
    reference = _video(10)
    first_scene = _cast(_video(6))
    second_scene = torch.flip(_cast(_video(10)), dims=[2])
    target = torch.cat([first_scene, second_scene], dim=0)
    kwargs = dict(stat_mode="stable_anchor", window_size=1, fade_out=20, ref_anchor_frames=1)

    full = _run(reference, target, **kwargs)[-16:]
    head = _run(reference, target[:6], **kwargs)[-6:]

    assert float((_stats(full[:6])[:, 0] - _stats(head)[:, 0]).abs().max()) < 0.01
    post_cut = _stats(full)[6:, 0] - _stats(target)[6:, 0]
    assert float(post_cut.abs().max()) < 0.002


def test_stable_anchor_preserves_frame_contrast():
    torch.manual_seed(9)
    reference = _video(12)
    reference = ((reference - 0.5) * 0.60 + 0.5).clamp(0, 1)
    target = _video(12)

    out = _run(reference, target, stat_mode="stable_anchor")[-12:]
    input_std = _luma_std(target)
    output_std = _luma_std(out)

    ratio = output_std / input_std.clamp_min(1e-6)
    assert float((ratio - 1.0).abs().max()) < 0.05


def test_stable_anchor_caps_positive_luma_shift():
    x = torch.full((8, 8, 3), 0.5)
    y = x.clone()
    y[..., 0] += 0.10

    capped = Node._cap_luma_shift(x, y, "oklab", 0.10, 0.03, "cpu")
    shift = float(capped[..., 0].mean() - x[..., 0].mean())

    assert 0.0 < shift <= 0.031


def test_stable_anchor_spreads_first_frame_flash():
    torch.manual_seed(6)
    reference = _video(3)
    target = reference.clone()
    target[0] = (target[0] * 1.06).clamp(0, 1)

    out = _run(reference, target, stat_mode="stable_anchor")[-3:]
    input_luma = _stats(target)[:, 0]
    output_luma = _stats(out)[:, 0]
    input_excess = float(input_luma[0] - input_luma[1:].mean())
    output_excess = float(output_luma[0] - output_luma[1:].mean())

    assert input_excess > 0.01
    assert output_excess < input_excess * 0.45, (input_excess, output_excess)


def test_stable_anchor_does_not_smooth_across_first_frame_cut():
    torch.manual_seed(7)
    target = _video(6)
    target[1:] = (target[1:] * 0.70).clamp(0, 1)

    out = _run(target, target, stat_mode="stable_anchor")[-6:]
    input_luma = _stats(target)[:, 0]
    output_luma = _stats(out)[:, 0]

    assert float((output_luma[1:] - input_luma[1:]).abs().max()) < 0.002


def test_cut_detection_ignores_short_flash():
    curve = torch.tensor([0.593] * 2 + [0.677] * 2 + [0.593] * 8)

    assert not bool(Node._cut_flags(curve, 0.04).any())


def test_cut_detection_finds_hard_cut():
    curve = torch.tensor([0.60, 0.60, 0.45, 0.45, 0.45, 0.45])

    assert bool(Node._cut_flags(curve, 0.04)[1])


def test_temporal_smoothing_respects_cut():
    curve = torch.tensor([0.60, 0.60, 0.45, 0.45, 0.45, 0.45])
    cuts = Node._cut_flags(curve, 0.04)
    baseline = Node._temporal_baseline(curve, cuts, 0.04)
    mask = Node._peak_mask(curve - baseline, 0.004, cuts)

    assert not bool(mask.any())
    smoothed = Node._gaussian_smooth(curve, cuts)
    assert torch.allclose(smoothed[2:], curve[2:], atol=1e-6)


def test_seam_rule_uses_reference_end():
    assert not Node._first_frame_is_seam(0.62, 0.60, 0.004)
    assert Node._first_frame_is_seam(0.60, 0.65, 0.004)
    assert Node._first_frame_is_seam(0.60, 0.60, 0.004)


def test_stable_anchor_uses_seam_local_target_anchor():
    torch.manual_seed(8)
    reference = _video(20)
    target = _cast(_video(60))
    target[30:] = (target[30:] * 1.30).clamp(0, 1)

    full = _run(reference, target, stat_mode="stable_anchor")[-60:][:3]
    head = _run(reference, target[:3], stat_mode="stable_anchor")[-3:]
    full_luma = _stats(full)[:, 0]
    head_luma = _stats(head)[:, 0]

    assert float((full_luma - head_luma).abs().max()) < 0.02


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
    test_stable_anchor_spreads_transient_overexposure()
    test_scene_cut_detection_uses_content()
    test_scene_cut_limits_target_anchor_and_correction()
    test_scene_cut_cap_applies_to_all_modes()
    test_scene_cut_anchors_all_modes_use_first_scene()
    test_stable_anchor_preserves_frame_contrast()
    test_stable_anchor_caps_positive_luma_shift()
    test_stable_anchor_spreads_first_frame_flash()
    test_stable_anchor_does_not_smooth_across_first_frame_cut()
    test_cut_detection_ignores_short_flash()
    test_cut_detection_finds_hard_cut()
    test_temporal_smoothing_respects_cut()
    test_seam_rule_uses_reference_end()
    test_stable_anchor_uses_seam_local_target_anchor()
    test_stable_anchor_cross_channel_has_effect()
    print("video match v2 tests passed")
