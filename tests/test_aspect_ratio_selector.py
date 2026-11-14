"""Unit tests for AspectRatioSelectorNode, runnable without a ComfyUI server.

    cd ComfyUI
    ../python/python.exe custom_nodes/psl_comfyui_utilities/tests/test_aspect_ratio_selector.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

MODULE_PATH = REPO_ROOT / "nodes" / "aspect_ratio_selector.py"
spec = importlib.util.spec_from_file_location("psl_aspect_ratio_selector", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

Node = module.AspectRatioSelectorNode


def _run(**kwargs):
    out = Node.execute(**kwargs)
    return out.result, out.ui


def test_schema_defaults():
    schema = Node.GET_SCHEMA()
    assert schema.node_id == "AspectRatioSelectorNode"
    assert [output.id for output in schema.outputs] == ["width", "height", "actual_megapixels", "image"]
    # 位置敏感：前端按位置恢复 widgets_values，新 widget 必须排在 aspect_data_json 之后，
    # 否则旧工作流的比例 JSON 会被灌进 upscale_method
    assert [item.id for item in schema.inputs] == [
        "image", "total_megapixels", "min_step", "aspect_data_json", "upscale_method", "crop",
    ]
    inputs = {item.id: item.as_dict() for item in schema.inputs}
    assert inputs["total_megapixels"]["step"] == 16.0
    assert inputs["min_step"]["step"] == 8
    assert inputs["image"]["optional"] is True
    assert inputs["upscale_method"]["default"] == "lanczos"
    assert inputs["upscale_method"]["options"] == ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    assert inputs["crop"]["default"] == "disabled"
    assert inputs["crop"]["options"] == ["disabled", "center"]


def test_manual_ratio_without_image():
    result, ui = _run(
        total_megapixels=100.0,
        min_step=8,
        aspect_data_json=json.dumps({"w": 16, "h": 9}),
    )
    w, h, mp, image = result
    assert (w, h) == (1336, 752)
    assert abs(mp - 100.4672) < 1e-9
    assert image is None
    assert ui is None


def test_input_ratio_uses_image_size():
    image = torch.zeros(1, 480, 640, 3)
    result, ui = _run(
        image=image,
        total_megapixels=4.0,
        min_step=8,
        aspect_data_json=json.dumps({"useInput": True, "w": 1, "h": 9}),
    )
    w, h, _, out_image = result
    # 4:3 输入图，总像素 4 万 + 步长 8 → 232×176（JSON 里的 1:9 必须被忽略）
    assert (w, h) == (232, 176)
    assert tuple(out_image.shape) == (1, 176, 232, 3)
    assert ui == {"psl_input_size": [640, 480]}


def test_input_ratio_falls_back_without_image():
    result, ui = _run(
        total_megapixels=4.0,
        min_step=8,
        aspect_data_json=json.dumps({"useInput": True, "w": 1, "h": 1}),
    )
    w, h, _, image = result
    assert (w, h) == (200, 200)
    assert image is None
    assert ui is None


def test_image_keeps_input_ratio():
    # 输出宽高按选中比例算（1:1 → 200×200），图片却保持输入图 4:3 → 232×176
    image = torch.zeros(1, 480, 640, 3)
    result, ui = _run(
        image=image,
        total_megapixels=4.0,
        min_step=8,
        aspect_data_json=json.dumps({"w": 1, "h": 1}),
    )
    w, h, _, out_image = result
    assert (w, h) == (200, 200)
    assert tuple(out_image.shape) == (1, 176, 232, 3)
    assert ui == {"psl_input_size": [640, 480]}


def test_invalid_json_falls_back_to_16_9():
    result, _ = _run(total_megapixels=1.0, min_step=8, aspect_data_json="{not json")
    assert result[:2] == (136, 72)


def test_half_step_rounds_like_frontend():
    # 400 万、步长 32、1:1：2000/32 = 62.5，前端 Math.round → 63（2016），
    # 后端若用银行家舍入会得到 62（1984），这里锁死两者一致
    result, _ = _run(
        total_megapixels=400.0,
        min_step=32,
        aspect_data_json=json.dumps({"w": 1, "h": 1}),
    )
    assert result[:2] == (2016, 2016)


def test_crop_center_vs_stretch():
    # 输入 640x480（4:3）。总像素 4 万 + 步长 48 → 图片目标 240×192（1.25），
    # 与输入比例有偏差：center 先裁掉左右各 20 列再缩放，disabled 直接拉伸
    cols = torch.arange(640, dtype=torch.float32).view(1, 1, 640, 1)
    image = cols.expand(1, 480, 640, 3) / 640.0
    data = json.dumps({"w": 2, "h": 1})   # 决定的是 width/height 输出，与图片无关

    result, _ = _run(
        image=image, total_megapixels=4.0, min_step=48,
        upscale_method="bilinear", crop="center", aspect_data_json=data,
    )
    center = result[3]
    assert tuple(center.shape) == (1, 192, 240, 3)
    assert abs(float(center[0, 0, 0, 0]) - 20.75 / 640) < 0.01
    assert abs(float(center[0, 0, 239, 0]) - 618.25 / 640) < 0.01

    result, _ = _run(
        image=image, total_megapixels=4.0, min_step=48,
        upscale_method="bilinear", crop="disabled", aspect_data_json=data,
    )
    stretch = result[3]
    assert tuple(stretch.shape) == (1, 192, 240, 3)
    assert float(stretch[0, 0, 0, 0]) < 0.01
    assert float(stretch[0, 0, 239, 0]) > 0.99


if __name__ == "__main__":
    test_schema_defaults()
    test_manual_ratio_without_image()
    test_input_ratio_uses_image_size()
    test_input_ratio_falls_back_without_image()
    test_image_keeps_input_ratio()
    test_invalid_json_falls_back_to_16_9()
    test_half_step_rounds_like_frontend()
    test_crop_center_vs_stretch()
    print("aspect ratio selector tests passed")
