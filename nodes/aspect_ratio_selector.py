import json
import math
from comfy_api.latest import io


class AspectRatioSelectorNode(io.ComfyNode):
    """可视化分辨率选择器 — JS 前端拖拽矩形选比例，Python 后端计算整数 W×H 输出。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AspectRatioSelectorNode",
            display_name="Aspect Ratio Selector",
            category="image/aspect_ratio",
            inputs=[
                io.Float.Input(
                    "total_megapixels",
                    display_name="总像素数（万）",
                    default=100.0,
                    min=1.0,
                    max=10000.0,
                    step=1.0,
                    tooltip="目标总像素数，单位万。例如 100 = 100万像素 = 1 Megapixel",
                ),
                io.Int.Input(
                    "min_step",
                    display_name="最小步长",
                    default=8,
                    min=1,
                    max=64,
                    tooltip="输出的宽高必须是此值的整数倍。ComfyUI 中常设为 8 以适配 VAE 编码",
                ),
                io.String.Input(
                    "aspect_data_json",
                    display_name="比例数据",
                    optional=True,
                    tooltip="JS 前端写入的比例数据，格式 {\"w\": number, \"h\": number}",
                ),
            ],
            outputs=[
                io.Int.Output(
                    "width",
                    display_name="宽度",
                    tooltip="计算出的整数宽度，保证是最小步长的倍数",
                ),
                io.Int.Output(
                    "height",
                    display_name="高度",
                    tooltip="计算出的整数高度，保证是最小步长的倍数",
                ),
                io.Float.Output(
                    "actual_megapixels",
                    display_name="实际总像素（万）",
                    tooltip="取整后的实际总像素数（万），与设置值可能略有偏差",
                ),
            ],
        )

    @classmethod
    def execute(cls, total_megapixels=100.0, min_step=8, aspect_data_json="{}"):
        # 解析 JS 传来的宽高比
        try:
            data = json.loads(aspect_data_json) if aspect_data_json else {}
        except json.JSONDecodeError:
            data = {}

        aspect_w = data.get("w", 16)
        aspect_h = data.get("h", 9)

        total_px = total_megapixels * 10000

        # 从宽高比和总像素推导 W×H
        # w / h = aspect_w / aspect_h  =>  w = ratio * h
        # w * h = total_px  =>  ratio * h^2 = total_px  =>  h = sqrt(total_px / ratio)
        ratio = aspect_w / aspect_h
        base_h = math.sqrt(total_px / ratio)
        base_w = total_px / base_h

        # 取整到 min_step 的整数倍
        w = round(base_w / min_step) * min_step
        h = round(base_h / min_step) * min_step

        # 确保不小于 min_step
        w = max(w, min_step)
        h = max(h, min_step)

        actual_px = w * h
        actual_mp = actual_px / 10000

        return (w, h, round(actual_mp, 4))
