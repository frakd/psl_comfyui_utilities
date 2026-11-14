import json
import math

import comfy.utils
from comfy_api.latest import io

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
CROP_METHODS = ["disabled", "center"]
INPUT_ASPECT_UI_KEY = "psl_input_size"


def _parse_aspect_data(raw):
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _manual_ratio(data):
    """前端写入的宽高（自由/预设模式的比例；「同输入图」模式下是最近一次运行得到的比例）。"""
    try:
        w = float(data.get("w", 16))
        h = float(data.get("h", 9))
    except (TypeError, ValueError):
        return 16.0, 9.0
    if not (math.isfinite(w) and math.isfinite(h)) or w <= 0 or h <= 0:
        return 16.0, 9.0
    return w, h


def _compute_size(total_megapixels, min_step, aspect_w, aspect_h):
    total_px = float(total_megapixels) * 10000
    ratio = aspect_w / aspect_h
    base_h = math.sqrt(total_px / ratio)
    base_w = total_px / base_h
    # Python round() 是银行家舍入（62.5→62），前端 Math.round 是四舍五入（62.5→63）；
    # 统一成四舍五入，否则预览显示的尺寸会和实际输出差一个步长
    w = math.floor(base_w / min_step + 0.5) * min_step
    h = math.floor(base_h / min_step + 0.5) * min_step
    return max(w, min_step), max(h, min_step)


class AspectRatioSelectorNode(io.ComfyNode):
    """可视化分辨率选择器 — JS 前端拖拽矩形选比例，Python 后端计算整数 W×H 输出。

    可选接入一张图片：image 输出按总像素数等比缩放，保持输入图自身宽高比；
    节点选中的比例只决定 width/height 输出。连接图片后预设里会多出「同输入图」。
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AspectRatioSelectorNode",
            display_name="Aspect Ratio Selector",
            category="image/aspect_ratio",
            description=(
                "可视化分辨率选择器：拖矩形选宽高比，按总像素数换算成整数 W×H。"
                "可选接入图片：image 输出按总像素数等比缩放并保持输入图自身比例，与选中的比例无关；"
                "连接图片后预设列表会多出「同输入图」，只决定 width/height 输出的比例。"
                "未接图片时 image 输出为空。"
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    display_name="输入图（可选）",
                    optional=True,
                    tooltip="可选输入图。连接后 image 输出为它按『总像素数』等比缩放的结果（保持输入图自身比例），预设里会多出「同输入图」",
                ),
                io.Float.Input(
                    "total_megapixels",
                    display_name="总像素数（万）",
                    default=100.0,
                    min=1.0,
                    max=10000.0,
                    step=16.0,
                    tooltip="目标总像素数，单位万。例如 100 = 100万像素 = 1 Megapixel",
                ),
                io.Int.Input(
                    "min_step",
                    display_name="最小步长",
                    default=8,
                    min=1,
                    max=64,
                    step=8,
                    tooltip="输出的宽高必须是此值的整数倍。ComfyUI 中常设为 8 以适配 VAE 编码",
                ),
                # 新 widget 必须追加在 aspect_data_json 之后：前端按位置恢复
                # widgets_values，插在它前面会让旧工作流的比例 JSON 错位进新 widget
                io.String.Input(
                    "aspect_data_json",
                    display_name="比例数据",
                    optional=True,
                    tooltip="JS 前端写入的比例数据，不要手填",
                ),
                io.Combo.Input(
                    "upscale_method",
                    display_name="缩放算法",
                    options=UPSCALE_METHODS,
                    default="lanczos",
                    tooltip="缩放输入图时使用的插值算法，与官方缩放图像节点一致（未接输入图时隐藏）",
                ),
                io.Combo.Input(
                    "crop",
                    display_name="裁剪",
                    options=CROP_METHODS,
                    default="disabled",
                    tooltip="图片按输入图自身比例缩放到目标像素数；最小步长取整会带来细微比例差，disabled 直接拉伸，center 先按目标比例中心裁掉边缘再缩放",
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
                io.Image.Output(
                    "image",
                    display_name="图像",
                    tooltip="输入图按总像素数等比缩放的结果（保持输入图自身比例）；未连接输入图时为空",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image=None,
        total_megapixels=100.0,
        min_step=8,
        upscale_method="lanczos",
        crop="disabled",
        aspect_data_json="{}",
    ):
        data = _parse_aspect_data(aspect_data_json)

        input_w = input_h = 0
        if image is not None:
            input_h, input_w = int(image.shape[1]), int(image.shape[2])

        # 「同输入图」只在真的拿到图时生效；拿不到（未接线/上游被静音）就退回前端记录的最近比例
        if data.get("useInput") and input_w > 0 and input_h > 0:
            aspect_w, aspect_h = float(input_w), float(input_h)
        else:
            aspect_w, aspect_h = _manual_ratio(data)

        w, h = _compute_size(total_megapixels, min_step, aspect_w, aspect_h)

        out_image = None
        ui = None
        if image is not None:
            # 图片输出永远保持输入图自身比例，只把尺寸缩放到总像素数附近；
            # 节点选中的比例只决定 width/height 输出，不影响这张图
            img_w, img_h = _compute_size(total_megapixels, min_step, input_w, input_h)
            samples = image.movedim(-1, 1)
            out_image = comfy.utils.common_upscale(samples, img_w, img_h, upscale_method, crop).movedim(1, -1)
            ui = {INPUT_ASPECT_UI_KEY: [input_w, input_h]}

        return io.NodeOutput(w, h, round(w * h / 10000, 4), out_image, ui=ui)
