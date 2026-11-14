"""Value Curve — 把曲线或手动列表输出成纯数组字符串。

前端 ui/value_curve.js 提供与 LoRA Step Schedule 相同的曲线面板：
拖动圆点会把列表写回 strength_list（手动模式），点“清空”恢复按曲线生成。
"""

from comfy_api.latest import io

from .value_curve_core import CURVE_NAMES, format_values, resolve_values


class PSLValueCurve(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PSLValueCurve",
            display_name="Value Curve (Array String)",
            category="utils/curve",
            search_aliases=[
                "curve",
                "value curve",
                "array",
                "list",
                "string",
                "曲线",
                "数值",
                "数组",
                "序列",
            ],
            description=(
                "数值曲线：按开始/结束强度、曲线和点数生成数值序列，输出为不带方括号的"
                "纯数组字符串（如 0.3, 0.6, 1.0）。也可以手填强度列表；下方曲线面板拖动圆点"
                "会实时写回列表，点“清空”恢复按曲线生成。输出可直接接到接受数字列表字符串的输入。"
            ),
            inputs=[
                io.Float.Input(
                    "start_strength",
                    display_name="开始强度",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="曲线第一个点的数值（手动列表非空时仅用于参考曲线）。",
                ),
                io.Float.Input(
                    "end_strength",
                    display_name="结束强度",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="曲线最后一个点的数值（手动列表非空时仅用于参考曲线）。",
                ),
                io.Combo.Input(
                    "curve",
                    display_name="曲线",
                    options=list(CURVE_NAMES),
                    default="linear",
                    tooltip=(
                        "单调曲线：linear 线性；ease_in 二次加速；ease_out 二次减速；"
                        "ease_in_out smoothstep；cosine 余弦。"
                    ),
                ),
                io.Int.Input(
                    "points",
                    display_name="曲线点数",
                    default=20,
                    min=2,
                    max=200,
                    step=1,
                    tooltip="曲线生成多少个数值点；手动列表非空时只影响面板里的虚线参考曲线。",
                ),
                io.String.Input(
                    "strength_list",
                    display_name="数值列表",
                    default="",
                    multiline=False,
                    placeholder="留空=按曲线生成；也可填 1.0, 0.8, 0.5",
                    tooltip=(
                        "手动数值列表，逗号/空格/换行分隔；非空时优先于开始/结束强度和曲线。"
                        "节点下方面板拖动圆点会实时写回这里；点“清空”恢复曲线生成。"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    "values",
                    display_name="数组字符串",
                    tooltip="逗号+空格分隔的数值列表，如 0.35, 0.42, 0.5（无方括号）",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        start_strength=1.0,
        end_strength=1.0,
        curve="linear",
        points=20,
        strength_list="",
    ):
        values, _ = resolve_values(
            start_strength, end_strength, curve, points, strength_list
        )
        return io.NodeOutput(format_values(values))
