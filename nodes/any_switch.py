"""Any Switch — 可视化多路开关：勾选第几路就输出第几路，或随机选一路。

输入是 Any 类型的可增长列表（io.Autogrow，核心 ComfyAnd/Or 同款机制）：
前端渲染器自动管理槽位（连满自动加、断开自动收），execute 收到的
``inputs`` 字典只包含**已连接**的输入，键是模板序数名（实测展开为
input0、input1…，0 基无下划线；兼容带组前缀的 inputs.input0 与旧式
input_0 两种写法——序数一律取尾部数字）。

选择语义：
- 手动模式：输出 selection 指向的那一路；该路未连接或未选择则输出 None
  （下游收到 None 等价于没连线，需要容忍 None）。
- 随机模式：输出随机一路。随机发生在**前端排队时**（serializeValue 把随机
  结果写进 selection）——selection 进 prompt，缓存签名每次都变，于是本节点
  与下游每次都重新运行，这正是随机模式要的效果；后端不掷骰子、只在
  selection 无效时（前端脚本缺失的 API 场景）兜底随机一次。

选中状态与模式都是普通 widget：进 prompt（参与缓存签名）也随工作流文件
持久化，刷新/换工作流不丢。前端 ui.psl_pick 回传实际输出的序数，用于
把勾选高亮同步到随机命中的那一路。

条目名（面板里那行「输入 N」）由前端双击改名，存 labels_json 隐藏 widget：
前端把它的 options.serialize 置 false，于是它不进 prompt（改名不该让下游
缓存失效），但 widget 值照旧随工作流文件保存。backend 只是声明了这个输入
以免前端脚本未生效时多传 kwargs 报错，正常收不到、也不消费。
"""

import random
import re

from comfy_api.latest import io

_MAX_INPUTS = 10
_ORDINAL_RE = re.compile(r"(\d+)$")


def _ordinal(key: str):
    """从 autogrow 键（input_3 / inputs.input_3 皆可）解析 0 基序数。"""
    m = _ORDINAL_RE.search(key)
    return int(m.group(1)) if m else None


class AnySwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.Autogrow.TemplatePrefix(
            input=io.AnyType.Input("input"),
            prefix="input",
            min=1,
            max=_MAX_INPUTS,
        )
        return io.Schema(
            node_id="AnySwitch",
            search_aliases=["switch", "any", "切换", "开关", "选择", "随机", "random"],
            display_name="Any Switch",
            category="utils/switch",
            description=(
                "可视化多路开关：面板里勾选第几路就输出第几路（单选），不选则输出"
                " None。输入是 Any 可增长列表——在画布上连满后会自动长出下一个输入"
                "口。模式可切换为随机：每次运行从已连接的输入里随机挑一路输出，"
                "且每次都会重新运行（随机发生在排队时，会改变缓存签名）。"
                "下游节点需要容忍 None。"
            ),
            is_output_node=True,
            inputs=[
                io.Autogrow.Input(
                    "inputs",
                    template=template,
                    optional=True,
                    tooltip="任意类型的可增长输入列表。面板里勾选哪路就输出哪路。",
                ),
                io.Combo.Input(
                    "mode",
                    options=["manual", "random"],
                    default="manual",
                    tooltip=(
                        "manual：手动勾选，输出勾选的那一路（由前端面板控制）。\n"
                        "random：每次运行随机输出一路已连接的输入；随机在排队时进行，"
                        "因此每次都会从本节点起重新运行。"
                    ),
                ),
                io.String.Input(
                    "selection",
                    default="",
                    tooltip=(
                        "当前选中的输入序数（0 基，字符串），由前端面板维护；"
                        "空串表示未选择。不要手填。"
                    ),
                ),
                io.String.Input(
                    "labels_json",
                    default="{}",
                    optional=True,
                    tooltip=(
                        "各路输入的条目名（JSON 对象，键是 0 基序数、值是用户双击改的名字），"
                        "由前端维护并随工作流文件持久化；前端会把它标记为不参与 prompt "
                        "序列化，所以后端正常收不到，仅作兼容保留。"
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[io.AnyType.Output("output", display_name="输出")],
        )

    @classmethod
    def execute(cls, inputs: dict = None, mode="manual", selection="", labels_json="{}"):
        del labels_json   # UI-only（条目名）：后端不消费，形参仅防前端脚本未生效时报多余 kwarg
        values = dict(inputs or {})

        pick = None   # 命中的 0 基序数；None 表示无有效选择
        if mode == "random" and not any(_ordinal(k) is not None and str(_ordinal(k)) == selection.strip()
                                        for k in values):
            # selection 无效（前端脚本缺失的 API 场景）才兜底随机；正常情况
            # 前端已在排队时把随机结果写进 selection，这里如实执行它。
            if values:
                keys = sorted(values.keys(), key=lambda k: (_ordinal(k) is None, _ordinal(k) or 0))
                pick = _ordinal(random.choice(keys))
        elif selection.strip().lstrip("-").isdigit():
            want = str(int(selection.strip()))
            for k in values:
                if str(_ordinal(k)) == want:
                    pick = int(want)
                    break

        out = None
        if pick is not None:
            for k, v in values.items():
                if _ordinal(k) == pick:
                    out = v
                    break
        # ui 值包一层 list：execution 合并多个 ui 时按 key 把值当可迭代展开
        return io.NodeOutput(out, ui={"psl_pick": [pick]})
