"""Text Container — 文本容器：按分隔符把一段文本切成若干条，点选决定 6 路 STRING 输出。

缓存正确性（与 Image Container / Refreshable Text Box 同源）：
- prompt 里与列表有关的字面量只有 selection_json（按输出槽顺序排列的选中槽索引）、
  manual_json（手动槽内容）和 upstream_mode（是否接入上游）。完整列表只活在前端的
  ui_state_json 里：前端用 serializeValue 固定提交空串，所以它在 prompt 里是一个
  不变的常量，不进缓存签名；完整状态本身随工作流文件持久化。上游接入期间
  manual_json 也恒为空数组，因此列表怎么刷新都不改 prompt。
- 因此「运行到此节点」刷新列表、点选、拖动行高都不会改变 prompt 字节，本节点与
  下游缓存不会因此失效。真正改变输出的操作（换选中项、改手动内容、换分隔符、
  换缺失槽档位）才改变 prompt —— 这是应该失效的。
- 上游文本走连线，不在 prompt 字面量里；上游变了由连线祖先签名负责失效。

输出语义（与 Image Container 一致）：
- 固定 6 路输出。列表可以超过 6 条（手动槽填满会自动追加），但最多同时选中 6 条，
  选中顺序就是输出槽 1..6 的顺序。
- 选中的槽没有内容（索引超出可分割条目 / 手动槽为空）时，该路输出按
  unselected_output 产出 none / block / ""。未选中的槽同理。
- block 用 ExecutionBlocker(None) 静默阻断下游并继续传染；none 产出 None，
  下游该入参收到 None；empty 产出空串，下游一定会跑。

手动槽与上游的切换由前端维护的 upstream_mode 决定（auto / linked / manual）：
- linked：以上游为准。text_in 为 None（上游输出 None）按空文本处理，手动槽不参与。
- manual：未接入上游，使用 manual_json。
- auto（默认，供 API 直接调用）：text_in 非 None 视为 linked，否则视为 manual。
前端会明确发送 linked / manual，不依赖 text_in 是否为 None 来猜——这样「上游
接了但本次输出 None」不会误回退到手动槽，列表显示与后端输出始终一致。

前端只维护一份当前列表：上游运行结果会直接覆盖手动预填内容，断开上游后这份
列表就变成可编辑的手动内容。上游显示期间 manual_json 提交常量空数组以保缓存。
"""

import json

from comfy_api.latest import io
from comfy_execution.graph_utils import ExecutionBlocker

SLOT_COUNT = 6
DEFAULT_DELIMITER = "\\n"  # 字面 backslash + n；前端默认显示 \n
INERT_MODES = ("none", "block", "empty")
UPSTREAM_MODES = ("auto", "linked", "manual")

_ESCAPES = {
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "\\": "\\",
}


def _normalize_newlines(text):
    # 统一换行，保证 "\n\n" 这类分隔符在 CRLF 文本上也能命中
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _decode_delimiter(raw):
    """把 \\n / \\t / \\r / \\\\ 转成实际字符；其余反斜杠原样保留。"""
    out = []
    i = 0
    while i < len(raw):
        char = raw[i]
        if char == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                i += 2
                continue
        out.append(char)
        i += 1
    return _normalize_newlines("".join(out))


def _split_text(text, delimiter):
    """按分隔符切分，自动去掉每段首尾空白并丢弃空段。"""
    normalized = _normalize_newlines(text)
    delim = _decode_delimiter(delimiter)
    parts = normalized.split(delim) if delim else [normalized]
    return [part.strip() for part in parts if part.strip()]


def _load_string_list(raw, field):
    try:
        value = json.loads(raw) if raw else []
    except json.JSONDecodeError as e:
        raise ValueError(f"{field} 不是合法 JSON: {e}") from e
    if not isinstance(value, list):
        raise ValueError(f"{field} 必须是字符串数组，实际是 {type(value).__name__}")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field}[{index}] 不是字符串，实际是 {type(item).__name__}")
    return value


def _load_selection(raw):
    try:
        value = json.loads(raw) if raw else []
    except json.JSONDecodeError as e:
        raise ValueError(f"selection_json 不是合法 JSON: {e}") from e
    if not isinstance(value, list):
        raise ValueError(f"selection_json 必须是整数数组，实际是 {type(value).__name__}")
    selection = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"selection_json[{index}] 不是整数，实际是 {type(item).__name__}")
        if item < 0:
            raise ValueError(f"selection_json[{index}] 不能为负数: {item}")
        selection.append(item)
    return selection


def _inert(mode):
    if mode == "block":
        return ExecutionBlocker(None)
    if mode == "empty":
        return ""
    return None


class TextContainer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextContainer",
            display_name="Text Container (6 Outputs)",
            category="text/container",
            description=(
                "文本容器：把上游文本按分隔符切成若干条，点选条目决定 6 路 STRING 输出。"
                "节点默认显示 6 个空占位槽；点「运行到此节点」会用上游文本刷新列表，"
                "上游没接入时可以直接在占位槽里输入内容，填满最后一个槽会自动追加新槽。"
                "条目被选中后右上角显示序号，序号就是它占用的输出槽（最多同时选中 6 条，"
                "列表本身可以超过 6 条）。"
                "工具条三档：多选（默认，什么都不选，按点击先后占用输出槽）、"
                "单选（同时只选一条）、顺序（单选 + 每次排队自动推进到下一条）。"
                "选中的槽没有内容时，该路输出按「缺失槽输出」开关处理："
                "none 输出 None；block 静默阻断下游；empty 输出空串。"
                "接入上游并运行会用上游分割结果直接覆盖手动预填内容；断开上游后"
                "这份列表变成可编辑的手动槽。"
                "刷新列表只改前端显示，不改变 prompt，因此不会让运行缓存失效。"
            ),
            # 输出节点：让「运行到此节点」可用，也让本节点不依赖下游连线就能执行。
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "text_in",
                    display_name="上游文本",
                    optional=True,
                    default=None,
                    force_input=True,
                    tooltip=(
                        "要分割的文本。是否接入上游由前端标记：接入后以上游为准，"
                        "上游输出 None 按空文本处理；未接入时改用节点里的手动槽。"
                    ),
                ),
                io.String.Input(
                    "delimiter",
                    display_name="分隔符",
                    default=DEFAULT_DELIMITER,
                    tooltip=(
                        "单个分隔符字符串，默认 \\n（换行）。支持 \\t、\\r、\\\\ 转义；"
                        "可以填多个字符（如 \\n\\n 按空行分割）；留空表示不分割。"
                        "分割后每段自动去掉首尾空白，空段会被丢弃。"
                    ),
                ),
                io.Combo.Input(
                    "unselected_output",
                    display_name="缺失槽输出",
                    options=list(INERT_MODES),
                    default="none",
                    tooltip=(
                        "未选中的槽、以及选中了但没有内容的槽（索引超出可分割条目、"
                        "或手动槽为空）该产出什么。\n\n"
                        "none（默认）：输出 None，下游该入参收到 None，节点照常执行。"
                        "前提是下游把该入参当可选并会判断 None。\n\n"
                        "block：输出 ExecutionBlocker(None)，静默阻断连到该槽的节点，"
                        "并继续往更下游传染（不报错、不弹窗）。\n\n"
                        "empty：输出空字符串，下游一定会跑。适合下游只接受字符串、"
                        "且能自行处理空串的情况。"
                    ),
                ),
                io.String.Input(
                    "selection_json",
                    default="[]",
                    tooltip="按输出槽顺序排列的选中槽索引（JSON 数组），由前端维护，不要手填。",
                ),
                io.String.Input(
                    "manual_json",
                    default="[]",
                    tooltip="手动槽内容（JSON 字符串数组），由前端维护，随工作流持久化。",
                ),
                io.String.Input(
                    "ui_state_json",
                    default="{}",
                    optional=True,
                    tooltip=(
                        "容器 UI 状态（当前列表、模式、行高、选中项、来源），由前端维护。"
                        "前端固定提交空串，所以它不参与运行缓存；完整值随工作流文件持久化。"
                    ),
                ),
                io.Combo.Input(
                    "upstream_mode",
                    options=list(UPSTREAM_MODES),
                    default="auto",
                    tooltip=(
                        "上游接入状态，由前端维护：linked = 已接入、manual = 未接入；"
                        "auto 留给直接调用 API 的场景（text_in 非 None 视为 linked）。"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(f"text_{index}", display_name=f"文本 {index}")
                for index in range(1, SLOT_COUNT + 1)
            ],
        )

    @classmethod
    def execute(
        cls,
        text_in=None,
        delimiter=DEFAULT_DELIMITER,
        unselected_output="none",
        selection_json="[]",
        manual_json="[]",
        ui_state_json="{}",
        upstream_mode="auto",
    ):
        # ui_state_json 是 UI 状态，前端固定提交空串；这里忽略它。
        del ui_state_json

        if unselected_output not in INERT_MODES:
            raise ValueError(
                f"unselected_output 必须是 {INERT_MODES} 之一，实际是 {unselected_output!r}"
            )
        if upstream_mode not in UPSTREAM_MODES:
            raise ValueError(
                f"upstream_mode 必须是 {UPSTREAM_MODES} 之一，实际是 {upstream_mode!r}"
            )
        selection = _load_selection(selection_json)[:SLOT_COUNT]
        manual = _load_string_list(manual_json, "manual_json")

        linked = text_in is not None if upstream_mode == "auto" else upstream_mode == "linked"
        if linked:
            source = "upstream"
            items = _split_text(text_in or "", delimiter)
        else:
            source = "manual"
            items = manual

        slots = []
        for slot in range(SLOT_COUNT):
            if slot >= len(selection):
                slots.append(_inert(unselected_output))
                continue
            index = selection[slot]
            if index >= len(items):
                slots.append(_inert(unselected_output))
                continue
            text = items[index]
            # 上游条目在分割时已经 trim 并丢弃空段；手动槽保留原样，只把纯空白当空。
            if source == "manual" and not text.strip():
                slots.append(_inert(unselected_output))
                continue
            slots.append(text)

        return io.NodeOutput(
            *slots,
            ui={
                "source": [source],
                # 上游时回传分割结果（前端据此刷新列表）；手动时回传 None，
                # 前端保留手动槽原样，不能把手动内容当成“刚分割出来的”。
                "split": [items if source == "upstream" else None],
            },
        )
