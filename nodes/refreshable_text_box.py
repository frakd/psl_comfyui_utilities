"""Refreshable Text Box — 显示会自刷新的文本，且不让刷新污染运行缓存。

痛点：文本框节点每次运行后把上游文本写回自己的 widget，于是下一次运行时 prompt
里这个字面量变了，运行缓存从本节点起全部失效——即使上游文本根本没变。

为什么只能在前端解决（读码确认）：
- 缓存签名把 prompt 中每个节点 inputs 的字面量原样纳入，见
  comfy_execution/caching.py:120-127（连线才是 ("ANCESTOR", ...) 引用，字面量是值）。
- 后端**没有**任何「某个输入不参与缓存」的机制；IS_CHANGED 也救不了——变化的
  字面量照样进签名，它只能增加失效、不能减少。

所以由 ui/refreshable_text_box.js 给文本 widget 挂 serializeValue：未锁定时一律
发送空串常量 => prompt 字节稳定 => 本节点与下游缓存不失效；锁定后才发送真实文本
（此时失效是预期的，用户就是要让改过的文本生效）。

widget.serialize（存进工作流 json）与 widget.options.serialize（是否进 prompt）
是两个独立开关（见前端 widgets.ts:501-512，存盘走 widget.value 而非 serializeValue），
因此文本照样能持久化，不必为此牺牲。
"""

from comfy_api.latest import io


class RefreshableTextBox(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RefreshableTextBox",
            display_name="Refreshable Text Box",
            category="text",
            description=(
                "可自刷新的文本框：运行时把上游文本刷进文本框显示，并原样输出。"
                "「允许覆盖」开启时刷新只改显示、不改进 prompt 的值，因此本节点与下游的"
                "运行缓存不会因为文本变化而失效；关掉（锁定）后文本框内容才会真正生效并输出。"
                "典型用法：开着允许覆盖 → 运行到此节点 → 编辑文本 → 锁定 → 再运行。"
            ),
            # 必须是输出节点：前端 isOutputNode 过滤 + 后端 execution.py:1181-1183
            # 只把 OUTPUT_NODE 节点当作部分执行的根，否则「运行到此节点」不可用。
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "text_in",
                    display_name="上游文本",
                    optional=True,
                    default="",
                    force_input=True,
                    tooltip="上游来的文本。连线输入，不连则视为空串。",
                ),
                io.String.Input(
                    "locked_text",
                    display_name="文本",
                    multiline=True,
                    default="",
                    tooltip=(
                        "文本框内容。「允许覆盖」开启时它只显示、不参与运行缓存；"
                        "锁定后它的内容会真正提交并作为本节点的输出。"
                    ),
                ),
                io.Boolean.Input(
                    "allow_overwrite",
                    display_name="允许覆盖",
                    default=True,
                    tooltip=(
                        "开启：每次运行用上游文本刷新显示，且提交给后端的值固定为空串，"
                        "因此不会让运行缓存失效。\n"
                        "关闭（锁定）：文本框当前内容生效并输出，切换这一下会让缓存从本节点起失效一次"
                        "（这是预期的——你就是要让新文本生效）。"
                    ),
                ),
            ],
            outputs=[
                io.String.Output("text", display_name="文本"),
            ],
        )

    @classmethod
    def execute(cls, text_in="", locked_text="", allow_overwrite=True):
        text = text_in if allow_overwrite else locked_text
        # 回传给前端刷新显示；缓存命中时本节点不会执行，也就不会刷新——
        # 而缓存命中恰恰意味着上游文本没变，本来也无需刷新。
        return io.NodeOutput(text, ui={"text": [text]})
