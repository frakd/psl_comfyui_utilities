"""Image Container — 多图容器：上传若干图片，点选决定输出顺序，6 个 IMAGE 输出 + 1 个 batch。

未被选中的输出槽返回 ExecutionBlocker(None)：execution.py:270-278 只在下游节点的
某个输入是 blocker 时跳过该节点，兄弟槽照常执行，message 为 None 则完全静默
（不发 execution_error、不弹提示），见 execution.py:525-542。

两个 widget 的分工是缓存正确性的关键：
- selected_json 进 prompt，是后端唯一需要的东西（有序的被选文件名）。
- pool_json 只承载 UI 状态（容器里全部图片、每行图片数），由前端 JS 设
  options.serialize=false，因此**不进 prompt**、只随工作流文件持久化。
  否则你在容器里上传/排序但没改选中集时，下游缓存也会白丢一遍
  （缓存签名把 prompt 里每个字面量原样纳入，见 comfy_execution/caching.py:120-127）。
"""

import json
import logging

import torch

import folder_paths
import comfy.model_management
from comfy_api.latest import io, InputImpl
from comfy_execution.graph_utils import ExecutionBlocker

SLOT_COUNT = 6


class ImageContainer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        outputs = [
            io.Image.Output(f"image_{i}", display_name=f"图像 {i}")
            for i in range(1, SLOT_COUNT + 1)
        ]
        outputs.append(io.Image.Output("batch", display_name="批次（需同尺寸）"))
        return io.Schema(
            node_id="ImageContainer",
            display_name="Image Container (6 Outputs)",
            category="image/container",
            description=(
                "图片容器：点按钮上传、把图片文件拖到节点上、或 Ctrl+V 粘贴剪贴板图片。"
                "单击缩略图选中它，按点击先后占用输出槽 1..6，左上角角标显示它占第几个槽；"
                "再点一次取消输出，后面的角标自动进位。长按左键拖拽只调整容器内的显示顺序，"
                "不影响角标编号。缩略图右上角 × 把该图从容器移除。"
                f"前 {SLOT_COUNT} 个输出各出一张图，未选中的槽静默阻断连到它的下游；"
                f"第 7 个 batch 输出把选中的图合成一个批次，尺寸不一致时跳过合成"
                f"（该槽静默阻断，前 {SLOT_COUNT} 个单图输出不受影响）。"
            ),
            # 必须是输出节点：既让「运行到此节点」可用（execution.py:1181-1183 只把
            # OUTPUT_NODE 当作部分执行的根），也让它不依赖下游连线就能被执行。
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "selected_json",
                    default="[]",
                    tooltip="有序的被选图片列表（JSON），由前端维护，不要手填。",
                ),
                io.String.Input(
                    "pool_json",
                    default="{}",
                    optional=True,
                    tooltip=(
                        "容器 UI 状态（全部图片 + 每行图片数），由前端维护。"
                        "前端会把它标记为不参与 prompt 序列化，只随工作流文件保存，"
                        "所以后端正常情况下收不到它。"
                    ),
                ),
            ],
            outputs=outputs,
        )

    @classmethod
    def _load(cls, entry):
        """读取一张图片，返回 [1,H,W,C]。多图/动图只取第一帧。"""
        name = entry.get("name") or ""
        subfolder = entry.get("subfolder") or ""
        file_type = entry.get("type") or "input"
        if not name:
            raise ValueError("selected_json 里有一个条目缺少 name 字段")

        rel = f"{subfolder}/{name}" if subfolder else name
        # get_annotated_filepath 解析 [input]/[output]/[temp] 标注，并自带路径穿越防护
        # （folder_paths.py:352-356：abspath 后必须仍在 base_dir 内，否则抛 ValueError）
        path = folder_paths.get_annotated_filepath(f"{rel} [{file_type}]")

        dtype = comfy.model_management.intermediate_dtype()
        device = comfy.model_management.intermediate_device()
        components = InputImpl.VideoFromFile(path).get_components()
        if components.images.shape[0] == 0:
            raise ValueError(f"图片无法解码: {rel!r}")
        return components.images[:1].to(device=device, dtype=dtype)

    @classmethod
    def execute(cls, selected_json="[]", pool_json="{}"):
        # pool_json 是 UI 状态，正常情况下前端不会发过来（options.serialize=false）；
        # 保留形参是为了万一前端脚本没生效时也不会因多余 kwarg 崩掉。
        del pool_json

        try:
            selected = json.loads(selected_json) if selected_json else []
        except json.JSONDecodeError as e:
            raise ValueError(f"selected_json 不是合法 JSON: {e}") from e
        if not isinstance(selected, list):
            raise ValueError(f"selected_json 必须是数组，实际是 {type(selected).__name__}")

        entries = selected[:SLOT_COUNT]
        loaded = [cls._load(e) for e in entries]

        slots = [loaded[i] if i < len(loaded) else ExecutionBlocker(None) for i in range(SLOT_COUNT)]

        if not loaded:
            return io.NodeOutput(*slots, ExecutionBlocker(None))

        height, width = loaded[0].shape[1], loaded[0].shape[2]
        mismatched = [
            f"{entries[i].get('name')} ({t.shape[2]}x{t.shape[1]})"
            for i, t in enumerate(loaded)
            if (t.shape[1], t.shape[2]) != (height, width)
        ]
        if mismatched:
            # 尺寸不一致就不尝试合成：batch 槽静默阻断，前 6 个单图输出照常工作。
            # 这里不用字面量 None —— 输出值没有类型校验，None 会原样送进下游 FUNCTION
            # 再崩在那里（execution.py:185-193），报错点会指向无辜的下游节点。
            logging.warning(
                "[ImageContainer] batch 输出跳过：选中图片尺寸不一致，第一张 %dx%d，不一致的有 %s",
                width, height, ", ".join(mismatched),
            )
            return io.NodeOutput(*slots, ExecutionBlocker(None))
        return io.NodeOutput(*slots, torch.cat(loaded, dim=0))
