"""Image Container — 多图容器：上传若干图片，点选决定输出顺序，6 个 IMAGE 输出 + 1 个 batch。

未被选中的输出槽产出什么由 unselected_output 决定，三种语义的差别在于下游怎么反应：

- none（默认）：产出 None。下游那个入参收到 None，节点照常执行——适合多条线汇入
  同一个多图输入节点，等于「这次少给一张」。前提是下游把该入参当可选并会判断 None。
- block：产出 ExecutionBlocker(None)。execution.py:270-278 遍历下游所有输入，只要
  有一个是 blocker 就 break、FUNCTION 根本不调用，且 execution.py:405-406 会把它扩展
  成该节点的全部输出继续往下传染；message 为 None 则完全静默（execution.py:525-542）。
  所以**多个槽汇入同一个节点时，只要有一个没选，那个节点整个不跑**。
- placeholder：产出纯黑图。下游一定会跑，但会把黑图当真图处理。

batch 槽在「没选图」或「选中图尺寸不一致」时按同一规则产出（不打日志，结果本身就看得出来）。

两个 widget 的分工是缓存正确性的关键：
- selected_json 进 prompt，是后端唯一需要的东西（有序的被选文件名）。
- pool_json 只承载 UI 状态（容器里全部图片、每行图片数），由前端 JS 设
  options.serialize=false，因此**不进 prompt**、只随工作流文件持久化。
  否则你在容器里上传/排序但没改选中集时，下游缓存也会白丢一遍
  （缓存签名把 prompt 里每个字面量原样纳入，见 comfy_execution/caching.py:120-127）。
"""

import json

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
                f"前 {SLOT_COUNT} 个输出各出一张图，未选中的槽按「未选中槽输出」开关处理；"
                f"第 7 个 batch 输出把选中的图合成一个批次，尺寸不一致时跳过合成，"
                f"前 {SLOT_COUNT} 个单图输出不受影响。"
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
                io.Combo.Input(
                    "unselected_output",
                    display_name="未选中槽输出",
                    options=["none", "block", "placeholder"],
                    default="none",
                    tooltip=(
                        "没被选中的输出槽（以及尺寸不一致时的 batch 槽）该产出什么。\n\n"
                        "none（默认）：该槽不产出值，下游对应的入参收到 None。"
                        "适合多条线汇入同一个多图输入节点——下游照常运行，只是少几张图。"
                        "前提是下游那个入参是可选的且它会判断 None；若下游把它当必填，会在 None 上崩。\n\n"
                        "block：该槽静默阻断下游，连到它的节点整个被跳过（不报错、不弹窗），"
                        "并继续往更下游传染。适合每个槽各喂一条独立分支、没选就不想让那条分支跑。"
                        "注意：若多个槽连进同一个节点，只要有一个没选，那个节点就整个不跑。\n\n"
                        "placeholder：该槽输出一张与第一张选中图同尺寸的纯黑图。下游一定会跑，"
                        "但它会把黑图当真图处理，批量结果里会多出黑帧且没有任何提示。"
                        "只在下游对 None 和阻断都不兼容时才用。"
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
    def _inert(cls, mode, ref):
        """未选中槽、以及无法合成 batch 时的产出。ref 是任一张已加载的图，用来对齐形状。"""
        if mode == "block":
            return ExecutionBlocker(None)
        if mode == "placeholder":
            if ref is None:
                return torch.zeros(
                    (1, 1, 1, 3),
                    dtype=comfy.model_management.intermediate_dtype(),
                    device=comfy.model_management.intermediate_device(),
                )
            return torch.zeros((1, ref.shape[1], ref.shape[2], ref.shape[3]),
                               dtype=ref.dtype, device=ref.device)
        # none：不产出值。下游来取这个槽时会拿到 None —— 与 execution.py:189-191 的
        # mark_missing 等价（那条路同样把输入置为 (None,)），所以不必真的少返回值。
        return None

    @classmethod
    def execute(cls, selected_json="[]", pool_json="{}", unselected_output="none"):
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
        ref = loaded[0] if loaded else None

        # 必须补齐 SLOT_COUNT + 1 个位置：输出按位置对应，少返回值会让 batch
        # 错位落到 image_{N+1} 上。
        slots = [
            loaded[i] if i < len(loaded) else cls._inert(unselected_output, ref)
            for i in range(SLOT_COUNT)
        ]

        batch = cls._inert(unselected_output, ref)
        if loaded:
            height, width = loaded[0].shape[1], loaded[0].shape[2]
            # 尺寸不一致就不合成，batch 槽按当前档位产出惰性值。不打日志：
            # 结果本身就能看出来（batch 下游没动），刷屏不值得。
            if all((t.shape[1], t.shape[2]) == (height, width) for t in loaded):
                batch = torch.cat(loaded, dim=0)

        return io.NodeOutput(*slots, batch)
