"""Mask Paint Output — 预览上游图片、唤出官方遮罩编辑器、输出 IMAGE + MASK。

官方遮罩编辑器的保存流程是硬编码的：它把绘制结果作为 PNG 上传到 input 目录
（遮罩存在该 PNG 的 alpha 通道里），然后把文件名写回**名为 image 的 widget**，
值形如 "clipspace-painted-masked-<时间戳>.png [input]"。既没有保存回调钩子，
也无法让它写到别的 widget 名。

因此上游 IMAGE 插槽不能叫 image —— graphToPrompt 先写 widget 再遍历 inputs，
连线会用 [源节点, slot] 覆盖同名 widget，画好的遮罩文件名会被冲掉。插槽改叫
image_input，把 image 这个名字让给编辑器。

IMAGE 与 MASK 永远取自**同一个文件**（image widget 指向的那张 PNG）：RGB -> IMAGE，
alpha -> MASK。只有 widget 为空时才退回上游 image_input（此时 MASK 全零）。上游换了
新图不会静默覆盖当前绘制结果 —— 接管由前端在 stop_here（取图模式）下清空 widget 完成，
后端不猜。两者分辨率不一致时 MASK 退回全零：错位的遮罩比空遮罩更危险，而空遮罩在
预览里一眼可见，不必打日志。

预览 PNG 带 alpha（= 1 - MASK），所以运行前后节点显示的都是同一张「画过遮罩」的图；
若预览写成不透明，运行一次就会看起来「遮罩没了」。注意 IMAGE 输出仍是完整 RGB，
遮罩走 MASK 插槽，合成由下游节点负责。
"""

import os
import random

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.model_management
from comfy_api.latest import io, InputImpl
from comfy_execution.graph_utils import ExecutionBlocker


class MaskPaintOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MaskPaintOutput",
            display_name="Mask Paint Output",
            category="image/mask",
            description=(
                "预览底图并在其上绘制遮罩，输出 IMAGE 与 MASK。"
                "两者永远来自同一张绘制/粘贴文件（RGB -> IMAGE，alpha -> MASK），"
                "没有该文件时才退回上游输入（MASK 全零），所以显示的和输出的一定是同一对。"
                "右键本节点选「用遮罩编辑器打开」（或图片工具栏的遮罩按钮）唤出官方绘制工具。"
                "stop_here 开启时本节点静默阻断下游并进入取图模式：上游出了新图会自动接管"
                "（当前图与遮罩作废，好在新图上重画）；关闭后进入输出模式，当前图与遮罩被锁定，"
                "上游怎么变都不影响本节点。"
            ),
            is_output_node=True,
            inputs=[
                io.Image.Input(
                    "image_input",
                    display_name="图像",
                    optional=True,
                    tooltip=(
                        "上游底图（单张或一批），可选。仅在节点当前没有绘制/粘贴图时生效；"
                        "上游换了新图想让本节点接管，请在 stop_here 开启（取图模式）下运行一次，"
                        "前端会作废当前图与遮罩、改用新图。"
                    ),
                ),
                io.String.Input(
                    "image",
                    default="",
                    optional=True,
                    tooltip=(
                        "官方遮罩编辑器保存后自动写入的文件名，形如 "
                        "clipspace-painted-masked-1234.png [input]；「粘贴底图」按钮也写这里。"
                        "不要手填（前端已隐藏该输入框）。该文件的 RGB 就是 IMAGE、alpha 就是 MASK，"
                        "跨运行保留。重新粘贴/拖入底图，或在取图模式下让上游接管，都会替换掉它。"
                    ),
                ),
                io.Boolean.Input(
                    "stop_here",
                    default=True,
                    tooltip=(
                        "开启 = 取图模式：静默阻断下游（不保存、不报错），方便先画遮罩；"
                        "上游出了新图会自动接管，当前图与遮罩随之作废。\n"
                        "关闭 = 输出模式：当前图与遮罩被锁定，上游变化不再影响本节点，"
                        "下游正常执行；上游因为签名未变会命中缓存、不会重算。"
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("image", display_name="图像"),
                io.Mask.Output("mask", display_name="遮罩"),
            ],
        )

    @classmethod
    def _load_painted(cls, name):
        """读取编辑器保存的 PNG：RGB -> IMAGE，alpha -> MASK（与 nodes.py 的 LoadImage 同语义）。"""
        # get_annotated_filepath 解析 [input]/[output]/[temp] 标注，并自带路径穿越防护
        path = folder_paths.get_annotated_filepath(name)
        dtype = comfy.model_management.intermediate_dtype()
        device = comfy.model_management.intermediate_device()

        components = InputImpl.VideoFromFile(path).get_components()
        if components.images.shape[0] == 0:
            raise ValueError(f"遮罩文件无法解码为图像: {name!r}")
        images = components.images.to(device=device, dtype=dtype)
        if components.alpha is not None:
            mask = (1.0 - components.alpha[..., -1]).to(device=device, dtype=dtype)
        else:
            mask = torch.zeros(images.shape[:3], dtype=dtype, device=device)
        return images, mask

    @classmethod
    def _preview(cls, images, mask):
        """写进 temp 目录并返回 ui.images —— 前端据此刷新节点预览。

        alpha 写回 1 - mask，与遮罩编辑器保存后显示的那张 PNG 观感一致：画过的地方透明。
        否则运行一次预览就变成不透明的完整图，看起来像「遮罩被刷掉了」。

        node.imgs 非空是官方遮罩编辑器能加载本节点图片、以及右键菜单出现
        Open in MaskEditor 的前提（isImageNode 判定）。
        """
        output_dir = folder_paths.get_temp_directory()
        prefix = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        full_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, output_dir, images[0].shape[1], images[0].shape[2]
        )
        results = []
        for i, image in enumerate(images):
            arr = np.clip(255.0 * image.cpu().numpy(), 0, 255)
            alpha = np.clip(255.0 * (1.0 - mask[i].cpu().numpy()), 0, 255)[..., None]
            file = f"{filename}_{counter:05}_.png"
            Image.fromarray(
                np.concatenate([arr, alpha], axis=-1).astype(np.uint8), "RGBA"
            ).save(os.path.join(full_dir, file), compress_level=1)
            results.append({"filename": file, "subfolder": subfolder, "type": "temp"})
            counter += 1
        return results

    @classmethod
    def execute(cls, image_input=None, image="", stop_here=True):
        painted = cls._load_painted(image) if image else None

        # 绘制/粘贴的文件优先：它就是节点当前显示的图，输出必须与显示一致。
        # 上游只在没有该文件时兜底；上游换新图由前端在取图模式下清空 widget 接管。
        if painted is not None:
            images = painted[0]
        elif image_input is not None:
            images = image_input
        else:
            raise ValueError(
                "MaskPaintOutput 没有可输出的图片：既没连接上游 image_input，"
                "也没粘贴或绘制过底图。"
            )

        # MASK：取绘制文件的 alpha。尺寸对不上就当没画过——错位的遮罩比空遮罩更危险。
        # 不打日志：预览里一眼就能看出遮罩是空的。
        mask = torch.zeros(images.shape[:3], dtype=images.dtype, device=images.device)
        if painted is not None:
            painted_mask = painted[1]
            if tuple(painted_mask.shape[1:3]) == tuple(images.shape[1:3]):
                mask = painted_mask.to(dtype=images.dtype, device=images.device)
                if images.shape[0] > 1 and mask.shape[0] == 1:
                    mask = mask.expand(images.shape[0], -1, -1).contiguous()

        # ui 与阻断值是两条独立通道（execution.py 里分别从 uis 与 results 收集），
        # 所以阻断下游的同时图片照样能显示出来，用户才有东西可画。
        ui = {"images": cls._preview(images, mask)}

        if stop_here:
            # message=None => execution_block_cb 不发 execution_error，下游静默跳过。
            # 注意不能改用 io.NodeOutput(block_execution=...)：那条路径要求非 None，
            # 会弹一次 ExecutionBlocked 提示。
            return io.NodeOutput(ExecutionBlocker(None), ExecutionBlocker(None), ui=ui)
        return io.NodeOutput(images, mask, ui=ui)
