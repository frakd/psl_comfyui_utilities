"""Mask Paint Output — 预览上游图片、唤出官方遮罩编辑器、输出 IMAGE + MASK。

官方遮罩编辑器的保存流程是硬编码的：它把绘制结果作为 PNG 上传到 input 目录
（遮罩存在该 PNG 的 alpha 通道里），然后把文件名写回**名为 image 的 widget**，
值形如 "clipspace-painted-masked-<时间戳>.png [input]"。既没有保存回调钩子，
也无法让它写到别的 widget 名。

因此上游 IMAGE 插槽不能叫 image —— graphToPrompt 先写 widget 再遍历 inputs，
连线会用 [源节点, slot] 覆盖同名 widget，画好的遮罩文件名会被冲掉。插槽改叫
image_input，把 image 这个名字让给编辑器。

IMAGE 与 MASK 的来源是**拆开**的：IMAGE 以上游 image_input 为权威（没连上游才用
绘制文件的 RGB），MASK 始终取绘制文件的 alpha。若让上游整体覆盖 image widget，
遮罩就会在每次运行时被冲掉（RGB 和 alpha 存在同一个 PNG 里），节点将永远输出不了
遮罩。拆开后上游换图会自动跟上，遮罩则跨运行保留；两者分辨率不一致时退回全零遮罩
并打 warning —— 错位的遮罩比空遮罩更危险。
"""

import logging
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
                "两个输出的来源是分开的：IMAGE 以上游输入为权威（没连上游才用粘贴/绘制文件的 RGB），"
                "MASK 始终取绘制文件的 alpha，因此重跑不会被上游图冲掉。"
                "右键本节点选 Open in MaskEditor（或图片工具栏的遮罩按钮）唤出官方绘制工具。"
                "stop_here 开启时本节点会静默阻断下游：整条工作流跑到这里就停，画完遮罩再关掉它继续，"
                "此时上游节点签名未变、命中缓存不会重算。"
            ),
            is_output_node=True,
            inputs=[
                io.Image.Input(
                    "image_input",
                    display_name="图像",
                    optional=True,
                    tooltip=(
                        "上游底图（单张或一批），可选。连着就以它为 IMAGE 输出的权威来源，"
                        "每次运行都会跟上上游的变化；不连则用粘贴/绘制进来的那张图的 RGB。"
                    ),
                ),
                io.String.Input(
                    "image",
                    default="",
                    optional=True,
                    tooltip=(
                        "官方遮罩编辑器保存后自动写入的文件名，形如 "
                        "clipspace-painted-masked-1234.png [input]；「粘贴底图」按钮也写这里。"
                        "不要手填。该文件的 alpha 通道就是 MASK 输出的来源，跨运行保留。"
                        "清空即丢弃遮罩（MASK 退回全零）。"
                    ),
                ),
                io.Boolean.Input(
                    "stop_here",
                    default=True,
                    tooltip=(
                        "开启时本节点静默阻断下游执行：整条工作流跑到这里就停下，"
                        "不保存、不报错，方便你先画遮罩。画完关掉它再运行即可继续，"
                        "上游因为签名未变会命中缓存、不会重算。"
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
    def _preview(cls, images):
        """写进 temp 目录并返回 ui.images —— 前端据此填充 node.imgs。

        node.imgs 非空是官方遮罩编辑器能加载本节点图片、以及右键菜单出现
        Open in MaskEditor 的前提（isImageNode 判定）。
        """
        output_dir = folder_paths.get_temp_directory()
        prefix = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        full_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, output_dir, images[0].shape[1], images[0].shape[2]
        )
        results = []
        for image in images:
            arr = 255.0 * image.cpu().numpy()
            file = f"{filename}_{counter:05}_.png"
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
                os.path.join(full_dir, file), compress_level=1
            )
            results.append({"filename": file, "subfolder": subfolder, "type": "temp"})
            counter += 1
        return results

    @classmethod
    def execute(cls, image_input=None, image="", stop_here=True):
        painted = cls._load_painted(image) if image else None

        # IMAGE：上游为权威；没连上游才用绘制/粘贴文件的 RGB。
        # 这样上游换了图会自动跟上，而遮罩存在另一个通道里不会被冲掉。
        if image_input is not None:
            images = image_input
        elif painted is not None:
            images = painted[0]
        else:
            raise ValueError(
                "MaskPaintOutput 没有可输出的图片：既没连接上游 image_input，"
                "也没粘贴或绘制过底图。"
            )

        # MASK：始终取绘制文件的 alpha。尺寸对不上就当没画过——错位的遮罩比空遮罩更危险。
        mask = torch.zeros(images.shape[:3], dtype=images.dtype, device=images.device)
        if painted is not None:
            painted_mask = painted[1]
            if tuple(painted_mask.shape[1:3]) == tuple(images.shape[1:3]):
                mask = painted_mask.to(dtype=images.dtype, device=images.device)
                if images.shape[0] > 1 and mask.shape[0] == 1:
                    mask = mask.expand(images.shape[0], -1, -1).contiguous()
            else:
                logging.warning(
                    "[MaskPaintOutput] 遮罩尺寸 %dx%d 与当前底图 %dx%d 不一致，已退回全零遮罩；"
                    "请在新尺寸上重新绘制。",
                    painted_mask.shape[2], painted_mask.shape[1],
                    images.shape[2], images.shape[1],
                )

        # ui 与阻断值是两条独立通道（execution.py 里分别从 uis 与 results 收集），
        # 所以阻断下游的同时图片照样能显示出来，用户才有东西可画。
        ui = {"images": cls._preview(images)}

        if stop_here:
            # message=None => execution_block_cb 不发 execution_error，下游静默跳过。
            # 注意不能改用 io.NodeOutput(block_execution=...)：那条路径要求非 None，
            # 会弹一次 ExecutionBlocked 提示。
            return io.NodeOutput(ExecutionBlocker(None), ExecutionBlocker(None), ui=ui)
        return io.NodeOutput(images, mask, ui=ui)
