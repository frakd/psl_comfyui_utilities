from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# ui/ 下的前端脚本由 ComfyUI 挂到 /extensions/personal_comfyui_utilities/，
# 脚本内 '../../scripts/app.js' 的相对深度与原 Lora_testing/ui/ 一致，无需改写。
WEB_DIRECTORY = "./ui"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
