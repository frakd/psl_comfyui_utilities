from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# ui/ 下的前端脚本由 ComfyUI 挂到 /extensions/<包目录名>/（现为 psl_comfyui_utilities），
# 脚本内 '../../scripts/app.js' 的相对深度不变，无需改写。
WEB_DIRECTORY = "./ui"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
