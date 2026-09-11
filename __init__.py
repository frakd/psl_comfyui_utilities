from typing_extensions import override
from comfy_api.latest import ComfyExtension

from .nodes import AudioHarshnessSuppressor


class AudioFilterExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [AudioHarshnessSuppressor]


async def comfy_entrypoint():
    return AudioFilterExtension()


NODE_CLASS_MAPPINGS = {
    "AudioHarshnessSuppressor": AudioHarshnessSuppressor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioHarshnessSuppressor": "Audio Harshness Suppressor",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
