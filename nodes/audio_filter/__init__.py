from typing_extensions import override
from comfy_api.latest import ComfyExtension

from .nodes import AudioHarshnessSuppressor


class AudioFilterExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [AudioHarshnessSuppressor]


async def comfy_entrypoint():
    return AudioFilterExtension()
