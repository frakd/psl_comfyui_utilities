from .aspect_ratio_selector import AspectRatioSelectorNode
from .audio_filter import AudioHarshnessSuppressor
from .deep_vram_cleanup import DeepVRAMCleanup
from .h3_av_lora import MiniMaxH3AVLoRAStack
from .model_reroute import ModelReroute

_NODE_CLASSES = (
    DeepVRAMCleanup,
    AudioHarshnessSuppressor,
    MiniMaxH3AVLoRAStack,
    AspectRatioSelectorNode,
    ModelReroute,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# GET_SCHEMA() 会顺带校验 schema，写错在加载时就炸，而不是等到运行节点时。
# node_id 只由各类的 define_schema 决定，这里不再重复维护一份键名。
for _cls in _NODE_CLASSES:
    _schema = _cls.GET_SCHEMA()
    NODE_CLASS_MAPPINGS[_schema.node_id] = _cls
    NODE_DISPLAY_NAME_MAPPINGS[_schema.node_id] = _schema.display_name
