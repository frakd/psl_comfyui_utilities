from .ab_audio_compare import AudioABCompare
from .any_switch import AnySwitch
from .aspect_ratio_selector import AspectRatioSelectorNode
from .audio_filter import AudioHarshnessSuppressor
from .cached_clip_text_encode import CachedCLIPTextEncode
from .deep_vram_cleanup import DeepVRAMCleanup
from .h3_av_lora import MiniMaxH3AVLoRAStack
from .image_container import ImageContainer
from .lora_step_schedule import PSLLoRAStepSchedule
from .mask_paint_output import MaskPaintOutput
from .memory_monitor import MemoryMonitorEnd, MemoryMonitorStart
from .model_reroute import ModelReroute
from .persistent_output import PersistentOutput
from .psl_video_memory import PSLVideoCombineMemory
from .refreshable_text_box import RefreshableTextBox
from .video_match_v2 import PSLAdvancedVideoMatchV2

_NODE_CLASSES = (
    DeepVRAMCleanup,
    MemoryMonitorStart,
    MemoryMonitorEnd,
    AudioHarshnessSuppressor,
    AudioABCompare,
    CachedCLIPTextEncode,
    AnySwitch,
    MiniMaxH3AVLoRAStack,
    AspectRatioSelectorNode,
    ModelReroute,
    MaskPaintOutput,
    PersistentOutput,
    RefreshableTextBox,
    ImageContainer,
    PSLLoRAStepSchedule,
    PSLVideoCombineMemory,
    PSLAdvancedVideoMatchV2,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 在节点搜索框里输入这些词中的任意一个就能筛出本包全部节点。
# psl 是包名 psl_comfyui_utilities 的缩写，敲起来短；personal 保留作同义词。
_SEARCH_ALIASES = ("psl", "personal")


def _inject_search_alias(cls):
    """V3 节点的 object_info 只从 schema 取 search_aliases。

    server.py:819-820 对 V3 节点直接 return GET_NODE_INFO_V1()，走不到 859 行那句
    读 SEARCH_ALIASES 类属性的代码（那行只对 V1 节点生效）；而 GET_NODE_INFO_V1
    的 search_aliases 来自 schema（_io.py:2017，未声明时为 None）。
    所以统一前缀必须进 schema。在这里包一层 define_schema 注入，免得每个节点
    都要记得写、以后新增节点漏掉。前端 nodeSearchService.ts:17 的 Fuse keys 是
    ['name', 'display_name', 'search_aliases']，category 不在其中，只能走别名。
    """
    original = cls.define_schema.__func__

    @classmethod
    def define_schema(kls):
        schema = original(kls)
        schema.search_aliases = list(dict.fromkeys([*_SEARCH_ALIASES, *(schema.search_aliases or [])]))
        return schema

    cls.define_schema = define_schema


# GET_SCHEMA() 会顺带校验 schema，写错在加载时就炸，而不是等到运行节点时。
# node_id 只由各类的 define_schema 决定，这里不再重复维护一份键名。
for _cls in _NODE_CLASSES:
    _inject_search_alias(_cls)
    _schema = _cls.GET_SCHEMA()
    NODE_CLASS_MAPPINGS[_schema.node_id] = _cls
    NODE_DISPLAY_NAME_MAPPINGS[_schema.node_id] = _schema.display_name
