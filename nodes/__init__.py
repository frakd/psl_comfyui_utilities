from .aspect_ratio_selector import AspectRatioSelectorNode
from .audio_filter import AudioHarshnessSuppressor
from .deep_vram_cleanup import DeepVRAMCleanup
from .h3_av_lora import MiniMaxH3AVLoRAStack
from .image_container import ImageContainer
from .mask_paint_output import MaskPaintOutput
from .model_reroute import ModelReroute
from .refreshable_text_box import RefreshableTextBox

_NODE_CLASSES = (
    DeepVRAMCleanup,
    AudioHarshnessSuppressor,
    MiniMaxH3AVLoRAStack,
    AspectRatioSelectorNode,
    ModelReroute,
    MaskPaintOutput,
    RefreshableTextBox,
    ImageContainer,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 在节点搜索框里输入这个词就能筛出本包全部节点。
_SEARCH_ALIAS = "personal"

# GET_SCHEMA() 会顺带校验 schema，写错在加载时就炸，而不是等到运行节点时。
# node_id 只由各类的 define_schema 决定，这里不再重复维护一份键名。
for _cls in _NODE_CLASSES:
    _schema = _cls.GET_SCHEMA()
    NODE_CLASS_MAPPINGS[_schema.node_id] = _cls
    NODE_DISPLAY_NAME_MAPPINGS[_schema.node_id] = _schema.display_name

    # V3 Schema 的 search_aliases 传不到 server.py:859 读取的 SEARCH_ALIASES 类属性
    # —— GET_SCHEMA() 同步了 _DESCRIPTION/_CATEGORY/_OUTPUT_NODE 等一堆，唯独漏了它，
    # _io.py 里也没有对应的 classproperty。所以在此把 schema 里声明的别名与统一前缀
    # 合并后显式挂上。前端 nodeSearchService.ts:17 的 Fuse keys 是
    # ['name', 'display_name', 'search_aliases']，category 不在其中，只能走别名。
    _cls.SEARCH_ALIASES = list(dict.fromkeys([
        _SEARCH_ALIAS,
        *(_schema.search_aliases or []),
        *getattr(_cls, "SEARCH_ALIASES", []),
    ]))
