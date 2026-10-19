"""CLIP Text Encode (Cached) — 文本编码结果缓存节点。

痛点：LoRA 开关、元数据同步、文本框刷新都会让上游节点重跑；ComfyUI 的缓存签名
把输入字面量与祖先签名一起纳入（comfy_execution/caching.py:117-127），所以即使
最终文本一模一样，下游 CLIPTextEncode 也会重新跑一次。对 Qwen3.5-9B 这类大文本
编码器，这一次重编码就是数秒。

本节点在 encode 外面加一层按 (CLIP 对象, text) 索引的小 LRU：命中时直接返回上次
的 conditioning，不重新 tokenize/encode；未命中才走原始逻辑。节点本身仍会被
ComfyUI 执行（上游签名变了），但执行成本降到一次字典查找。

不做全局 monkey-patch：只替换显式使用本节点的编码，行为可见、可回滚。
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict

from comfy_api.latest import io

_MAX_CACHE_ENTRIES = 4
_MISS = object()

_cache: "OrderedDict[tuple, tuple[weakref.ReferenceType, list]]" = OrderedDict()
_cache_lock = threading.Lock()


def _clip_cache_key(clip, text: str):
    """(CLIP 对象标识, layer_idx, patch 数, 文本)。"""
    patcher = getattr(clip, "patcher", None)
    patches = getattr(patcher, "patches", None)
    try:
        patch_count = len(patches) if patches is not None else 0
    except TypeError:
        patch_count = 0
    return (id(clip), getattr(clip, "layer_idx", None), patch_count, text)


def _lookup(clip, text: str):
    key = _clip_cache_key(clip, text)
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return _MISS
        clip_ref, conditioning = entry
        if clip_ref() is not clip:
            # id 被回收后重用：旧条目作废，避免把别的 CLIP 的编码结果当成命中。
            _cache.pop(key, None)
            return _MISS
        _cache.move_to_end(key)
        return conditioning


def _store(clip, text: str, conditioning) -> None:
    try:
        clip_ref = weakref.ref(clip)
    except TypeError:
        # 极少数不可弱引用的对象：放弃缓存，避免把大模型钉在内存里。
        return
    key = _clip_cache_key(clip, text)
    with _cache_lock:
        _cache[key] = (clip_ref, conditioning)
        _cache.move_to_end(key)
        while len(_cache) > _MAX_CACHE_ENTRIES:
            _cache.popitem(last=False)


def _has_hooks(conditioning) -> bool:
    """带 hooks 的 conditioning 不缓存：hooks 可能绑定当前 CLIP/调度状态。"""
    try:
        for item in conditioning:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                meta = item[1]
                if isinstance(meta, dict) and meta.get("hooks") is not None:
                    return True
    except TypeError:
        return True
    return False


def _encode(clip, text: str):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def _clear_cache() -> None:
    """测试/诊断用：清空缓存。"""
    with _cache_lock:
        _cache.clear()


class CachedCLIPTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CachedCLIPTextEncode",
            display_name="CLIP Text Encode (Cached)",
            category="conditioning",
            search_aliases=[
                "clip",
                "text",
                "encode",
                "cache",
                "缓存",
                "文本编码",
                "提示词编码",
            ],
            description=(
                "与核心 CLIP Text Encode 等价，但按 (CLIP 对象, 文本) 缓存编码结果。"
                "上游（LoRA 开关、元数据同步、文本框刷新等）重跑导致本节点再次执行时，"
                "只要 CLIP 对象和文本没变，就直接返回上次的 conditioning，不再跑文本编码器。"
                "适合 Qwen 等大文本编码器；文本或 CLIP 真正变化时照常重新编码。"
                "带 hooks 的 conditioning 不缓存；可用 cache_mode=disabled 临时绕过。"
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    display_name="CLIP",
                    tooltip="用于编码文本的 CLIP 模型。与核心 CLIP Text Encode 的 clip 输入一致。",
                ),
                io.String.Input(
                    "text",
                    display_name="文本",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="要编码的提示词。与核心 CLIP Text Encode 的 text 完全一致。",
                ),
                io.Combo.Input(
                    "cache_mode",
                    display_name="缓存",
                    options=["enabled", "disabled"],
                    default="enabled",
                    tooltip=(
                        "enabled：同一 CLIP 对象 + 同一文本命中缓存，直接返回上次的 conditioning。\n"
                        "disabled：每次重新编码，用于排查缓存或 CLIP 被原地修改的场景。"
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output("conditioning", display_name="条件"),
            ],
        )

    @classmethod
    def execute(cls, clip, text="", cache_mode="enabled"):
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node your checkpoint does not "
                "contain a valid clip or text encoder model."
            )
        text = "" if text is None else text

        if cache_mode != "enabled":
            return io.NodeOutput(_encode(clip, text))

        cached = _lookup(clip, text)
        if cached is not _MISS:
            return io.NodeOutput(cached)

        conditioning = _encode(clip, text)
        if not _has_hooks(conditioning):
            _store(clip, text, conditioning)
        return io.NodeOutput(conditioning)
