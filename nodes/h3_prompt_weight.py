# -*- coding: utf-8 -*-
"""MiniMax H3 Image prompt preset weighting for the official ComfyUI model.

The node merges a workflow-persisted list of preset snippets into the positive
prompt, then applies KJ-style per-token attention weighting inside the H3 DiT.

H3 packs its sequence as [text | reference images | target image] and the
prompt is the last part of the text segment, behind any picture blocks. Token
positions are therefore anchored to the end of the encoded text (offset =
context length - prompt length), which also covers edit images and the
empty-image mode.

Optional images/vae mirror the official MiniMaxH3ImageConditioning node. The
empty-image mode mirrors KJNodes: a blank picture block is prepended to the
conditioning only for an early sampling window, controlled by start/end
percent.
"""

import logging
import math
import threading
import weakref
from collections import OrderedDict

import comfy.model_management
import comfy.patcher_extension
import torch

from comfy_api.latest import io

from .prompt_weight_core import (
    MERGE_SEPARATORS,
    cache_key,
    extract_weighted_terms,
    merge_prompt,
    parse_presets,
    positions_from_ids,
)

_NODE = "MiniMaxH3ImagePromptWeightPreset"
_WRAPPER_KEY = _NODE
_STATE_KEY = "minimax_h3_prompt_preset"

# <|im_start|>, user, newline, <|im_end|> in the Qwen3-VL vocab used by H3.
_H3_SPECIALS = (151644, 872, 198, 151645)
_MAX_RESOLUTION = 16384
_EMPTY_IMAGE_SIZE = 512
_VISION_SHORT_EDGES = ["auto", "384", "512", "768", "1024", "1280", "1536", "1792", "2048"]
_OFFICIAL_VISION_EDGES = {"auto", "1024"}
_VISION_TOKEN_WARN = 2048

_MISS = object()


class _WeakValueCache:
    def __init__(self, max_entries=4):
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def get(self, key, identity):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return _MISS
            refs, value = entry
            if len(refs) != len(identity) or any(ref() is not obj for ref, obj in zip(refs, identity)):
                self._entries.pop(key, None)
                return _MISS
            self._entries.move_to_end(key)
            return value

    def put(self, key, value, identity):
        try:
            refs = tuple(weakref.ref(obj) for obj in identity)
        except TypeError as exc:
            logging.debug("%s: cannot cache a non-weakref-able input (%s)", _NODE, exc)
            return
        with self._lock:
            self._entries[key] = (refs, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


_COND_CACHE = _WeakValueCache()
_PICTURE_CACHE = _WeakValueCache()


def _collect_images(images):
    if not isinstance(images, dict):
        return []
    names = sorted(images, key=lambda name: int(name.rsplit("_", 1)[-1]))
    return [images[name] for name in names if images.get(name) is not None]


def _token_ids(tokens):
    key = next(iter(tokens))
    return [item[0] for item in tokens[key][0]]


def _empty_h3_image_latent(width, height, image_list):
    if not (width and height):
        try:
            from comfy_extras.nodes_minimax_h3 import (
                IMAGE_REF_SHORT_EDGE, IMAGE_REF_VAE_MAX_SHORT_EDGE, IMAGE_SIZE, _size_short_edge,
            )
        except ImportError as exc:
            raise RuntimeError(f"{_NODE} requires a ComfyUI version with MiniMax H3 Image support") from exc
        if image_list:
            image = image_list[0]
            size = _size_short_edge(image.shape[2], image.shape[1],
                                    IMAGE_REF_SHORT_EDGE, IMAGE_REF_VAE_MAX_SHORT_EDGE)
        else:
            size = (IMAGE_SIZE, IMAGE_SIZE)
        width, height = width or size[0], height or size[1]
    return {"samples": torch.zeros(
        [1, 24, 1, height // 16, width // 16],
        device=comfy.model_management.intermediate_device(),
    )}


def _encode_conditioning(clip, prompt, width, height, vae, images, cache_enabled):
    image_list = _collect_images(images)
    identity = tuple([clip, *image_list] + ([vae] if vae is not None else []))
    # Width/height only change the zero target latent, never the text/image
    # conditioning, so they stay out of the encode-cache key.
    extra = (
        "h3",
        len(image_list),
        tuple((id(image), tuple(image.shape)) for image in image_list),
    )
    key = cache_key(clip, prompt, extra)
    if cache_enabled:
        cached = _COND_CACHE.get(key, identity)
        if cached is not _MISS:
            return cached, _empty_h3_image_latent(width, height, image_list)

    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning
    except ImportError as exc:
        raise RuntimeError(f"{_NODE} requires a ComfyUI version with MiniMax H3 Image support") from exc

    cond = MiniMaxH3ImageConditioning.execute(
        clip, prompt, width, height, vae=vae, images=images
    ).result[0]
    if cache_enabled:
        _COND_CACHE.put(key, cond, identity)
    return cond, _empty_h3_image_latent(width, height, image_list)


def _estimate_vision_tokens(vision_images):
    try:
        from comfy.text_encoders.qwen_vl import smart_resize
    except ImportError:
        return None
    total = 0
    for image in vision_images:
        height, width = int(image.shape[1]), int(image.shape[2])
        h_bar, w_bar = smart_resize(height, width, 32, 3136, 12845056)
        total += max(1, (h_bar // 32) * (w_bar // 32))
    return total


def _encode_custom_conditioning(clip, prompt, width, height, vae, images, vision_edge, cache_enabled):
    image_list = _collect_images(images)
    identity = tuple([clip, *image_list] + ([vae] if vae is not None else []))
    extra = (
        "h3-custom",
        int(vision_edge),
        len(image_list),
        tuple((id(image), tuple(image.shape)) for image in image_list),
    )
    key = cache_key(clip, prompt, extra)
    if cache_enabled:
        cached = _COND_CACHE.get(key, identity)
        if cached is not _MISS:
            return cached, _empty_h3_image_latent(width, height, image_list)

    try:
        from comfy_extras.nodes_minimax_h3 import (
            IMAGE_REF_SHORT_EDGE, IMAGE_REF_VAE_MAX_SHORT_EDGE, _image_refs, _size_short_edge,
        )
    except ImportError as exc:
        raise RuntimeError(f"{_NODE} requires a ComfyUI version with MiniMax H3 Image support") from exc

    vae_size = lambda w, h: _size_short_edge(w, h, IMAGE_REF_SHORT_EDGE, IMAGE_REF_VAE_MAX_SHORT_EDGE)
    vision_size = lambda w, h: _size_short_edge(w, h, vision_edge, vision_edge)
    image_list, vision_images, refs = _image_refs(images, vae, vision_size, vae_size)
    total_tokens = _estimate_vision_tokens(vision_images)
    if total_tokens is not None:
        logging.info("%s: custom vision short edge %d -> ~%d vision tokens",
                     _NODE, vision_edge, total_tokens)
        if total_tokens > _VISION_TOKEN_WARN:
            logging.warning("%s: custom vision short edge %d produces ~%d vision tokens; "
                            "the text prompt may be diluted", _NODE, vision_edge, total_tokens)

    cond = clip.encode_from_tokens_scheduled(
        clip.tokenize(prompt, images=vision_images, picture_label="Picture %d: ")
    )
    if refs:
        import node_helpers
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": refs})
    if cache_enabled:
        _COND_CACHE.put(key, cond, identity)
    return cond, _empty_h3_image_latent(width, height, image_list)


def _vision_positions_by_length(conditioning):
    mapping = {}
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        states, metadata = item[0], item[1]
        tags = metadata.get("minimax_token_tags") if isinstance(metadata, dict) else None
        if not torch.is_tensor(tags) or tags.shape[-1] != states.shape[1]:
            continue
        positions = (tags == 0).nonzero(as_tuple=False).flatten()
        if positions.numel():
            mapping[int(states.shape[1])] = positions
    return mapping


def _encode_empty_picture(clip, cache_enabled):
    identity = (clip,)
    key = cache_key(clip, "\0minimax_h3_empty_picture")
    if cache_enabled:
        cached = _PICTURE_CACHE.get(key, identity)
        if cached is not _MISS:
            return cached

    cond = clip.encode_from_tokens_scheduled(
        clip.tokenize("", images=[torch.zeros(1, _EMPTY_IMAGE_SIZE, _EMPTY_IMAGE_SIZE, 3)],
                      picture_label="Picture %d: ")
    )
    if not cond or not isinstance(cond[0], (list, tuple)) or len(cond[0]) < 2:
        raise RuntimeError(f"{_NODE}: the empty picture produced an unexpected conditioning")
    picture, picture_dict = cond[0]
    if not torch.is_tensor(picture) or not isinstance(picture_dict, dict):
        raise RuntimeError(f"{_NODE}: the empty picture produced an unexpected conditioning")
    value = (picture, picture_dict)
    if cache_enabled:
        _PICTURE_CACHE.put(key, value, identity)
    return value


def _apply_empty_image(cond, picture, picture_dict, start, end):
    picture_tags = picture_dict.get("minimax_token_tags")
    if picture_tags is None:
        logging.warning("%s: the empty picture has no token tags; empty-image mode skipped", _NODE)
        return cond
    if picture_tags.shape[-1] != picture.shape[1]:
        logging.warning("%s: the empty picture tags do not match its rows; empty-image mode skipped", _NODE)
        return cond

    with_picture = []
    for item in cond:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            logging.warning("%s: unexpected conditioning entry; empty-image mode skipped", _NODE)
            return cond
        text_states, metadata = item[0], item[1]
        tags = metadata.get("minimax_token_tags") if isinstance(metadata, dict) else None
        if not torch.is_tensor(text_states) or tags is None:
            logging.warning("%s: conditioning has no H3 token tags; empty-image mode skipped", _NODE)
            return cond
        if tags.shape[-1] != text_states.shape[1]:
            logging.warning("%s: conditioning tags do not match its rows; empty-image mode skipped", _NODE)
            return cond

        merged_tags = torch.cat([picture_tags, tags])
        with_picture.append([
            torch.cat([picture, text_states], dim=1),
            {**metadata, "minimax_token_tags": merged_tags,
             "start_percent": start, "end_percent": end},
        ])
        if start > 0.0:
            with_picture.append([text_states, {**metadata, "start_percent": 0.0, "end_percent": start}])
        if end < 1.0:
            with_picture.append([text_states, {**metadata, "start_percent": end, "end_percent": 1.0}])
    return with_picture


def _build_weight_tensors(value_pos, value_factor, bias_pos, bias_value, offset, seq_len, device, dtype,
                          vision_positions=None, image_strength=1.0):
    multiplier = None
    if value_pos:
        positions = torch.tensor(value_pos, device=device) + offset
        if int(positions.max()) >= seq_len:
            return None
        multiplier = torch.ones(1, 1, seq_len, 1, device=device, dtype=dtype)
        multiplier[0, 0, positions, 0] = torch.tensor(value_factor, device=device, dtype=dtype)
    use_image_bias = vision_positions is not None and image_strength != 1.0
    bias = None
    if bias_pos or use_image_bias:
        bias = torch.zeros(1, 1, 1, seq_len, device=device, dtype=dtype)
        if bias_pos:
            positions = torch.tensor(bias_pos, device=device) + offset
            if int(positions.max()) >= seq_len:
                return None
            bias[0, 0, 0, positions] = torch.tensor(bias_value, device=device, dtype=dtype)
        if use_image_bias:
            positions = vision_positions.to(device, dtype=torch.long)
            positions = positions[positions < seq_len]
            if positions.numel():
                weight = math.log(image_strength) if image_strength > 0.0 else math.log(1e-6)
                bias[0, 0, 0, positions] += weight
    return multiplier, bias


def _make_forward_wrapper(pairs, prompt_len, name, image_strength=1.0, vision_positions_by_len=None):
    value_pos = [position for position, factor, _ in pairs if factor != 1.0]
    value_factor = [factor for _, factor, _ in pairs if factor != 1.0]
    bias_pos = [position for position, _, bias in pairs if bias != 0.0]
    bias_value = [bias for _, _, bias in pairs if bias != 0.0]
    warned = set()
    tensor_cache = {}

    def warn_once(key, message):
        if key not in warned:
            warned.add(key)
            logging.warning(f"{name}: {message}")

    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        # H3 Image denoises one positive sample at a time; a CFG sampler would
        # stack a negative row the model itself rejects, so only [0] is valid.
        if transformer_options.get("cond_or_uncond") == [0]:
            offset = context.shape[1] - prompt_len
            if offset >= 0:
                payload = kwargs.get("minimax_payload") or {}
                layout = payload.get("layout") if isinstance(payload, dict) else None
                seq_len = getattr(layout, "seq_len", None)
                vision_positions = None
                if image_strength != 1.0 and vision_positions_by_len:
                    vision_positions = vision_positions_by_len.get(int(context.shape[1]))
                tensors = None
                if isinstance(seq_len, int) and seq_len > 0:
                    cache_key_ = (seq_len, context.device, context.dtype, image_strength,
                                  id(vision_positions) if vision_positions is not None else None)
                    tensors = tensor_cache.get(cache_key_)
                    if tensors is None:
                        tensors = _build_weight_tensors(
                            value_pos, value_factor, bias_pos, bias_value,
                            offset, seq_len, context.device, context.dtype,
                            vision_positions, image_strength,
                        )
                        if tensors is not None:
                            tensor_cache[cache_key_] = tensors
                if tensors is not None:
                    # Precomputed outside the per-block allocation scopes: blocks
                    # only do an in-place multiply and reuse the bias mask.
                    transformer_options[_STATE_KEY] = (tensors[0], tensors[1], None, None, None, None)
                elif not value_pos and not bias_pos and vision_positions is None:
                    transformer_options[_STATE_KEY] = (None, None, None, None, None, None)
                elif seq_len is not None:
                    warn_once("layout", f"packed layout length {seq_len} cannot hold the prompt positions; "
                                        "weighting skipped")
                else:
                    if image_strength != 1.0 and vision_positions is not None:
                        warn_once("vision", "image_strength needs the packed layout; it is ignored")
                    # Fallback for a sampler that does not expose the packed layout.
                    device = context.device
                    transformer_options[_STATE_KEY] = (
                        None, None,
                        torch.tensor(value_pos, device=device) + offset if value_pos else None,
                        torch.tensor(value_factor, device=device)[:, None] if value_pos else None,
                        torch.tensor(bias_pos, device=device) + offset if bias_pos else None,
                        torch.tensor(bias_value, device=device) if bias_pos else None,
                    )
            else:
                warn_once("length", f"conditioning length {context.shape[1]} is shorter than the prompt "
                                    f"({prompt_len}); weighting skipped")
        return executor(x, timestep, context, transformer_options, **kwargs)

    return wrapper


def _make_attention_override(weights, comfy_attention, previous):
    multiplier, bias_tensor, value_idx, value_factor, bias_idx, bias_value = weights

    def override(func, q, k, v, heads, mask=None, **kwargs):
        if multiplier is not None:
            current = multiplier
            if current.dtype != v.dtype:
                current = current.to(v.dtype)
            if v.ndim == 3 and current.ndim == 4:
                current = current.view(current.shape[0], current.shape[2], current.shape[3])
            v.mul_(current)
        elif value_idx is not None:
            factor = value_factor.to(v.dtype)
            v[..., value_idx, :] = v[..., value_idx, :] * factor
        if bias_tensor is not None and mask is None:
            key_count = k.shape[-2] if kwargs.get("skip_reshape") else k.shape[1]
            mask = bias_tensor
            if mask.shape[-1] != key_count:
                if mask.shape[-1] < key_count:
                    mask = torch.nn.functional.pad(mask, (0, key_count - mask.shape[-1]))
                else:
                    mask = mask[..., :key_count]
        elif bias_idx is not None and mask is None:
            key_count = k.shape[-2] if kwargs.get("skip_reshape") else k.shape[1]
            mask = q.new_zeros(1, 1, 1, key_count)
            mask[..., bias_idx] = bias_value.to(q.dtype)
        if previous is not None:
            return previous(func, q, k, v, heads, mask=mask, **kwargs)
        # Keep the checkpoint's own backend (H3 Image ships comfy_kitchen_int8):
        # int8 attention accepts an additive mask; sol/sage fall back internally.
        attention = comfy_attention.function if comfy_attention is not None else None
        return (attention or func)(q, k, v, heads, mask=mask, **kwargs)

    return override


def _make_block_patch(comfy_attention, previous):
    def patch(args, extra):
        options = args["transformer_options"]
        weights = options.get(_STATE_KEY)
        if weights is not None:
            options = {
                **options,
                "optimized_attention_override": _make_attention_override(
                    weights, comfy_attention, options.get("optimized_attention_override")
                ),
            }
            args = {**args, "transformer_options": options}
        if previous is not None:
            return previous(args, extra)
        return extra["original_block"](args)

    return patch


def _patch_model(model, pairs, prompt_len, image_strength=1.0, vision_positions_by_len=None):
    patched = model.clone()
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        _WRAPPER_KEY,
        _make_forward_wrapper(pairs, prompt_len, _NODE, image_strength, vision_positions_by_len),
    )
    diffusion_model = patched.get_model_object("diffusion_model")
    if not hasattr(diffusion_model, "blocks"):
        raise ValueError(f"{_NODE}: model is not a MiniMax H3 diffusion model")
    existing = patched.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {}) or {}
    for index, block in enumerate(diffusion_model.blocks):
        comfy_attention = getattr(getattr(block, "attn", None), "comfy_attention", None)
        patched.set_model_patch_replace(
            _make_block_patch(comfy_attention, existing.get(("double_block", index))),
            "dit", "double_block", index,
        )
    return patched


class MiniMaxH3ImagePromptWeightPreset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=_NODE,
            display_name="MiniMax H3 Image Prompt Weight (Presets)",
            category="MiniMax H3/prompt",
            description="Merges a workflow-persisted list of plain preset snippets into the positive prompt and "
                        "applies KJ-style per-token attention weighting inside the H3 Image DiT; weights apply to "
                        "every occurrence of the matched phrase. Entry weight 1 = neutral, >1 emphasizes, <1 "
                        "weakens, 0 removes, <0 subtracts the concept. The merged-text encoding is cached by CLIP, "
                        "patch state, text, image identities and VAE, so weight changes still apply without "
                        "re-encoding. Optional images/vae mirror the official MiniMaxH3ImageConditioning node; "
                        "vision_short_edge customizes the Qwen vision copy for experiments (auto/1024 keep the "
                        "official path), and image_strength biases vision tokens only in custom modes. "
                        "empty_image prepends a blank picture block for an early sampling window. Wire model, "
                        "conditioning and latent to the sampler.",
            inputs=[
                io.Clip.Input("clip"),
                io.Model.Input("model"),
                io.String.Input("text", multiline=True, dynamic_prompts=True, default="",
                                tooltip="用户正面提示词。这里仍可使用 (phrase:weight) 语法；列表中的预设条目会合并到这段文本的开头或结尾。"),
                io.Combo.Input("merge_order", options=["before", "after"], default="before",
                               tooltip="预设条目合并到用户提示词的前面（before）还是后面（after）。条目之间用空格连接，按列表顺序排列。"),
                io.Combo.Input("cache_mode", display_name="缓存", options=["enabled", "disabled"], default="enabled",
                               tooltip="enabled：同一 CLIP + 同一合并文本 + 同一图片/VAE 命中缓存，跳过 Qwen 文本编码器；"
                                       "disabled：每次重新编码。只改预设权重不会使缓存失效。"),
                io.String.Input("presets_data", default="[]", socketless=True, advanced=True,
                                tooltip="由节点前端管理的 JSON 条目列表 [{enabled, text, weight}]；一般不需要手动修改。"),
                io.Int.Input("width", default=0, min=0, max=_MAX_RESOLUTION, step=32,
                             tooltip="0 使用 image_1 的尺寸；没有图片时使用 2048。"),
                io.Int.Input("height", default=0, min=0, max=_MAX_RESOLUTION, step=32),
                io.Boolean.Input("empty_image", display_name="空图条件", default=False,
                                 label_on="on", label_off="off",
                                 tooltip="纯文生图时，在采样早期给文本编码器一个空白图片槽位，软化 H3 偏硬的文生图渲染。"
                                         "接入参考图时该选项自动失效。"),
                io.Float.Input("empty_start", default=0.0, min=0.0, max=1.0, step=0.001,
                               tooltip="空图条件生效的起始 sigma 百分比。"),
                io.Float.Input("empty_end", default=0.25, min=0.0, max=1.0, step=0.001,
                               tooltip="空图条件生效的结束 sigma 百分比。0.25 约等于 6 步里的前 2 步。"),
                io.Vae.Input("vae", optional=True,
                             tooltip="可选。接入后参考图会额外编码成 H3 的 reference latents；不接则图片只进入文本编码器。"),
                io.Autogrow.Input("images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(input=io.Image.Input("image"),
                                                                      prefix="image_", min=0, max=100),
                                  tooltip="可选。参考图；image_1 是编辑底图，提示词里可用 <Picture 1> 引用。"),
                io.Combo.Input("vision_short_edge", display_name="视觉短边", options=_VISION_SHORT_EDGES,
                               default="auto",
                               tooltip="auto/1024 完全走官方 H3 Image 路径；其他值是实验性：只改 Qwen vision 编码的短边，"
                                       "VAE reference latent 仍按官方 1024~2048 编码。接图时生效。"),
                io.Float.Input("image_strength", display_name="图片条件权重", default=1.0, min=0.0, max=4.0,
                               step=0.05,
                               tooltip="仅在视觉短边为自定义值时生效，作用在 vision token 的 attention logit 上："
                                       "0 静音，1 中性，>1 强化。VAE reference latent 不受影响。"),
                io.Combo.Input("merge_separator", display_name="分隔符", options=list(MERGE_SEPARATORS),
                               default="none",
                               tooltip="preset 条目之间、以及 preset 块与用户提示词之间的分隔符。H3 对中文条目建议 none，"
                                       "空格可能让模型把条目文字画进画面；英文条目可用 space 或 newline。"),
            ],
            outputs=[io.Model.Output(display_name="model"),
                     io.Conditioning.Output(display_name="positive"),
                     io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, clip, model, text="", merge_order="before", cache_mode="enabled", presets_data="[]",
                width=0, height=0, empty_image=False, empty_start=0.0, empty_end=0.25,
                vae=None, images=None, vision_short_edge="auto", image_strength=1.0,
                merge_separator="none"):
        entries = parse_presets(presets_data)
        clean_text, user_terms = extract_weighted_terms(text or "")
        separator = MERGE_SEPARATORS.get(str(merge_separator), "")
        merged = merge_prompt(clean_text, entries, merge_order, separator=separator)
        terms = user_terms + entries
        image_list = _collect_images(images)

        if image_list and empty_image:
            logging.warning("%s: empty_image is ignored when reference images are wired", _NODE)
        use_empty_image = bool(empty_image) and not image_list
        start = max(0.0, min(1.0, float(empty_start)))
        end = max(0.0, min(1.0, float(empty_end)))

        edge_value = str(vision_short_edge)
        if edge_value in _OFFICIAL_VISION_EDGES:
            custom_edge = None
        else:
            try:
                custom_edge = int(edge_value)
            except ValueError as exc:
                raise ValueError(f"{_NODE}: invalid vision_short_edge {edge_value!r}") from exc
        vision_strength = float(image_strength)
        if custom_edge is None and vision_strength != 1.0:
            logging.info("%s: image_strength only applies to a custom vision_short_edge; ignored", _NODE)
            vision_strength = 1.0

        diffusion_model = model.get_model_object("diffusion_model") if model is not None else None
        if diffusion_model is None or not hasattr(diffusion_model, "blocks"):
            raise ValueError(f"{_NODE}: model is not a MiniMax H3 diffusion model")

        cache_enabled = cache_mode == "enabled"
        if custom_edge is None:
            cond, latent = _encode_conditioning(clip, merged, width, height, vae, images, cache_enabled)
        else:
            cond, latent = _encode_custom_conditioning(
                clip, merged, width, height, vae, images, custom_edge, cache_enabled
            )

        ids = _token_ids(clip.tokenize(merged))
        prompt_len = len(ids)
        pairs = positions_from_ids(clip, ids, prompt_len, terms, _H3_SPECIALS)

        if use_empty_image:
            if end > start:
                picture, picture_dict = _encode_empty_picture(clip, cache_enabled)
                cond = _apply_empty_image(cond, picture, picture_dict, start, end)
            else:
                logging.warning("%s: empty_end (%s) must be greater than empty_start (%s); "
                                "empty-image mode skipped", _NODE, end, start)

        vision_positions_by_len = None
        if custom_edge is not None and vision_strength != 1.0:
            vision_positions_by_len = _vision_positions_by_length(cond)

        if not pairs and not vision_positions_by_len:
            return io.NodeOutput(model, cond, latent)

        logging.info("%s: weighting %d text positions; vision_short_edge=%s; image_strength=%s",
                     _NODE, len(pairs), custom_edge if custom_edge is not None else "official",
                     vision_strength)
        return io.NodeOutput(
            _patch_model(model, pairs, prompt_len, vision_strength, vision_positions_by_len),
            cond, latent,
        )
