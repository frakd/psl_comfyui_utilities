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
import threading
import weakref
from collections import OrderedDict

import comfy.patcher_extension
import torch

from comfy_api.latest import io

from .prompt_weight_core import (
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


def _encode_conditioning(clip, prompt, width, height, vae, images, cache_enabled):
    image_list = _collect_images(images)
    identity = tuple([clip, *image_list] + ([vae] if vae is not None else []))
    extra = (
        "h3",
        int(width),
        int(height),
        len(image_list),
        tuple((id(image), tuple(image.shape)) for image in image_list),
    )
    key = cache_key(clip, prompt, extra)
    if cache_enabled:
        cached = _COND_CACHE.get(key, identity)
        if cached is not _MISS:
            return cached

    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning
    except ImportError as exc:
        raise RuntimeError(f"{_NODE} requires a ComfyUI version with MiniMax H3 Image support") from exc

    cond, latent = MiniMaxH3ImageConditioning.execute(
        clip, prompt, width, height, vae=vae, images=images
    ).result
    if cache_enabled:
        _COND_CACHE.put(key, (cond, latent), identity)
    return cond, latent


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


def _make_forward_wrapper(pairs, prompt_len, name):
    value_pos = [position for position, factor, _ in pairs if factor != 1.0]
    value_factor = [factor for _, factor, _ in pairs if factor != 1.0]
    bias_pos = [position for position, _, bias in pairs if bias != 0.0]
    bias_value = [bias for _, _, bias in pairs if bias != 0.0]
    warned = set()

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
                device = context.device
                transformer_options[_STATE_KEY] = (
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
    value_idx, value_factor, bias_idx, bias_value = weights

    def override(func, q, k, v, heads, mask=None, **kwargs):
        if value_idx is not None:
            factor = value_factor.to(v.dtype)
            v[..., value_idx, :] = v[..., value_idx, :] * factor
        if bias_idx is not None and mask is None:
            key_count = k.shape[-2] if kwargs.get("skip_reshape") else k.shape[1]
            bias = q.new_zeros(1, 1, 1, key_count)
            bias[..., bias_idx] = bias_value.to(q.dtype)
            # The default comfy backend understands additive masks; a custom
            # sparse/int8 backend may drop them, so avoid the preferred backend
            # when an emphasis bias has to ride along.
            return func(q, k, v, heads, mask=bias, **kwargs)
        if previous is not None:
            return previous(func, q, k, v, heads, mask=mask, **kwargs)
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


def _patch_model(model, pairs, prompt_len):
    patched = model.clone()
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        _WRAPPER_KEY,
        _make_forward_wrapper(pairs, prompt_len, _NODE),
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
            ],
            outputs=[io.Model.Output(display_name="model"),
                     io.Conditioning.Output(display_name="positive"),
                     io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, clip, model, text="", merge_order="before", cache_mode="enabled", presets_data="[]",
                width=0, height=0, empty_image=False, empty_start=0.0, empty_end=0.25,
                vae=None, images=None):
        entries = parse_presets(presets_data)
        clean_text, user_terms = extract_weighted_terms(text or "")
        merged = merge_prompt(clean_text, entries, merge_order)
        terms = user_terms + entries
        image_list = _collect_images(images)

        if image_list and empty_image:
            logging.warning("%s: empty_image is ignored when reference images are wired", _NODE)
        use_empty_image = bool(empty_image) and not image_list
        start = max(0.0, min(1.0, float(empty_start)))
        end = max(0.0, min(1.0, float(empty_end)))

        diffusion_model = model.get_model_object("diffusion_model") if model is not None else None
        if diffusion_model is None or not hasattr(diffusion_model, "blocks"):
            raise ValueError(f"{_NODE}: model is not a MiniMax H3 diffusion model")

        cache_enabled = cache_mode == "enabled"
        cond, latent = _encode_conditioning(clip, merged, width, height, vae, images, cache_enabled)

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

        if not pairs:
            return io.NodeOutput(model, cond, latent)

        logging.info("%s: weighting %d text positions", _NODE, len(pairs))
        return io.NodeOutput(_patch_model(model, pairs, prompt_len), cond, latent)
