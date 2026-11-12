# -*- coding: utf-8 -*-
"""MOSS-Image prompt preset weighting for the official ComfyUI model.

The node merges a workflow-persisted list of preset snippets into the user
prompt, then applies per-token attention weighting:

* weight > 1 adds an additive bias to the token's attention logits;
* weight < 1 scales the token's value vector (0 removes it, negatives subtract).

The conditioning carries the emphasis bias through `context_attention_bias`
when the core model supports it; value scaling is applied by an attention
override inside every transformer block.

Optional edit inputs mirror the official TextEncodeMOSSImageEdit node: images
can be sent to the text encoder as vision tokens (with the same "Picture N:"
prefix and 384 px cap) and, when a VAE is wired, encoded as reference latents
for the edit LoRA. The per-token weights ride in the same attention bias as
the official image_strength, so text weighting and image conditioning apply
together.
"""

import inspect
import logging
import math

import comfy.patcher_extension
import torch

from comfy_api.latest import io

from .prompt_weight_core import (
    encode_layout,
    extract_weighted_terms,
    merge_prompt,
    parse_presets,
    positions_from_ids,
)

try:
    from comfy_extras.nodes_moss_image import EDIT_FORMATS
except ImportError:  # unit tests run without ComfyUI's comfy_extras on sys.path
    EDIT_FORMATS = {
        "ai-toolkit": ("index_timestep_zero_isolated", True),
        "ai-toolkit (kv_cache off)": ("index_timestep_zero", True),
        "Moss-Image-Edit": ("index_timestep_zero_text_start", False),
    }

_NODE = "MOSSImagePromptWeightPreset"
_WRAPPER_KEY = _NODE
_STATE_KEY = "moss_prompt_preset"

_VISION_SIZE = 384
_VISION_PREFIX = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
_DEFAULT_EDIT_FORMAT = "Moss-Image-Edit"


def _positive_rows(transformer_options, batch):
    codes = transformer_options.get("cond_or_uncond")
    if not isinstance(codes, (list, tuple)) or not codes or len(codes) > batch:
        return None
    if batch % len(codes) != 0:
        return None
    chunk = batch // len(codes)
    return [index for index in range(batch) if codes[index // chunk] == 0]


def _has_context_bias(diffusion_model):
    try:
        return "context_attention_bias" in inspect.signature(type(diffusion_model)._forward).parameters
    except (AttributeError, TypeError, ValueError):
        return False


def _collect_images(images):
    if not isinstance(images, dict):
        return []
    return [images[name] for name in sorted(images) if images.get(name) is not None]


def _resize_for_vision(image):
    import comfy.utils

    # The text encoder sees each reference downscaled to at most 384x384 pixels,
    # exactly like the official editor.
    scale = min(1.0, math.sqrt(_VISION_SIZE * _VISION_SIZE / (image.shape[1] * image.shape[2])))
    return comfy.utils.common_upscale(
        image.movedim(-1, 1), round(image.shape[2] * scale), round(image.shape[1] * scale), "area", "disabled"
    ).movedim(1, -1)


def _vision_slot_counter():
    import comfy.text_encoders.qwen_vl

    counts = {}

    def slots(token):
        data = token.get("data")
        if not torch.is_tensor(data):
            return None
        key = (id(data), tuple(data.shape))
        if key in counts:
            return counts[key]
        try:
            # Qwen3.5 merges 2x2 patches of 16 px into one conditioning token.
            _, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(data, patch_size=16)
            count = int(grid[0, 1] * grid[0, 2]) // 4
        except Exception as exc:
            counts[key] = None
            logging.warning(f"{_NODE}: cannot measure a vision token ({exc}); per-token weighting skipped")
            return None
        counts[key] = count
        return count

    return slots


def _image_bias(ids, strength):
    if not ids or strength == 1.0:
        return None
    weight = math.log(strength)
    if not math.isfinite(weight):
        return None
    # Vision slots are the only negative markers in the expanded layout; log()
    # scales their attention share, text tokens stay unbiased here.
    return torch.tensor([[weight if slot < 0 else 0.0 for slot in ids]])


def _attach_references(cond, vae, images, method):
    if vae is None or not images:
        return cond
    import node_helpers

    refs = [vae.encode(image[:, :, :, :3]) for image in images]
    return node_helpers.conditioning_set_values(
        cond, {"reference_latents": refs, "reference_latents_method": method}
    )


def _make_forward_wrapper(pairs, text_len):
    value_pos = [position for position, factor, _ in pairs if factor != 1.0]
    value_factor = [factor for _, factor, _ in pairs if factor != 1.0]
    bias_pos = [position for position, _, bias in pairs if bias != 0.0]
    bias_value = [bias for _, _, bias in pairs if bias != 0.0]
    warned = set()

    def warn_once(key, message):
        if key not in warned:
            warned.add(key)
            logging.warning(f"{_NODE}: {message}")

    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        batch, _, height, width = x.shape
        rows = _positive_rows(transformer_options, batch)
        if context.shape[1] != text_len:
            warn_once("length", f"conditioning length {context.shape[1]} is not this node's {text_len}; "
                                "weighting skipped (wire this node's conditioning output to the sampler)")
        elif rows is None:
            warn_once("rows", "cond_or_uncond is missing or uneven; cannot separate positive from negative, "
                              "weighting skipped")
        elif rows:
            device = x.device
            image_tokens = height * width
            v_idx = torch.tensor(value_pos, device=device) + image_tokens if value_pos else None
            v_factor = torch.tensor(value_factor, device=device, dtype=x.dtype) if value_pos else None
            bias = None
            if bias_pos:
                bias = torch.zeros(batch, 1, 1, image_tokens + text_len, device=device, dtype=x.dtype)
                positions = torch.tensor(bias_pos, device=device) + image_tokens
                values = torch.tensor(bias_value, device=device, dtype=x.dtype)
                for row in rows:
                    bias[row, 0, 0, positions] = values
            transformer_options[_STATE_KEY] = (rows, v_idx, v_factor, bias)
        return executor(x, timestep, context, transformer_options, **kwargs)

    return wrapper


def _make_attention_override(weights, previous):
    rows, v_idx, v_factor, bias = weights

    def override(func, q, k, v, heads, mask=None, **kwargs):
        if v_idx is not None:
            factor = v_factor.to(v.dtype)
            # The core attention reshapes v to [B, heads, N, dim] only after this
            # override, so v normally arrives as [B, N, dim]. Keep the 4D branch.
            factor = factor[:, None, None] if v.ndim == 4 else factor[:, None]
            for row in rows:
                v[row, v_idx] = v[row, v_idx] * factor
        if bias is not None and mask is None:
            key_count = k.shape[-2] if kwargs.get("skip_reshape") else k.shape[1]
            current = bias
            if current.shape[-1] < key_count:
                current = torch.nn.functional.pad(current, (0, key_count - current.shape[-1]))
            elif current.shape[-1] > key_count:
                current = current[..., :key_count]
            # A backend that ignores attn_mask through **kwargs cannot apply the
            # bias; comfy's own backends either take it or fall back to sdpa.
            return func(q, k, v, heads, mask=current.to(q.dtype), **kwargs)
        if previous is not None:
            return previous(func, q, k, v, heads, mask=mask, **kwargs)
        return func(q, k, v, heads, mask=mask, **kwargs)

    return override


def _make_block_patch(previous):
    def patch(args, extra):
        options = args["transformer_options"]
        weights = options.get(_STATE_KEY)
        if weights is not None:
            options = {
                **options,
                "optimized_attention_override": _make_attention_override(
                    weights, options.get("optimized_attention_override")
                ),
            }
            args = {**args, "transformer_options": options}
        if previous is not None:
            return previous(args, extra)
        return extra["original_block"](args)

    return patch


def _patch_model(model, pairs, text_len):
    patched = model.clone()
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        _WRAPPER_KEY,
        _make_forward_wrapper(pairs, text_len),
    )
    diffusion_model = patched.get_model_object("diffusion_model")
    existing = patched.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    for index in range(len(diffusion_model.layers)):
        patched.set_model_patch_replace(
            _make_block_patch(existing.get(("single_block", index))), "dit", "single_block", index
        )
    return patched


def _apply_conditioning_bias(cond, pairs, cond_len):
    bias_terms = [(position, bias) for position, _, bias in pairs if bias != 0.0]
    if not bias_terms:
        return cond, pairs
    added = torch.zeros(1, cond_len)
    for position, value in bias_terms:
        added[0, position] = value

    combined = []
    for tensor, metadata in cond:
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        existing = metadata.get("context_attention_bias")
        if torch.is_tensor(existing):
            if existing.shape[-1] == added.shape[-1]:
                # The official edit encoder also uses this field for image_strength;
                # both biases are additive, so keep both.
                metadata["context_attention_bias"] = existing.to(added.dtype) + added.to(existing.device)
            else:
                logging.warning(f"{_NODE}: existing context_attention_bias has length {existing.shape[-1]}, "
                                f"expected {added.shape[-1]}; replacing it")
                metadata["context_attention_bias"] = added
        elif existing is not None:
            logging.warning(f"{_NODE}: existing context_attention_bias is not a tensor; replacing it")
            metadata["context_attention_bias"] = added
        else:
            metadata["context_attention_bias"] = added
        combined.append([tensor, metadata])
    return combined, [(position, factor, 0.0) for position, factor, _ in pairs if factor != 1.0]


class MOSSImagePromptWeightPreset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=_NODE,
            display_name="MOSS-Image Prompt Weight (Presets)",
            category="MOSS-Image/prompt",
            description="Merges a workflow-persisted list of plain preset snippets into the positive prompt and "
                        "applies KJ-style per-token attention weighting; weights apply to every occurrence of the "
                        "matched phrase. Entry weight 1 = neutral, >1 emphasizes, <1 weakens, 0 removes, <0 "
                        "subtracts the concept. The merged-text encoding is cached by CLIP, patch state and text, "
                        "so upstream re-runs skip the text encoder while weight changes still apply. Optional "
                        "images/vae/edit_format/image_strength mirror the official TextEncodeMOSSImageEdit node, so "
                        "text weighting and image conditioning apply in one node. Requires the official ComfyUI "
                        "MOSS-Image model; wire both model and conditioning outputs to the sampler.",
            inputs=[
                io.Clip.Input("clip"),
                io.Model.Input("model"),
                io.String.Input("text", multiline=True, dynamic_prompts=True, default="",
                                tooltip="用户正面提示词。这里仍可使用 (phrase:weight) 语法；列表中的预设条目会合并到这段文本的开头或结尾。"),
                io.Combo.Input("merge_order", options=["before", "after"], default="before",
                               tooltip="预设条目合并到用户提示词的前面（before）还是后面（after）。条目之间用空格连接，按列表顺序排列。"),
                io.Combo.Input("cache_mode", display_name="缓存", options=["enabled", "disabled"], default="enabled",
                               tooltip="enabled：同一 CLIP + 同一合并文本命中缓存，直接复用上次的 conditioning，"
                                       "跳过 Qwen 文本编码器；disabled：每次重新编码。\n"
                                       "只改预设权重不会使缓存失效，权重仍会重新计算并施加。"),
                io.String.Input("presets_data", default="[]", socketless=True, advanced=True,
                                tooltip="由节点前端管理的 JSON 条目列表 [{enabled, text, weight}]；text 是纯文本，weight 由滑块设置（-2..2，负值反向缩放 value 以抑制概念），一般不需要手动修改。"),
                io.Vae.Input("vae", optional=True,
                             tooltip="可选。接入后参考图会编码成 reference latents 供编辑 LoRA 使用；不接则图片只进入文本编码器（与官方编辑节点一致）。"),
                io.Combo.Input("edit_format", display_name="编辑格式", options=list(EDIT_FORMATS),
                               default=_DEFAULT_EDIT_FORMAT,
                               tooltip="编辑 LoRA 训练时的布局：ai-toolkit 的两种（参考图同时进入文本编码器）或 Moss-Image-Edit。"
                                       "仅在接入图片时生效，未接图片时该选项与 image_strength 会自动隐藏。"),
                io.Autogrow.Input("images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(input=io.Image.Input("image"),
                                                                      prefix="image_", min=0, max=10),
                                  tooltip="可选。参考图按接入尺寸使用，建议先缩放；文本编码器看到的是缩到最长边 384 像素的副本。"),
                io.Float.Input("image_strength", default=1.0, min=0.0, max=100.0, step=0.05,
                               tooltip="图片在文本编码器注意力里的权重（ai-toolkit 格式，或不接 VAE 时）：0 忽略，1 不变，>1 强调。"
                                       "Moss-Image-Edit 格式下图片不进入文本编码器，该值不生效。"),
            ],
            outputs=[io.Model.Output(display_name="model"),
                     io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(cls, clip, model, text="", merge_order="before", cache_mode="enabled", presets_data="[]",
                vae=None, edit_format=_DEFAULT_EDIT_FORMAT, images=None, image_strength=1.0):
        entries = parse_presets(presets_data)
        clean_text, user_terms = extract_weighted_terms(text or "")
        merged = merge_prompt(clean_text, entries, merge_order)

        image_list = _collect_images(images)
        method, vision = EDIT_FORMATS.get(str(edit_format), EDIT_FORMATS[_DEFAULT_EDIT_FORMAT])
        strength = float(image_strength)
        # ai-toolkit formats, or a missing VAE, send the images through the text
        # encoder; strength 0 removes them there entirely.
        use_vision = bool(image_list) and (vision or vae is None) and strength > 0.0

        prompt = merged
        vl = []
        slots = None
        cache_extra = ()
        cache_identity = ()
        if image_list:
            cache_extra = (str(edit_format), len(image_list),
                           tuple((id(image), tuple(image.shape)) for image in image_list))
            cache_identity = tuple(image_list)
        if use_vision:
            vl = [_resize_for_vision(image) for image in image_list]
            prompt = "".join(_VISION_PREFIX.format(index + 1) for index in range(len(vl))) + merged
            slots = _vision_slot_counter()

        cond, ids, cond_len = encode_layout(
            clip, prompt, cache=cache_mode == "enabled", images=vl or None,
            vision_slots=slots, extra=cache_extra, identity=cache_identity,
        )
        layout_ok = ids is not None and (slots is None or len(ids) == cond_len)
        if ids is None:
            logging.warning(f"{_NODE}: the prompt contains vision tokens this node cannot measure; "
                            "per-token weighting skipped")
        elif not layout_ok:
            logging.warning(f"{_NODE}: expanded prompt has {len(ids)} positions but the conditioning has "
                            f"{cond_len}; per-token weighting skipped")
        if vl and layout_ok:
            bias = _image_bias(ids, strength)
            if bias is not None:
                import node_helpers
                cond = node_helpers.conditioning_set_values(cond, {"context_attention_bias": bias})
        cond = _attach_references(cond, vae, image_list, method)

        pairs = positions_from_ids(clip, ids, cond_len, user_terms + entries) if layout_ok else []
        if not pairs:
            return io.NodeOutput(model, cond)

        diffusion_model = model.get_model_object("diffusion_model")
        if (diffusion_model is None or not hasattr(diffusion_model, "_forward")
                or not hasattr(diffusion_model, "layers") or not hasattr(diffusion_model, "text_embedder")):
            raise ValueError(f"{_NODE}: model is not the official ComfyUI MOSS-Image diffusion model; "
                             "use the official MOSS-Image loader")

        logging.info("%s: weighting %d text positions", _NODE, len(pairs))
        if _has_context_bias(diffusion_model):
            cond, pairs = _apply_conditioning_bias(cond, pairs, cond_len)
            if not pairs:
                return io.NodeOutput(model, cond)
        return io.NodeOutput(_patch_model(model, pairs, cond_len), cond)
