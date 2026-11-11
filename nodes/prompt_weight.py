# -*- coding: utf-8 -*-
"""MOSS-Image prompt preset weighting for the official ComfyUI model.

The node merges a workflow-persisted list of preset snippets into the user
prompt, then applies per-token attention weighting:

* weight > 1 adds an additive bias to the token's attention logits;
* weight < 1 scales the token's value vector (0 removes it, negatives subtract).

The conditioning carries the emphasis bias through `context_attention_bias`
when the core model supports it; value scaling is applied by an attention
override inside every transformer block.
"""

import inspect
import logging

import comfy.patcher_extension
import torch

from comfy_api.latest import io

from .prompt_weight_core import extract_weighted_terms, locate_pairs, merge_prompt, parse_presets

_NODE = "MOSSImagePromptWeightPreset"
_WRAPPER_KEY = _NODE
_STATE_KEY = "moss_prompt_preset"


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
                        "so upstream re-runs skip the text encoder while weight changes still apply. Requires the "
                        "official ComfyUI MOSS-Image model; wire both model and conditioning outputs to the sampler.",
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
            ],
            outputs=[io.Model.Output(display_name="model"),
                     io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(cls, clip, model, text="", merge_order="before", cache_mode="enabled", presets_data="[]"):
        entries = parse_presets(presets_data)
        clean_text, user_terms = extract_weighted_terms(text or "")
        merged = merge_prompt(clean_text, entries, merge_order)
        cond, pairs, cond_len = locate_pairs(clip, merged, user_terms + entries, cache=cache_mode == "enabled")
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
