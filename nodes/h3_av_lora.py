import logging
import re

import torch
import torch.nn.functional as F

import folder_paths
import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.utils
import comfy.weight_adapter
from comfy.patcher_extension import PatcherInjection
from comfy_api.latest import io


MAX_LORAS = 8


class _LoraConfig:
    def __init__(self, lora_name, strength, audio_weight, video_weight):
        self.lora_name = lora_name
        self.strength = float(strength)
        self.audio_weight = float(audio_weight)
        self.video_weight = float(video_weight)


class _LoraEntry:
    def __init__(self, config, adapter, tag_weights, sparse=False, scalar=1.0):
        self.config = config
        self.adapter = adapter
        # per-tag delta weight, indexed by the packed-row tag:
        # (0 video/cond/ref-image, 1 text, 2 audio)
        self.tag_weights = tuple(float(w) for w in tag_weights)
        # residual entries only cover a minority of rows; compute their delta on
        # gathered rows instead of the full activation
        self.sparse = bool(sparse)
        self.scalar = float(scalar)


def _module_path(key):
    path = key
    if path.startswith("diffusion_model."):
        path = path[len("diffusion_model."):]
    if path.endswith(".weight"):
        path = path[:-len(".weight")]
    return path


def _get_module(dm, key):
    module = dm
    for part in _module_path(key).split("."):
        if module is None:
            return None
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part, None)
    return module


def _classify_key(key):
    path = _module_path(key)
    if re.match(r"blocks\.\d+\.(attn|mlp)\..+", path):
        return "token"
    if re.match(r"blocks\.\d+\.adaln_proj\.linear", path):
        return "adaln"
    if re.match(r"final_layer\.adaln_proj\.linear", path):
        return "final_adaln"
    if path.endswith("final_layer.video_out") or path.endswith("video_patch_proj"):
        return "video_scalar"
    if path.endswith("final_layer.audio_out") or path.endswith("audio_patch_proj"):
        return "audio_scalar"
    return "global"


def _adapter_pair(adapter, device, dtype, keep_on_gpu):
    if not keep_on_gpu:
        v = adapter.weights
        return v[1].to(device=device, dtype=dtype), v[0].to(device=device, dtype=dtype)
    cache = getattr(adapter, "_h3_av_pair_cache", None)
    if cache is not None and cache[0] == device and cache[1] == dtype:
        return cache[2], cache[3]
    v = adapter.weights
    down = v[1].to(device=device, dtype=dtype)
    up = v[0].to(device=device, dtype=dtype)
    adapter._h3_av_pair_cache = (device, dtype, down, up)
    return down, up


def _linear_lora(adapter, x, keep_on_gpu=False):
    v = adapter.weights
    up, down = v[0], v[1]
    if up.ndim != 2 or down.ndim != 2:
        return None, 1.0
    if down.shape[1] != x.shape[-1] or up.shape[1] != down.shape[0]:
        return None, 1.0
    down, up = _adapter_pair(adapter, x.device, x.dtype, keep_on_gpu)
    rank = down.shape[0]
    scale = (v[2] / rank) if v[2] is not None else 1.0
    return F.linear(F.linear(x, down), up), scale


def _swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul_(up)


def _is_supported_lora_adapter(adapter):
    if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
        return True
    v = adapter.weights
    return len(v) <= 3 or (v[3] is None and v[4] is None and v[5] is None)


_MINIMAX_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


def _minimax_lora_base(key):
    for suffix in _MINIMAX_LORA_SUFFIXES:
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return None


def _linear_lora_pair_matches(target_shape, lora_a, lora_b):
    if len(target_shape) != 2 or len(lora_a.shape) != 2 or len(lora_b.shape) != 2:
        return True
    return (
        lora_a.shape[0] == lora_b.shape[1]
        and lora_a.shape[1] == target_shape[1]
        and lora_b.shape[0] == target_shape[0]
    )


def _prepare_minimax_h3_lora(inner, lora):
    if inner is None or not hasattr(inner, "state_dict"):
        return lora, 0

    model_state = inner.state_dict()
    normalized = {}
    renamed = 0
    for key, value in lora.items():
        normalized_key = key
        base = _minimax_lora_base(key)
        if base is not None and not base.startswith("diffusion_model."):
            candidate = "diffusion_model.{}".format(key)
            candidate_base = _minimax_lora_base(candidate)
            if "{}.weight".format(candidate_base) in model_state:
                normalized_key = candidate
                renamed += 1
        normalized[normalized_key] = value

    incompatible_bases = set()
    bases = {
        base
        for key in normalized
        if (base := _minimax_lora_base(key)) is not None
    }
    for base in bases:
        target = model_state.get("{}.weight".format(base))
        lora_a = normalized.get("{}.lora_A.weight".format(base))
        lora_b = normalized.get("{}.lora_B.weight".format(base))
        if target is None or lora_a is None or lora_b is None:
            continue
        if not _linear_lora_pair_matches(target.shape, lora_a, lora_b):
            incompatible_bases.add(base)

    if incompatible_bases:
        for base in incompatible_bases:
            normalized.pop("{}.lora_A.weight".format(base), None)
            normalized.pop("{}.lora_B.weight".format(base), None)
            normalized.pop("{}.alpha".format(base), None)
        logging.warning(
            "[H3 AV LoRA] skipped %d adapter pair(s) whose shapes do not match "
            "the current H3 model (full-model/curve AdaLN mismatch)",
            len(incompatible_bases),
        )

    if renamed:
        logging.info(
            "[H3 AV LoRA] normalized %d native MiniMax H3 LoRA key(s) for ComfyUI.",
            renamed,
        )
    return normalized, len(incompatible_bases)


def _adapter_shape_matches(adapter, module):
    if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
        return True
    target = getattr(module, "weight", None)
    if target is None or not hasattr(target, "shape"):
        return True
    v = adapter.weights
    if len(v) < 2:
        return False
    up, down = v[0], v[1]
    if up.ndim != 2 or down.ndim != 2:
        return False
    if up.shape[1] != down.shape[0]:
        return False
    if len(target.shape) == 2:
        return up.shape[0] == target.shape[0] and down.shape[1] == target.shape[1]
    return True


def _move_adapter_to_device(adapter):
    if not hasattr(adapter, "weights") or adapter.weights is None:
        return
    device = comfy.model_management.get_torch_device()
    weights = []
    for w in adapter.weights:
        if isinstance(w, torch.Tensor):
            try:
                w = w.to(device=device)
            except Exception:
                pass
        weights.append(w)
    adapter.weights = tuple(weights)


# Masks only depend on the packed-row tags and the per-tag weights; the same
# mod_segments object is shared by every block within one forward, so cache the
# built mask and let all blocks reuse it. A new segments object (next sampling
# step) invalidates the cache.
_MASK_CACHE = {}
_MASK_CACHE_SEGMENTS = None


def _build_token_mask(n, segments, tag_weights, device, dtype):
    mask = torch.full((n,), float(tag_weights[1]), device=device, dtype=dtype)
    table = torch.tensor(tag_weights, device=device, dtype=dtype)
    for a, b, row in segments:
        if isinstance(row, torch.Tensor):
            tags = (row % 3).to(device=device, dtype=torch.long)
            mask[a:b] = table[tags]
        else:
            mask[a:b] = float(tag_weights[int(row) % 3])
    return mask


def _build_modality_mask(n, segments, tag_weights, device, dtype):
    mask = torch.full((n, 3), float(tag_weights[1]), device=device, dtype=dtype)
    for a, b, row in segments:
        if isinstance(row, torch.Tensor):
            idx = (row // 3).long()
            tags = (row % 3).long()
            for i in torch.unique(idx):
                tag = int(tags[idx == i][0])
                mask[int(i), tag] = float(tag_weights[tag])
        else:
            row_index = int(row)
            mask[row_index // 3, row_index % 3] = float(tag_weights[row_index % 3])
    return mask


def _cached_mask(kind, n, segments, tag_weights, device, dtype):
    global _MASK_CACHE_SEGMENTS
    if segments is not _MASK_CACHE_SEGMENTS:
        _MASK_CACHE.clear()
        _MASK_CACHE_SEGMENTS = segments
    key = (kind, n, tag_weights, device, dtype)
    mask = _MASK_CACHE.get(key)
    if mask is None:
        builder = _build_token_mask if kind == "token" else _build_modality_mask
        mask = builder(n, segments, tag_weights, device, dtype)
        _MASK_CACHE[key] = mask
    return mask


def _build_single_final_mask(n, seg, weight, device, dtype=torch.float32):
    mask = torch.ones(n, device=device, dtype=dtype)
    if seg is None:
        return mask
    row = seg[2]
    if isinstance(row, torch.Tensor):
        mask[row.unique().long()] = float(weight)
    else:
        mask[int(row)] = float(weight)
    return mask


class _AVModuleBypass:
    _warned_shape_keys = set()
    _warned_chunk_keys = set()

    def __init__(self, module, entries, mask_mode="token", chunks=1, keep_on_gpu=False):
        self.module = module
        self.original_forward = None
        self.entries = entries
        self.mask_mode = mask_mode
        self.masks = None
        self.scalars = None
        self.chunks = int(chunks)
        self.chunk_offset = None
        self.chunk_total = None
        self.keep_on_gpu = bool(keep_on_gpu)
        self.shape_warned = False

    def install(self):
        if getattr(self.module, "_h3_av_bypasses", None) is None:
            self.module._h3_av_original_forward = self.module.forward
            self.module._h3_av_bypasses = []
            self.module.forward = self._dispatch
        if self not in self.module._h3_av_bypasses:
            self.module._h3_av_bypasses.append(self)
        return True

    def eject(self):
        bypasses = getattr(self.module, "_h3_av_bypasses", None)
        if bypasses is None or self not in bypasses:
            return
        bypasses.remove(self)
        if bypasses:
            self.module.forward = bypasses[0]._dispatch
            return
        self.module.forward = self.module._h3_av_original_forward
        del self.module._h3_av_bypasses
        del self.module._h3_av_original_forward

    def _dispatch(self, x, *args, **kwargs):
        base = self.module._h3_av_original_forward(x, *args, **kwargs)
        for wrapper in self.module._h3_av_bypasses:
            base = wrapper._apply(x, base)
        return base

    def set_masks(self, masks):
        self.masks = masks

    def set_scalars(self, scalars):
        self.scalars = scalars

    def clear_masks(self):
        self.masks = None

    def set_chunk(self, offset, total):
        self.chunk_offset = offset
        self.chunk_total = total

    def clear_chunk(self):
        self.chunk_offset = None
        self.chunk_total = None

    def _apply_mask_scale_legacy(self, delta, entry, idx, scale):
        if self.mask_mode in ("token", "adaln"):
            mask = self.masks[idx] if self.masks is not None else None
            if mask is not None:
                m = mask.to(device=delta.device, dtype=delta.dtype)
                if (m.ndim == 2 and delta.ndim == 2
                        and delta.shape[0] == m.shape[0]
                        and delta.shape[1] % m.shape[1] == 0):
                    delta.view(delta.shape[0], m.shape[1], -1).mul_(m.unsqueeze(-1))
                    return scale
                if (self.chunk_offset is not None and self.chunk_total is not None
                        and m.ndim == 1 and m.shape[0] == self.chunk_total):
                    start = self.chunk_offset
                    end = min(start + delta.shape[0], m.shape[0])
                    if start < end:
                        m = m[start:end]
                if delta.ndim == 2 and delta.shape[0] == m.shape[0]:
                    delta.mul_(m.unsqueeze(-1))
                elif delta.ndim == 3 and delta.shape[1] == m.shape[0]:
                    delta.mul_(m.view(1, -1, 1))
                elif delta.ndim == 1 and delta.shape[0] == m.shape[0]:
                    delta.mul_(m)
                else:
                    divisor = None
                    if self.chunks > 1 and m.shape[0] == delta.shape[0] * self.chunks:
                        divisor = self.chunks
                    elif m.shape[0] and delta.shape[0] and m.shape[0] % delta.shape[0] == 0:
                        divisor = m.shape[0] // delta.shape[0]
                    if divisor is not None:
                        scale /= float(divisor)
                        warning_key = (entry.config.lora_name, tuple(m.shape), tuple(delta.shape), divisor)
                        if warning_key not in _AVModuleBypass._warned_chunk_keys:
                            _AVModuleBypass._warned_chunk_keys.add(warning_key)
                            logging.warning(
                                "[H3 AV LoRA] chunked module detected for %s: mask %s, delta %s, "
                                "falling back to per-chunk divisor %d without AV mask",
                                entry.config.lora_name, tuple(m.shape), tuple(delta.shape), divisor
                            )
                    else:
                        warning_key = (entry.config.lora_name, tuple(m.shape), tuple(delta.shape))
                        if warning_key not in _AVModuleBypass._warned_shape_keys:
                            _AVModuleBypass._warned_shape_keys.add(warning_key)
                            logging.warning(
                                "[H3 AV LoRA] mask shape %s does not match delta %s for %s; skipping this module delta",
                                tuple(m.shape), tuple(delta.shape), entry.config.lora_name
                            )
                        return None
        elif self.mask_mode == "scalar":
            if self.scalars is not None:
                scale *= float(self.scalars[idx])
        return scale

    def _resolve_token_mask(self, idx, n):
        if self.masks is None:
            return None
        mask = self.masks[idx]
        if mask is None or mask.ndim != 1:
            return None
        if mask.shape[0] == n:
            return mask
        if (self.chunk_offset is not None and self.chunk_total is not None
                and mask.shape[0] == self.chunk_total):
            start = self.chunk_offset
            end = min(start + n, mask.shape[0])
            if start < end:
                return mask[start:end]
        return None

    def _add_sparse_delta(self, base, x, entry, mask):
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        if rows.numel() == 0:
            return True
        if rows.numel() * 4 > mask.shape[0]:
            return False
        delta, lora_scale = _linear_lora(entry.adapter, x.index_select(0, rows), self.keep_on_gpu)
        if delta is None:
            logging.warning(
                "[H3 AV LoRA] skipping incompatible LoRA delta for %s (%s)",
                entry.config.lora_name, type(self.module).__name__
            )
            return True
        if delta.device != base.device or delta.dtype != base.dtype:
            delta = delta.to(device=base.device, dtype=base.dtype)
        weights = mask.index_select(0, rows).unsqueeze(-1)
        if weights.dtype != delta.dtype:
            weights = weights.to(delta.dtype)
        scale = float(entry.config.strength) * float(lora_scale)
        delta.mul_(weights)
        if scale != 1.0:
            delta.mul_(scale)
        base.index_add_(0, rows, delta)
        return True

    def _add_masked_delta(self, base, delta, idx, scale):
        if self.mask_mode == "scalar":
            if self.scalars is not None:
                scale *= float(self.scalars[idx])
            base.add_(delta, alpha=scale)
            return base
        mask = self.masks[idx] if self.masks is not None else None
        if mask is None:
            base.add_(delta, alpha=scale)
            return base
        if mask.dtype != delta.dtype:
            mask = mask.to(delta.dtype)
        if (mask.ndim == 2 and delta.ndim == 2
                and delta.shape[0] == mask.shape[0]
                and delta.shape[1] % mask.shape[1] == 0):
            if delta.shape[1] == mask.shape[1]:
                base.addcmul_(delta, mask, value=scale)
            else:
                base.view(delta.shape[0], mask.shape[1], -1).addcmul_(
                    delta.view(delta.shape[0], mask.shape[1], -1), mask.unsqueeze(-1), value=scale
                )
            return base
        if mask.ndim == 1 and delta.ndim == 2:
            m = mask
            if (self.chunk_offset is not None and self.chunk_total is not None
                    and m.shape[0] == self.chunk_total and m.shape[0] != delta.shape[0]):
                start = self.chunk_offset
                end = min(start + delta.shape[0], m.shape[0])
                if start < end:
                    m = m[start:end]
            if m.shape[0] == delta.shape[0]:
                base.addcmul_(delta, m.unsqueeze(-1), value=scale)
                return base
        scale = self._apply_mask_scale_legacy(delta, self.entries[idx], idx, scale)
        if scale is not None:
            base.add_(delta, alpha=scale)
        return base

    def _apply(self, x, base):
        if not self.entries:
            return base

        # Accumulate each non-LoRA delta into the fresh module output to avoid
        # holding multiple full-size activations at once.
        non_lora_total = None
        for idx, entry in enumerate(self.entries):
            adapter = entry.adapter
            if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
                continue
            adapter.multiplier = 1.0
            try:
                delta = adapter.h(x, base)
            except Exception as e:
                logging.warning("[H3 AV LoRA] adapter delta failed for %s: %s", entry.config.lora_name, e)
                continue
            if delta is None:
                continue
            if delta.device != base.device or delta.dtype != base.dtype:
                delta = delta.to(device=base.device, dtype=base.dtype)

            scale = float(entry.config.strength)
            scale = self._apply_mask_scale_legacy(delta, entry, idx, scale)
            if scale is None:
                continue
            if non_lora_total is None:
                if scale != 1.0:
                    delta.mul_(scale)
                non_lora_total = delta
            else:
                non_lora_total.add_(delta, alpha=scale)

        if non_lora_total is not None:
            base.add_(non_lora_total)

        for idx, entry in enumerate(self.entries):
            adapter = entry.adapter
            if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
                continue
            adapter.multiplier = 1.0
            if entry.sparse and self.mask_mode == "token":
                mask = self._resolve_token_mask(idx, x.shape[0])
                if mask is not None and self._add_sparse_delta(base, x, entry, mask):
                    continue
            delta, lora_scale = _linear_lora(adapter, x, self.keep_on_gpu)
            if delta is None:
                logging.warning(
                    "[H3 AV LoRA] skipping incompatible LoRA delta for %s (%s)",
                    entry.config.lora_name, type(self.module).__name__
                )
                continue
            if delta.device != base.device or delta.dtype != base.dtype:
                delta = delta.to(device=base.device, dtype=base.dtype)

            scale = float(entry.config.strength) * float(lora_scale)
            base = self._add_masked_delta(base, delta, idx, scale)

        return base


class _H3BlockAVPatch:
    def __init__(self, block, child_wrappers):
        self.block = block
        self.original_forward = None
        self.child_wrappers = child_wrappers

    def install(self):
        if getattr(self.block, "_h3_av_block_patches", None) is None:
            self.block._h3_av_block_original_forward = self.block.forward
            self.block._h3_av_block_patches = []
            self.block.forward = self._dispatch
        if self not in self.block._h3_av_block_patches:
            self.block._h3_av_block_patches.append(self)
        return True

    def eject(self):
        patches = getattr(self.block, "_h3_av_block_patches", None)
        if patches is None or self not in patches:
            return
        patches.remove(self)
        if patches:
            self.block.forward = patches[0]._dispatch
            return
        self.block.forward = self.block._h3_av_block_original_forward
        del self.block._h3_av_block_patches
        del self.block._h3_av_block_original_forward

    def _dispatch(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, **kwargs):
        try:
            for patch in self.block._h3_av_block_patches:
                patch._prepare(x, t_emb, mod_segments)
            return self.block._h3_av_block_original_forward(
                x, t_emb, mod_segments, rope_freqs, transformer_options, **kwargs
            )
        finally:
            for patch in self.block._h3_av_block_patches:
                for wrapper in patch.child_wrappers:
                    wrapper.clear_masks()

    def _prepare(self, x, t_emb, mod_segments):
        for wrapper in self.child_wrappers:
            if wrapper.mask_mode == "token":
                wrapper.set_masks([
                    _cached_mask("token", x.shape[0], mod_segments, entry.tag_weights, x.device, x.dtype)
                    for entry in wrapper.entries
                ])
            elif wrapper.mask_mode == "adaln":
                wrapper.set_masks([
                    _cached_mask("adaln", t_emb.shape[0], mod_segments, entry.tag_weights, t_emb.device, t_emb.dtype)
                    for entry in wrapper.entries
                ])


class _H3FinalLayerAVPatch:
    def __init__(self, final_layer, child_wrappers):
        self.final_layer = final_layer
        self.original_forward = None
        self.child_wrappers = child_wrappers

    def install(self):
        if getattr(self.final_layer, "_h3_av_final_patches", None) is None:
            self.final_layer._h3_av_final_original_forward = self.final_layer.forward
            self.final_layer._h3_av_final_patches = []
            self.final_layer.forward = self._dispatch
        if self not in self.final_layer._h3_av_final_patches:
            self.final_layer._h3_av_final_patches.append(self)
        return True

    def eject(self):
        patches = getattr(self.final_layer, "_h3_av_final_patches", None)
        if patches is None or self not in patches:
            return
        patches.remove(self)
        if patches:
            self.final_layer.forward = patches[0]._dispatch
            return
        self.final_layer.forward = self.final_layer._h3_av_final_original_forward
        del self.final_layer._h3_av_final_patches
        del self.final_layer._h3_av_final_original_forward

    def _dispatch(self, x, t_emb, video_seg, audio_seg, *args, **kwargs):
        patches = self.final_layer._h3_av_final_patches
        video_masks = [patch._build_masks(t_emb, video_seg, 0) for patch in patches]
        audio_masks = [patch._build_masks(t_emb, audio_seg, 2) for patch in patches]

        try:
            for patch, per_wrapper in zip(patches, video_masks):
                patch._set_masks(per_wrapper)
            video_out = self.final_layer._h3_av_final_original_forward(
                x, t_emb, video_seg, audio_seg, *args, **kwargs)[0]
        finally:
            for patch in patches:
                patch._clear_masks()

        try:
            for patch, per_wrapper in zip(patches, audio_masks):
                patch._set_masks(per_wrapper)
            audio_out = self.final_layer._h3_av_final_original_forward(
                x, t_emb, video_seg, audio_seg, *args, **kwargs)[1]
        finally:
            for patch in patches:
                patch._clear_masks()
        return [video_out, audio_out]

    def _build_masks(self, t_emb, seg, tag):
        return [
            [
                _build_single_final_mask(t_emb.shape[0], seg, entry.tag_weights[tag], t_emb.device, t_emb.dtype)
                for entry in wrapper.entries
            ]
            for wrapper in self.child_wrappers
        ]

    def _set_masks(self, per_wrapper):
        for wrapper, masks in zip(self.child_wrappers, per_wrapper):
            wrapper.set_masks(masks)

    def _clear_masks(self):
        for wrapper in self.child_wrappers:
            wrapper.clear_masks()


class _H3MLPAVPatch:
    def __init__(self, mlp, chunks=1):
        self.mlp = mlp
        self.chunks = int(chunks)
        self.original_forward = None

    def install(self):
        if getattr(self.mlp, "_h3_av_mlp_refcount", None) is None:
            self.original_forward = self.mlp.forward
            self.mlp._h3_av_mlp_original_forward = self.original_forward
            self.mlp._h3_av_mlp_refcount = 1
            self.mlp.forward = self._call
        else:
            self.mlp._h3_av_mlp_refcount += 1
        return True

    def eject(self):
        refcount = getattr(self.mlp, "_h3_av_mlp_refcount", 0)
        if refcount <= 0:
            return
        refcount -= 1
        self.mlp._h3_av_mlp_refcount = refcount
        if refcount > 0:
            return
        self.mlp.forward = self.mlp._h3_av_mlp_original_forward
        del self.mlp._h3_av_mlp_original_forward
        del self.mlp._h3_av_mlp_refcount

    def _set_chunk(self, offset, total):
        for module in (self.mlp.fc1, self.mlp.fc2):
            for wrapper in getattr(module, "_h3_av_bypasses", ()):
                wrapper.set_chunk(offset, total)

    def _clear_chunk(self):
        for module in (self.mlp.fc1, self.mlp.fc2):
            for wrapper in getattr(module, "_h3_av_bypasses", ()):
                wrapper.clear_chunk()

    def _forward_chunk(self, x):
        return self.mlp.fc2(_swiglu(self.mlp.fc1(x)))

    def _call(self, x):
        if self.chunks > 1 and x.shape[0] > 1:
            out = torch.empty_like(x)
            offset = 0
            for chunk in torch.chunk(x, self.chunks, dim=0):
                self._set_chunk(offset, x.shape[0])
                try:
                    out[offset:offset + chunk.shape[0]] = self._forward_chunk(chunk)
                finally:
                    self._clear_chunk()
                offset += chunk.shape[0]
            return out
        self._set_chunk(0, x.shape[0])
        try:
            return self._forward_chunk(x)
        finally:
            self._clear_chunk()


class _H3AVInjection:
    def __init__(self, wrappers, mlp_patches, block_patches, final_patches):
        self.wrappers = wrappers
        self.mlp_patches = mlp_patches
        self.block_patches = block_patches
        self.final_patches = final_patches
        self.installed_wrappers = []
        self.installed_mlps = []
        self.installed_blocks = []
        self.installed_finals = []

    def inject(self, model_patcher):
        for wrapper in self.wrappers:
            for entry in wrapper.entries:
                # LoRA tensors are cast per module in _linear_lora; pre-moving
                # every adapter to the GPU at injection adds avoidable VRAM pressure.
                if not isinstance(entry.adapter, comfy.weight_adapter.LoRAAdapter):
                    _move_adapter_to_device(entry.adapter)
            if wrapper.install():
                self.installed_wrappers.append(wrapper)
        for patch in self.mlp_patches:
            if patch.install():
                self.installed_mlps.append(patch)
        for patch in self.block_patches:
            if patch.install():
                self.installed_blocks.append(patch)
        for patch in self.final_patches:
            if patch.install():
                self.installed_finals.append(patch)

    def eject(self, model_patcher):
        for patch in reversed(self.installed_finals):
            patch.eject()
        for patch in reversed(self.installed_blocks):
            patch.eject()
        for patch in reversed(self.installed_mlps):
            patch.eject()
        for wrapper in reversed(self.installed_wrappers):
            wrapper.eject()
        self.installed_wrappers.clear()
        self.installed_mlps.clear()
        self.installed_blocks.clear()
        self.installed_finals.clear()


class MiniMaxH3AVLoRAStack(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        inputs = [
            io.Model.Input("model"),
            io.Int.Input(
                "chunks",
                default=1,
                min=1,
                max=16,
                step=1,
                optional=True,
                tooltip="Set to the same value as SolAttn's MiniMax H3 Chunk FeedForward when enabled.",
            ),
        ]
        for i in range(1, MAX_LORAS + 1):
            inputs.append(io.Combo.Input(f"lora_{i}", options=lora_list, default="None", optional=True))
            inputs.append(io.Float.Input(f"strength_{i}", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True))
            inputs.append(io.Float.Input(f"audio_weight_{i}", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True))
            inputs.append(io.Float.Input(f"video_weight_{i}", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True))
        # 新增输入一律追加在末尾：widget 值按位序保存，插在前面会让旧工作流的 LoRA 参数整体错位
        inputs.append(io.Boolean.Input(
            "keep_lora_on_gpu",
            default=False,
            optional=True,
            tooltip="Cache the per-LoRA matmul tensors in VRAM so sampling stops re-uploading them every step. "
                    "Costs roughly the LoRA size in bf16 VRAM; turn on when stacking several large LoRAs.",
        ))
        return io.Schema(
            node_id="MiniMaxH3AVLoRAStack",
            display_name="MiniMax H3 AV LoRA Stack",
            category="MiniMax-H3/LoRA",
            description=(
                "Stack MiniMax H3 LoRAs with per-LoRA audio/video weights. "
                "Each LoRA is merged into the weights at its video weight and only the "
                "audio/text rows are corrected at runtime, so per-modality weighting "
                "stays cheap even with several stacked LoRAs."
            ),
            inputs=inputs,
            outputs=[
                io.Model.Output("model", display_name="model"),
                io.String.Output("loaded_info", display_name="loaded_info"),
            ],
        )

    @classmethod
    def execute(cls, model, **kwargs):
        chunks = max(1, int(kwargs.get("chunks", 1)))
        keep_on_gpu = bool(kwargs.get("keep_lora_on_gpu", False))
        inner = getattr(model, "model", None)
        dm = getattr(inner, "diffusion_model", None) if inner is not None else None
        if dm is None or not hasattr(dm, "blocks"):
            raise ValueError("MiniMaxH3AVLoRAStack requires a MiniMax H3 diffusion model.")

        new_model = model.clone()
        entries_by_key = {}
        info = []
        merged_total = 0

        for i in range(1, MAX_LORAS + 1):
            lora_name = kwargs.get(f"lora_{i}", "None")
            if lora_name in (None, "None"):
                continue
            strength = float(kwargs.get(f"strength_{i}", 1.0))
            audio_weight = float(kwargs.get(f"audio_weight_{i}", 1.0))
            video_weight = float(kwargs.get(f"video_weight_{i}", 1.0))
            if strength == 0.0 or (audio_weight == 0.0 and video_weight == 0.0):
                continue

            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                info.append(f"{lora_name}: missing file")
                logging.warning("[H3 AV LoRA] missing file: %s", lora_name)
                continue

            try:
                lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
            except Exception as e:
                info.append(f"{lora_name}: load error {e}")
                logging.warning("[H3 AV LoRA] failed to load %s: %s", lora_name, e)
                continue

            lora_dict = comfy.lora_convert.convert_lora(lora_dict)
            lora_dict, skipped_adaln = _prepare_minimax_h3_lora(inner, lora_dict)
            if skipped_adaln:
                info.append(
                    f"{lora_name}: skipped {skipped_adaln} AdaLN adapter pair(s) "
                    "incompatible with current H3 model"
                )

            key_map = comfy.lora.model_lora_keys_unet(inner, {})
            loaded = comfy.lora.load_lora(lora_dict, key_map, log_missing=False)
            config = _LoraConfig(lora_name, strength, audio_weight, video_weight)

            if abs(audio_weight - video_weight) < 1e-6:
                patched = new_model.add_patches(loaded, strength * audio_weight)
                merged_total += len(patched)
                info.append(f"{lora_name}: merged {len(patched)} modules")
                continue

            # Unequal weights: merge the video component into the weights (video
            # rows dominate the packed sequence) and keep only the per-row delta
            # where a tag's weight differs from the merged one.
            plans = []
            for key, adapter in loaded.items():
                if not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
                    continue
                if not _is_supported_lora_adapter(adapter):
                    info.append(f"{lora_name}: skipped unsupported LoRA variant {key} (mid/dora/reshape)")
                    continue
                module = _get_module(dm, key)
                if module is None:
                    info.append(f"{lora_name}: skipped missing module {key}")
                    continue
                if not _adapter_shape_matches(adapter, module):
                    info.append(f"{lora_name}: skipped shape mismatch {key}")
                    continue
                mode = _classify_key(key)
                plain_lora = isinstance(adapter, comfy.weight_adapter.LoRAAdapter)
                if mode in ("token", "adaln"):
                    if plain_lora and video_weight != 0.0:
                        plans.append((key, adapter, "merge_residual", video_weight))
                    else:
                        plans.append((key, adapter, "runtime", None))
                elif mode == "video_scalar":
                    if video_weight != 0.0:
                        plans.append((key, adapter, "merge_scalar", video_weight))
                elif mode == "audio_scalar":
                    if audio_weight != 0.0:
                        plans.append((key, adapter, "merge_scalar", audio_weight))
                elif mode == "global" and plain_lora:
                    plans.append((key, adapter, "merge_scalar", 1.0))
                else:
                    plans.append((key, adapter, "runtime", None))

            merge_groups = {}
            for key, adapter, kind, factor in plans:
                if factor is None:
                    continue
                merge_groups.setdefault(round(float(factor), 6), {})[key] = adapter

            merged_keys = set()
            for factor, subset in merge_groups.items():
                merged_keys.update(new_model.add_patches(subset, strength * factor))

            count_merged = 0
            count_runtime = 0
            for key, adapter, kind, factor in plans:
                mode = _classify_key(key)
                if kind == "merge_residual":
                    if key in merged_keys:
                        entry = _LoraEntry(
                            config, adapter,
                            (0.0, 1.0 - video_weight, audio_weight - video_weight),
                            sparse=True,
                        )
                        count_merged += 1
                    else:
                        entry = _LoraEntry(config, adapter, (video_weight, 1.0, audio_weight))
                        count_runtime += 1
                    entries_by_key.setdefault(key, []).append(entry)
                elif kind == "merge_scalar":
                    if key in merged_keys:
                        count_merged += 1
                    else:
                        entry = _LoraEntry(config, adapter, (video_weight, 1.0, audio_weight))
                        entry.scalar = {
                            "video_scalar": video_weight,
                            "audio_scalar": audio_weight,
                        }.get(mode, 1.0)
                        entries_by_key.setdefault(key, []).append(entry)
                        count_runtime += 1
                else:
                    entry = _LoraEntry(config, adapter, (video_weight, 1.0, audio_weight))
                    if mode in ("video_scalar", "audio_scalar", "global"):
                        entry.scalar = {
                            "video_scalar": video_weight,
                            "audio_scalar": audio_weight,
                        }.get(mode, 1.0)
                    entries_by_key.setdefault(key, []).append(entry)
                    count_runtime += 1

            merged_total += count_merged
            info.append(f"{lora_name}: merged {count_merged} modules, runtime {count_runtime}")

        if not entries_by_key:
            if merged_total == 0:
                info.append("no compatible LoRA modules loaded")
            return (new_model, "; ".join(info))

        wrappers = []
        block_map = {}
        mlp_map = {}
        final_map = {}

        for key, entries in entries_by_key.items():
            module = _get_module(dm, key)
            if module is None:
                info.append(f"skipped missing module: {key}")
                continue

            mode = _classify_key(key)
            if mode in ("token", "adaln"):
                chunked = chunks > 1 and ".mlp." in _module_path(key)
                wrapper = _AVModuleBypass(
                    module,
                    entries,
                    mask_mode=mode,
                    chunks=chunks if chunked else 1,
                    keep_on_gpu=keep_on_gpu,
                )
                wrappers.append(wrapper)
                m = re.match(r"blocks\.(\d+)", _module_path(key))
                if m is None:
                    info.append(f"skipped unexpected block key: {key}")
                    continue
                block_info = block_map.setdefault(int(m.group(1)), {"wrappers": []})
                block_info["wrappers"].append(wrapper)
                if ".mlp." in _module_path(key):
                    mlp_info = mlp_map.setdefault(int(m.group(1)), {"wrappers": []})
                    mlp_info["wrappers"].append(wrapper)
            elif mode == "final_adaln":
                wrapper = _AVModuleBypass(module, entries, mask_mode="adaln", keep_on_gpu=keep_on_gpu)
                wrappers.append(wrapper)
                final_info = final_map.setdefault("adaln", {"wrappers": []})
                final_info["wrappers"].append(wrapper)
            else:
                wrapper = _AVModuleBypass(module, entries, mask_mode="scalar", keep_on_gpu=keep_on_gpu)
                wrapper.set_scalars([entry.scalar for entry in entries])
                wrappers.append(wrapper)

        block_patches = []
        for block_id, block_info in sorted(block_map.items()):
            block = dm.blocks[block_id]
            block_patches.append(_H3BlockAVPatch(block, block_info["wrappers"]))

        mlp_patches = []
        for block_id, mlp_info in sorted(mlp_map.items()):
            block = dm.blocks[block_id]
            has_fc2 = any(wrapper.module is block.mlp.fc2 for wrapper in mlp_info["wrappers"])
            if chunks > 1 or has_fc2:
                mlp_patches.append(_H3MLPAVPatch(block.mlp, chunks=chunks))

        final_patches = []
        if "adaln" in final_map:
            final_patches.append(_H3FinalLayerAVPatch(dm.final_layer, final_map["adaln"]["wrappers"]))

        injection = _H3AVInjection(wrappers, mlp_patches, block_patches, final_patches)
        # new_model 是本次执行新建的 clone，key 天然每次唯一；V3 下 cls 也是每次执行的
        # 临时克隆，两者都只用于命名空间，不承载跨执行的稳定性。
        injection_key = "h3_av_lora_{}_{}".format(id(cls), id(new_model))
        new_model.set_injections(injection_key, [PatcherInjection(injection.inject, injection.eject)])
        info.append(f"wrapped {len(wrappers)} modules / {len(mlp_patches)} MLP paths / {len(block_patches)} blocks")
        return (new_model, "; ".join(info))
