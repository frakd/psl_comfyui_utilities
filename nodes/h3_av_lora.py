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
        self.uid = id(self)


class _LoraEntry:
    def __init__(self, config, adapter):
        self.config = config
        self.adapter = adapter
        self.mask_index = 0


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


def _linear_lora(adapter, x):
    v = adapter.weights
    up, down, alpha = v[0], v[1], v[2]
    if up.ndim != 2 or down.ndim != 2:
        return None, 1.0
    if down.shape[1] != x.shape[-1] or up.shape[1] != down.shape[0]:
        return None, 1.0
    down = down.to(device=x.device, dtype=x.dtype)
    up = up.to(device=x.device, dtype=x.dtype)
    rank = down.shape[0]
    scale = (alpha / rank) if alpha is not None else 1.0
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


def _build_token_mask(n, segments, audio_weight, video_weight, device):
    mask = torch.ones(n, device=device, dtype=torch.float32)
    for a, b, row in segments:
        if isinstance(row, torch.Tensor):
            seg_mask = torch.ones(b - a, device=row.device, dtype=torch.float32)
            tags = row % 3
            seg_mask[tags == 2] = audio_weight
            seg_mask[tags == 0] = video_weight
            mask[a:b] = seg_mask.to(device)
        else:
            tag = int(row) % 3
            value = audio_weight if tag == 2 else (video_weight if tag == 0 else 1.0)
            mask[a:b] = value
    return mask


def _build_modality_mask(n, segments, audio_weight, video_weight, device):
    mask = torch.ones(n, 3, device=device, dtype=torch.float32)
    for a, b, row in segments:
        if isinstance(row, torch.Tensor):
            idx = (row // 3).long()
            tags = (row % 3).long()
            for i in torch.unique(idx):
                tag = int(tags[idx == i][0])
                value = audio_weight if tag == 2 else (video_weight if tag == 0 else 1.0)
                mask[int(i), tag] = value
        else:
            row_index = int(row)
            tag = row_index % 3
            value = audio_weight if tag == 2 else (video_weight if tag == 0 else 1.0)
            mask[row_index // 3, tag] = value
    return mask


def _build_single_final_mask(n, seg, weight, device):
    mask = torch.ones(n, device=device, dtype=torch.float32)
    if seg is None:
        return mask
    row = seg[2]
    if isinstance(row, torch.Tensor):
        mask[row.unique().long()] = weight
    else:
        mask[int(row)] = weight
    return mask


class _AVModuleBypass:
    _warned_shape_keys = set()
    _warned_chunk_keys = set()

    def __init__(self, module, entries, mask_mode="token", scalar_mode="global", chunks=1):
        self.module = module
        self.original_forward = None
        self.entries = entries
        self.mask_mode = mask_mode
        self.scalar_mode = scalar_mode
        self.masks = None
        self.scalars = None
        self.chunks = int(chunks)
        self.chunk_offset = None
        self.chunk_total = None
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

    def _apply_mask_scale(self, delta, entry, idx, scale):
        if self.mask_mode in ("token", "adaln"):
            mask = self.masks[entry.mask_index] if self.masks is not None else None
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

    def _apply(self, x, base):
        if not self.entries:
            return base

        # Accumulate each LoRA delta into the fresh module output to avoid holding
        # multiple full-size activations at once.
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
            scale = self._apply_mask_scale(delta, entry, idx, scale)
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
            delta, lora_scale = _linear_lora(adapter, x)
            if delta is None:
                logging.warning(
                    "[H3 AV LoRA] skipping incompatible LoRA delta for %s (%s)",
                    entry.config.lora_name, type(self.module).__name__
                )
                continue
            if delta.device != base.device or delta.dtype != base.dtype:
                delta = delta.to(device=base.device, dtype=base.dtype)

            scale = float(entry.config.strength) * float(lora_scale)
            scale = self._apply_mask_scale(delta, entry, idx, scale)
            if scale is None:
                continue
            base.add_(delta, alpha=scale)

        return base


class _H3BlockAVPatch:
    def __init__(self, block, child_wrappers, configs):
        self.block = block
        self.original_forward = None
        self.child_wrappers = child_wrappers
        self.configs = configs

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
        token_masks = [
            _build_token_mask(x.shape[0], mod_segments, c.audio_weight, c.video_weight, x.device)
            for c in self.configs
        ]
        modality_masks = [
            _build_modality_mask(t_emb.shape[0], mod_segments, c.audio_weight, c.video_weight, t_emb.device)
            for c in self.configs
        ]
        for wrapper in self.child_wrappers:
            if wrapper.mask_mode == "token":
                wrapper.set_masks(token_masks)
            elif wrapper.mask_mode == "adaln":
                wrapper.set_masks(modality_masks)


class _H3FinalLayerAVPatch:
    def __init__(self, final_layer, child_wrappers, configs):
        self.final_layer = final_layer
        self.original_forward = None
        self.child_wrappers = child_wrappers
        self.configs = configs

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

    def _dispatch(self, x, t_emb, video_seg, audio_seg):
        patches = self.final_layer._h3_av_final_patches
        video_masks = [patch._build_masks(t_emb, video_seg, "video") for patch in patches]
        audio_masks = [patch._build_masks(t_emb, audio_seg, "audio") for patch in patches]

        try:
            for patch, masks in zip(patches, video_masks):
                patch._set_masks(masks)
            video_out, _ = self.final_layer._h3_av_final_original_forward(x, t_emb, video_seg, audio_seg)
        finally:
            for patch in patches:
                patch._clear_masks()

        try:
            for patch, masks in zip(patches, audio_masks):
                patch._set_masks(masks)
            _, audio_out = self.final_layer._h3_av_final_original_forward(x, t_emb, video_seg, audio_seg)
        finally:
            for patch in patches:
                patch._clear_masks()
        return video_out, audio_out

    def _build_masks(self, t_emb, seg, kind):
        weights = [
            c.video_weight if kind == "video" else c.audio_weight
            for c in self.configs
        ]
        return [
            _build_single_final_mask(t_emb.shape[0], seg, weight, t_emb.device)
            for weight in weights
        ]

    def _set_masks(self, masks):
        for wrapper in self.child_wrappers:
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
                # LoRA tensors are cast per module in _linear_lora; pre-moving every
                # adapter to the GPU at injection adds avoidable VRAM pressure.
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
        return io.Schema(
            node_id="MiniMaxH3AVLoRAStack",
            display_name="MiniMax H3 AV LoRA Stack",
            category="MiniMax-H3/LoRA",
            description=(
                "Stack MiniMax H3 LoRAs with per-LoRA audio/video weights. "
                "For standard H3 LoRAs the node applies token-row masks inside the "
                "shared DiT blocks; optional audio/video-specific keys are split by key."
            ),
            inputs=inputs,
            outputs=[
                io.Model.Output("model", display_name="model"),
                io.String.Output("loaded_info", display_name="loaded_info"),
            ],
        )

    @staticmethod
    def _config_index(configs, config):
        for i, c in enumerate(configs):
            if c.uid == config.uid:
                return i
        configs.append(config)
        return len(configs) - 1

    @classmethod
    def execute(cls, model, **kwargs):
        chunks = max(1, int(kwargs.get("chunks", 1)))
        inner = getattr(model, "model", None)
        dm = getattr(inner, "diffusion_model", None) if inner is not None else None
        if dm is None or not hasattr(dm, "blocks"):
            raise ValueError("MiniMaxH3AVLoRAStack requires a MiniMax H3 diffusion model.")

        new_model = model.clone()
        entries_by_key = {}
        info = []
        use_merge = True
        merge_entries = []

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
            equal_weights = abs(audio_weight - video_weight) < 1e-6
            if not equal_weights:
                use_merge = False
            elif use_merge:
                merge_entries.append((lora_name, loaded, strength * audio_weight))
            config = _LoraConfig(lora_name, strength, audio_weight, video_weight)
            count = 0
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
                entries_by_key.setdefault(key, []).append(_LoraEntry(config, adapter))
                count += 1
            info.append(f"{lora_name}: {count} modules")

        if use_merge and merge_entries:
            total_merged = 0
            for merge_name, loaded, effective_strength in merge_entries:
                patched = new_model.add_patches(loaded, effective_strength)
                info.append(f"{merge_name}: merged {len(patched)} modules")
                total_merged += len(patched)
            if total_merged == 0:
                info.append("no compatible LoRA modules loaded")
            return (new_model, "; ".join(info))

        if not entries_by_key:
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
                wrapper = _AVModuleBypass(module, entries, mask_mode="adaln")
                wrappers.append(wrapper)
                final_info = final_map.setdefault("adaln", {"wrappers": []})
                final_info["wrappers"].append(wrapper)
            else:
                scalar_mode = mode
                wrapper = _AVModuleBypass(module, entries, mask_mode="scalar", scalar_mode=scalar_mode)
                scalar_values = []
                for entry in entries:
                    if mode == "video_scalar":
                        scalar_values.append(entry.config.video_weight)
                    elif mode == "audio_scalar":
                        scalar_values.append(entry.config.audio_weight)
                    else:
                        scalar_values.append(1.0)
                wrapper.set_scalars(scalar_values)
                wrappers.append(wrapper)

        block_patches = []
        for block_id, block_info in sorted(block_map.items()):
            configs = []
            for wrapper in block_info["wrappers"]:
                for entry in wrapper.entries:
                    entry.mask_index = cls._config_index(configs, entry.config)
            block = dm.blocks[block_id]
            block_patches.append(_H3BlockAVPatch(block, block_info["wrappers"], configs))

        mlp_patches = []
        for block_id, mlp_info in sorted(mlp_map.items()):
            block = dm.blocks[block_id]
            has_fc2 = any(wrapper.module is block.mlp.fc2 for wrapper in mlp_info["wrappers"])
            if chunks > 1 or has_fc2:
                mlp_patches.append(_H3MLPAVPatch(block.mlp, chunks=chunks))

        final_patches = []
        if "adaln" in final_map:
            configs = []
            for wrapper in final_map["adaln"]["wrappers"]:
                for entry in wrapper.entries:
                    entry.mask_index = cls._config_index(configs, entry.config)
            final_patches.append(_H3FinalLayerAVPatch(dm.final_layer, final_map["adaln"]["wrappers"], configs))

        injection = _H3AVInjection(wrappers, mlp_patches, block_patches, final_patches)
        # new_model 是本次执行新建的 clone，key 天然每次唯一；V3 下 cls 也是每次执行的
        # 临时克隆，两者都只用于命名空间，不承载跨执行的稳定性。
        injection_key = "h3_av_lora_{}_{}".format(id(cls), id(new_model))
        new_model.set_injections(injection_key, [PatcherInjection(injection.inject, injection.eject)])
        info.append(f"wrapped {len(wrappers)} modules / {len(mlp_patches)} MLP paths / {len(block_patches)} blocks")
        return (new_model, "; ".join(info))
