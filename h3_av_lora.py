import logging
import re

import torch
import torch.nn.functional as F

import folder_paths
import comfy.lora
import comfy.model_management
import comfy.utils
import comfy.weight_adapter
from comfy.patcher_extension import PatcherInjection


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


def _linear_lora_delta(adapter, x):
    v = adapter.weights
    up, down, alpha = v[0], v[1], v[2]
    if up.ndim != 2 or down.ndim != 2:
        return None
    if down.shape[1] != x.shape[-1] or up.shape[1] != down.shape[0]:
        return None
    down = down.to(device=x.device, dtype=x.dtype)
    up = up.to(device=x.device, dtype=x.dtype)
    rank = down.shape[0]
    scale = (alpha / rank) if alpha is not None else 1.0
    return F.linear(F.linear(x, down), up) * scale


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


def _build_row_mask(n, segments, audio_weight, video_weight, device):
    mask = torch.ones(n, device=device, dtype=torch.float32)
    for a, b, row in segments:
        if isinstance(row, torch.Tensor):
            idx = (row // 3).long()
            tags = row % 3
            for i in torch.unique(idx):
                vals = tags[idx == i]
                if bool((vals == 2).any()):
                    mask[int(i)] = audio_weight
                elif bool((vals == 0).any()):
                    mask[int(i)] = video_weight
        else:
            mask[int(row) // 3] = audio_weight if (int(row) % 3) == 2 else video_weight if (int(row) % 3) == 0 else 1.0
    return mask


def _build_final_row_mask(n, video_seg, audio_seg, audio_weight, video_weight, device):
    mask = torch.ones(n, device=device, dtype=torch.float32)
    if video_seg is not None:
        row = video_seg[2]
        if isinstance(row, torch.Tensor):
            mask[row.unique().long()] = video_weight
        else:
            mask[int(row)] = video_weight
    if audio_seg is not None:
        row = audio_seg[2]
        if isinstance(row, torch.Tensor):
            mask[row.unique().long()] = audio_weight
        else:
            mask[int(row)] = audio_weight
    return mask


class _AVModuleBypass:
    _warned_shape_keys = set()
    _warned_chunk_keys = set()

    def __init__(self, module, entries, mask_mode="token", scalar_mode="global", chunks=1):
        self.module = module
        self.original_forward = module.forward
        self.entries = entries
        self.mask_mode = mask_mode
        self.scalar_mode = scalar_mode
        self.masks = None
        self.scalars = None
        self.chunks = int(chunks)
        self.shape_warned = False

    def install(self):
        if getattr(self.module, "_h3_av_bypass", None) is not None:
            logging.warning("[H3 AV LoRA] module already wrapped: %s", self.module)
            return False
        self.module._h3_av_bypass = self
        self.module.forward = self._call
        return True

    def eject(self):
        if getattr(self.module, "_h3_av_bypass", None) is self:
            self.module.forward = self.original_forward
            self.module._h3_av_bypass = None

    def set_masks(self, masks):
        self.masks = masks

    def set_scalars(self, scalars):
        self.scalars = scalars

    def clear_masks(self):
        self.masks = None

    def _call(self, x, *args, **kwargs):
        base = self.original_forward(x, *args, **kwargs)
        if not self.entries:
            return base

        total = None
        for idx, entry in enumerate(self.entries):
            adapter = entry.adapter
            adapter.multiplier = 1.0
            delta = _linear_lora_delta(adapter, x) if isinstance(adapter, comfy.weight_adapter.LoRAAdapter) else None
            if delta is None:
                if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
                    logging.warning(
                        "[H3 AV LoRA] skipping incompatible LoRA delta for %s (%s)",
                        entry.config.lora_name, type(self.module).__name__
                    )
                    continue
                try:
                    delta = adapter.h(x, base)
                except Exception as e:
                    logging.warning("[H3 AV LoRA] adapter delta failed for %s: %s", entry.config.lora_name, e)
                    continue
            if delta is None:
                continue

            scale = float(entry.config.strength)
            if self.mask_mode in ("token", "adaln"):
                mask = self.masks[entry.mask_index] if self.masks is not None else None
                if mask is not None:
                    m = mask.to(device=base.device, dtype=base.dtype)
                    if delta.ndim == 2 and delta.shape[0] == m.shape[0]:
                        delta = delta * m.unsqueeze(-1)
                    elif delta.ndim == 3 and delta.shape[1] == m.shape[0]:
                        delta = delta * m.view(1, -1, 1)
                    elif delta.ndim == 1 and delta.shape[0] == m.shape[0]:
                        delta = delta * m
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
                                    "using per-chunk divisor %d without AV mask",
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
                            continue
            elif self.mask_mode == "scalar":
                if self.scalars is not None:
                    scale *= float(self.scalars[idx])

            scaled = delta * scale
            total = scaled if total is None else total + scaled

        if total is None:
            return base
        return base + total


class _H3BlockAVPatch:
    def __init__(self, block, child_wrappers, configs):
        self.block = block
        self.original_forward = block.forward
        self.child_wrappers = child_wrappers
        self.configs = configs

    def install(self):
        if getattr(self.block, "_h3_av_block_patch", None) is not None:
            logging.warning("[H3 AV LoRA] block already wrapped: %s", self.block)
            return False
        self.block._h3_av_block_patch = self
        self.block.forward = self._call
        return True

    def eject(self):
        if getattr(self.block, "_h3_av_block_patch", None) is self:
            self.block.forward = self.original_forward
            self.block._h3_av_block_patch = None

    def _call(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, **kwargs):
        token_masks = [
            _build_token_mask(x.shape[0], mod_segments, c.audio_weight, c.video_weight, x.device)
            for c in self.configs
        ]
        row_masks = [
            _build_row_mask(t_emb.shape[0], mod_segments, c.audio_weight, c.video_weight, t_emb.device)
            for c in self.configs
        ]
        for wrapper in self.child_wrappers:
            if wrapper.mask_mode == "token":
                wrapper.set_masks(token_masks)
            elif wrapper.mask_mode == "adaln":
                wrapper.set_masks(row_masks)
        try:
            return self.original_forward(x, t_emb, mod_segments, rope_freqs, transformer_options, **kwargs)
        finally:
            for wrapper in self.child_wrappers:
                wrapper.clear_masks()


class _H3FinalLayerAVPatch:
    def __init__(self, final_layer, child_wrappers, configs):
        self.final_layer = final_layer
        self.original_forward = final_layer.forward
        self.child_wrappers = child_wrappers
        self.configs = configs

    def install(self):
        if getattr(self.final_layer, "_h3_av_final_patch", None) is not None:
            logging.warning("[H3 AV LoRA] final layer already wrapped: %s", self.final_layer)
            return False
        self.final_layer._h3_av_final_patch = self
        self.final_layer.forward = self._call
        return True

    def eject(self):
        if getattr(self.final_layer, "_h3_av_final_patch", None) is self:
            self.final_layer.forward = self.original_forward
            self.final_layer._h3_av_final_patch = None

    def _call(self, x, t_emb, video_seg, audio_seg):
        row_masks = [
            _build_final_row_mask(t_emb.shape[0], video_seg, audio_seg, c.audio_weight, c.video_weight, t_emb.device)
            for c in self.configs
        ]
        for wrapper in self.child_wrappers:
            wrapper.set_masks(row_masks)
        try:
            return self.original_forward(x, t_emb, video_seg, audio_seg)
        finally:
            for wrapper in self.child_wrappers:
                wrapper.clear_masks()


class _H3AVInjection:
    def __init__(self, wrappers, block_patches, final_patches):
        self.wrappers = wrappers
        self.block_patches = block_patches
        self.final_patches = final_patches
        self.installed_wrappers = []
        self.installed_blocks = []
        self.installed_finals = []

    def inject(self, model_patcher):
        for wrapper in self.wrappers:
            for entry in wrapper.entries:
                _move_adapter_to_device(entry.adapter)
            if wrapper.install():
                self.installed_wrappers.append(wrapper)
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
        for wrapper in reversed(self.installed_wrappers):
            wrapper.eject()
        self.installed_wrappers.clear()
        self.installed_blocks.clear()
        self.installed_finals.clear()


class MiniMaxH3AVLoRAStack:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        optional = {
            "chunks": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Set to the same value as SolAttn's MiniMax H3 Chunk FeedForward when enabled.",
                },
            ),
        }
        for i in range(1, MAX_LORAS + 1):
            optional[f"lora_{i}"] = (lora_list, {"default": "None"})
            optional[f"strength_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
            )
            optional[f"audio_weight_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
            )
            optional[f"video_weight_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
            )
        return {
            "required": {"model": ("MODEL",)},
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "loaded_info")
    FUNCTION = "load_loras"
    CATEGORY = "MiniMax-H3/LoRA"
    DESCRIPTION = (
        "Stack MiniMax H3 LoRAs with per-LoRA audio/video weights. "
        "For standard H3 LoRAs the node applies token-row masks inside the "
        "shared DiT blocks; optional audio/video-specific keys are split by key."
    )

    @staticmethod
    def _config_index(configs, config):
        for i, c in enumerate(configs):
            if c.uid == config.uid:
                return i
        configs.append(config)
        return len(configs) - 1

    def load_loras(self, model, **kwargs):
        chunks = max(1, int(kwargs.get("chunks", 1)))
        inner = getattr(model, "model", None)
        dm = getattr(inner, "diffusion_model", None) if inner is not None else None
        if dm is None or not hasattr(dm, "blocks"):
            raise ValueError("MiniMaxH3AVLoRAStack requires a MiniMax H3 diffusion model.")

        new_model = model.clone()
        entries_by_key = {}
        info = []

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

            key_map = comfy.lora.model_lora_keys_unet(inner, {})
            loaded = comfy.lora.load_lora(lora_dict, key_map, log_missing=False)
            config = _LoraConfig(lora_name, strength, audio_weight, video_weight)
            count = 0
            for key, adapter in loaded.items():
                if not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
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

        if not entries_by_key:
            info.append("no compatible LoRA modules loaded")
            return (new_model, "; ".join(info))

        wrappers = []
        block_map = {}
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
                    entry.mask_index = self._config_index(configs, entry.config)
            block = dm.blocks[block_id]
            block_patches.append(_H3BlockAVPatch(block, block_info["wrappers"], configs))

        final_patches = []
        if "adaln" in final_map:
            configs = []
            for wrapper in final_map["adaln"]["wrappers"]:
                for entry in wrapper.entries:
                    entry.mask_index = self._config_index(configs, entry.config)
            final_patches.append(_H3FinalLayerAVPatch(dm.final_layer, final_map["adaln"]["wrappers"], configs))

        injection = _H3AVInjection(wrappers, block_patches, final_patches)
        new_model.set_injections("h3_av_lora", [PatcherInjection(injection.inject, injection.eject)])
        info.append(f"wrapped {len(wrappers)} modules / {len(block_patches)} blocks")
        return (new_model, "; ".join(info))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3AVLoRAStack": MiniMaxH3AVLoRAStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3AVLoRAStack": "MiniMax H3 AV LoRA Stack",
}