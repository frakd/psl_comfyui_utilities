# -*- coding: utf-8 -*-
"""验证 H3 AV LoRA 的合并+残差路径与逐 token 掩码路径数学等价。"""
import importlib.util
import sys
import types
from pathlib import Path

NODES_DIR = Path(__file__).resolve().parents[1] / "nodes"
FAIL = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f"  <- {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(name)


try:
    import torch
except ImportError:
    print("SKIP  需要 torch")
    raise SystemExit(0)

# ---- stubs -----------------------------------------------------------------
comfy = types.ModuleType("comfy")
comfy.__path__ = []
sys.modules["comfy"] = comfy

weight_adapter = types.ModuleType("comfy.weight_adapter")


class WeightAdapterBase:
    def __init__(self, weights=None, multiplier=1.0):
        self.weights = weights
        self.multiplier = multiplier


class LoRAAdapter(WeightAdapterBase):
    pass


weight_adapter.WeightAdapterBase = WeightAdapterBase
weight_adapter.LoRAAdapter = LoRAAdapter
sys.modules["comfy.weight_adapter"] = weight_adapter
comfy.weight_adapter = weight_adapter

patcher_extension = types.ModuleType("comfy.patcher_extension")


class PatcherInjection:
    def __init__(self, inject, eject):
        self.inject = inject
        self.eject = eject


patcher_extension.PatcherInjection = PatcherInjection
sys.modules["comfy.patcher_extension"] = patcher_extension
comfy.patcher_extension = patcher_extension

model_management = types.ModuleType("comfy.model_management")
model_management.get_torch_device = lambda: torch.device("cpu")
sys.modules["comfy.model_management"] = model_management
comfy.model_management = model_management

lora_mod = types.ModuleType("comfy.lora")
lora_mod.model_lora_keys_unet = lambda inner, key_map: key_map
lora_mod.load_lora = lambda lora, key_map, log_missing=True: {}
sys.modules["comfy.lora"] = lora_mod
comfy.lora = lora_mod

lora_convert = types.ModuleType("comfy.lora_convert")
lora_convert.convert_lora = lambda lora: lora
sys.modules["comfy.lora_convert"] = lora_convert
comfy.lora_convert = lora_convert

utils = types.ModuleType("comfy.utils")
utils.load_torch_file = lambda *a, **k: {}
sys.modules["comfy.utils"] = utils
comfy.utils = utils

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_filename_list = lambda kind: []
folder_paths.get_full_path = lambda kind, name: None
sys.modules["folder_paths"] = folder_paths

comfy_api = types.ModuleType("comfy_api")
comfy_api.__path__ = []
latest = types.ModuleType("comfy_api.latest")
latest.io = types.SimpleNamespace(ComfyNode=object, Schema=object)
comfy_api.latest = latest
sys.modules["comfy_api"] = comfy_api
sys.modules["comfy_api.latest"] = latest

spec = importlib.util.spec_from_file_location("h3_av_lora", NODES_DIR / "h3_av_lora.py")
h3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h3)

# ---- helpers ---------------------------------------------------------------
torch.manual_seed(0)
IN, OUT, RANK, N = 8, 6, 4, 20
# tags: rows 0..13 video (mod row 0 + 3k), 14..16 text (tag 1), 17..19 audio (tag 2)
SEGMENTS = [(0, 14, 0), (14, 17, 1), (17, 20, 2)]
TAGS = torch.tensor([0] * 14 + [1] * 3 + [2] * 3)
W = torch.randn(OUT, IN)
X = torch.randn(N, IN)
DOWN = torch.randn(RANK, IN)
UP = torch.randn(OUT, RANK)
ALPHA = 4.0
STRENGTH = 0.8
VIDEO_W, AUDIO_W = 0.5, -0.25
REF_WEIGHTS = torch.tensor([VIDEO_W, 1.0, AUDIO_W])[TAGS]


def reference():
    delta = X @ DOWN.T @ UP.T
    return X @ W.T + (STRENGTH * (ALPHA / RANK)) * delta * REF_WEIGHTS.unsqueeze(-1)


def make_adapter():
    return LoRAAdapter((UP.clone(), DOWN.clone(), ALPHA))


def make_wrapper(tag_weights, sparse):
    adapter = make_adapter()
    config = h3._LoraConfig("test", STRENGTH, AUDIO_W, VIDEO_W)
    entry = h3._LoraEntry(config, adapter, tag_weights, sparse=sparse)
    wrapper = h3._AVModuleBypass(object(), [entry], mask_mode="token")
    mask = h3._build_token_mask(N, SEGMENTS, tag_weights, torch.device("cpu"), torch.float32)
    wrapper.set_masks([mask])
    return wrapper


REF = reference()

# 1. mask 构建
mask = h3._build_token_mask(N, SEGMENTS, (VIDEO_W, 1.0, AUDIO_W), torch.device("cpu"), torch.float32)
expected = torch.tensor([VIDEO_W] * 14 + [1.0] * 3 + [AUDIO_W] * 3)
check("token mask per-tag values", torch.allclose(mask, expected), mask.tolist())

row = torch.tensor([0, 1, 2, 3, 4, 5])
mask_t = h3._build_token_mask(6, [(0, 6, row)], (0.5, 1.0, -0.25), torch.device("cpu"), torch.float32)
expected_t = torch.tensor([0.5, 1.0, -0.25, 0.5, 1.0, -0.25])
check("token mask per-token mod rows", torch.allclose(mask_t, expected_t), mask_t.tolist())

modal = h3._build_modality_mask(1, [(0, 14, 0), (14, 17, 1), (17, 20, 2)], (0.5, 1.0, -0.25),
                                torch.device("cpu"), torch.float32)
check("modality mask", modal[0, 0].item() == 0.5 and modal[0, 1].item() == 1.0 and modal[0, 2].item() == -0.25,
      modal.tolist())

# 2. 全掩码密集路径 == 参考
wrapper = make_wrapper((VIDEO_W, 1.0, AUDIO_W), sparse=False)
base = X @ W.T
out = wrapper._apply(X, base.clone())
check("dense full-mask matches reference", torch.allclose(out, REF, atol=2e-6), (out - REF).abs().max().item())

# 3. 合并视频权重 + 残差稀疏路径 == 参考
W_MERGED = W + STRENGTH * VIDEO_W * (UP @ DOWN)
wrapper = make_wrapper((0.0, 1.0 - VIDEO_W, AUDIO_W - VIDEO_W), sparse=True)
base = X @ W_MERGED.T
out = wrapper._apply(X, base.clone())
check("merged + sparse residual matches reference", torch.allclose(out, REF, atol=2e-6),
      (out - REF).abs().max().item())

# 4. 残差行占比过高时退回密集路径，仍然等价
wrapper = make_wrapper((0.0, 1.0 - VIDEO_W, AUDIO_W - VIDEO_W), sparse=True)
wrapper._add_sparse_delta_orig = wrapper._add_sparse_delta
wrapper._add_sparse_delta = lambda *a, **k: False
base = X @ W_MERGED.T
out = wrapper._apply(X, base.clone())
check("merged + dense residual matches reference", torch.allclose(out, REF, atol=2e-6),
      (out - REF).abs().max().item())

# 5. 分块 MLP 下 mask 偏移对齐
wrapper = make_wrapper((0.0, 1.0 - VIDEO_W, AUDIO_W - VIDEO_W), sparse=True)
out = torch.empty(N, OUT)
offset = 0
for chunk in torch.chunk(X, 2, dim=0):
    wrapper.set_chunk(offset, N)
    try:
        out[offset:offset + chunk.shape[0]] = wrapper._apply(chunk, (chunk @ W_MERGED.T).clone())
    finally:
        wrapper.clear_chunk()
    offset += chunk.shape[0]
check("chunked sparse residual matches reference", torch.allclose(out, REF, atol=2e-6),
      (out - REF).abs().max().item())

# 6. keep_on_gpu 缓存路径结果一致
wrapper = make_wrapper((0.0, 1.0 - VIDEO_W, AUDIO_W - VIDEO_W), sparse=True)
wrapper.keep_on_gpu = True
base = X @ W_MERGED.T
out = wrapper._apply(X, base.clone())
check("keep_on_gpu cache matches reference", torch.allclose(out, REF, atol=2e-6),
      (out - REF).abs().max().item())
check("keep_on_gpu cached tensors", getattr(wrapper.entries[0].adapter, "_h3_av_pair_cache", None) is not None)

# 7. 分类
check("classify token", h3._classify_key("diffusion_model.blocks.3.attn.qkv_proj.weight") == "token")
check("classify adaln", h3._classify_key("diffusion_model.blocks.3.adaln_proj.linear.weight") == "adaln")
check("classify final adaln", h3._classify_key("diffusion_model.final_layer.adaln_proj.linear.weight") == "final_adaln")
check("classify audio head", h3._classify_key("diffusion_model.final_layer.audio_out.weight") == "audio_scalar")
check("classify token refiner global",
      h3._classify_key("diffusion_model.token_refiner.blocks.0.attn.qkv_proj.weight") == "global")

print()
if FAIL:
    print(f"{len(FAIL)} FAILED: {FAIL}")
    raise SystemExit(1)
print("all checks passed")

# ---- execute-level smoke test (fake H3 model + fake patcher) ----------------
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN, 2 * IN, bias=False)
        self.fc2 = nn.Linear(IN, IN, bias=False)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(IN, 3 * IN, bias=False)
        self.out_proj = nn.Linear(3 * IN, IN, bias=False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()
        self.mlp = _MLP()


class _Adaln(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(IN, 6 * IN, bias=True)


class _Final(nn.Module):
    def __init__(self):
        super().__init__()
        self.adaln_proj = _Adaln()
        self.audio_out = nn.Linear(IN, 2, bias=True)
        self.video_out = nn.Linear(IN, 4, bias=True)


class _DM(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block()])
        self.final_layer = _Final()
        self.token_refiner = nn.Module()
        self.token_refiner.blocks = nn.ModuleList([_Block()])


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = _DM()


class _FakePatcher:
    def __init__(self, inner):
        self.model = inner
        self.patch_calls = []
        self.injections = {}

    def clone(self):
        return _FakePatcher(self.model)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        self.patch_calls.append((dict(patches), strength_patch))
        return list(patches)

    def set_injections(self, key, injections):
        self.injections[key] = injections


def _adapter_for(module):
    out_f, in_f = module.weight.shape
    return LoRAAdapter((torch.randn(out_f, RANK), torch.randn(RANK, in_f), float(RANK)))


def _build_loaded(inner):
    dm = inner.diffusion_model
    keys = {
        "diffusion_model.blocks.0.attn.qkv_proj": dm.blocks[0].attn.qkv_proj,
        "diffusion_model.blocks.0.mlp.fc1": dm.blocks[0].mlp.fc1,
        "diffusion_model.blocks.0.mlp.fc2": dm.blocks[0].mlp.fc2,
        "diffusion_model.token_refiner.blocks.0.attn.qkv_proj": dm.token_refiner.blocks[0].attn.qkv_proj,
        "diffusion_model.final_layer.adaln_proj.linear": dm.final_layer.adaln_proj.linear,
        "diffusion_model.final_layer.audio_out": dm.final_layer.audio_out,
    }
    return {k: _adapter_for(m) for k, m in keys.items()}


def _run_execute(**weights):
    inner = _Inner()
    loaded = _build_loaded(inner)
    h3.folder_paths.get_full_path = lambda kind, name: "fake.safetensors"
    h3.comfy.utils.load_torch_file = lambda *a, **k: {"anything": 1}
    h3.comfy.lora.load_lora = lambda *a, **k: loaded
    model = _FakePatcher(inner)
    kwargs = {"lora_1": "fake.safetensors", "strength_1": 1.0}
    kwargs.update(weights)
    new_model, info = h3.MiniMaxH3AVLoRAStack.execute(model, **kwargs)
    if new_model.injections:
        for inj in new_model.injections.values():
            for one in inj:
                one.inject(new_model)
    return inner, loaded, new_model, info


def _tag_of(module):
    return module._h3_av_bypasses[0].entries[0].tag_weights


# 等权：整体合并，不注入任何 wrapper
inner, _, new_model, info = _run_execute(audio_weight_1=1.0, video_weight_1=1.0)
check("equal weights -> merge only", not new_model.injections and len(new_model.patch_calls) == 1,
      (len(new_model.patch_calls), list(new_model.injections)))
check("equal weights -> no bypass installed", not hasattr(inner.diffusion_model.blocks[0].attn.qkv_proj, "_h3_av_bypasses"))

# audio=0 / video=1：块内 token 层只对音频行做 -1 残差，token_refiner 全强度合并
inner, _, new_model, info = _run_execute(audio_weight_1=0.0, video_weight_1=1.0)
dm = inner.diffusion_model
check("residual tag weights", _tag_of(dm.blocks[0].attn.qkv_proj) == (0.0, 0.0, -1.0),
      _tag_of(dm.blocks[0].attn.qkv_proj))
check("fc1/fc2 residual installed",
      hasattr(dm.blocks[0].mlp.fc1, "_h3_av_bypasses") and hasattr(dm.blocks[0].mlp.fc2, "_h3_av_bypasses"))
check("mlp forward patched", getattr(dm.blocks[0].mlp, "_h3_av_mlp_refcount", 0) == 1)
merged_keys = {k for subset, _ in new_model.patch_calls for k in subset}
check("token_refiner merged globally", "diffusion_model.token_refiner.blocks.0.attn.qkv_proj" in merged_keys)
check("token_refiner has no bypass", not hasattr(dm.token_refiner.blocks[0].attn.qkv_proj, "_h3_av_bypasses"))
occurrences = sum(1 for subset, _ in new_model.patch_calls for k in subset
                  if k == "diffusion_model.blocks.0.attn.qkv_proj")
check("block token key merged once and kept on residual runtime",
      occurrences == 1 and hasattr(dm.blocks[0].attn.qkv_proj, "_h3_av_bypasses"), occurrences)
check("audio head with weight 0 skipped", "diffusion_model.final_layer.audio_out" not in merged_keys
      and not hasattr(dm.final_layer.audio_out, "_h3_av_bypasses"))
check("final adaln runtime installed", hasattr(dm.final_layer.adaln_proj.linear, "_h3_av_bypasses"))
check("info mentions merge/runtime", "merged" in info and "runtime" in info, info)

# video=0.5：token 层按 0.5 合并，残差为 (0, 0.5, -0.5)，全局项仍按 1.0
inner, _, new_model, info = _run_execute(audio_weight_1=0.0, video_weight_1=0.5)
dm = inner.diffusion_model
check("residual weights at video 0.5", _tag_of(dm.blocks[0].attn.qkv_proj) == (0.0, 0.5, -0.5),
      _tag_of(dm.blocks[0].attn.qkv_proj))
factors = sorted(round(s, 3) for _, s in new_model.patch_calls)
check("merge strengths split by tag group", factors == [0.5, 1.0], factors)
print()
if FAIL:
    print(f"{len(FAIL)} FAILED: {FAIL}")
    raise SystemExit(1)
print("all checks passed")
