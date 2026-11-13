# -*- coding: utf-8 -*-
"""验证 attention override 的 v 缩放形状。

官方 core 的 attention 在 override 之后才把 v reshape 成 [B, heads, N, dim]，
所以 override 里 v 实际是 [B, N, dim]；这个形状曾因因子写成 [V,1,1] 而广播失败。
需要 torch；没有 torch 的测试环境会跳过。
"""
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


# prompt_weight.py 只依赖 comfy 的 WrappersMP 和 comfy_api.latest.io 的基类；
# 这里桩掉它们，直接测试 attention override。
comfy = types.ModuleType("comfy")
comfy.__path__ = []
patcher = types.ModuleType("comfy.patcher_extension")


class _WrappersMP:
    DIFFUSION_MODEL = "diffusion_model"


patcher.WrappersMP = _WrappersMP
comfy.patcher_extension = patcher
sys.modules["comfy"] = comfy
sys.modules["comfy.patcher_extension"] = patcher

comfy_api = types.ModuleType("comfy_api")
comfy_api.__path__ = []
latest = types.ModuleType("comfy_api.latest")
latest.__path__ = []
io_mod = types.ModuleType("comfy_api.latest.io")


class _ComfyNode:
    pass


io_mod.ComfyNode = _ComfyNode
latest.io = io_mod
comfy_api.latest = latest
sys.modules["comfy_api"] = comfy_api
sys.modules["comfy_api.latest"] = latest
sys.modules["comfy_api.latest.io"] = io_mod

pkg = types.ModuleType("psl_weight_test")
pkg.__path__ = [str(NODES_DIR)]
sys.modules["psl_weight_test"] = pkg
spec = importlib.util.spec_from_file_location("psl_weight_test.prompt_weight", NODES_DIR / "prompt_weight.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def passthrough(q, k, v, heads, mask=None, **kwargs):
    return v


weights = ([0], torch.tensor([3, 4, 5, 6]), torch.tensor([0.5, 0.25, 0.1, 0.0]), None)

v3 = torch.ones(1, 8, 3072)
mod._make_attention_override(weights, None)(
    passthrough, torch.zeros(1, 8, 3072), torch.zeros(1, 8, 3072), v3, 24
)
check("3D v [B,N,dim] 按位置缩放",
      [round(v3[0, i, 0].item(), 3) for i in range(8)] == [1.0, 1.0, 1.0, 0.5, 0.25, 0.1, 0.0, 1.0],
      repr([round(v3[0, i, 0].item(), 3) for i in range(8)]))

v4 = torch.ones(1, 8, 2, 4)
mod._make_attention_override(weights, None)(
    passthrough, torch.zeros(1, 8, 2, 4), torch.zeros(1, 8, 2, 4), v4, 2
)
check("4D v [B,N,H,D] 保持兼容",
      [round(v4[0, i, 0, 0].item(), 3) for i in range(8)] == [1.0, 1.0, 1.0, 0.5, 0.25, 0.1, 0.0, 1.0],
      repr([round(v4[0, i, 0, 0].item(), 3) for i in range(8)]))

zero = mod._image_bias([-1, -1, 7], 0.0)
check("strength 0 静音视觉槽位",
      zero is not None and float(zero[0, 0]) < -10 and float(zero[0, 2]) == 0.0,
      repr(zero.tolist() if zero is not None else None))
check("bias 相加", torch.allclose(mod._sum_bias(zero, torch.ones(1, 3)), zero + 1))
check("None 项不改变结果", mod._sum_bias(None, zero) is zero)

captured = {}


class FakeAttention:
    def __init__(self, function):
        self.function = function


def preferred(_q, _k, v, _heads, mask=None, **_kwargs):
    captured["mask"] = mask
    return v


def forbidden(*_args, **_kwargs):
    raise AssertionError("global attention backend should not be used")


bias = torch.zeros(1, 3)
bias[0, 2] = 3.0
mod._make_attention_override(([0], None, None, bias), FakeAttention(preferred), None)(
    forbidden, torch.zeros(1, 3, 4), torch.zeros(1, 3, 4), torch.ones(1, 3, 4), 2,
)
check("MOSS 有 bias 时保留 checkpoint attention 后端",
      captured.get("mask") is not None and captured["mask"][0, 2].item() == 3.0,
      repr(captured.get("mask")))

if FAIL:
    print("失败的用例：" + ", ".join(FAIL))
    raise SystemExit(1)
print("全部通过")
