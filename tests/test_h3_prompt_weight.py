# -*- coding: utf-8 -*-
"""验证 H3 preset 节点的纯逻辑：token 定位、empty-image 组装、offset 与 attention override。"""
import importlib.util
import re
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

comfy = types.ModuleType("comfy")
comfy.__path__ = []
patcher = types.ModuleType("comfy.patcher_extension")


class _WrappersMP:
    DIFFUSION_MODEL = "diffusion_model"


patcher.WrappersMP = _WrappersMP
comfy.patcher_extension = patcher
model_management = types.ModuleType("comfy.model_management")
model_management.intermediate_device = lambda: "cpu"
comfy.model_management = model_management
sys.modules["comfy"] = comfy
sys.modules["comfy.patcher_extension"] = patcher
sys.modules["comfy.model_management"] = model_management

comfy_api = types.ModuleType("comfy_api")
comfy_api.__path__ = []
latest = types.ModuleType("comfy_api.latest")
latest.__path__ = []
io_mod = types.ModuleType("comfy_api.latest.io")


class _ComfyNode:
    pass


class _NodeOutput:
    def __init__(self, *args, **_kwargs):
        self.args = args

    @property
    def result(self):
        return self.args


io_mod.ComfyNode = _ComfyNode
io_mod.NodeOutput = _NodeOutput
latest.io = io_mod
comfy_api.latest = latest
sys.modules["comfy_api"] = comfy_api
sys.modules["comfy_api.latest"] = latest
sys.modules["comfy_api.latest.io"] = io_mod

pkg = types.ModuleType("psl_h3_test")
pkg.__path__ = [str(NODES_DIR)]
sys.modules["psl_h3_test"] = pkg

core_spec = importlib.util.spec_from_file_location("psl_h3_test.prompt_weight_core",
                                                   NODES_DIR / "prompt_weight_core.py")
core = importlib.util.module_from_spec(core_spec)
sys.modules[core_spec.name] = core
core_spec.loader.exec_module(core)

h3_spec = importlib.util.spec_from_file_location("psl_h3_test.h3_prompt_weight",
                                                 NODES_DIR / "h3_prompt_weight.py")
h3 = importlib.util.module_from_spec(h3_spec)
sys.modules[h3_spec.name] = h3
h3_spec.loader.exec_module(h3)


# ---------- H3 token 定位 ----------
class FakeTokenizer:
    def __init__(self):
        self.word_ids = {}

    def _word(self, token):
        if token not in self.word_ids:
            self.word_ids[token] = 100 + len(self.word_ids)
        return self.word_ids[token]

    def encode(self, text):
        ids = []
        for part in re.findall(r"<\|[^|]+\|>|\n|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text):
            if part == "<|im_start|>":
                ids.append(151644)
            elif part == "<|im_end|>":
                ids.append(151645)
            elif part == "user":
                ids.append(872)
            elif part == "\n":
                ids.append(198)
            else:
                ids.append(self._word(part.lower()))
        return ids


class FakeClip:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def tokenize(self, text, **_kwargs):
        return {"h3": [[(token, 1.0) for token in self.tokenizer.encode(text)]]}


clip = FakeClip()
text = "a photo of a cat, masterpiece, a cat on a sofa"
ids = [item[0] for item in clip.tokenize(text)["h3"][0]]
pairs = core.positions_from_ids(
    clip, ids, len(ids), [("cat", -1.0), ("masterpiece", 1.5)], h3._H3_SPECIALS
)
cat_ids = [item[0] for item in clip.tokenize("cat")["h3"][0]]
cat_positions = []
for index in range(len(ids)):
    if ids[index:index + len(cat_ids)] == cat_ids:
        cat_positions.extend(range(index, index + len(cat_ids)))
masterpiece_ids = [item[0] for item in clip.tokenize("masterpiece")["h3"][0]]
masterpiece_positions = []
for index in range(len(ids)):
    if ids[index:index + len(masterpiece_ids)] == masterpiece_ids:
        masterpiece_positions.extend(range(index, index + len(masterpiece_ids)))
expected = []
for position in cat_positions:
    expected.append((position, -1.0, 0.0))
for position in masterpiece_positions:
    expected.append((position, 1.0, 1.0))
check("H3 specials 定位短语并合并重复位置", pairs == sorted(expected), repr(pairs))


# ---------- empty-image 条件组装 ----------
text_states = torch.zeros(1, 5, 4)
metadata = {
    "minimax_token_tags": torch.ones(5, dtype=torch.long),
    "minimax_refs": ["reference"],
}
picture = torch.zeros(1, 2, 4)
picture_dict = {"minimax_token_tags": torch.zeros(2, dtype=torch.long)}
cond = h3._apply_empty_image([(text_states, metadata)], picture, picture_dict, 0.1, 0.6)
check("empty-image 拆成三段", len(cond) == 3, str(len(cond)))
check("empty-image 行数拼接", cond[0][0].shape[1] == 7, str(tuple(cond[0][0].shape)))
check("empty-image tags 顺序", cond[0][1]["minimax_token_tags"].tolist() == [0, 0, 1, 1, 1, 1, 1],
      repr(cond[0][1]["minimax_token_tags"].tolist()))
check("empty-image 保留其他 metadata", cond[0][1].get("minimax_refs") == ["reference"],
      repr(cond[0][1].get("minimax_refs")))
check("empty-image 区间", [(entry[1].get("start_percent"), entry[1].get("end_percent")) for entry in cond]
      == [(0.1, 0.6), (0.0, 0.1), (0.6, 1.0)],
      repr([(entry[1].get("start_percent"), entry[1].get("end_percent")) for entry in cond]))

full = h3._apply_empty_image([(text_states, metadata)], picture, picture_dict, 0.0, 1.0)
check("empty-image 全程只留一段", len(full) == 1, str(len(full)))

bad_picture = {"minimax_token_tags": torch.zeros(3, dtype=torch.long)}
check("tags 不匹配时退回原 cond",
      h3._apply_empty_image([(text_states, metadata)], picture, bad_picture, 0.0, 0.25)
      is not None and len(h3._apply_empty_image([(text_states, metadata)], picture, bad_picture, 0.0, 0.25)) == 1)


# ---------- forward wrapper 的 prompt 尾部偏移 ----------
def executor(_x, _timestep, _context, transformer_options, **_kwargs):
    return transformer_options


wrapper = h3._make_forward_wrapper([(1, 0.5, 0.0), (3, 1.0, 2.0)], 4, "test")
options = {"cond_or_uncond": [0]}
result = wrapper(executor, None, None, torch.zeros(1, 6, 4), options)
weights = result.get(h3._STATE_KEY)
check("positive 条件写入权重", weights is not None, repr(weights))
if weights is not None:
    check("无 layout 时回退到位置索引", weights[0] is None and weights[1] is None, repr(weights))
    check("value 位置加 offset", weights[2].tolist() == [3], repr(weights[2].tolist()))
    check("value 因子形状", tuple(weights[3].shape) == (1, 1), repr(tuple(weights[3].shape)))
    check("bias 位置加 offset", weights[4].tolist() == [5], repr(weights[4].tolist()))
    check("bias 数值", weights[5].tolist() == [2.0], repr(weights[5].tolist()))

precomputed = wrapper(
    executor, None, None, torch.zeros(1, 6, 4),
    {"cond_or_uncond": [0]},
    minimax_payload={"layout": types.SimpleNamespace(seq_len=8)},
)
state = precomputed.get(h3._STATE_KEY)
check("有 layout 时预计算乘子/mask", state is not None and state[0] is not None and state[1] is not None,
      repr(state))
if state is not None and state[0] is not None:
    check("预计算 value 乘子", state[0].shape == (1, 1, 8, 1) and state[0][0, 0, 3, 0].item() == 0.5
          and state[0][0, 0, 2, 0].item() == 1.0, repr(state[0][0, 0, :, 0].tolist()))
    check("预计算 bias mask", state[1][0, 0, 0, 5].item() == 2.0 and state[1][0, 0, 0, 4].item() == 0.0,
          repr(state[1].tolist()))
    check("预计算后不再使用位置索引", state[2] is None and state[3] is None and state[4] is None,
          repr(state))

skipped = wrapper(executor, None, None, torch.zeros(1, 6, 4), {"cond_or_uncond": [0, 1]})
check("非纯正条件不写入权重", h3._STATE_KEY not in skipped, repr(skipped))
short = wrapper(executor, None, None, torch.zeros(1, 3, 4), {"cond_or_uncond": [0]})
check("conditioning 短于 prompt 时不写入权重", h3._STATE_KEY not in short, repr(short))


# ---------- attention override ----------
def passthrough(_q, _k, v, _heads, mask=None, **_kwargs):
    return v if mask is None else mask


multiplier = torch.ones(1, 1, 8, 1)
multiplier[0, 0, 3, 0] = 0.5
value_weights = (multiplier, None, None, None, None, None)
v = torch.ones(1, 2, 8, 4)
h3._make_attention_override(value_weights, None, None)(
    passthrough, torch.zeros(1, 2, 8, 4), torch.zeros(1, 2, 8, 4), v, 2, skip_reshape=True
)
check("4D v 按预计算乘子缩放", v[0, 0, 3, 0].item() == 0.5 and v[0, 0, 2, 0].item() == 1.0,
      repr(v[0, 0, :4, 0].tolist()))

bias_tensor = torch.zeros(1, 1, 1, 8)
bias_tensor[0, 0, 0, 4] = 2.0
bias_weights = (None, bias_tensor, None, None, None, None)
bias = h3._make_attention_override(bias_weights, None, None)(
    passthrough, torch.zeros(1, 2, 8, 4), torch.zeros(1, 2, 8, 4), torch.ones(1, 2, 8, 4),
    2, skip_reshape=True,
)
check("预计算 additive mask 直接复用",
      tuple(bias.shape) == (1, 1, 1, 8) and bias[0, 0, 0, 4].item() == 2.0
      and bias[0, 0, 0, 3].item() == 0.0, repr(bias.tolist()))

captured = {}


class FakeAttention:
    def __init__(self, function):
        self.function = function


def preferred(_q, _k, v, _heads, mask=None, **_kwargs):
    captured["mask"] = mask
    return v


def forbidden(*_args, **_kwargs):
    raise AssertionError("global attention backend should not be used")


h3._make_attention_override(bias_weights, FakeAttention(preferred), None)(
    forbidden, torch.zeros(1, 2, 8, 4), torch.zeros(1, 2, 8, 4), torch.ones(1, 2, 8, 4),
    2, skip_reshape=True,
)
check("有 bias 时保留 checkpoint 的 attention 后端",
      captured.get("mask") is not None and captured["mask"][0, 0, 0, 4].item() == 2.0,
      repr(captured.get("mask")))

# ---------- 图片顺序 ----------
left = torch.zeros(1, 2, 2, 3)
right = torch.zeros(1, 2, 2, 3)
images = {"image_10": left, "image_2": right}
ordered = h3._collect_images(images)
check("参考图按数字顺序", ordered[0] is right and ordered[1] is left,
      repr([id(image) for image in ordered]))


# ---------- execute 全流程（假 H3 core / clip / model） ----------
class FakeH3Conditioning:
    calls = 0

    @classmethod
    def execute(cls, clip, prompt, width, height, vae=None, images=None):
        cls.calls += 1
        rows = len(clip.tokenize(prompt)["h3"][0])
        cond = [(torch.zeros(1, rows, 4), {"minimax_token_tags": torch.ones(rows, dtype=torch.long)})]
        latent = {"samples": torch.zeros(1, 24, 1, max(1, int(height) // 16), max(1, int(width) // 16))}
        return types.SimpleNamespace(result=(cond, latent))


class ExecuteClip(FakeClip):
    def __init__(self):
        super().__init__()
        self.picture_calls = 0

    def encode_from_tokens_scheduled(self, _tokens):
        self.picture_calls += 1
        return [[torch.zeros(1, 2, 4), {"minimax_token_tags": torch.zeros(2, dtype=torch.long)}]]


class FakeModel:
    def __init__(self):
        self.wrappers = {}
        self.block_patches = []
        self.diffusion_model = types.SimpleNamespace(
            blocks=[types.SimpleNamespace(attn=types.SimpleNamespace(comfy_attention=None))]
        )
        self.model_options = {"transformer_options": {"patches_replace": {"dit": {}}}}

    def get_model_object(self, _name):
        return self.diffusion_model

    def clone(self):
        return self

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.pop((wrapper_type, key), None)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers[(wrapper_type, key)] = wrapper

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.block_patches.append((name, block_name, number, patch))


comfy_extras = types.ModuleType("comfy_extras")
comfy_extras.__path__ = []
fake_minimax = types.ModuleType("comfy_extras.nodes_minimax_h3")
fake_minimax.MiniMaxH3ImageConditioning = FakeH3Conditioning
comfy_extras.nodes_minimax_h3 = fake_minimax
sys.modules["comfy_extras"] = comfy_extras
sys.modules["comfy_extras.nodes_minimax_h3"] = fake_minimax

execute_clip = ExecuteClip()
execute_model = FakeModel()
output = h3.MiniMaxH3ImagePromptWeightPreset.execute(
    clip=execute_clip, model=execute_model, text="(cat:1.5)", presets_data="[]",
    width=64, height=64, empty_image=True, empty_start=0.0, empty_end=0.25,
)
result = output.result
check("execute 返回 model/cond/latent", len(result) == 3, str(len(result)))
if len(result) == 3:
    patched, cond, latent = result
    check("empty-image 已拼接到 cond", cond[0][0].shape[1] == 3, str(tuple(cond[0][0].shape)))
    check("cond 区间正确", (cond[0][1].get("start_percent"), cond[0][1].get("end_percent")) == (0.0, 0.25),
          repr((cond[0][1].get("start_percent"), cond[0][1].get("end_percent"))))
    check("latent 尺寸按 width/height", tuple(latent["samples"].shape) == (1, 24, 1, 4, 4),
          str(tuple(latent["samples"].shape)))
    check("注册 diffusion wrapper", ("diffusion_model", h3._WRAPPER_KEY) in patched.wrappers,
          repr(list(patched.wrappers)))
    check("注册 double_block patch", any(item[:3] == ("dit", "double_block", 0) for item in patched.block_patches),
          repr(patched.block_patches))

check("首次 execute 编码一次", FakeH3Conditioning.calls == 1, str(FakeH3Conditioning.calls))
check("首次 execute 编码一次空图", execute_clip.picture_calls == 1, str(execute_clip.picture_calls))

h3.MiniMaxH3ImagePromptWeightPreset.execute(
    clip=execute_clip, model=execute_model, text="(cat:0.3)", presets_data="[]",
    width=64, height=64, empty_image=True, empty_start=0.0, empty_end=0.25,
)
check("只改权重不重编码", FakeH3Conditioning.calls == 1, str(FakeH3Conditioning.calls))
check("只改权重不重编码空图", execute_clip.picture_calls == 1, str(execute_clip.picture_calls))

resized = h3.MiniMaxH3ImagePromptWeightPreset.execute(
    clip=execute_clip, model=execute_model, text="(cat:0.3)", presets_data="[]",
    width=128, height=96, empty_image=True, empty_start=0.0, empty_end=0.25,
).result
check("只改宽高不重编码", FakeH3Conditioning.calls == 1, str(FakeH3Conditioning.calls))
check("只改宽高重建 latent", tuple(resized[2]["samples"].shape) == (1, 24, 1, 6, 8),
      str(tuple(resized[2]["samples"].shape)))

h3.MiniMaxH3ImagePromptWeightPreset.execute(
    clip=execute_clip, model=execute_model, text="dog", presets_data="[]",
    width=64, height=64, empty_image=True, empty_start=0.0, empty_end=0.25,
)
check("文本变化后重新编码", FakeH3Conditioning.calls == 2, str(FakeH3Conditioning.calls))


if FAIL:
    print("失败的用例：" + ", ".join(FAIL))
    raise SystemExit(1)
print("全部通过")
