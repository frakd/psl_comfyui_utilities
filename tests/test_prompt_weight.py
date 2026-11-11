# -*- coding: utf-8 -*-
"""验证 prompt_weight_core：预设解析、文本合并、权重公式、短语定位。不需要 torch/ComfyUI。"""
import importlib.util
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FAIL = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f"  <- {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(name)


spec = importlib.util.spec_from_file_location("prompt_weight_core_ut", REPO_ROOT / "nodes" / "prompt_weight_core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


# ---------- 权重解析 ----------
clean, terms = core.extract_weighted_terms("a cat, (blurry:-1), (masterpiece:1.2)")
check("去掉权重标记后保留短语", clean == "a cat, blurry, masterpiece", repr(clean))
check("提取权重条目", terms == [("blurry", -1.0), ("masterpiece", 1.2)], repr(terms))

check("w=2 -> (1, +2)", core.weight_pair(2.0) == (1.0, 2.0), repr(core.weight_pair(2.0)))
check("w=0.5 -> (0.5, 0)", core.weight_pair(0.5) == (0.5, 0.0), repr(core.weight_pair(0.5)))
check("w=0 -> (0, 0)", core.weight_pair(0.0) == (0.0, 0.0), repr(core.weight_pair(0.0)))
check("w=-1 -> (-1, 0)", core.weight_pair(-1.0) == (-1.0, 0.0), repr(core.weight_pair(-1.0)))
check("w=-2 -> (-2, 0)", core.weight_pair(-2.0) == (-2.0, 0.0), repr(core.weight_pair(-2.0)))
check("w=1 -> no-op", core.weight_pair(1.0) == (1.0, 0.0), repr(core.weight_pair(1.0)))

merged = core.merge_pairs([(1, 2.0, 0.0), (1, 3.0, 0.0), (2, 1.0, 1.0), (2, 1.0, 2.0), (3, 2.0, 0.0), (3, 0.5, 0.0)])
check("重复位置因子相乘/bias 相加，并过滤 no-op",
      merged == [(1, 6.0, 0.0), (2, 1.0, 3.0)], repr(merged))


# ---------- 预设 JSON ----------
raw = json.dumps([
    {"enabled": True, "text": "  a   cat ", "weight": 2.5},
    {"enabled": False, "text": "hidden", "weight": 1.0},
    {"enabled": True, "text": "   ", "weight": 1.0},
    {"enabled": True, "text": "blurry", "weight": "bad"},
    "simple",
    {"enabled": True, "text": "negative", "weight": -3},
])
check("预设解析/启用/空条目/权重钳制/字符串条目",
      core.parse_presets(raw) == [("a cat", 2.0), ("blurry", 1.0), ("simple", 1.0), ("negative", -2.0)],
      repr(core.parse_presets(raw)))
check("非法 JSON 返回空", core.parse_presets("{oops") == [])
check("presets 包装对象", core.parse_presets(json.dumps({"presets": ["x"]})) == [("x", 1.0)])


# ---------- 合并顺序 ----------
check("预设在前", core.merge_prompt("user prompt", [("a", 1.0), ("b", 2.0)], "before") == "a b user prompt")
check("预设在后", core.merge_prompt("user prompt", [("a", 1.0), ("b", 2.0)], "after") == "user prompt a b")
check("没有预设", core.merge_prompt("user prompt", [], "before") == "user prompt")
check("没有用户文本", core.merge_prompt("", [("a", 1.0), ("b", 2.0)], "after") == "a b")


# ---------- 短语定位（假 clip，验证算法；真实 tokenizer 另行核对） ----------
class FakeTensor:
    def __init__(self, length):
        self.shape = (1, length, 4)


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
                ids.append(248045)
            elif part == "<|im_end|>":
                ids.append(248046)
            elif part == "\n":
                ids.append(198)
            elif part == "user":
                ids.append(846)
            elif part == "assistant":
                ids.append(5)
            elif part == "think":
                ids.append(6)
            else:
                ids.append(self._word(part))
        return ids


class FakeClip:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.encode_calls = 0

    def tokenize(self, text):
        if text.startswith("<|im_start|>"):
            ids = self.tokenizer.encode(text)
        else:
            ids = [248045, 846, 198] + self.tokenizer.encode(text) + [248046, 198, 248045, 5, 198, 6, 198]
        return {"fake": [[(token, 1.0, 0) for token in ids]]}

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        length = len(tokens["fake"][0])
        return [[FakeTensor(length), {}]]


clip = FakeClip()
cond, pairs, cond_len = core.locate_pairs(
    clip, "a cat blurry", [("a cat", 2.0), ("blurry", 0.5)]
)
check("定位到正确的 cond 位置和权重",
      pairs == [(3, 1.0, 2.0), (4, 1.0, 2.0), (5, 0.5, 0.0)], repr(pairs))
check("cond 长度与合并文本一致", cond_len == 6 + 7, repr(cond_len))
check("只定位用户内容区", cond[0][0].shape[1] == cond_len)

_, negative, _ = core.locate_pairs(clip, "a cat blurry", [("blurry", -1.0)])
check("负权重生成负的 value 因子", negative == [(5, -1.0, 0.0)], repr(negative))

_, missing, _ = core.locate_pairs(clip, "a cat", [("dog", 1.5)])
check("找不到的短语不产生权重", missing == [], repr(missing))


# ---------- 全关 / 无权重 ----------
check("全关条目解析为空",
      core.parse_presets(json.dumps([{"enabled": False, "text": "女孩"}])) == [])
clean_no_weight, no_terms = core.extract_weighted_terms("a photo of a cat")
check("无权重时 clean 文本不变", clean_no_weight == "a photo of a cat" and no_terms == [],
      repr((clean_no_weight, no_terms)))
_, no_pairs, _ = core.locate_pairs(FakeClip(), "a photo of a cat", [])
check("无条目/无权重时不产生注意力权重", no_pairs == [], repr(no_pairs))


# ---------- 编码缓存 ----------
core.clear_cache()
cache_clip = FakeClip()
core.locate_pairs(cache_clip, "a cat blurry", [("a cat", 2.0)])
core.locate_pairs(cache_clip, "a cat blurry", [("blurry", 0.5)])
check("相同 CLIP+文本命中编码缓存", cache_clip.encode_calls == 1, str(cache_clip.encode_calls))

core.locate_pairs(cache_clip, "a cat blurry", [("blurry", 0.5)], cache=False)
check("cache=False 强制重新编码", cache_clip.encode_calls == 2, str(cache_clip.encode_calls))


class HookClip(FakeClip):
    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        return [[FakeTensor(len(tokens["fake"][0])), {"hooks": object()}]]


core.clear_cache()
hook_clip = HookClip()
core.locate_pairs(hook_clip, "a cat", [])
core.locate_pairs(hook_clip, "a cat", [])
check("带 hooks 的 conditioning 不缓存", hook_clip.encode_calls == 2, str(hook_clip.encode_calls))


class ChangingTokenClip(FakeClip):
    def __init__(self):
        super().__init__()
        self.salt = 0

    def tokenize(self, text):
        tokens = super().tokenize(text)
        if self.salt:
            tokens["fake"][0].insert(3, (self.salt, 1.0, 0))
        return tokens


core.clear_cache()
changing = ChangingTokenClip()
core.locate_pairs(changing, "a photo of a cat", [])
changing.salt = 999
core.locate_pairs(changing, "a photo of a cat", [])
check("tokenizer 变化时缓存重新编码", changing.encode_calls == 2, str(changing.encode_calls))


if FAIL:
    print("失败的用例：" + ", ".join(FAIL))
    raise SystemExit(1)
print("全部通过")
