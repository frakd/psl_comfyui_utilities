"""Unit tests for CachedCLIPTextEncode, runnable without a ComfyUI server.

    cd ComfyUI
    python custom_nodes/psl_comfyui_utilities/tests/test_cached_clip_text_encode.py
"""

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

MODULE_PATH = REPO_ROOT / "nodes" / "cached_clip_text_encode.py"
spec = importlib.util.spec_from_file_location("psl_cached_clip_text_encode", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

Cached = module.CachedCLIPTextEncode


class FakeClip:
    def __init__(self):
        self.calls = 0
        self.layer_idx = None
        self.patcher = types.SimpleNamespace(patches={})

    def tokenize(self, text):
        return [text]

    def encode_from_tokens_scheduled(self, tokens):
        self.calls += 1
        return [[object(), {"pooled_output": object()}]]


class FakeHookClip(FakeClip):
    def encode_from_tokens_scheduled(self, tokens):
        self.calls += 1
        return [[object(), {"hooks": object()}]]


def run(clip, text, cache_mode="enabled"):
    out = Cached.execute(clip=clip, text=text, cache_mode=cache_mode)
    return out.result[0]


def test_same_clip_and_text_hits_cache():
    module._clear_cache()
    clip = FakeClip()

    first = run(clip, "hello")
    second = run(clip, "hello")

    assert clip.calls == 1, clip.calls
    assert first is second


def test_different_text_reencodes():
    module._clear_cache()
    clip = FakeClip()

    run(clip, "one")
    run(clip, "two")

    assert clip.calls == 2


def test_different_clip_object_misses():
    module._clear_cache()
    first_clip = FakeClip()
    second_clip = FakeClip()

    run(first_clip, "same text")
    run(second_clip, "same text")

    assert first_clip.calls == 1
    assert second_clip.calls == 1


def test_disabled_mode_bypasses_cache():
    module._clear_cache()
    clip = FakeClip()

    run(clip, "hello")
    run(clip, "hello", cache_mode="disabled")

    assert clip.calls == 2


def test_hooks_conditioning_is_not_cached():
    module._clear_cache()
    clip = FakeHookClip()

    run(clip, "hello")
    run(clip, "hello")

    assert clip.calls == 2


def test_layer_idx_and_patch_count_participate_in_key():
    module._clear_cache()
    clip = FakeClip()

    run(clip, "x")
    clip.layer_idx = 1
    run(clip, "x")
    assert clip.calls == 2

    clip.layer_idx = None
    clip.patcher.patches["lora"] = object()
    run(clip, "x")
    assert clip.calls == 3


def test_patches_uuid_change_invalidates_same_patch_count():
    module._clear_cache()
    clip = FakeClip()
    clip.patcher.patches = {"same": object()}
    clip.patcher.patches_uuid = "uuid-0"

    run(clip, "x")
    assert clip.calls == 1

    # 模拟原地 add_patches：patch 数量不变，但 patcher 的 weights uuid 变了
    clip.patcher.patches = {"same": object()}
    clip.patcher.patches_uuid = "uuid-1"
    run(clip, "x")
    assert clip.calls == 2


def test_lru_evicts_oldest_entry():
    module._clear_cache()
    clip = FakeClip()

    for index in range(module._MAX_CACHE_ENTRIES + 1):
        run(clip, f"text-{index}")
    assert clip.calls == module._MAX_CACHE_ENTRIES + 1

    run(clip, f"text-{module._MAX_CACHE_ENTRIES}")
    assert clip.calls == module._MAX_CACHE_ENTRIES + 1

    run(clip, "text-0")
    assert clip.calls == module._MAX_CACHE_ENTRIES + 2


def test_none_clip_is_reported():
    module._clear_cache()
    try:
        Cached.execute(clip=None, text="hello")
    except RuntimeError as error:
        assert "clip input is invalid" in str(error)
    else:
        raise AssertionError("None clip was silently accepted")


def test_schema_exposes_conditioning_output():
    schema = Cached.GET_SCHEMA()

    assert schema.node_id == "CachedCLIPTextEncode"
    assert [output.id for output in schema.outputs] == ["conditioning"]
    assert [item.id for item in schema.inputs] == ["clip", "text", "cache_mode"]


if __name__ == "__main__":
    test_same_clip_and_text_hits_cache()
    test_different_text_reencodes()
    test_different_clip_object_misses()
    test_disabled_mode_bypasses_cache()
    test_hooks_conditioning_is_not_cached()
    test_layer_idx_and_patch_count_participate_in_key()
    test_patches_uuid_change_invalidates_same_patch_count()
    test_lru_evicts_oldest_entry()
    test_none_clip_is_reported()
    test_schema_exposes_conditioning_output()
    print("cached clip text encode tests passed")
