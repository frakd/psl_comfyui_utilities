# -*- coding: utf-8 -*-
"""Text and token-position helpers for the MOSS-Image prompt preset node.

The module is deliberately free of torch/ComfyUI imports so the merge and
phrase-location logic can be unit tested. `clip` is duck-typed and only needs
`tokenize()` and `encode_from_tokens_scheduled()`.
"""

import json
import logging
import re
import threading
import weakref
from collections import OrderedDict

# Qwen3.5 chat template: <|im_start|>, user, newline, <|im_end|>.
SPECIALS = (248045, 846, 198, 248046)

_WEIGHT_RE = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")
# The slider exposes -2..2: >1 emphasizes, 1 is neutral, 0 removes, <0 subtracts.
_MIN_ENTRY_WEIGHT = -2.0
_MAX_ENTRY_WEIGHT = 2.0

# Conditioning cache: merged text in, (cond, token ids, cond_len) out.
_MAX_CACHE_ENTRIES = 4
_MISS = object()
_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _cache_key(clip, text):
    patcher = getattr(clip, "patcher", None)
    patches = getattr(patcher, "patches", None)
    try:
        patch_count = len(patches) if patches is not None else 0
    except TypeError:
        patch_count = 0
    return (
        id(clip),
        getattr(clip, "layer_idx", None),
        getattr(patcher, "patches_uuid", None),
        patch_count,
        text,
    )


def _conditioning_has_hooks(conditioning):
    try:
        for item in conditioning:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                metadata = item[1]
                if isinstance(metadata, dict) and metadata.get("hooks") is not None:
                    return True
    except TypeError:
        return True
    return False


def _cache_get(clip, text):
    key = _cache_key(clip, text)
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return _MISS
        clip_ref, value = entry
        if clip_ref() is not clip:
            _CACHE.pop(key, None)
            return _MISS
        _CACHE.move_to_end(key)
        return value


def _cache_put(clip, text, value):
    try:
        clip_ref = weakref.ref(clip)
    except TypeError:
        return
    key = _cache_key(clip, text)
    with _CACHE_LOCK:
        _CACHE[key] = (clip_ref, value)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_CACHE_ENTRIES:
            _CACHE.popitem(last=False)


def clear_cache():
    with _CACHE_LOCK:
        _CACHE.clear()


def _encode_cached(clip, text, enabled):
    tokens = clip.tokenize(text)
    ids = _token_ids(tokens)
    if enabled:
        cached = _cache_get(clip, text)
        if cached is not _MISS:
            cond, cached_ids, cond_len = cached
            # Re-tokenizing is cheap; a changed tokenizer/template must not
            # return a conditioning computed from the old token ids.
            if cached_ids == ids:
                return cached
    cond = clip.encode_from_tokens_scheduled(tokens)
    cond_len = cond[0][0].shape[1]
    if enabled and not _conditioning_has_hooks(cond):
        _cache_put(clip, text, (cond, ids, cond_len))
    return cond, ids, cond_len


def _warn(message):
    logging.warning(f"MOSS-Image prompt presets: {message}")


def _normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_presets(raw):
    """Return [(text, weight)] for the enabled, non-empty list entries."""
    if isinstance(raw, str):
        try:
            data = json.loads(raw or "[]")
        except (TypeError, ValueError) as exc:
            _warn(f"presets_data is not valid JSON ({exc}); no presets applied")
            return []
    elif raw is None:
        data = []
    else:
        data = raw

    if isinstance(data, dict):
        data = data.get("presets", [])
    if not isinstance(data, list):
        _warn("presets_data must be a JSON list; no presets applied")
        return []

    entries = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            text, weight, enabled = item, 1.0, True
        elif isinstance(item, dict):
            text = item.get("text", "")
            enabled = bool(item.get("enabled", True))
            try:
                weight = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                _warn(f"entry {index} has a non-numeric weight; using 1.0")
                weight = 1.0
        else:
            _warn(f"entry {index} is not an object; skipped")
            continue

        text = _normalize_text(text)
        if not enabled or not text:
            continue
        weight = max(_MIN_ENTRY_WEIGHT, min(_MAX_ENTRY_WEIGHT, weight))
        entries.append((text, weight))
    return entries


def extract_weighted_terms(text):
    """Split (phrase:weight) terms out of free text; returns (clean_text, terms)."""
    text = str(text or "")
    terms = [(match.group(1).strip(), float(match.group(2))) for match in _WEIGHT_RE.finditer(text)]
    clean = _WEIGHT_RE.sub(lambda match: match.group(1), text).strip()
    return clean, terms


def merge_prompt(user_text, entries, order="before"):
    """Put the entry block before or after the user prompt; weights stay out of the text."""
    user_text = str(user_text or "").strip()
    # Spaces keep a trailing ')' from merging with the separator comma in BPE,
    # so the phrase keeps the same tokens as in its standalone template.
    block = " ".join(text for text, _ in entries if text)
    if not block:
        return user_text
    if not user_text:
        return block
    return f"{block} {user_text}" if str(order).lower() != "after" else f"{user_text} {block}"


def weight_pair(weight):
    """(value multiplier, attention-logit bias) for a KJ-style weight."""
    weight = float(weight)
    if weight > 1.0:
        return 1.0, (weight - 1.0) * 2.0
    return weight, 0.0


def merge_pairs(pairs):
    """Collapse repeated positions: value multipliers multiply, logit biases add."""
    merged = {}
    for position, value_factor, logit_bias in pairs:
        old_factor, old_bias = merged.get(position, (1.0, 0.0))
        merged[position] = (old_factor * value_factor, old_bias + logit_bias)
    return [
        (position, factor, bias)
        for position, (factor, bias) in sorted(merged.items())
        if factor != 1.0 or bias != 0.0
    ]


def _token_ids(tokens):
    key = next(iter(tokens))
    return [item[0] for item in tokens[key][0]]


def _user_span(ids, specials=SPECIALS):
    im_start, user, newline, im_end = specials
    for index in range(max(0, len(ids) - 2)):
        if ids[index] == im_start and ids[index + 1] == user and ids[index + 2] == newline:
            end = index + 3
            while end < len(ids) and ids[end] != im_end:
                end += 1
            if end < len(ids):
                return index + 3, end
    return None, None


def _find_subsequence(sequence, subsequence, start, end):
    size = len(subsequence)
    if not size:
        return []
    hits = []
    for index in range(max(0, start), min(len(sequence), end) - size + 1):
        if sequence[index:index + size] == subsequence:
            hits.append(index)
    return hits


def _phrase_candidates(clip, phrase, specials=SPECIALS):
    # The official tokenizer skips its chat template when the text already starts
    # with <|im_start|>; wrapping the phrase this way keeps a leading space intact.
    candidates = []
    for variant in (" " + phrase, phrase):
        ids = _token_ids(clip.tokenize(f"<|im_start|>user\n{variant}<|im_end|>"))
        start, end = _user_span(ids, specials)
        if start is None:
            continue
        sub = ids[start:end]
        if sub and sub not in candidates:
            candidates.append(sub)
    return candidates


def locate_pairs(clip, merged_text, terms, specials=SPECIALS, cache=True):
    """Encode the merged text and locate every phrase's conditioning positions.

    The encoder result is cached by (CLIP object, patch state, merged text);
    cache=False forces a fresh encode. Returns
    (conditioning, [(position, value_factor, logit_bias)], text_length).
    """
    cond, ids, cond_len = _encode_cached(clip, merged_text, cache)
    if not ids:
        return cond, [], cond_len

    visible_start = max(0, len(ids) - cond_len)
    start, end = _user_span(ids, specials)
    if start is None:
        start, end = visible_start, len(ids)

    pairs = []
    for phrase, weight in terms:
        phrase = str(phrase or "").strip()
        if not phrase:
            continue
        hits = set()
        for candidate in _phrase_candidates(clip, phrase, specials):
            for index in _find_subsequence(ids, candidate, start, end):
                hits.update(range(index, index + len(candidate)))
        positions = [index - visible_start for index in sorted(hits)]
        positions = [position for position in positions if 0 <= position < cond_len]
        if not positions:
            _warn(f"phrase {phrase[:60]!r} not found in the prompt; its weight is ignored")
            continue
        value_factor, logit_bias = weight_pair(weight)
        pairs.extend((position, value_factor, logit_bias) for position in positions)

    return cond, merge_pairs(pairs), cond_len
