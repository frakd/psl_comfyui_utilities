# -*- coding: utf-8 -*-
"""Text and token-position helpers for the MOSS-Image prompt preset node.

The module is deliberately free of torch/ComfyUI imports so the merge and
phrase-location logic can be unit tested. `clip` is duck-typed and only needs
`tokenize()` and `encode_from_tokens_scheduled()`.
"""

import json
import logging
import math
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
MERGE_SEPARATORS = {"space": " ", "none": "", "newline": "\n", "comma": ", "}

# Conditioning cache: merged text in, (cond, token ids, cond_len) out.
_MAX_CACHE_ENTRIES = 4
_MISS = object()
_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()

# Expanded vision tokens occupy several positions; the marker keeps them from
# ever matching a phrase while preserving one list index per conditioning row.
_VISION_SLOT = -1


def _cache_key(clip, text, extra=()):
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
        tuple(extra),
    )


def cache_key(clip, text, extra=()):
    """Stable encode-cache key; callers must still verify object identity."""
    return _cache_key(clip, text, extra)


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


def _cache_get(clip, text, extra, identity):
    key = _cache_key(clip, text, extra)
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return _MISS
        clip_ref, identity_refs, value = entry
        # An id() key alone can alias a recycled object, so the weakrefs must
        # still resolve to the exact tensors (or CLIP) that produced the entry.
        if clip_ref() is not clip or any(ref() is not obj for ref, obj in zip(identity_refs, identity)):
            _CACHE.pop(key, None)
            return _MISS
        _CACHE.move_to_end(key)
        return value


def _cache_put(clip, text, value, extra, identity):
    try:
        clip_ref = weakref.ref(clip)
        identity_refs = [weakref.ref(obj) for obj in identity]
    except TypeError:
        return
    key = _cache_key(clip, text, extra)
    with _CACHE_LOCK:
        _CACHE[key] = (clip_ref, identity_refs, value)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_CACHE_ENTRIES:
            _CACHE.popitem(last=False)


def clear_cache():
    with _CACHE_LOCK:
        _CACHE.clear()


def _token_entries(tokens):
    key = next(iter(tokens))
    return tokens[key][0]


def _expand_entries(entries, vision_slots=None):
    """Token ids with every vision entry expanded into its conditioning slots.

    Returns None when a vision entry is present but its slot count is unknown;
    positions would then be wrong, so the caller must skip weighting.
    """
    ids = []
    for entry in entries:
        token = entry[0]
        if not isinstance(token, dict):
            ids.append(token)
            continue
        count = vision_slots(token) if vision_slots is not None else None
        if not isinstance(count, int) or count < 1:
            return None
        ids.extend([_VISION_SLOT] * count)
    return ids


def _encode_cached(clip, text, enabled, images=None, vision_slots=None, extra=(), identity=()):
    tokens = clip.tokenize(text, images=list(images)) if images else clip.tokenize(text)
    ids = _expand_entries(_token_entries(tokens), vision_slots)
    if enabled and ids is not None:
        cached = _cache_get(clip, text, extra, identity)
        if cached is not _MISS:
            cond, cached_ids, cond_len = cached
            # Re-tokenizing is cheap; a changed tokenizer/template or a
            # different image must not return the old conditioning.
            if cached_ids == ids:
                return cached
    cond = clip.encode_from_tokens_scheduled(tokens)
    cond_len = cond[0][0].shape[1]
    if enabled and ids is not None and not _conditioning_has_hooks(cond):
        _cache_put(clip, text, (cond, ids, cond_len), extra, identity)
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


def parse_strength_schedule(raw):
    """Split a list of strengths into equal trajectory slices.

    [0.4, 0.8, 1.2] -> [(0.0, 1/3, 0.4), (1/3, 2/3, 0.8), (2/3, 1.0, 1.2)].
    Empty, invalid or non-positive-length lists return []; values clamp to 0..100.
    """
    if isinstance(raw, (list, tuple)):
        data = list(raw)
    else:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            _warn(f"strength_schedule is not valid JSON ({exc}); schedule ignored")
            return []
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        data = [data]
    if not isinstance(data, list):
        _warn("strength_schedule must be a JSON list of numbers; schedule ignored")
        return []

    values = []
    for index, item in enumerate(data):
        try:
            value = float(item)
        except (TypeError, ValueError):
            _warn(f"strength_schedule entry {index} is not a number; schedule ignored")
            return []
        if not math.isfinite(value):
            _warn(f"strength_schedule entry {index} is not finite; schedule ignored")
            return []
        values.append(max(0.0, min(100.0, value)))
    if not values:
        return []

    width = 1.0 / len(values)
    return [(index * width, (index + 1) * width, value) for index, value in enumerate(values)]


def extract_weighted_terms(text):
    """Split (phrase:weight) terms out of free text; returns (clean_text, terms)."""
    text = str(text or "")
    terms = [(match.group(1).strip(), float(match.group(2))) for match in _WEIGHT_RE.finditer(text)]
    clean = _WEIGHT_RE.sub(lambda match: match.group(1), text).strip()
    return clean, terms


def merge_prompt(user_text, entries, order="before", separator=" ", boundary_separator=" "):
    """Put the entry block before or after the user prompt; weights stay out of the text.

    `separator` joins the entries; `boundary_separator` joins the entry block and
    the user text. The default boundary space keeps a trailing ')' from merging
    with the separator comma in BPE.
    """
    user_text = str(user_text or "").strip()
    block = str(separator or "").join(text for text, _ in entries if text)
    if not block:
        return user_text
    if not user_text:
        return block
    boundary = str(boundary_separator or "")
    return f"{block}{boundary}{user_text}" if str(order).lower() != "after" else f"{user_text}{boundary}{block}"


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


def encode_layout(clip, text, cache=True, images=None, vision_slots=None,
                  extra=(), identity=()):
    """Tokenize (optionally with images), encode, and expand the token layout.

    `images` are passed to `clip.tokenize` as vision inputs; `vision_slots`
    maps a vision token to the number of conditioning positions it occupies.
    The result is cached by (CLIP, patch state, text, extra) and identity-checked
    against the CLIP object and `identity` tensors. Returns
    (conditioning, expanded token ids or None, conditioning length).
    """
    return _encode_cached(clip, text, cache, images, vision_slots, extra, identity)


def positions_from_ids(clip, ids, cond_len, terms, specials=SPECIALS):
    """[(position, value_factor, logit_bias)] for every phrase occurrence."""
    if not ids:
        return []

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

    return merge_pairs(pairs)


def locate_pairs(clip, merged_text, terms, specials=SPECIALS, cache=True, images=None,
                 vision_slots=None, cache_extra=(), cache_identity=()):
    """Encode the merged text and locate every phrase's conditioning positions.

    The encoder result is cached by (CLIP object, patch state, merged text,
    cache_extra); cache=False forces a fresh encode. When vision images are
    used, the expanded layout must match the conditioning length exactly,
    otherwise weighting is skipped. Returns
    (conditioning, [(position, value_factor, logit_bias)], text_length).
    """
    cond, ids, cond_len = encode_layout(clip, merged_text, cache=cache, images=images,
                                        vision_slots=vision_slots, extra=cache_extra,
                                        identity=cache_identity)
    if ids is None:
        _warn("the prompt contains vision tokens this node cannot measure; weighting skipped")
        return cond, [], cond_len
    if vision_slots is not None and len(ids) != cond_len:
        _warn(f"expanded prompt has {len(ids)} positions but the conditioning has {cond_len}; "
              "weighting skipped")
        return cond, [], cond_len
    return cond, positions_from_ids(clip, ids, cond_len, terms, specials), cond_len
