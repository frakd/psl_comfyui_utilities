"""Pure value-curve helpers shared by the Value Curve node (no torch/ComfyUI imports)."""

import math
import re
from typing import Any

CURVE_NAMES = ("linear", "ease_in", "ease_out", "ease_in_out", "cosine")
MAX_LIST_POINTS = 2000
_LIST_SEPARATORS = re.compile(r"[,;\s，、]+")


def curve_value(curve: str, t: float) -> float:
    """Monotonic 0 -> 1 curve; kept in sync with ui/value_curve.js."""
    t = min(1.0, max(0.0, float(t)))
    if curve == "linear":
        return t
    if curve == "ease_in":
        return t * t
    if curve == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if curve == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    if curve == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    raise ValueError(f"未知曲线类型: {curve!r}（可选: {', '.join(CURVE_NAMES)}）")


def generate_values(
    start_strength: float,
    end_strength: float,
    curve: str,
    points: int,
) -> list[float]:
    points = int(points)
    if points < 2:
        raise ValueError("points 至少为 2")
    if curve not in CURVE_NAMES:
        raise ValueError(f"未知曲线类型: {curve!r}（可选: {', '.join(CURVE_NAMES)}）")
    start = float(start_strength)
    end = float(end_strength)
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("开始/结束强度必须是有限数值")

    span = points - 1
    values = []
    for i in range(points):
        t = i / span
        values.append(start + (end - start) * curve_value(curve, t))
    return values


def parse_value_list(text: Any) -> list[float]:
    """Parse a manual list (brackets optional); empty text means "generate from the curve"."""
    text = "" if text is None else str(text).strip()
    if not text:
        return []
    if text[0] in "[(":
        text = text[1:]
    if text and text[-1] in "])":
        text = text[:-1]

    values = []
    for token in _LIST_SEPARATORS.split(text):
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"强度列表含非数字项 {token!r}；请用逗号/空格分隔数字"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"强度列表含非有限数值: {token!r}")
        values.append(value)

    if len(values) > MAX_LIST_POINTS:
        raise ValueError(f"强度列表过长（{len(values)} 项，上限 {MAX_LIST_POINTS}）")
    return values


def resolve_values(
    start_strength: float,
    end_strength: float,
    curve: str,
    points: int,
    value_list: Any,
) -> tuple[list[float], bool]:
    manual = parse_value_list(value_list)
    if manual:
        return manual, True
    return generate_values(start_strength, end_strength, curve, points), False


def format_value(value: float, digits: int = 3) -> str:
    """Match the JS panel's trimNumber: half-up rounding, trailing zeros removed."""
    scale = 10 ** digits
    rounded = math.floor(float(value) * scale + 0.5) / scale
    if rounded == 0:
        rounded = 0.0
    text = f"{rounded:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def format_values(values) -> str:
    return ", ".join(format_value(value) for value in values)
