"""Unit tests for value_curve_core, runnable without a ComfyUI server.

    cd ComfyUI
    ../python/python.exe custom_nodes/psl_comfyui_utilities/tests/test_value_curve.py
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "nodes" / "value_curve_core.py"
spec = importlib.util.spec_from_file_location("value_curve_core_ut", MODULE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_generate_linear():
    assert core.generate_values(0.0, 1.0, "linear", 3) == [0.0, 0.5, 1.0]
    assert core.generate_values(0.35, 0.35, "linear", 4) == [0.35, 0.35, 0.35, 0.35]
    assert core.generate_values(1.0, 0.0, "linear", 3) == [1.0, 0.5, 0.0]


def test_curve_endpoints_and_monotonicity():
    for curve in core.CURVE_NAMES:
        values = core.generate_values(0.0, 1.0, curve, 5)
        assert values[0] == 0.0
        assert abs(values[-1] - 1.0) < 1e-9
        assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_parse_value_lists():
    assert core.parse_value_list("0.3, 0.6, 1.2") == [0.3, 0.6, 1.2]
    assert core.parse_value_list("[0.3, 0.6]") == [0.3, 0.6]
    assert core.parse_value_list("(0.3 0.6)") == [0.3, 0.6]
    assert core.parse_value_list("0.3，0.6;1.2") == [0.3, 0.6, 1.2]
    assert core.parse_value_list("  ") == []
    for bad in ("0.3, x", "0.3, nan"):
        try:
            core.parse_value_list(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid list must raise: {bad!r}")


def test_resolve_prefers_manual():
    values, manual = core.resolve_values(0.0, 1.0, "linear", 3, "0.2, 0.4")
    assert manual is True
    assert values == [0.2, 0.4]
    values, manual = core.resolve_values(0.0, 1.0, "linear", 3, "")
    assert manual is False
    assert values == [0.0, 0.5, 1.0]


def test_format_values():
    assert core.format_values([0, 0.3333333, 0.5, 1]) == "0, 0.333, 0.5, 1"
    assert core.format_values([-0.0, -0.0004]) == "0, 0"
    assert core.format_values([0.1235, 0.35, -0.5]) == "0.124, 0.35, -0.5"


if __name__ == "__main__":
    test_generate_linear()
    test_curve_endpoints_and_monotonicity()
    test_parse_value_lists()
    test_resolve_prefers_manual()
    test_format_values()
    print("value curve tests passed")
