"""Unit tests for TextContainer, runnable without a ComfyUI server.

    cd ComfyUI
    python custom_nodes/psl_comfyui_utilities/tests/test_text_container.py
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

MODULE_PATH = REPO_ROOT / "nodes" / "text_container.py"
spec = importlib.util.spec_from_file_location("psl_text_container", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from comfy_execution.graph_utils import ExecutionBlocker  # noqa: E402

TextContainer = module.TextContainer
NONE = object()


def run(
    text_in=None,
    delimiter="\\n",
    unselected_output="none",
    selection=None,
    manual=None,
    upstream_mode="auto",
):
    output = TextContainer.execute(
        text_in=text_in,
        delimiter=delimiter,
        unselected_output=unselected_output,
        selection_json=json.dumps(selection or []),
        manual_json=json.dumps(manual or []),
        upstream_mode=upstream_mode,
    )
    return output.result, output.ui


def split_of(ui):
    return ui["split"][0]


def test_schema_shape():
    schema = TextContainer.GET_SCHEMA()

    assert schema.node_id == "TextContainer"
    assert schema.is_output_node is True
    assert [output.id for output in schema.outputs] == [f"text_{i}" for i in range(1, 7)]
    assert [item.id for item in schema.inputs] == [
        "text_in",
        "delimiter",
        "unselected_output",
        "selection_json",
        "manual_json",
        "ui_state_json",
        "upstream_mode",
    ]


def test_split_trims_and_drops_empty_parts():
    out, ui = run(" a \r\n\r\n b \r\n   \r\n c ", selection=[0, 1, 2])

    assert split_of(ui) == ["a", "b", "c"]
    assert out[:3] == ("a", "b", "c")
    assert out[3:] == (None, None, None)


def test_escape_sequences_and_multichar_delimiter():
    _, ui = run("a\tb\tc", delimiter="\\t")
    assert split_of(ui) == ["a", "b", "c"]

    _, ui = run("a\n\nb\nc", delimiter="\\n\\n")
    assert split_of(ui) == ["a", "b\nc"]

    _, ui = run("a::b::c", delimiter="::")
    assert split_of(ui) == ["a", "b", "c"]


def test_empty_delimiter_means_no_split():
    _, ui = run(" a \n b ", delimiter="")
    assert split_of(ui) == ["a \n b"]


def test_crlf_is_normalized_before_blank_line_split():
    _, ui = run("a\r\n\r\nb\r\nc", delimiter="\\n\\n")
    assert split_of(ui) == ["a", "b\nc"]


def test_selection_order_maps_to_output_slots():
    out, _ = run("a\nb\nc", selection=[2, 0, 1])

    assert out[:3] == ("c", "a", "b")
    assert out[3:] == (None, None, None)


def test_missing_selected_slot_uses_inert_mode():
    for mode, expected in (("none", None), ("empty", "")):
        out, _ = run("only", selection=[0, 3], unselected_output=mode)
        assert out[0] == "only"
        assert out[1] == expected
        assert out[2:] == (expected,) * 4

    out, _ = run("only", selection=[0, 3], unselected_output="block")
    assert out[0] == "only"
    assert isinstance(out[1], ExecutionBlocker)
    assert out[1].message is None
    assert all(isinstance(value, ExecutionBlocker) for value in out[2:])


def test_unselected_slots_use_inert_mode():
    out, _ = run("a\nb", selection=[1])

    assert out[0] == "b"
    assert out[1:] == (None,) * 5


def test_entries_beyond_six_are_selectable():
    out, _ = run("\n".join(str(i) for i in range(8)), selection=[7])

    assert out[0] == "7"


def test_selection_is_truncated_to_six_slots():
    out, _ = run("a\nb\nc\nd\ne\nf\ng", selection=list(range(8)))

    assert out == ("a", "b", "c", "d", "e", "f")


def test_manual_mode_preserves_text_and_treats_blank_as_missing():
    out, ui = run(
        text_in=None,
        manual=["a", "", " c ", "   "],
        selection=[0, 1, 2, 3],
        upstream_mode="manual",
    )

    assert ui["source"][0] == "manual"
    assert split_of(ui) is None
    assert out[0] == "a"
    assert out[1] is None
    assert out[2] == " c "
    assert out[3] is None
    assert out[4:] == (None, None)


def test_manual_whitespace_only_blocks_or_empties():
    out, _ = run(manual=["  "], selection=[0], unselected_output="empty", upstream_mode="manual")
    assert out[0] == ""

    out, _ = run(manual=["  "], selection=[0], unselected_output="block", upstream_mode="manual")
    assert isinstance(out[0], ExecutionBlocker)


def test_linked_mode_ignores_manual_content():
    out, ui = run(
        text_in="upstream",
        manual=["prefill"],
        selection=[0],
        upstream_mode="linked",
    )

    assert ui["source"][0] == "upstream"
    assert out[0] == "upstream"


def test_linked_mode_with_none_text_is_empty_not_manual_fallback():
    out, ui = run(
        text_in=None,
        manual=["prefill"],
        selection=[0],
        upstream_mode="linked",
    )

    assert ui["source"][0] == "upstream"
    assert split_of(ui) == []
    assert out == (None,) * 6


def test_linked_mode_with_empty_string_is_empty():
    out, ui = run(text_in="", manual=["prefill"], selection=[0], upstream_mode="linked")

    assert ui["source"][0] == "upstream"
    assert split_of(ui) == []
    assert out == (None,) * 6


def test_auto_mode_infers_from_text_in():
    out, ui = run(text_in="a\nb", manual=["prefill"], selection=[0], upstream_mode="auto")
    assert ui["source"][0] == "upstream"
    assert out[0] == "a"

    out, ui = run(text_in=None, manual=["prefill"], selection=[0], upstream_mode="auto")
    assert ui["source"][0] == "manual"
    assert split_of(ui) is None
    assert out[0] == "prefill"


def test_empty_selection_outputs_all_inert():
    out, _ = run("a\nb", selection=[], unselected_output="empty")

    assert out == ("",) * 6


def test_invalid_json_is_reported():
    for field, raw in (
        ("selection_json", "{oops"),
        ("manual_json", "{\"not\": \"a list\"}"),
    ):
        kwargs = dict(selection_json="[]", manual_json="[]")
        kwargs[field] = raw
        try:
            TextContainer.execute(**kwargs)
        except ValueError as error:
            assert field in str(error)
        else:
            raise AssertionError(f"{field} accepted invalid JSON")

    for raw in ('["0"]', "[true]", "[-1]", "[1.5]"):
        try:
            TextContainer.execute(selection_json=raw, manual_json="[]")
        except ValueError as error:
            assert "selection_json" in str(error)
        else:
            raise AssertionError(f"selection_json accepted {raw!r}")


def test_invalid_modes_are_reported():
    try:
        TextContainer.execute(selection_json="[]", manual_json="[]", unselected_output="whatever")
    except ValueError as error:
        assert "unselected_output" in str(error)
    else:
        raise AssertionError("invalid unselected_output was accepted")

    try:
        TextContainer.execute(selection_json="[]", manual_json="[]", upstream_mode="whatever")
    except ValueError as error:
        assert "upstream_mode" in str(error)
    else:
        raise AssertionError("invalid upstream_mode was accepted")


if __name__ == "__main__":
    test_schema_shape()
    test_split_trims_and_drops_empty_parts()
    test_escape_sequences_and_multichar_delimiter()
    test_empty_delimiter_means_no_split()
    test_crlf_is_normalized_before_blank_line_split()
    test_selection_order_maps_to_output_slots()
    test_missing_selected_slot_uses_inert_mode()
    test_unselected_slots_use_inert_mode()
    test_entries_beyond_six_are_selectable()
    test_selection_is_truncated_to_six_slots()
    test_manual_mode_preserves_text_and_treats_blank_as_missing()
    test_manual_whitespace_only_blocks_or_empties()
    test_linked_mode_ignores_manual_content()
    test_linked_mode_with_none_text_is_empty_not_manual_fallback()
    test_linked_mode_with_empty_string_is_empty()
    test_auto_mode_infers_from_text_in()
    test_empty_selection_outputs_all_inert()
    test_invalid_json_is_reported()
    test_invalid_modes_are_reported()
    print("text container tests passed")
