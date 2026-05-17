"""Tests for the pure helpers in :mod:`core.llm.base`.

These run on the raw LLM response *before* JSON validation. A bug
here means models that wrap their answer in markdown fences or
think-blocks fall through as parse errors.
"""

from __future__ import annotations

from halal_trader.core.llm.base import _clean_json_body, strip_thinking

# ── strip_thinking ──────────────────────────────────────────


def test_strip_thinking_no_tags_returns_empty_chain():
    thinking, body = strip_thinking("just an answer")
    assert thinking == ""
    assert body == "just an answer"


def test_strip_thinking_single_block_extracted():
    """The ``<think>`` block ends up in `thinking`; the body is what
    remains after the tags are removed."""
    thinking, body = strip_thinking("<think>some reasoning</think>\nfinal answer")
    assert thinking == "some reasoning"
    assert body == "final answer"


def test_strip_thinking_multiple_blocks_joined_with_blank_lines():
    """Several `<think>` chunks get concatenated with blank-line
    separators so log readers can scan them."""
    thinking, body = strip_thinking("<think>step 1</think>middle<think>step 2</think>end")
    assert "step 1" in thinking
    assert "step 2" in thinking
    assert thinking.count("\n\n") >= 1
    assert "middle" in body
    assert "end" in body


def test_strip_thinking_handles_whitespace_only_block():
    """A pure-whitespace `<think>` doesn't contribute to the chain."""
    thinking, body = strip_thinking("<think>   </think>real answer")
    assert thinking == ""
    assert body == "real answer"


def test_strip_thinking_strips_outer_whitespace_from_body():
    _, body = strip_thinking("<think>x</think>\n\n  hi  \n\n")
    assert body == "hi"


# ── _clean_json_body ────────────────────────────────────────


def test_clean_passes_plain_json_through():
    assert _clean_json_body('{"a": 1}') == '{"a": 1}'


def test_clean_strips_triple_backtick_fences():
    """Models often wrap their JSON in ```json … ``` fences."""
    raw = '```json\n{"a": 1}\n```'
    assert _clean_json_body(raw) == '{"a": 1}'


def test_clean_strips_unmarked_triple_backtick_fences():
    raw = '```\n{"a": 1}\n```'
    assert _clean_json_body(raw) == '{"a": 1}'


def test_clean_drops_leading_prose_before_first_brace():
    """If the model adds a prose preamble, slice it off so the parser
    doesn't choke on the chatter."""
    raw = 'Sure! Here is the JSON: {"a": 1}'
    out = _clean_json_body(raw)
    assert out.startswith("{")
    assert "Sure" not in out


def test_clean_keeps_brace_at_position_zero():
    """If the JSON starts at index 0 already, no slicing happens."""
    out = _clean_json_body('{"a": 1}\nbye')
    assert out.startswith("{")


def test_clean_handles_empty_input():
    assert _clean_json_body("") == ""


def test_clean_handles_whitespace_only_input():
    assert _clean_json_body("   \n\n  ") == ""
