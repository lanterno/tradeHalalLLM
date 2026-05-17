"""Tests for the agent's `_flatten` history-rendering helper.

Used as the fallback when an LLM provider doesn't support multi-turn
tool history natively. Compresses the agent's conversation into a
single user-facing blob the model can consume.
"""

from __future__ import annotations

from halal_trader.core.llm.agent import _flatten


def test_empty_history_returns_empty_string():
    assert _flatten([]) == ""


def test_single_user_message():
    out = _flatten([{"role": "user", "content": "hi"}])
    assert out == "hi"


def test_assistant_role_treated_like_other_non_tool_roles():
    """Anything that isn't a `tool_result` falls through to the
    plain `content` rendering."""
    out = _flatten([{"role": "assistant", "content": "some plan"}])
    assert "some plan" in out


def test_default_role_is_user_when_missing():
    """Defensive: a stripped-down message with only `content` still
    renders rather than crashing on KeyError."""
    out = _flatten([{"content": "no role"}])
    assert "no role" in out


def test_tool_result_renders_with_tool_marker():
    """`tool_result` rows get a `[tool NAME result]` prefix so the
    flattened blob preserves the tool/answer association."""
    out = _flatten([{"role": "tool_result", "tool": "fetch_news", "content": "Bullish news"}])
    assert "[tool fetch_news result]" in out
    assert "Bullish news" in out


def test_tool_result_missing_tool_uses_question_mark():
    """A defensive `?` keeps the marker shape stable when the tool
    name is missing — better than crashing."""
    out = _flatten([{"role": "tool_result", "content": "hello"}])
    assert "[tool ? result]" in out


def test_tool_result_missing_content_renders_empty():
    out = _flatten([{"role": "tool_result", "tool": "x"}])
    assert "[tool x result]" in out


def test_messages_joined_with_blank_lines():
    """Stable separator so the model sees distinct turns."""
    out = _flatten(
        [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
    )
    assert "first\n\nsecond" == out


def test_mixed_user_and_tool_result_round_trip():
    out = _flatten(
        [
            {"role": "user", "content": "what's the news?"},
            {"role": "tool_result", "tool": "fetch_news", "content": "ETF approved"},
            {"role": "user", "content": "any sells?"},
        ]
    )
    assert "what's the news?" in out
    assert "[tool fetch_news result]" in out
    assert "ETF approved" in out
    assert "any sells?" in out


def test_non_string_content_coerced_to_str():
    """Defensive: the content field might be an int / dict — `str(...)`
    keeps the flatten from raising."""
    out = _flatten([{"role": "user", "content": 42}])
    assert "42" in out
