"""Tests for the halal compliance explainer (Wave L)."""

from __future__ import annotations

from halal_trader.halal.explainer import explain_screening


def _receipt(decision: str, criteria: dict, source: str = "coingecko_rules") -> dict:
    return {
        "trade": {"symbol": "BTCUSDT"},
        "screening": {
            "decision": decision,
            "source": source,
            "criteria": criteria,
            "cache_hit": False,
        },
    }


def test_explain_pass_includes_decision_and_category() -> None:
    receipt = _receipt(
        "halal",
        {"category": "layer-1", "market_cap": 800_000_000_000, "interest_bearing": False},
    )
    out = explain_screening(receipt)
    assert out.decision == "halal"
    assert "HALAL" in out.body_md
    assert "layer-1" in out.body_md
    assert "$800,000,000,000" in out.body_md
    assert any("section-1" in s for s in out.sources)


def test_explain_fail_lists_failures() -> None:
    receipt = _receipt(
        "not_halal",
        {"failures": ["interest-bearing yield", "non-halal sector"]},
    )
    out = explain_screening(receipt)
    assert out.decision == "not_halal"
    assert "❌ interest-bearing yield" in out.body_md
    assert any("section-3" in s for s in out.sources)


def test_explain_doubtful_points_to_exception_queue() -> None:
    receipt = _receipt("doubtful", {"reason": "newly-listed token, no Sharia ruling"})
    out = explain_screening(receipt)
    assert out.decision == "doubtful"
    assert "exception queue" in out.body_md.lower()


def test_explain_handles_minimal_criteria() -> None:
    """Even an empty criteria dict produces a stable explanation."""
    receipt = _receipt("halal", {})
    out = explain_screening(receipt)
    assert "HALAL" in out.body_md
    assert "BTCUSDT" in out.body_md


def test_explain_includes_notes_as_blockquote() -> None:
    receipt = _receipt("halal", {"notes": "Approved by scholar council 2026-04-01."})
    out = explain_screening(receipt)
    assert "> Approved by scholar council" in out.body_md
