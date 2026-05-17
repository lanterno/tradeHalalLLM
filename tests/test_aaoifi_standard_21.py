"""Tests for halal/aaoifi_standard_21.py — Round-5 Wave 1.A."""

from __future__ import annotations

import pytest

from halal_trader.halal.aaoifi_standard_21 import (
    CLAUSES,
    ClauseCitation,
    CoverageSummary,
    ScreenerRule,
    StandardClause,
    clause_by_id,
    clauses_for_rule,
    coverage_summary,
    render_citations,
    render_clause,
    render_coverage_matrix,
)


def test_screener_rule_string_values():
    assert ScreenerRule.SECTOR_HALAL_ACTIVITY.value == "sector_halal_activity"
    assert ScreenerRule.NO_PROHIBITED_REVENUE.value == "no_prohibited_revenue"
    assert ScreenerRule.REVENUE_PURITY_THRESHOLD.value == "revenue_purity_threshold"
    assert ScreenerRule.DEBT_RATIO_LIMIT.value == "debt_ratio_limit"
    assert ScreenerRule.INTEREST_INCOME_LIMIT.value == "interest_income_limit"
    assert ScreenerRule.LIQUID_ASSETS_RATIO.value == "liquid_assets_ratio"
    assert ScreenerRule.LISTED_ON_RECOGNISED_EXCHANGE.value == "listed_on_recognised_exchange"
    assert ScreenerRule.SHAREHOLDER_LIABILITY_LIMITED.value == "shareholder_liability_limited"
    assert ScreenerRule.NO_PREFERRED_SHARES.value == "no_preferred_shares"
    assert ScreenerRule.NO_MARGIN_TRADING.value == "no_margin_trading"
    assert ScreenerRule.NO_SHORT_SELLING.value == "no_short_selling"
    assert ScreenerRule.DELIVERY_VERSUS_PAYMENT.value == "delivery_versus_payment"
    assert ScreenerRule.PURIFICATION_REQUIRED.value == "purification_required"
    assert ScreenerRule.DISCLOSURE_OF_NON_HALAL_INCOME.value == "disclosure_of_non_halal_income"
    assert ScreenerRule.SCHOLAR_REVIEW_FOR_AMBIGUOUS.value == "scholar_review_for_ambiguous"


def test_clauses_non_empty():
    assert len(CLAUSES) > 0


def test_clauses_sorted_by_id():
    keys = [tuple(int(seg) for seg in c.clause_id.split(".")) for c in CLAUSES]
    assert keys == sorted(keys)


def test_clauses_unique_ids():
    ids = [c.clause_id for c in CLAUSES]
    assert len(ids) == len(set(ids))


def test_every_clause_has_rule():
    for clause in CLAUSES:
        assert isinstance(clause.rule, ScreenerRule)


def test_clause_by_id_known():
    clause = clause_by_id("3.2")
    assert clause is not None
    assert clause.rule is ScreenerRule.DEBT_RATIO_LIMIT


def test_clause_by_id_unknown_returns_none():
    assert clause_by_id("99.99") is None


def test_clauses_for_rule_returns_matching():
    matches = clauses_for_rule(ScreenerRule.NO_PROHIBITED_REVENUE)
    assert len(matches) >= 1
    for c in matches:
        assert c.rule is ScreenerRule.NO_PROHIBITED_REVENUE


def test_clauses_for_rule_unused_rule_returns_empty():
    # Pick a rule and see; if every rule is used at least once, this still
    # exercises the empty-fast-path defensively.
    used_rules = {c.rule for c in CLAUSES}
    unused = set(ScreenerRule) - used_rules
    if unused:
        assert clauses_for_rule(next(iter(unused))) == ()


def test_standard_clause_validation_empty_id():
    with pytest.raises(ValueError):
        StandardClause(
            clause_id="",
            title="x",
            rule=ScreenerRule.DEBT_RATIO_LIMIT,
            summary="x",
        )


def test_standard_clause_validation_bad_id_format():
    with pytest.raises(ValueError):
        StandardClause(
            clause_id="3-2-1",
            title="x",
            rule=ScreenerRule.DEBT_RATIO_LIMIT,
            summary="x",
        )


def test_standard_clause_validation_empty_title():
    with pytest.raises(ValueError):
        StandardClause(
            clause_id="9.9",
            title="   ",
            rule=ScreenerRule.DEBT_RATIO_LIMIT,
            summary="x",
        )


def test_standard_clause_validation_empty_summary():
    with pytest.raises(ValueError):
        StandardClause(
            clause_id="9.9",
            title="x",
            rule=ScreenerRule.DEBT_RATIO_LIMIT,
            summary="",
        )


def test_standard_clause_validation_empty_sample_test():
    with pytest.raises(ValueError):
        StandardClause(
            clause_id="9.9",
            title="x",
            rule=ScreenerRule.DEBT_RATIO_LIMIT,
            summary="x",
            sample_test="   ",
        )


def test_standard_clause_immutable():
    c = CLAUSES[0]
    with pytest.raises(AttributeError):
        c.title = "changed"  # type: ignore[misc]


def test_coverage_summary_full_engagement():
    every_rule = {c.rule for c in CLAUSES}
    summary = coverage_summary(every_rule)
    assert summary.total_clauses == len(CLAUSES)
    assert summary.engaged_clauses == len(CLAUSES)


def test_coverage_summary_empty_engagement():
    summary = coverage_summary(set())
    assert summary.engaged_clauses == 0
    assert summary.total_clauses == len(CLAUSES)


def test_coverage_summary_partial():
    summary = coverage_summary({ScreenerRule.DEBT_RATIO_LIMIT})
    assert summary.engaged_clauses == len(clauses_for_rule(ScreenerRule.DEBT_RATIO_LIMIT))


def test_coverage_summary_rejects_negative():
    with pytest.raises(ValueError):
        CoverageSummary(total_clauses=-1, rules_engaged=frozenset(), engaged_clauses=0)


def test_coverage_summary_rejects_engaged_gt_total():
    with pytest.raises(ValueError):
        CoverageSummary(total_clauses=1, rules_engaged=frozenset(), engaged_clauses=2)


def test_render_clause_format():
    clause = clause_by_id("3.2")
    assert clause is not None
    rendered = render_clause(clause)
    assert "§3.2" in rendered
    assert clause.title in rendered
    assert clause.rule.value in rendered


def test_render_coverage_matrix_default():
    out = render_coverage_matrix()
    assert "AAOIFI Standard 21" in out
    assert "engaged" in out
    # First clause should appear
    assert f"§{CLAUSES[0].clause_id}" in out


def test_render_coverage_matrix_with_full_summary_marks_all():
    every_rule = frozenset(c.rule for c in CLAUSES)
    summary = CoverageSummary(
        total_clauses=len(CLAUSES),
        rules_engaged=every_rule,
        engaged_clauses=len(CLAUSES),
    )
    out = render_coverage_matrix(summary)
    # No unmarked rows (no two-space prefix line)
    for line in out.splitlines()[2:]:
        if line.strip():
            assert line.startswith("✅")


def test_render_no_secret_leak():
    """Render output must never include forbidden tokens."""
    out = render_coverage_matrix()
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


def test_clause_citation_known_clause():
    cit = ClauseCitation(
        clause_id="3.2",
        rule=ScreenerRule.DEBT_RATIO_LIMIT,
        pass_fail=True,
        note="ratio=12%",
    )
    assert cit.pass_fail


def test_clause_citation_unknown_clause_rejected():
    with pytest.raises(ValueError):
        ClauseCitation(
            clause_id="99.99",
            rule=ScreenerRule.DEBT_RATIO_LIMIT,
            pass_fail=True,
        )


def test_clause_citation_rule_mismatch_rejected():
    with pytest.raises(ValueError):
        ClauseCitation(
            clause_id="3.2",
            rule=ScreenerRule.SECTOR_HALAL_ACTIVITY,  # 3.2 maps to debt_ratio_limit
            pass_fail=True,
        )


def test_render_citations_pass_and_fail():
    cits = [
        ClauseCitation(clause_id="3.2", rule=ScreenerRule.DEBT_RATIO_LIMIT, pass_fail=True),
        ClauseCitation(
            clause_id="3.3", rule=ScreenerRule.INTEREST_INCOME_LIMIT, pass_fail=False, note="6%"
        ),
    ]
    out = render_citations(cits)
    assert "✅" in out
    assert "❌" in out
    assert "§3.2" in out
    assert "§3.3" in out


def test_render_citations_empty():
    assert render_citations([]) == ""


def test_e2e_default_engagement_engages_majority():
    summary = coverage_summary(
        {
            ScreenerRule.SECTOR_HALAL_ACTIVITY,
            ScreenerRule.NO_PROHIBITED_REVENUE,
            ScreenerRule.REVENUE_PURITY_THRESHOLD,
            ScreenerRule.DEBT_RATIO_LIMIT,
            ScreenerRule.INTEREST_INCOME_LIMIT,
            ScreenerRule.LIQUID_ASSETS_RATIO,
            ScreenerRule.PURIFICATION_REQUIRED,
        }
    )
    # Should be the majority of clauses but not all (mechanics + scholar-review still gaps)
    assert summary.engaged_clauses < summary.total_clauses
    assert summary.engaged_clauses >= summary.total_clauses // 2


def test_replay_consistency():
    """Catalogue is import-time frozen so two coverage calls return the same numbers."""
    a = coverage_summary({ScreenerRule.DEBT_RATIO_LIMIT})
    b = coverage_summary({ScreenerRule.DEBT_RATIO_LIMIT})
    assert a == b
