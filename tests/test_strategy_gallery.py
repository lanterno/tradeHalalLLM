"""Tests for the public strategy gallery curation engine."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.web.strategy_gallery import (
    GalleryViolationError,
    HalalStrictnessLevel,
    PublicMetrics,
    StrategyEntry,
    StrategyVisibility,
    assemble_lineage,
    compute_simplicity_score,
    hash_author,
    render_entry,
    validate_for_publication,
)

_SALT = b"abcdef0123456789-test-salt-32-bytes"


def _metrics(
    *,
    sharpe_ratio: float = 1.5,
    win_rate_pct: float = 55.0,
    max_drawdown_pct: float = 12.0,
    total_trades: int = 100,
    time_period_days: int = 180,
) -> PublicMetrics:
    return PublicMetrics(
        sharpe_ratio=sharpe_ratio,
        win_rate_pct=win_rate_pct,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=total_trades,
        time_period_days=time_period_days,
    )


def _entry(
    *,
    strategy_id: str = "strat-001",
    anonymous_author: str = "anon-aaaa1111bbbb2222",
    name: str = "Halal Crypto Momentum",
    version: int = 1,
    summary: str = "Momentum on halal crypto pairs.",
    halal_strictness: HalalStrictnessLevel = HalalStrictnessLevel.MODERATE,
    simplicity_score: float = 75.0,
    visibility: StrategyVisibility = StrategyVisibility.PRIVATE,
    metrics: PublicMetrics | None = None,
    parent_fork_id: str | None = None,
    opt_in_publication: bool = False,
) -> StrategyEntry:
    return StrategyEntry(
        strategy_id=strategy_id,
        anonymous_author=anonymous_author,
        name=name,
        version=version,
        summary=summary,
        halal_strictness=halal_strictness,
        simplicity_score=simplicity_score,
        visibility=visibility,
        metrics=metrics,
        parent_fork_id=parent_fork_id,
        opt_in_publication=opt_in_publication,
    )


# ---------------------------------------------------------------------------
# PublicMetrics validation
# ---------------------------------------------------------------------------


def test_metrics_rejects_win_rate_above_100() -> None:
    with pytest.raises(ValueError, match="win_rate_pct"):
        _metrics(win_rate_pct=101.0)


def test_metrics_rejects_negative_drawdown() -> None:
    with pytest.raises(ValueError, match="max_drawdown_pct"):
        _metrics(max_drawdown_pct=-1.0)


def test_metrics_rejects_negative_total_trades() -> None:
    with pytest.raises(ValueError, match="total_trades"):
        _metrics(total_trades=-1)


def test_metrics_rejects_zero_period() -> None:
    with pytest.raises(ValueError, match="time_period_days"):
        _metrics(time_period_days=0)


def test_metrics_accepts_valid_values() -> None:
    m = _metrics()
    assert m.sharpe_ratio == 1.5
    assert m.win_rate_pct == 55.0


# ---------------------------------------------------------------------------
# StrategyEntry validation
# ---------------------------------------------------------------------------


def test_entry_rejects_empty_strategy_id() -> None:
    with pytest.raises(ValueError, match="strategy_id"):
        _entry(strategy_id="")


def test_entry_rejects_empty_author() -> None:
    with pytest.raises(ValueError, match="anonymous_author"):
        _entry(anonymous_author="")


def test_entry_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _entry(name="")


def test_entry_rejects_zero_version() -> None:
    with pytest.raises(ValueError, match="version"):
        _entry(version=0)


def test_entry_rejects_simplicity_above_100() -> None:
    with pytest.raises(ValueError, match="simplicity_score"):
        _entry(simplicity_score=101.0)


def test_entry_rejects_negative_simplicity() -> None:
    with pytest.raises(ValueError, match="simplicity_score"):
        _entry(simplicity_score=-1.0)


def test_entry_rejects_empty_parent_fork_id_when_set() -> None:
    with pytest.raises(ValueError, match="parent_fork_id"):
        _entry(parent_fork_id="")


def test_entry_accepts_none_parent_fork_id() -> None:
    e = _entry(parent_fork_id=None)
    assert e.parent_fork_id is None


# ---------------------------------------------------------------------------
# Default visibility
# ---------------------------------------------------------------------------


def test_default_visibility_is_private() -> None:
    """Pin: PRIVATE is the default — no implicit publication."""

    e = _entry()
    assert e.visibility is StrategyVisibility.PRIVATE


def test_default_opt_in_is_false() -> None:
    """Pin: opt_in_publication defaults False — explicit opt-in only."""

    e = _entry()
    assert e.opt_in_publication is False


# ---------------------------------------------------------------------------
# hash_author — anonymous-author-token
# ---------------------------------------------------------------------------


def test_hash_author_starts_with_anon_prefix() -> None:
    token = hash_author("alice", salt=_SALT)
    assert token.startswith("anon-")


def test_hash_author_contains_no_original_id() -> None:
    """Pin: hashed token doesn't contain the original user_id."""

    token = hash_author("alice", salt=_SALT)
    assert "alice" not in token


def test_hash_author_deterministic_within_salt() -> None:
    a = hash_author("alice", salt=_SALT)
    b = hash_author("alice", salt=_SALT)
    assert a == b


def test_hash_author_different_users_different_tokens() -> None:
    a = hash_author("alice", salt=_SALT)
    b = hash_author("bob", salt=_SALT)
    assert a != b


def test_hash_author_different_salts_different_tokens() -> None:
    """Pin: different salt → different token (re-publication anti-linking)."""

    salt_a = b"a" * 16
    salt_b = b"b" * 16
    a = hash_author("alice", salt=salt_a)
    b = hash_author("alice", salt=salt_b)
    assert a != b


def test_hash_author_rejects_short_salt() -> None:
    with pytest.raises(ValueError, match="salt"):
        hash_author("alice", salt=b"short")


def test_hash_author_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        hash_author("", salt=_SALT)


# ---------------------------------------------------------------------------
# Simplicity score
# ---------------------------------------------------------------------------


def test_simplicity_score_clean_strategy() -> None:
    """Pin: 50 LOC + 5 symbols + depth 1 → score = 100 (boundary)."""

    score = compute_simplicity_score(lines_of_code=50, symbol_list_size=5, reasoning_depth=1)
    assert score == 100.0


def test_simplicity_score_typical_strategy() -> None:
    """100 LOC + 5 symbols + depth 2 → ~92.5 (clean but not minimal)."""

    score = compute_simplicity_score(lines_of_code=100, symbol_list_size=5, reasoning_depth=2)
    assert 85 < score < 95


def test_simplicity_score_complex_strategy() -> None:
    """500 LOC + 50 symbols + depth 5 → ~32 (complex)."""

    score = compute_simplicity_score(lines_of_code=500, symbol_list_size=50, reasoning_depth=5)
    assert 25 < score < 40


def test_simplicity_score_very_complex_strategy() -> None:
    """1000+ LOC + 100+ symbols + depth 10+ → 0 (very complex)."""

    score = compute_simplicity_score(lines_of_code=2000, symbol_list_size=200, reasoning_depth=20)
    assert score == 0.0


def test_simplicity_score_clamped_at_zero() -> None:
    score = compute_simplicity_score(
        lines_of_code=10_000, symbol_list_size=1000, reasoning_depth=100
    )
    assert score == 0.0


def test_simplicity_score_clamped_at_100() -> None:
    """Pin: very small LOC / symbols / depth doesn't push score above 100."""

    score = compute_simplicity_score(lines_of_code=10, symbol_list_size=1, reasoning_depth=1)
    assert score == 100.0


def test_simplicity_score_rejects_negative_loc() -> None:
    with pytest.raises(ValueError, match="lines_of_code"):
        compute_simplicity_score(lines_of_code=-1, symbol_list_size=5, reasoning_depth=1)


def test_simplicity_score_rejects_negative_symbols() -> None:
    with pytest.raises(ValueError, match="symbol_list_size"):
        compute_simplicity_score(lines_of_code=100, symbol_list_size=-1, reasoning_depth=1)


def test_simplicity_score_rejects_negative_depth() -> None:
    with pytest.raises(ValueError, match="reasoning_depth"):
        compute_simplicity_score(lines_of_code=100, symbol_list_size=5, reasoning_depth=-1)


# ---------------------------------------------------------------------------
# validate_for_publication — gates
# ---------------------------------------------------------------------------


def test_private_entry_passes_silently() -> None:
    """Pin: PRIVATE entries skip publication checks."""

    e = _entry(visibility=StrategyVisibility.PRIVATE, opt_in_publication=False)
    # No exception
    validate_for_publication(e)


def test_public_listed_without_opt_in_raises() -> None:
    """Pin: explicit opt-in required for public listing."""

    e = _entry(
        visibility=StrategyVisibility.PUBLIC_LISTED,
        opt_in_publication=False,
        metrics=_metrics(),
    )
    with pytest.raises(GalleryViolationError, match="opt_in_publication"):
        validate_for_publication(e)


def test_public_unlisted_without_opt_in_raises() -> None:
    e = _entry(
        visibility=StrategyVisibility.PUBLIC_UNLISTED,
        opt_in_publication=False,
    )
    with pytest.raises(GalleryViolationError, match="opt_in_publication"):
        validate_for_publication(e)


def test_public_with_opt_in_passes() -> None:
    e = _entry(
        visibility=StrategyVisibility.PUBLIC_LISTED,
        opt_in_publication=True,
        metrics=_metrics(),
    )
    # No exception
    validate_for_publication(e)


def test_summary_with_email_pii_raises() -> None:
    """Pin: PII in summary blocks publication."""

    e = _entry(
        visibility=StrategyVisibility.PUBLIC_LISTED,
        opt_in_publication=True,
        metrics=_metrics(),
        summary="Contact alice@example.com for forks.",
    )
    with pytest.raises(GalleryViolationError, match="PII-shaped"):
        validate_for_publication(e)


def test_summary_with_ssn_pii_raises() -> None:
    e = _entry(
        visibility=StrategyVisibility.PUBLIC_LISTED,
        opt_in_publication=True,
        metrics=_metrics(),
        summary="SSN 123-45-6789 referenced.",
    )
    with pytest.raises(GalleryViolationError, match="PII-shaped"):
        validate_for_publication(e)


def test_summary_with_eth_address_pii_raises() -> None:
    e = _entry(
        visibility=StrategyVisibility.PUBLIC_LISTED,
        opt_in_publication=True,
        metrics=_metrics(),
        summary=("Address 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb6 in test."),
    )
    with pytest.raises(GalleryViolationError, match="PII-shaped"):
        validate_for_publication(e)


def test_name_with_pii_raises() -> None:
    """Pin: PII in name also blocks (operators sometimes embed personal info)."""

    e = _entry(
        visibility=StrategyVisibility.PUBLIC_LISTED,
        opt_in_publication=True,
        metrics=_metrics(),
        name="Strategy by alice@example.com",
    )
    with pytest.raises(GalleryViolationError, match="PII-shaped"):
        validate_for_publication(e)


def test_clean_summary_passes() -> None:
    e = _entry(
        visibility=StrategyVisibility.PUBLIC_LISTED,
        opt_in_publication=True,
        metrics=_metrics(),
        summary="Standard momentum on top-50 halal-screened pairs.",
    )
    # No exception
    validate_for_publication(e)


def test_public_listed_without_metrics_raises() -> None:
    """Pin: PUBLIC_LISTED requires metrics for sortability."""

    e = _entry(
        visibility=StrategyVisibility.PUBLIC_LISTED,
        opt_in_publication=True,
        metrics=None,
    )
    with pytest.raises(GalleryViolationError, match="metrics"):
        validate_for_publication(e)


def test_public_unlisted_without_metrics_passes() -> None:
    """Pin: PUBLIC_UNLISTED doesn't require metrics (unlisted = direct-URL share)."""

    e = _entry(
        visibility=StrategyVisibility.PUBLIC_UNLISTED,
        opt_in_publication=True,
        metrics=None,
    )
    # No exception
    validate_for_publication(e)


# ---------------------------------------------------------------------------
# assemble_lineage — fork chain
# ---------------------------------------------------------------------------


def test_lineage_root_strategy_is_just_self() -> None:
    e = _entry(strategy_id="root", parent_fork_id=None)
    chain = assemble_lineage((e,), target_id="root")
    assert chain == (e,)


def test_lineage_two_level_fork() -> None:
    root = _entry(strategy_id="root", parent_fork_id=None)
    fork = _entry(strategy_id="fork", parent_fork_id="root", name="Fork v1")
    chain = assemble_lineage((root, fork), target_id="fork")
    assert chain == (root, fork)


def test_lineage_multi_level_fork() -> None:
    root = _entry(strategy_id="root", parent_fork_id=None)
    a = _entry(strategy_id="a", parent_fork_id="root", name="A")
    b = _entry(strategy_id="b", parent_fork_id="a", name="B")
    chain = assemble_lineage((root, a, b), target_id="b")
    assert chain == (root, a, b)


def test_lineage_target_not_found_raises() -> None:
    e = _entry(strategy_id="root")
    with pytest.raises(KeyError):
        assemble_lineage((e,), target_id="missing")


def test_lineage_detached_parent_stops_walk() -> None:
    """Pin: orphaned fork (parent_fork_id missing from entries) stops at the orphan."""

    fork = _entry(strategy_id="orphan-fork", parent_fork_id="missing-parent")
    chain = assemble_lineage((fork,), target_id="orphan-fork")
    # walk stops at the fork itself; chain has just one entry
    assert chain == (fork,)


def test_lineage_cycle_raises() -> None:
    """Pin: cycle detected → GalleryViolationError, not infinite loop."""

    a = _entry(strategy_id="a", parent_fork_id="b")
    b = _entry(strategy_id="b", parent_fork_id="a")
    with pytest.raises(GalleryViolationError, match="cycle"):
        assemble_lineage((a, b), target_id="a")


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_metrics_is_frozen() -> None:
    m = _metrics()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.sharpe_ratio = 99.0  # type: ignore[misc]


def test_entry_is_frozen() -> None:
    e = _entry()
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.visibility = StrategyVisibility.PUBLIC_LISTED  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_visibility_string_values() -> None:
    assert StrategyVisibility.PRIVATE.value == "private"
    assert StrategyVisibility.PUBLIC_UNLISTED.value == "public_unlisted"
    assert StrategyVisibility.PUBLIC_LISTED.value == "public_listed"


def test_halal_strictness_string_values() -> None:
    assert HalalStrictnessLevel.BASIC.value == "basic"
    assert HalalStrictnessLevel.MODERATE.value == "moderate"
    assert HalalStrictnessLevel.STRICT.value == "strict"
    assert HalalStrictnessLevel.MAX_STRICT.value == "max_strict"


# ---------------------------------------------------------------------------
# Render output — pinned no-PII contract
# ---------------------------------------------------------------------------


def test_render_includes_anonymous_author() -> None:
    e = _entry(anonymous_author="anon-1234567890abcdef")
    text = render_entry(e)
    assert "anon-1234567890abcdef" in text


def test_render_visibility_emoji() -> None:
    private = render_entry(_entry(visibility=StrategyVisibility.PRIVATE))
    listed = render_entry(_entry(visibility=StrategyVisibility.PUBLIC_LISTED))
    unlisted = render_entry(_entry(visibility=StrategyVisibility.PUBLIC_UNLISTED))
    assert "🔒" in private
    assert "🌍" in listed
    assert "🔗" in unlisted


def test_render_strictness_emoji() -> None:
    basic = render_entry(_entry(halal_strictness=HalalStrictnessLevel.BASIC))
    strict = render_entry(_entry(halal_strictness=HalalStrictnessLevel.STRICT))
    max_strict = render_entry(_entry(halal_strictness=HalalStrictnessLevel.MAX_STRICT))
    assert "🟢" in basic
    assert "🟠" in strict
    assert "🔴" in max_strict


def test_render_includes_metrics_when_present() -> None:
    e = _entry(metrics=_metrics(sharpe_ratio=1.5, win_rate_pct=55.0))
    text = render_entry(e)
    assert "Sharpe 1.50" in text
    assert "win 55.0%" in text


def test_render_omits_metrics_when_absent() -> None:
    e = _entry(metrics=None)
    text = render_entry(e)
    assert "performance" not in text


def test_render_includes_parent_fork_when_set() -> None:
    e = _entry(parent_fork_id="parent-strategy-id")
    text = render_entry(e)
    assert "forked from" in text
    assert "parent-strategy-id" in text


def test_render_omits_parent_fork_when_absent() -> None:
    e = _entry(parent_fork_id=None)
    text = render_entry(e)
    assert "forked from" not in text


def test_render_does_not_leak_user_id() -> None:
    """Pin: render never includes operator's raw user_id."""

    # The engine doesn't have access to raw user_id (it works on the
    # already-anonymised entry), but we verify there's no path for
    # one to appear.
    e = _entry(
        anonymous_author="anon-aaaa1111bbbb2222",
        name="Strategy",
    )
    text = render_entry(e)
    # Only the anonymous token should appear, not anything user-shaped
    assert "user_id" not in text
    assert "@" not in text  # no email-shaped strings


def test_render_includes_simplicity_score() -> None:
    e = _entry(simplicity_score=87.5)
    text = render_entry(e)
    assert "87.5/100" in text


# ---------------------------------------------------------------------------
# End-to-end realistic flows
# ---------------------------------------------------------------------------


def test_typical_publication_flow() -> None:
    """Operator creates a strategy, opts in, publishes."""

    author_token = hash_author("alice", salt=_SALT)
    metrics = _metrics(sharpe_ratio=1.8, win_rate_pct=60.0)
    score = compute_simplicity_score(lines_of_code=120, symbol_list_size=8, reasoning_depth=2)
    entry = StrategyEntry(
        strategy_id="strat-001",
        anonymous_author=author_token,
        name="Halal Crypto Trend",
        version=1,
        summary="Trend following on top-30 halal-screened crypto pairs.",
        halal_strictness=HalalStrictnessLevel.MODERATE,
        simplicity_score=score,
        visibility=StrategyVisibility.PUBLIC_LISTED,
        metrics=metrics,
        opt_in_publication=True,
    )
    # No exception
    validate_for_publication(entry)


def test_full_fork_lineage_render() -> None:
    """Walk a 3-deep fork chain and verify the render."""

    root = _entry(strategy_id="root", name="Root")
    fork1 = _entry(strategy_id="f1", parent_fork_id="root", name="Fork 1")
    fork2 = _entry(strategy_id="f2", parent_fork_id="f1", name="Fork 2")
    chain = assemble_lineage((root, fork1, fork2), target_id="f2")
    assert len(chain) == 3
    assert chain[0].strategy_id == "root"
    assert chain[2].strategy_id == "f2"


def test_publication_blocked_for_pii_in_summary_realistic() -> None:
    """Operator accidentally includes their email — publication blocked.

    The operator must rewrite the summary; the engine doesn't auto-
    redact (operators want to know about the leak, not have it
    silently scrubbed).
    """

    author_token = hash_author("alice", salt=_SALT)
    entry = StrategyEntry(
        strategy_id="strat-002",
        anonymous_author=author_token,
        name="Halal Momentum",
        version=1,
        summary="Reach me at alice@example.com for support questions.",
        halal_strictness=HalalStrictnessLevel.STRICT,
        simplicity_score=80.0,
        visibility=StrategyVisibility.PUBLIC_LISTED,
        metrics=_metrics(),
        opt_in_publication=True,
    )
    with pytest.raises(GalleryViolationError, match="PII-shaped"):
        validate_for_publication(entry)


def test_simplicity_ordering_for_gallery_sort() -> None:
    """Pin: simpler strategies sort higher."""

    simple = compute_simplicity_score(lines_of_code=50, symbol_list_size=5, reasoning_depth=1)
    medium = compute_simplicity_score(lines_of_code=200, symbol_list_size=10, reasoning_depth=2)
    complex_score = compute_simplicity_score(
        lines_of_code=600, symbol_list_size=30, reasoning_depth=5
    )
    assert simple > medium > complex_score
