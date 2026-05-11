"""Strategy publishing gate — Round-5 Wave 21.A.

Before a strategy is listed in the marketplace, the operator must
verify it has earned the right to be public. This module is the
**pre-publish verification gate**:

1. **Backtest record** — at least 252 bars (≈1 trading year) with
   computed Sharpe ≥ `min_sharpe` and max-drawdown ≤ `max_drawdown`.
2. **Live-paper attribution** — at least `min_paper_days` (default 90)
   of live-paper trading where the strategy executed at least
   `min_paper_trades` real fills.
3. **Halal-screen** — the strategy's universe must pass a caller-
   supplied halal-screen predicate. The gate doesn't itself maintain
   the screen list.
4. **Author identity** — the platform's KYC-passed author registry
   must include the author_id.

The gate returns a `GateVerdict` — APPROVED / REJECTED / PROVISIONAL.
PROVISIONAL is used when ≥ 1 check is borderline; ops/operator decides.

Pinned semantics:

- **Closed-set GateVerdict** ladder.
- **Closed-set GateFailure** ladder enumerates every reason a strategy
  can be rejected.
- **Min thresholds are operator-tunable** but defaults are
  conservative — better to reject too many than to admit a weak
  strategy.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import Enum


class GateVerdict(str, Enum):
    """Closed-set verdict ladder."""

    APPROVED = "approved"
    PROVISIONAL = "provisional"
    REJECTED = "rejected"


class GateFailure(str, Enum):
    """Closed-set failure-reason ladder."""

    INSUFFICIENT_BACKTEST = "insufficient_backtest"
    LOW_SHARPE = "low_sharpe"
    HIGH_DRAWDOWN = "high_drawdown"
    INSUFFICIENT_PAPER_DAYS = "insufficient_paper_days"
    INSUFFICIENT_PAPER_TRADES = "insufficient_paper_trades"
    AUTHOR_NOT_REGISTERED = "author_not_registered"
    UNIVERSE_HARAM = "universe_haram"
    PAPER_DRAWDOWN_HIGH = "paper_drawdown_high"
    PAPER_SHARPE_LOW = "paper_sharpe_low"


@dataclass(frozen=True)
class BacktestRecord:
    """Summarised backtest stats."""

    n_bars: int
    sharpe: float
    max_drawdown_pct: float
    """Non-negative; magnitude of peak-to-trough decline."""
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if self.n_bars < 0:
            raise ValueError("n_bars must be ≥ 0")
        if self.max_drawdown_pct < 0:
            raise ValueError("max_drawdown_pct must be non-negative")
        if self.max_drawdown_pct > 1.0:
            raise ValueError("max_drawdown_pct must be in [0, 1]")
        if self.end_date < self.start_date:
            raise ValueError("end_date must be ≥ start_date")
        if not -5.0 <= self.sharpe <= 10.0:
            raise ValueError("sharpe outside reasonable bounds")


@dataclass(frozen=True)
class PaperRecord:
    """Live-paper attribution record."""

    started_on: date
    """When the strategy went live on paper."""
    last_active_on: date
    n_trades: int
    sharpe: float
    max_drawdown_pct: float

    def __post_init__(self) -> None:
        if self.last_active_on < self.started_on:
            raise ValueError("last_active_on must be ≥ started_on")
        if self.n_trades < 0:
            raise ValueError("n_trades must be ≥ 0")
        if not 0.0 <= self.max_drawdown_pct <= 1.0:
            raise ValueError("max_drawdown_pct must be in [0, 1]")
        if not -5.0 <= self.sharpe <= 10.0:
            raise ValueError("sharpe outside reasonable bounds")

    def days_live(self) -> int:
        return (self.last_active_on - self.started_on).days


@dataclass(frozen=True)
class GatePolicy:
    """Operator-tunable thresholds."""

    min_backtest_bars: int = 252
    min_backtest_sharpe: float = 0.50
    max_backtest_drawdown: float = 0.30
    """Higher than this → REJECTED."""
    min_paper_days: int = 90
    min_paper_trades: int = 20
    min_paper_sharpe: float = 0.30
    max_paper_drawdown: float = 0.25
    """Provisional thresholds — within `provisional_band` of the limits."""
    provisional_band: float = 0.10
    """Fraction of the cutoff; if metric is within band, PROVISIONAL."""

    def __post_init__(self) -> None:
        if self.min_backtest_bars <= 0:
            raise ValueError("min_backtest_bars must be positive")
        if self.min_paper_days <= 0:
            raise ValueError("min_paper_days must be positive")
        if self.min_paper_trades <= 0:
            raise ValueError("min_paper_trades must be positive")
        if not 0.0 < self.max_backtest_drawdown <= 1.0:
            raise ValueError("max_backtest_drawdown must be in (0, 1]")
        if not 0.0 < self.max_paper_drawdown <= 1.0:
            raise ValueError("max_paper_drawdown must be in (0, 1]")
        if not 0.0 < self.provisional_band < 1.0:
            raise ValueError("provisional_band must be in (0, 1)")


@dataclass(frozen=True)
class StrategyApplication:
    """A submission for publishing."""

    application_id: str
    strategy_id: str
    author_id: str
    universe_tickers: tuple[str, ...]
    backtest: BacktestRecord
    paper: PaperRecord
    submitted_at: date

    def __post_init__(self) -> None:
        if not self.application_id or not self.application_id.strip():
            raise ValueError("application_id must be non-empty")
        if not self.strategy_id or not self.strategy_id.strip():
            raise ValueError("strategy_id must be non-empty")
        if not self.author_id or not self.author_id.strip():
            raise ValueError("author_id must be non-empty")
        if not self.universe_tickers:
            raise ValueError("universe_tickers must be non-empty")
        for t in self.universe_tickers:
            if not t or not t.strip():
                raise ValueError("ticker entries must be non-empty")


@dataclass(frozen=True)
class GateResult:
    """Output of `evaluate`."""

    application_id: str
    verdict: GateVerdict
    failures: tuple[GateFailure, ...]
    provisional_reasons: tuple[GateFailure, ...]
    notes: tuple[str, ...]


def evaluate(
    application: StrategyApplication,
    *,
    is_author_registered: Callable[[str], bool],
    is_ticker_halal: Callable[[str], bool],
    policy: GatePolicy | None = None,
) -> GateResult:
    """Evaluate the application against all gate criteria."""
    pol = policy if policy is not None else GatePolicy()
    failures: list[GateFailure] = []
    provisional: list[GateFailure] = []
    notes: list[str] = []

    # Author identity.
    if not is_author_registered(application.author_id):
        failures.append(GateFailure.AUTHOR_NOT_REGISTERED)
        notes.append("author_id not in platform's KYC-passed registry")

    # Universe halal-screen.
    haram_tickers = [t for t in application.universe_tickers if not is_ticker_halal(t)]
    if haram_tickers:
        failures.append(GateFailure.UNIVERSE_HARAM)
        notes.append(f"{len(haram_tickers)} ticker(s) failed halal screen")

    # Backtest length.
    bt = application.backtest
    if bt.n_bars < pol.min_backtest_bars:
        failures.append(GateFailure.INSUFFICIENT_BACKTEST)
        notes.append(f"backtest {bt.n_bars} bars < min {pol.min_backtest_bars}")

    # Backtest Sharpe — hard floor vs provisional band.
    sharpe_band = pol.min_backtest_sharpe * (1 - pol.provisional_band)
    if bt.sharpe < sharpe_band:
        failures.append(GateFailure.LOW_SHARPE)
        notes.append(f"backtest Sharpe {bt.sharpe:.2f} below provisional band ({sharpe_band:.2f})")
    elif bt.sharpe < pol.min_backtest_sharpe:
        provisional.append(GateFailure.LOW_SHARPE)
        notes.append(
            f"backtest Sharpe {bt.sharpe:.2f} in provisional band "
            f"[{sharpe_band:.2f}, {pol.min_backtest_sharpe:.2f})"
        )

    # Backtest drawdown.
    dd_band = pol.max_backtest_drawdown * (1 + pol.provisional_band)
    if bt.max_drawdown_pct > dd_band:
        failures.append(GateFailure.HIGH_DRAWDOWN)
        notes.append(f"backtest DD {bt.max_drawdown_pct * 100:.2f}% > band {dd_band * 100:.2f}%")
    elif bt.max_drawdown_pct > pol.max_backtest_drawdown:
        provisional.append(GateFailure.HIGH_DRAWDOWN)
        notes.append(
            f"backtest DD {bt.max_drawdown_pct * 100:.2f}% in provisional "
            f"band ({pol.max_backtest_drawdown * 100:.2f}, "
            f"{dd_band * 100:.2f}]"
        )

    # Paper days.
    pp = application.paper
    if pp.days_live() < pol.min_paper_days:
        failures.append(GateFailure.INSUFFICIENT_PAPER_DAYS)
        notes.append(f"paper {pp.days_live()}d < min {pol.min_paper_days}")

    # Paper trades.
    if pp.n_trades < pol.min_paper_trades:
        failures.append(GateFailure.INSUFFICIENT_PAPER_TRADES)
        notes.append(f"paper {pp.n_trades} trades < min {pol.min_paper_trades}")

    # Paper Sharpe / DD.
    p_sharpe_band = pol.min_paper_sharpe * (1 - pol.provisional_band)
    if pp.sharpe < p_sharpe_band:
        failures.append(GateFailure.PAPER_SHARPE_LOW)
        notes.append(f"paper Sharpe {pp.sharpe:.2f} below band ({p_sharpe_band:.2f})")
    elif pp.sharpe < pol.min_paper_sharpe:
        provisional.append(GateFailure.PAPER_SHARPE_LOW)
        notes.append(f"paper Sharpe {pp.sharpe:.2f} in provisional band")

    p_dd_band = pol.max_paper_drawdown * (1 + pol.provisional_band)
    if pp.max_drawdown_pct > p_dd_band:
        failures.append(GateFailure.PAPER_DRAWDOWN_HIGH)
        notes.append(f"paper DD {pp.max_drawdown_pct * 100:.2f}% > band {p_dd_band * 100:.2f}%")
    elif pp.max_drawdown_pct > pol.max_paper_drawdown:
        provisional.append(GateFailure.PAPER_DRAWDOWN_HIGH)
        notes.append(f"paper DD {pp.max_drawdown_pct * 100:.2f}% in provisional band")

    if failures:
        verdict = GateVerdict.REJECTED
    elif provisional:
        verdict = GateVerdict.PROVISIONAL
    else:
        verdict = GateVerdict.APPROVED
    return GateResult(
        application_id=application.application_id,
        verdict=verdict,
        failures=tuple(failures),
        provisional_reasons=tuple(provisional),
        notes=tuple(notes),
    )


_VERDICT_EMOJI: dict[GateVerdict, str] = {
    GateVerdict.APPROVED: "✅",
    GateVerdict.PROVISIONAL: "🟡",
    GateVerdict.REJECTED: "❌",
}


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_result(result: GateResult) -> str:
    head = (
        f"{_VERDICT_EMOJI[result.verdict]} {result.application_id}: {result.verdict.value.upper()}"
    )
    if result.failures:
        head += f" ({len(result.failures)} fail)"
    if result.provisional_reasons:
        head += f" ({len(result.provisional_reasons)} provisional)"
    lines = [head]
    for note in result.notes:
        lines.append(f"  • {note}")
    return "\n".join(lines)
