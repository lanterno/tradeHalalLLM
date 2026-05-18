"""Maysir (gambling) screen for high-volatility / fundamentals-poor names.

Round-5 Wave 1.G primitive. Standard halal screening checks debt
ratios + revenue purity + sector; it does NOT catch the case
where a name is technically halal-compliant on paper but trades
like gambling rather than investment. Penny stocks with no
business model, meme stocks driven by Reddit pumps, names with
zero analyst coverage and 200% short interest — these embody
maysir (gambling), which is independently prohibited in Shariah
regardless of the company's debt ratios.

The maysir screen is a structural overlay on top of the standard
screener: a name passes the standard halal screen AND the maysir
screen → tradable; passes standard but fails maysir → flagged
to operator with the specific maysir signals.

Picked a closed-set signal ladder + tunable policy thresholds
because (a) maysir patterns evolve (today's meme dynamics weren't
on anyone's radar five years ago), so the catalogue of signals is
documentation that operators read + scholars reference; (b) the
thresholds (penny <$5, retail-flow >70%, etc.) are operator-tunable
because some operators want stricter screens; (c) the scoring is
rule-based + interpretable — no black-box "trust this score" —
which matches the auditability standard of Wave 11.B SSB
governance.

Pinned semantics:
- **Closed-set MaysirRisk ladder.** NONE < LOW < MODERATE < HIGH
  < EXTREME. HIGH + EXTREME are non-tradable by default.
- **Closed-set MaysirSignal catalogue.** Adding a new signal is
  a code review change — keeps the catalogue documented + audited.
- **Signals are independent boolean detectors.** Each signal has
  a clear tunable threshold; the overall risk is a function of
  how many fire, weighted by severity.
- **Render output never includes the underlying alt-data feed
  details.** Only the signals + risk level + summary; raw Reddit
  posts / Robinhood holding lists go to operator-side debug log.
- **`is_tradable(assessment)` is the load-bearing gate.** Returns
  True only for risk in {NONE, LOW, MODERATE}; HIGH + EXTREME
  block the trade.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class MaysirRisk(str, Enum):
    """Maysir (gambling) risk ladder.

    Pinned string values for JSON / DB persistence stability.
    NONE < LOW < MODERATE < HIGH < EXTREME. HIGH + EXTREME are
    non-tradable by default.
    """

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class MaysirSignal(str, Enum):
    """Catalogue of maysir-pattern detector signals.

    Pinned string values. Adding a new signal is a code review
    change — keeps the catalogue documented.
    """

    PENNY_PRICE = "penny_price"  # Stock price below penny threshold
    EXTREME_SHORT_INTEREST = "extreme_short_interest"  # Squeeze risk
    RETAIL_FLOW_DOMINANT = "retail_flow_dominant"  # >X% retail flow
    NO_ANALYST_COVERAGE = "no_analyst_coverage"  # Zero analyst coverage
    EXTREME_VOLATILITY = "extreme_volatility"  # Realized vol > threshold
    MEME_PUMP_PATTERN = "meme_pump_pattern"  # Recent vertical price move
    ZERO_REVENUE = "zero_revenue"  # Pre-revenue / no business model


# Signal severity weights (NONE = 0, contributes nothing).
# Used to compute overall MaysirRisk from the set of fired signals.
_SIGNAL_WEIGHT: dict[MaysirSignal, int] = {
    MaysirSignal.ZERO_REVENUE: 3,  # Most severe: no business model at all
    MaysirSignal.PENNY_PRICE: 2,
    MaysirSignal.MEME_PUMP_PATTERN: 2,
    MaysirSignal.EXTREME_SHORT_INTEREST: 2,
    MaysirSignal.NO_ANALYST_COVERAGE: 1,
    MaysirSignal.RETAIL_FLOW_DOMINANT: 1,
    MaysirSignal.EXTREME_VOLATILITY: 1,
}


@dataclass(frozen=True)
class MaysirPolicy:
    """Operator-tunable maysir-pattern thresholds.

    Defaults are calibrated against the documented "meme stock"
    pattern (GME / AMC era) — penny <$5, short interest >50%,
    retail-flow >70%, vol >100% annualized, vertical 30d move
    >100%.
    """

    penny_price_threshold: float = 5.0  # USD; price below = penny
    extreme_short_interest_pct: float = 50.0  # >% of float short
    retail_flow_dominant_pct: float = 70.0  # >% of trading volume retail
    no_analyst_coverage_count: int = 0  # ==count flags signal
    extreme_volatility_annualized: float = 1.0  # >100% annualized vol
    meme_pump_30d_pct: float = 100.0  # >100% gain in 30d
    # Risk-level cutoffs (sum of fired signal weights):
    moderate_score_threshold: int = 2  # Score >= 2 = MODERATE
    high_score_threshold: int = 4  # Score >= 4 = HIGH
    extreme_score_threshold: int = 6  # Score >= 6 = EXTREME

    def __post_init__(self) -> None:
        if self.penny_price_threshold <= 0:
            raise ValueError("penny_price_threshold must be > 0")
        if not (0 < self.extreme_short_interest_pct <= 100):
            raise ValueError("extreme_short_interest_pct must be in (0, 100]")
        if not (0 < self.retail_flow_dominant_pct <= 100):
            raise ValueError("retail_flow_dominant_pct must be in (0, 100]")
        if self.no_analyst_coverage_count < 0:
            raise ValueError("no_analyst_coverage_count must be >= 0")
        if self.extreme_volatility_annualized <= 0:
            raise ValueError("extreme_volatility_annualized must be > 0")
        if self.meme_pump_30d_pct <= 0:
            raise ValueError("meme_pump_30d_pct must be > 0")
        if not (
            self.moderate_score_threshold < self.high_score_threshold < self.extreme_score_threshold
        ):
            raise ValueError("score thresholds must satisfy moderate < high < extreme")
        if self.moderate_score_threshold < 1:
            raise ValueError("moderate_score_threshold must be >= 1")


@dataclass(frozen=True)
class MaysirInputs:
    """Per-ticker inputs for the maysir screen.

    Operators provide via the screener pipeline; values come from
    Alpaca / Yahoo / Reddit alt-data adapters.
    """

    ticker: str
    stock_price: float
    short_interest_pct: float
    retail_flow_pct: float
    analyst_coverage_count: int
    realized_volatility_annualized: float
    price_change_30d_pct: float
    revenue_ttm_usd: float

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if self.stock_price <= 0:
            raise ValueError("stock_price must be > 0")
        if not (0 <= self.short_interest_pct <= 1000):
            # Short interest can exceed 100% (synthetic shorts);
            # cap at 1000 for sanity.
            raise ValueError("short_interest_pct must be in [0, 1000]")
        if not (0 <= self.retail_flow_pct <= 100):
            raise ValueError("retail_flow_pct must be in [0, 100]")
        if self.analyst_coverage_count < 0:
            raise ValueError("analyst_coverage_count must be >= 0")
        if self.realized_volatility_annualized < 0:
            raise ValueError("realized_volatility_annualized must be >= 0")
        if self.revenue_ttm_usd < 0:
            raise ValueError("revenue_ttm_usd must be >= 0")


@dataclass(frozen=True)
class MaysirAssessment:
    """Output of the maysir screen for one ticker."""

    ticker: str
    signals: frozenset[MaysirSignal]
    risk: MaysirRisk
    score: int

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if self.score < 0:
            raise ValueError("score must be non-negative")
        # Cross-check: NONE risk should have empty signals; non-NONE
        # should have at least one signal. Pinned via test.
        if self.risk is MaysirRisk.NONE and self.signals:
            raise ValueError("NONE risk must have empty signals")
        if self.risk is not MaysirRisk.NONE and not self.signals:
            raise ValueError("non-NONE risk requires at least one signal")


def _detect_signals(
    inputs: MaysirInputs,
    *,
    policy: MaysirPolicy,
) -> frozenset[MaysirSignal]:
    """Run each signal detector against the inputs."""

    signals: set[MaysirSignal] = set()
    if inputs.stock_price < policy.penny_price_threshold:
        signals.add(MaysirSignal.PENNY_PRICE)
    if inputs.short_interest_pct > policy.extreme_short_interest_pct:
        signals.add(MaysirSignal.EXTREME_SHORT_INTEREST)
    if inputs.retail_flow_pct > policy.retail_flow_dominant_pct:
        signals.add(MaysirSignal.RETAIL_FLOW_DOMINANT)
    if inputs.analyst_coverage_count <= policy.no_analyst_coverage_count:
        signals.add(MaysirSignal.NO_ANALYST_COVERAGE)
    if inputs.realized_volatility_annualized > policy.extreme_volatility_annualized:
        signals.add(MaysirSignal.EXTREME_VOLATILITY)
    if inputs.price_change_30d_pct > policy.meme_pump_30d_pct:
        signals.add(MaysirSignal.MEME_PUMP_PATTERN)
    if inputs.revenue_ttm_usd == 0:
        signals.add(MaysirSignal.ZERO_REVENUE)
    return frozenset(signals)


def _score_to_risk(score: int, *, policy: MaysirPolicy) -> MaysirRisk:
    """Map fired-signal weight sum to MaysirRisk via policy cutoffs."""

    if score == 0:
        return MaysirRisk.NONE
    if score >= policy.extreme_score_threshold:
        return MaysirRisk.EXTREME
    if score >= policy.high_score_threshold:
        return MaysirRisk.HIGH
    if score >= policy.moderate_score_threshold:
        return MaysirRisk.MODERATE
    return MaysirRisk.LOW


def screen_for_maysir(
    inputs: MaysirInputs,
    *,
    policy: MaysirPolicy = MaysirPolicy(),
) -> MaysirAssessment:
    """Run the maysir screen for one ticker.

    Returns the assessment with fired signals + computed risk +
    weighted score. Operators consult `is_tradable(assessment)`
    as the load-bearing gate.
    """

    signals = _detect_signals(inputs, policy=policy)
    score = sum(_SIGNAL_WEIGHT[s] for s in signals)
    risk = _score_to_risk(score, policy=policy)
    return MaysirAssessment(
        ticker=inputs.ticker,
        signals=signals,
        risk=risk,
        score=score,
    )


def screen_batch(
    inputs_list: Iterable[MaysirInputs],
    *,
    policy: MaysirPolicy = MaysirPolicy(),
) -> tuple[MaysirAssessment, ...]:
    """Run the screen across many tickers; sorted by ticker.

    Deterministic ordering — the dashboard tile + operator email
    summary expect diff-stable output.
    """

    assessments = [screen_for_maysir(i, policy=policy) for i in inputs_list]
    assessments.sort(key=lambda a: a.ticker)
    return tuple(assessments)


def is_tradable(assessment: MaysirAssessment) -> bool:
    """Whether the ticker passes the maysir screen.

    Load-bearing gate. True for risk in {NONE, LOW, MODERATE};
    False for HIGH + EXTREME.
    """

    return assessment.risk not in {MaysirRisk.HIGH, MaysirRisk.EXTREME}


def filter_blocked(
    assessments: Iterable[MaysirAssessment],
) -> tuple[MaysirAssessment, ...]:
    """Return only the assessments blocked by the maysir screen.

    Operators surface in the "rejected" tile of the dashboard.
    """

    return tuple(a for a in assessments if not is_tradable(a))


_RISK_EMOJI: dict[MaysirRisk, str] = {
    MaysirRisk.NONE: "✅",
    MaysirRisk.LOW: "🟢",
    MaysirRisk.MODERATE: "🟡",
    MaysirRisk.HIGH: "🟠",
    MaysirRisk.EXTREME: "🔴",
}


_SIGNAL_LABEL: dict[MaysirSignal, str] = {
    MaysirSignal.PENNY_PRICE: "penny price",
    MaysirSignal.EXTREME_SHORT_INTEREST: "extreme short interest",
    MaysirSignal.RETAIL_FLOW_DOMINANT: "retail-flow dominant",
    MaysirSignal.NO_ANALYST_COVERAGE: "no analyst coverage",
    MaysirSignal.EXTREME_VOLATILITY: "extreme volatility",
    MaysirSignal.MEME_PUMP_PATTERN: "meme-pump pattern",
    MaysirSignal.ZERO_REVENUE: "zero revenue",
}


def render_assessment(assessment: MaysirAssessment) -> str:
    """Format one assessment for ops display.

    No-secret-leak: shows only ticker + risk + signal labels.
    Raw alt-data feed details (Reddit posts, Robinhood holding
    lists) go to operator-side debug log.
    """

    emoji = _RISK_EMOJI[assessment.risk]
    parts = [f"{emoji} {assessment.ticker}: {assessment.risk.value}"]
    if assessment.signals:
        labels = sorted(_SIGNAL_LABEL[s] for s in assessment.signals)
        parts.append(f"({', '.join(labels)})")
    return " ".join(parts)


__all__ = [
    "MaysirAssessment",
    "MaysirInputs",
    "MaysirPolicy",
    "MaysirRisk",
    "MaysirSignal",
    "filter_blocked",
    "is_tradable",
    "render_assessment",
    "screen_batch",
    "screen_for_maysir",
]
