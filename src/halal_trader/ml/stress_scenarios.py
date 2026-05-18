"""Stress-test scenario library — Round-5 Wave 14.C.

A catalogue of historical + synthetic stress scenarios the bot can
apply to its current portfolio to estimate worst-case outcomes. The
existing `crypto/stress.py` covers crypto-specific synthetic
generators; this module is the **portfolio-level scenario catalogue**
suitable for both stocks + crypto, modelled as percentage shocks
applied to a snapshot of positions.

Pinned semantics:

- **Closed-set ScenarioKind ladder** — historical events
  (BLACK_MONDAY_1987 / DOTCOM_2000 / GFC_2008 / COVID_2020 /
  RATE_HIKES_2022) + synthetic shocks (VOL_SPIKE / LIQUIDITY_FREEZE
  / RATE_SHOCK / CURRENCY_DEVAL / BLACK_SWAN).
- **Asset-class-keyed shock vectors** — equities / crypto / sukuk /
  commodities / cash each carry separate shock magnitudes so a
  scenario hitting equities -30% can simultaneously bid sukuk +5%.
- **`apply_scenario`** is pure — caller passes positions + scenario
  → projected portfolio value.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    """Closed-set asset classes for scenario shocks."""

    EQUITIES = "equities"
    CRYPTO = "crypto"
    SUKUK = "sukuk"
    COMMODITIES = "commodities"
    CASH = "cash"


class ScenarioKind(str, Enum):
    """Closed-set stress scenarios."""

    BLACK_MONDAY_1987 = "black_monday_1987"
    DOTCOM_2000 = "dotcom_2000"
    GFC_2008 = "gfc_2008"
    COVID_2020 = "covid_2020"
    RATE_HIKES_2022 = "rate_hikes_2022"
    VOL_SPIKE = "vol_spike"
    LIQUIDITY_FREEZE = "liquidity_freeze"
    RATE_SHOCK = "rate_shock"
    CURRENCY_DEVAL = "currency_deval"
    BLACK_SWAN = "black_swan"


@dataclass(frozen=True)
class Scenario:
    """A single stress scenario."""

    kind: ScenarioKind
    description: str
    shocks: Mapping[AssetClass, float]  # signed pct change
    is_historical: bool

    def __post_init__(self) -> None:
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")
        for ac, shock in self.shocks.items():
            if not -1.0 <= shock <= 5.0:
                raise ValueError(f"shock for {ac.value} must be in [-1.0, 5.0]; got {shock}")


# Module-level catalogue of curated scenarios. Numbers based on
# historical realised drawdowns / news-cycle outcomes; they are
# deliberately approximate (operators can override).
def _catalogue() -> tuple[Scenario, ...]:
    return (
        Scenario(
            kind=ScenarioKind.BLACK_MONDAY_1987,
            description="Oct 19 1987 — equities -22% in one day.",
            shocks={
                AssetClass.EQUITIES: -0.22,
                AssetClass.CRYPTO: 0.0,  # didn't exist
                AssetClass.SUKUK: 0.02,
                AssetClass.COMMODITIES: -0.05,
                AssetClass.CASH: 0.0,
            },
            is_historical=True,
        ),
        Scenario(
            kind=ScenarioKind.DOTCOM_2000,
            description="Mar 2000 - Oct 2002 — Nasdaq -78%, S&P -49%.",
            shocks={
                AssetClass.EQUITIES: -0.50,
                AssetClass.CRYPTO: 0.0,
                AssetClass.SUKUK: 0.10,
                AssetClass.COMMODITIES: 0.05,
                AssetClass.CASH: 0.0,
            },
            is_historical=True,
        ),
        Scenario(
            kind=ScenarioKind.GFC_2008,
            description="Sep 2008 - Mar 2009 — global equities -50%, credit freeze.",
            shocks={
                AssetClass.EQUITIES: -0.50,
                AssetClass.CRYPTO: 0.0,
                AssetClass.SUKUK: -0.05,
                AssetClass.COMMODITIES: -0.40,
                AssetClass.CASH: 0.0,
            },
            is_historical=True,
        ),
        Scenario(
            kind=ScenarioKind.COVID_2020,
            description="Feb-Mar 2020 — equities -34%, crypto -50%, oil negative.",
            shocks={
                AssetClass.EQUITIES: -0.34,
                AssetClass.CRYPTO: -0.50,
                AssetClass.SUKUK: -0.05,
                AssetClass.COMMODITIES: -0.30,
                AssetClass.CASH: 0.0,
            },
            is_historical=True,
        ),
        Scenario(
            kind=ScenarioKind.RATE_HIKES_2022,
            description="2022 Fed hike cycle — equities -25%, crypto -75%, bonds -20%.",
            shocks={
                AssetClass.EQUITIES: -0.25,
                AssetClass.CRYPTO: -0.75,
                AssetClass.SUKUK: -0.20,
                AssetClass.COMMODITIES: 0.10,
                AssetClass.CASH: 0.0,
            },
            is_historical=True,
        ),
        Scenario(
            kind=ScenarioKind.VOL_SPIKE,
            description="Synthetic — VIX 50, equities -10%, crypto -20%.",
            shocks={
                AssetClass.EQUITIES: -0.10,
                AssetClass.CRYPTO: -0.20,
                AssetClass.SUKUK: 0.0,
                AssetClass.COMMODITIES: -0.05,
                AssetClass.CASH: 0.0,
            },
            is_historical=False,
        ),
        Scenario(
            kind=ScenarioKind.LIQUIDITY_FREEZE,
            description="Synthetic — bid-ask blowout, equities -15%, sukuk -10%.",
            shocks={
                AssetClass.EQUITIES: -0.15,
                AssetClass.CRYPTO: -0.30,
                AssetClass.SUKUK: -0.10,
                AssetClass.COMMODITIES: -0.10,
                AssetClass.CASH: 0.0,
            },
            is_historical=False,
        ),
        Scenario(
            kind=ScenarioKind.RATE_SHOCK,
            description="Synthetic — +200bps overnight, sukuk -25%, equities -8%.",
            shocks={
                AssetClass.EQUITIES: -0.08,
                AssetClass.CRYPTO: -0.15,
                AssetClass.SUKUK: -0.25,
                AssetClass.COMMODITIES: 0.0,
                AssetClass.CASH: 0.0,
            },
            is_historical=False,
        ),
        Scenario(
            kind=ScenarioKind.CURRENCY_DEVAL,
            description="Synthetic — base-currency 30% devaluation.",
            shocks={
                AssetClass.EQUITIES: -0.05,
                AssetClass.CRYPTO: 0.20,
                AssetClass.SUKUK: -0.10,
                AssetClass.COMMODITIES: 0.20,
                AssetClass.CASH: -0.30,
            },
            is_historical=False,
        ),
        Scenario(
            kind=ScenarioKind.BLACK_SWAN,
            description="Synthetic — 5-sigma everything-down event.",
            shocks={
                AssetClass.EQUITIES: -0.40,
                AssetClass.CRYPTO: -0.60,
                AssetClass.SUKUK: -0.20,
                AssetClass.COMMODITIES: -0.30,
                AssetClass.CASH: 0.0,
            },
            is_historical=False,
        ),
    )


SCENARIOS: tuple[Scenario, ...] = _catalogue()


def scenario_by_kind(kind: ScenarioKind) -> Scenario:
    for s in SCENARIOS:
        if s.kind is kind:
            return s
    raise KeyError(f"unknown scenario {kind!r}")


def historical_scenarios() -> tuple[Scenario, ...]:
    return tuple(s for s in SCENARIOS if s.is_historical)


def synthetic_scenarios() -> tuple[Scenario, ...]:
    return tuple(s for s in SCENARIOS if not s.is_historical)


@dataclass(frozen=True)
class Position:
    """A position in the portfolio for stress purposes."""

    symbol: str
    asset_class: AssetClass
    market_value: float

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.market_value < 0:
            raise ValueError("market_value must be non-negative")


@dataclass(frozen=True)
class StressResult:
    """Result of applying a scenario to a portfolio."""

    scenario_kind: ScenarioKind
    starting_value: float
    projected_value: float
    pct_change: float
    per_position: tuple[tuple[str, float, float], ...]  # (symbol, old, new)


def apply_scenario(positions: Iterable[Position], scenario: Scenario) -> StressResult:
    starting_value = 0.0
    projected_value = 0.0
    per_pos: list[tuple[str, float, float]] = []

    for p in positions:
        shock = scenario.shocks.get(p.asset_class, 0.0)
        new_value = p.market_value * (1.0 + shock)
        starting_value += p.market_value
        projected_value += new_value
        per_pos.append((p.symbol, p.market_value, new_value))

    pct_change = (projected_value - starting_value) / starting_value if starting_value > 0 else 0.0
    return StressResult(
        scenario_kind=scenario.kind,
        starting_value=starting_value,
        projected_value=projected_value,
        pct_change=pct_change,
        per_position=tuple(per_pos),
    )


def render_result(result: StressResult) -> str:
    arrow = "▼" if result.pct_change < 0 else "▲"
    return (
        f"{arrow} {result.scenario_kind.value}: "
        f"${result.starting_value:.2f}→${result.projected_value:.2f} "
        f"({result.pct_change * 100:+.2f}%)"
    )
