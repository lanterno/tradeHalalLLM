"""Compare the in-house screen with halal ETFs' holdings (decision D3).

Two directions, read differently:

* **ETF holds it, we reject it** -- a likely false negative of ours, or a
  methodology difference (the ETFs' boards use their own thresholds and
  average market caps). Each one is worth a look; this is the list that
  shows where the screen and the boards disagree.
* **We pass it, neither ETF holds it** -- only meaningful for large caps: the
  ETFs hold S&P 500 / FTSE USA names, so a halal small cap is expected to be
  absent. Large caps here are possible false positives, the dangerous kind.

Neither direction sees a false pass below the ETFs' size range, which is
where most of them were (screen v11's review: 98 debt-free REITs and
homebuilders, hotel REITs, a mortgage REIT). Two things help there:

* rejections of ETF-held names are split by kind (``rejection_kind``): an
  activity exclusion, the index veto, a ratio, or missing data. Only the
  ratio and data kinds are candidates for a methodology fix; an activity
  rejection the ETF disagrees with is a judgment to review, not a bug.
* **implied-debt suspects**: passes whose interest expense, read back at
  aaoifi.IMPLIED_RATE, implies debt at or over the 30% limit. It needs no
  ETF and no size floor. The screen already makes a pass doubtful when its
  debt tags explain under half of that implied debt; what is left here is
  the band in between, worth a look by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from halal_trader.compliance.aaoifi import DEBT_LIMIT
from halal_trader.core.num import to_float

RejectionKind = Literal["activity", "veto", "ratio", "data"]


@dataclass(frozen=True, slots=True)
class Verdict:
    symbol: str
    verdict: str
    reasons: list[str]
    market_cap: float | None
    implied_debt_ratio: float | None = None  # interest expense read back as debt / market cap
    debt_ratio: float | None = None


def rejection_kind(v: Verdict) -> RejectionKind:
    """Why the screen rejected a name: its activity, the index veto, a ratio, or missing data."""
    if any(r.startswith("business activity") for r in v.reasons):
        return "activity"
    if v.verdict == "doubtful":
        return "data"
    if any(r.startswith("excluded by") for r in v.reasons) and not any(
        ">=" in r for r in v.reasons
    ):
        return "veto"
    return "ratio"


@dataclass(frozen=True, slots=True)
class Validation:
    etf_names: int
    screened_etf_names: int
    agree: int
    etf_held_we_reject: list[Verdict] = field(default_factory=list)
    etf_held_we_doubt: list[Verdict] = field(default_factory=list)
    etf_held_not_screened: list[str] = field(default_factory=list)
    large_halal_not_in_etfs: list[Verdict] = field(default_factory=list)
    implied_debt_suspects: list[Verdict] = field(default_factory=list)

    @property
    def rejected_by_kind(self) -> dict[RejectionKind, list[Verdict]]:
        """ETF-held names we reject or doubt, by why."""
        out: dict[RejectionKind, list[Verdict]] = {}
        for v in (*self.etf_held_we_reject, *self.etf_held_we_doubt):
            out.setdefault(rejection_kind(v), []).append(v)
        return out

    @property
    def agreement(self) -> float:
        return self.agree / self.screened_etf_names if self.screened_etf_names else 0.0

    @property
    def missing_data(self) -> list[Verdict]:
        """Names an ETF holds that we doubt only because a figure is missing.

        Data, not judgement: a tag the screen does not read, or a company that
        moved to a new SEC registrant (compliance/successors.py). Each one is
        worth tracing, since it is otherwise left out of the core for no reason.
        """
        return [
            v
            for v in self.etf_held_we_doubt
            if v.reasons
            and all(r.startswith("not computable") for r in v.reasons)
            # A foreign issuer is doubtful by policy (screen v4), not for want of data.
            and not any("foreign issuer" in r for r in v.reasons)
        ]


def compare(
    verdicts: dict[str, Verdict], etf_tickers: set[str], *, large_cap: float = 50e9
) -> Validation:
    screened = [verdicts[t] for t in sorted(etf_tickers) if t in verdicts]
    return Validation(
        etf_names=len(etf_tickers),
        screened_etf_names=len(screened),
        agree=sum(1 for v in screened if v.verdict == "halal"),
        etf_held_we_reject=[v for v in screened if v.verdict == "not_halal"],
        etf_held_we_doubt=[v for v in screened if v.verdict == "doubtful"],
        etf_held_not_screened=sorted(t for t in etf_tickers if t not in verdicts),
        large_halal_not_in_etfs=sorted(
            (
                v
                for v in verdicts.values()
                if v.verdict == "halal"
                and v.symbol not in etf_tickers
                and (v.market_cap or 0.0) >= large_cap
            ),
            key=lambda v: -(v.market_cap or 0.0),
        ),
        implied_debt_suspects=sorted(
            (
                v
                for v in verdicts.values()
                if v.verdict == "halal" and (v.implied_debt_ratio or 0.0) >= DEBT_LIMIT
            ),
            key=lambda v: -(v.implied_debt_ratio or 0.0),
        ),
    )


def verdict_of(row: Any) -> Verdict:
    """A stored screen row (symbol, verdict, reasons, mc, implied, debt) as a Verdict."""
    return Verdict(
        row.symbol,
        row.verdict,
        list(row.reasons),
        to_float(row.mc),
        to_float(getattr(row, "implied", None)),
        to_float(getattr(row, "debt", None)),
    )


async def weekly_check(engine: Any, today: Any) -> Validation:
    """The newest screen against the halal ETFs' newest filed holdings (stored, not fetched).

    Run after each weekly screen. Under the strict option the screen may reject
    what an ETF holds, but it should never pass a large company that neither
    ETF holds: ``large_halal_not_in_etfs`` non-empty is a regression to report.
    """
    from sqlalchemy import text

    from halal_trader.compliance.index_veto import views_at

    views = await views_at(engine, today)
    tickers = set().union(*(v.tickers for v in views)) if views else set()
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, verdict, reasons, metrics->>'market_cap' AS mc, "
                "metrics->>'implied_debt_ratio' AS implied, metrics->>'debt_ratio' AS debt "
                "FROM halal_screen_current "
                "WHERE as_of = (SELECT max(as_of) FROM halal_screen_results WHERE as_of <= :d)"
            ),
            {"d": today},
        )
        verdicts = {r.symbol: verdict_of(r) for r in rows}
    return compare(verdicts, tickers)
