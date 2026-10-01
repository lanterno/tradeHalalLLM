"""Compare the in-house screen with halal ETFs' holdings (decision D3).

Two directions, read differently:

* **ETF holds it, we reject it** -- a likely false negative of ours, or a
  methodology difference (the ETFs' boards use their own thresholds and
  average market caps). Each one is worth a look; this is the list that
  decides whether Zoya production is worth buying.
* **We pass it, neither ETF holds it** -- only meaningful for large caps: the
  ETFs hold S&P 500 / FTSE USA names, so a halal small cap is expected to be
  absent. Large caps here are possible false positives, the dangerous kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Verdict:
    symbol: str
    verdict: str
    reasons: list[str]
    market_cap: float | None


@dataclass(frozen=True, slots=True)
class Validation:
    etf_names: int
    screened_etf_names: int
    agree: int
    etf_held_we_reject: list[Verdict] = field(default_factory=list)
    etf_held_we_doubt: list[Verdict] = field(default_factory=list)
    etf_held_not_screened: list[str] = field(default_factory=list)
    large_halal_not_in_etfs: list[Verdict] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        return self.agree / self.screened_etf_names if self.screened_etf_names else 0.0


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
    )
