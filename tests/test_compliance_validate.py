"""Validating the screen against halal ETFs' N-PORT holdings."""

from __future__ import annotations

from datetime import date

from halal_trader.compliance.etf_holdings import parse_nport
from halal_trader.compliance.validate import Verdict, compare

NPORT = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
  <formData>
    <genInfo><seriesId>S000067283</seriesId><repPdDate>2026-05-31</repPdDate></genInfo>
    <invstOrSecs>
      <invstOrSec><name>NVIDIA</name><title>NVIDIA Corp</title><cusip>67066G104</cusip>
        <identifiers><ticker value="NVDA"/></identifiers>
        <pctVal>13.5</pctVal><assetCat>EC</assetCat></invstOrSec>
      <invstOrSec><name>BRK</name><title>Berkshire B</title><cusip>084670702</cusip>
        <identifiers><ticker value="BRK/B"/></identifiers>
        <pctVal>1.0</pctVal><assetCat>EC</assetCat></invstOrSec>
      <invstOrSec><name>MMF</name><title>Money market</title><cusip>31846V336</cusip>
        <identifiers><ticker value="FGXXX"/></identifiers>
        <pctVal>1.8</pctVal><assetCat>STIV</assetCat></invstOrSec>
    </invstOrSecs>
  </formData>
</edgarSubmission>"""


def test_parse_nport_keeps_common_equity_with_tickers() -> None:
    series, end, holdings = parse_nport(NPORT)

    assert series == "S000067283"
    assert end == date(2026, 5, 31)
    assert [h.ticker for h in holdings] == ["NVDA", "BRK.B"]  # money-market fund dropped
    assert holdings[0].weight_pct == 13.5


def _v(sym: str, verdict: str, cap: float = 1e9) -> Verdict:
    return Verdict(sym, verdict, [f"{verdict} reason"], cap)


def test_compare_reports_both_directions() -> None:
    verdicts = {
        "NVDA": _v("NVDA", "halal"),
        "AAPL": _v("AAPL", "halal"),
        "INTC": _v("INTC", "not_halal"),  # ETF holds it, we reject it
        "XOM": _v("XOM", "doubtful"),
        "BIGCO": _v("BIGCO", "halal", cap=200e9),  # we pass a giant no ETF holds
        "SMALL": _v("SMALL", "halal", cap=2e9),  # small and absent: expected
    }
    etfs = {"NVDA", "AAPL", "INTC", "XOM", "TSLA"}

    v = compare(verdicts, etfs)

    assert (v.etf_names, v.screened_etf_names, v.agree) == (5, 4, 2)
    assert v.agreement == 0.5
    assert [x.symbol for x in v.etf_held_we_reject] == ["INTC"]
    assert [x.symbol for x in v.etf_held_we_doubt] == ["XOM"]
    assert v.etf_held_not_screened == ["TSLA"]
    assert [x.symbol for x in v.large_halal_not_in_etfs] == ["BIGCO"]
