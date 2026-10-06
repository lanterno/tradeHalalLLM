"""The belief board's plain words: evidence sentences, stances, decision reasons."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.api import plain
from halabot.api.queries import random_entry_win_rate


@pytest.mark.parametrize(
    ("source", "direction", "detail", "expected"),
    [
        ("indicator.relstrength", 1.0, "rel +4.26% vs SPY", "Outperforming SPY by 4.3%"),
        ("indicator.relstrength", -1.0, "rel -0.84% vs SPY", "Lagging SPY by 0.8%"),
        ("indicator.rsi", 1.0, "RSI 95", "RSI 95 · overbought"),
        ("indicator.rsi", 1.0, "RSI 70", "RSI 70 · overbought"),
        ("indicator.rsi", 0.2, "RSI 55", "RSI 55"),
        ("indicator.rsi", -1.0, "RSI 30", "RSI 30 · oversold"),
        (
            "indicator.alignment",
            0.8,
            "short +4.27% / long +8.47%",
            "Uptrend: +4.3% short, +8.5% long",
        ),
        (
            "indicator.alignment",
            -0.8,
            "short -1.50% / long -1.60%",
            "Downtrend: -1.5% short, -1.6% long",
        ),
        (
            "indicator.alignment",
            0.1,
            "short -0.10% / long +0.70%",
            "Mixed trend: -0.1% short, +0.7% long",
        ),
        ("indicator.multiframe", 0.85, "EMA stack 8/21/55 sep=0.64", "Moving averages stacked up"),
        (
            "indicator.multiframe",
            -0.5,
            "EMA stack 8/21/55 sep=0.20",
            "Moving averages stacked down",
        ),
        ("news", 0.6, "news(llm) +0.60: Big Chip Deal", "News: “Big Chip Deal”"),
        ("news", -0.5, "news polarity -0.50: Probe: Opened", "News: “Probe: Opened”"),
        (
            "forecaster",
            -0.4,
            "chronos 5-bar proj -4.123% snr=0.50",
            "Forecast -4.1% over the next 5 hours",
        ),
        ("forecaster", 0.3, "OLS slope proj +0.120%/bar R²=0.40", "Forecast trend +0.1% an hour"),
        ("indicator.structure", 0.5, "near support 163.10", "Near support $163.10"),
        (
            "indicator.volume",
            -0.3,
            "vol 2.5x conf move -0.52%",
            "Volume 2.5× normal behind a -0.5% move",
        ),
        ("anomaly", 0.0, "vol spike 2.3x baseline", "Volatility 2.3× its usual level"),
        # An unknown source, or a detail no rule parses, reads as itself.
        ("indicator.new", 0.4, "something new", "something new"),
        ("indicator.relstrength", 1.0, "rel n/a", "rel n/a"),
    ],
)
def test_evidence_sentences(source, direction, detail, expected):
    assert plain.evidence_sentence(source, direction, detail) == expected


def test_evidence_labels():
    assert plain.evidence_label("indicator.relstrength") == "Relative strength"
    assert plain.evidence_label("forecaster") == "Forecast"
    assert plain.evidence_label("indicator.order_flow") == "Order flow"


def test_rsi_caution_only_for_a_very_high_rsi_counted_in_favour():
    def ev(detail, direction=1.0):
        return [{"source": "indicator.rsi", "direction": direction, "detail": detail}]

    assert plain.rsi_caution(ev("RSI 79")) is None
    assert "RSI 80" in (plain.rsi_caution(ev("RSI 80")) or "")
    assert plain.rsi_caution(ev("RSI 90", direction=-1.0)) is None
    assert plain.rsi_caution([]) is None


BANDS = {"benchmark": "SPY", "entry_band": 0.35, "exit_band": 0.15}


@pytest.mark.parametrize(
    ("asset", "direction", "conviction", "strict", "expected"),
    [
        ("AVGO", "long_bias", 0.35, "halal", "long"),  # the entry band is inclusive
        ("AVGO", "long_bias", 0.3499, "halal", "leaning"),
        ("AVGO", "long_bias", 0.15, "halal", "leaning"),  # so is the exit band
        ("AVGO", "long_bias", 0.1499, "halal", "none"),
        ("AVGO", "neutral", 0.9, "halal", "none"),  # conviction without a long bias
        ("NVDA", "long_bias", 0.9, "not_halal", "excluded"),
        ("TSM", "long_bias", 0.9, "doubtful", "excluded"),
        ("CRM", "long_bias", 0.9, "unscreened", "excluded"),
        ("SPY", "long_bias", 0.9, "unscreened", "benchmark"),  # the index, never excluded
        ("SPY", "neutral", 0.0, "halal", "benchmark"),
    ],
)
def test_stance(asset, direction, conviction, strict, expected):
    got = plain.stance(
        asset=asset, direction=direction, conviction=conviction, strict=strict, **BANDS
    )
    assert got == expected


def reason(**payload):
    return plain.decision_reason(payload, entry_band=0.35, exit_band=0.15)


def test_decision_reasons():
    opening = {"side": "buy", "current_weight": 0.0, "target_weight": 0.07, "conviction_raw": 0.41}
    assert reason(**opening, reason="conviction") == "conviction crossed 35%"
    adding = {"side": "buy", "current_weight": 0.07, "target_weight": 0.1, "conviction_raw": 0.47}
    assert reason(**adding, reason="conviction") == "conviction rose to 47%"
    assert reason(side="buy", current_weight=0.07, target_weight=0.1) == "conviction rose"
    closing = {"side": "sell", "current_weight": 0.06, "target_weight": 0.0}
    assert reason(**closing, conviction_raw=0.08) == "conviction fell below 15%"
    assert reason(**closing) == "conviction fell below 15%"
    # Sold out while conviction was still above the exit band: not a fall.
    assert reason(**closing, conviction_raw=0.4) == "target fell to zero at conviction 40%"
    trim = {"side": "sell", "current_weight": 0.07, "target_weight": 0.02, "conviction_raw": 0.24}
    assert reason(**trim) == "conviction fell to 24%"
    assert (
        reason(**adding, reason="conviction (gross-normalized)")
        == "conviction rose to 47% · book scaled to its exposure cap"
    )


def test_forced_and_halted_decisions():
    forced = {"side": "sell", "target_weight": 0.0, "forced_exit": True}
    assert reason(**forced, reason="price_break") == (
        "forced exit: price broke its invalidation level"
    )
    assert reason(**forced, reason="compliance_lapsed") == "forced exit: its halal verdict lapsed"
    assert reason(**forced, reason="stop_hit") == "forced exit: stop hit"
    assert reason(**forced) == "forced exit"
    halted = {"side": "sell", "current_weight": 0.05, "target_weight": 0.0}
    assert reason(**halted, reason="risk halt: drawdown") == "risk halt: drawdown"


T0 = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)


def _series(closes):
    return [(T0 + timedelta(hours=i), c) for i, c in enumerate(closes)]


def test_random_entry_win_rate_holds_each_entry_for_the_hold():
    # Held 2 bars: 100->102 wins, 101->100 loses, 102->102.1 is under the 0.2% bar.
    rate, n = random_entry_win_rate(
        {"A": _series([100, 101, 102, 100, 102.1])}, hold=timedelta(hours=2), threshold=0.002
    )
    assert n == 3
    assert rate == pytest.approx(1 / 3)


def test_random_entry_win_rate_skips_entries_outside_the_session_and_empty_input():
    series = {"A": _series([100, 110, 121, 133])}
    rate, n = random_entry_win_rate(
        series, hold=timedelta(hours=1), threshold=0.002, in_session=lambda t: t == T0
    )
    assert (rate, n) == (1.0, 1)
    assert random_entry_win_rate({}, hold=timedelta(hours=1), threshold=0.0) == (None, 0)
    # A hold longer than the history leaves nothing to measure.
    assert random_entry_win_rate(series, hold=timedelta(days=1), threshold=0.0) == (None, 0)


def test_top_evidence_keeps_one_reading_per_indicator() -> None:
    from types import SimpleNamespace as E

    from halabot.api.queries import _top_evidence

    ev = [
        E(source="indicator.relstrength", direction=1.0, weight=0.5, detail="rel +4.30% vs SPY"),
        E(source="indicator.relstrength", direction=1.0, weight=0.3, detail="rel +3.10% vs SPY"),
        E(source="news", direction=0.5, weight=0.6, detail="a"),
        E(source="news", direction=0.5, weight=0.55, detail="b"),
        E(source="indicator.rsi", direction=1.0, weight=0.2, detail="RSI 70"),
    ]
    top = _top_evidence(ev)
    assert [e.detail for e in top] == ["rel +4.30% vs SPY", "a", "b", "RSI 70"]
