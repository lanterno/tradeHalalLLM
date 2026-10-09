"""Peer read-through and the exit test: who counts as a peer, and the registered split."""

from __future__ import annotations

from datetime import UTC, date, datetime

from halal_trader.events.earnings_signal import Release
from halal_trader.events.intraday import TRAIN_UNTIL, Headline, Outcome, exit_test
from halal_trader.events.peers import PEERS, peers_of

AT = datetime(2024, 5, 1, 20, 5, tzinfo=UTC)
SOFT, OIL = "SERVICES-PREPACKAGED SOFTWARE", "PETROLEUM REFINING"


def test_peers_are_the_largest_same_industry_names_not_reporting_nearby() -> None:
    names = {f"S{i}": SOFT for i in range(PEERS + 5)} | {"LEAD": SOFT, "XOM": OIL, "NEAR": SOFT}
    caps = {s: float(i + 1) for i, s in enumerate(names)} | {"NEAR": 1e12, "XOM": 1e12}
    leader = Release("LEAD", AT, 0.05, 0.05, "beat", "beats", None)
    peers = peers_of(
        leader,
        sic=names,
        caps_by_screen={date(2024, 3, 31): caps, date(2024, 6, 30): {}},  # the later one is unseen
        screen_dates=[date(2024, 3, 31), date(2024, 6, 30)],
        reported={"NEAR": [date(2024, 5, 3)]},  # its own earnings, two days later
    )
    assert len(peers) == PEERS and "LEAD" not in peers and "XOM" not in peers
    assert "NEAR" not in peers
    assert peers[0] == f"S{PEERS + 4}"  # the largest first


def test_the_exit_test_splits_at_the_registered_date() -> None:
    def outcome(day: date, score: float, same: float) -> Outcome:
        at = datetime(day.year, day.month, day.day, 15, tzinfo=UTC)
        return Outcome(Headline("X", at, score), same, 0.0, 0.0)

    early = [outcome(date(2026, 1, 5), -0.9, -0.01), outcome(date(2026, 1, 6), -0.9, -0.03)]
    late = [outcome(TRAIN_UNTIL, -0.9, 0.02), outcome(date(2026, 7, 1), -0.9, 0.04)]
    halves = exit_test(early + late)
    neg = {
        k: next(b for b in v if b.label.startswith("score <=") and b.horizon == "same_day")
        for k, v in halves.items()
    }
    assert (neg["train"].n, round(neg["train"].mean, 4)) == (2, -0.02)
    assert (neg["holdout"].n, round(neg["holdout"].mean, 4)) == (2, 0.03)
