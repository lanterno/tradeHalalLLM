"""One rule for when an SEC filing became public."""

from __future__ import annotations

from datetime import UTC, date, datetime

from halal_trader.compliance.sec import filed_at


def test_the_acceptance_time_wins_when_edgar_has_one() -> None:
    assert filed_at(date(2026, 8, 3), "2026-08-03T16:05:12.000Z") == datetime(
        2026, 8, 3, 16, 5, 12, tzinfo=UTC
    )


def test_without_one_a_filing_is_known_at_1700_new_york() -> None:
    assert filed_at(date(2026, 8, 3)) == datetime(2026, 8, 3, 21, 0, tzinfo=UTC)  # EDT
    assert filed_at(date(2026, 12, 3), "") == datetime(2026, 12, 3, 22, 0, tzinfo=UTC)  # EST
