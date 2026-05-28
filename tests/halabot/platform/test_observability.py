"""Correlation-id context binding (INV-5) + log setup."""

from __future__ import annotations

import logging

from halabot.platform.observability import (
    CorrelationFilter,
    correlation_scope,
    current_correlation_id,
    setup_logging,
)


def test_scope_binds_and_resets():
    assert current_correlation_id() is None
    with correlation_scope("abc"):
        assert current_correlation_id() == "abc"
        with correlation_scope("nested"):
            assert current_correlation_id() == "nested"
        assert current_correlation_id() == "abc"
    assert current_correlation_id() is None


def test_filter_attaches_id_or_dash():
    f = CorrelationFilter()
    rec = logging.LogRecord("n", logging.INFO, __file__, 1, "m", (), None)
    f.filter(rec)
    assert rec.correlation_id == "-"
    with correlation_scope("xyz"):
        rec2 = logging.LogRecord("n", logging.INFO, __file__, 1, "m", (), None)
        f.filter(rec2)
        assert rec2.correlation_id == "xyz"


def test_setup_logging_accepts_int_and_str():
    setup_logging(logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG
    setup_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING
    # Reset so we don't leave the root logger noisy for other tests.
    setup_logging(logging.INFO)
