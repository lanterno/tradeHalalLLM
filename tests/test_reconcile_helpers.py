"""Tests for the pure helpers in :mod:`core.reconcile`.

The full ``reconcile_crypto`` / ``reconcile_stocks`` flows hit the
broker + DB and live in DB-backed tests. The drift-math helpers and
the report dataclasses are pure and are what the operator's alert
text + dashboard summary actually render from.
"""

from halal_trader.core.reconcile import (
    Drift,
    ReconcileReport,
    _drift_pct,
    _summarize_drifts,
)

# ── _drift_pct ──────────────────────────────────────────────────


def test_drift_pct_zero_when_equal():
    assert _drift_pct(10.0, 10.0) == 0.0


def test_drift_pct_symmetric():
    """Symmetric drift: swapping db/broker gives the same magnitude."""
    a = _drift_pct(10.0, 11.0)
    b = _drift_pct(11.0, 10.0)
    assert abs(a - b) < 1e-12


def test_drift_pct_normalizes_to_larger_side():
    """|10 - 11| / max(10, 11) = 1/11 ≈ 0.0909."""
    assert abs(_drift_pct(10.0, 11.0) - (1.0 / 11.0)) < 1e-9


def test_drift_pct_handles_zero_inputs():
    """Both zero → no drift; one zero → 100% drift normalized to non-zero."""
    assert _drift_pct(0.0, 0.0) == 0.0
    # 1.0 vs 0.0 → |1| / max(0, 1) = 1.0
    assert _drift_pct(1.0, 0.0) == 1.0
    assert _drift_pct(0.0, 1.0) == 1.0


# ── _summarize_drifts ──────────────────────────────────────────


def test_summarize_empty_returns_header_only():
    out = _summarize_drifts([])
    assert "Reconciliation drift detected" in out


def test_summarize_renders_drift_per_row():
    drifts = [
        Drift(
            market="crypto",
            symbol="BTCUSDT",
            db_quantity=0.5,
            broker_quantity=0.45,
            drift_pct=0.1,
            drift_usd=2_500.0,
        )
    ]
    out = _summarize_drifts(drifts)
    assert "BTCUSDT" in out
    assert "crypto" in out
    assert "10.0%" in out
    assert "$2500" in out


def test_summarize_caps_at_five_drifts():
    """Long lists get truncated to keep the alert message readable."""
    drifts = [
        Drift(
            market="stocks",
            symbol=f"S{i}",
            db_quantity=10.0,
            broker_quantity=11.0,
            drift_pct=0.1,
        )
        for i in range(10)
    ]
    out = _summarize_drifts(drifts)
    # Only the first 5 should appear.
    for i in range(5):
        assert f"S{i}" in out
    for i in range(5, 10):
        assert f"S{i}" not in out


def test_summarize_omits_dollar_when_no_usd_estimate():
    drifts = [
        Drift(
            market="crypto",
            symbol="X",
            db_quantity=1.0,
            broker_quantity=2.0,
            drift_pct=0.5,
            drift_usd=None,
        )
    ]
    out = _summarize_drifts(drifts)
    assert "X" in out
    assert "$" not in out


# ── ReconcileReport.has_drift ───────────────────────────────────


def test_report_has_drift_false_when_empty():
    r = ReconcileReport(market="crypto")
    assert r.has_drift is False


def test_report_has_drift_true_with_any_drift_row():
    r = ReconcileReport(
        market="stocks",
        drifts=[
            Drift(
                market="stocks",
                symbol="AAPL",
                db_quantity=10.0,
                broker_quantity=11.0,
                drift_pct=0.1,
            )
        ],
    )
    assert r.has_drift is True
