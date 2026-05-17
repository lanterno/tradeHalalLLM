# `drift.breach` — model drift detector tripped

**Severity:** WARN
**Triggers when:** the equity-curve anomaly detector
(`ml/equity_anomaly.py`, Wave 4.I) reports `severity == "alert"`
on the per-trade or drawdown z-score (|z| ≥ 3 against the rolling
window).
**Acknowledgement window:** 60 minutes — drift is rarely a
single-cycle problem.

## Likely causes

1. **Regime change** — market shifted from the regime the model
   was trained on; per-trade returns are still random walks but
   shifted.
2. **Feature drift** — a producer (`crypto/indicators.py`,
   sentiment, regime) is computing a feature differently than
   training time. Confirm via the Wave 6.B feature schema diff.
3. **Random tail** — fat-tailed returns can produce a >3σ event
   on noise alone. The detector's cold-start guard suppresses
   most of this; what reaches the alert path is real.

## Diagnose

```bash
just logs-tail | grep "equity_anomaly"
```

…shows the z-score, severity, and direction. Cross-check with
the Wave 6.E fingerprint of the active model — if it changed
recently, suspect a model regression.

## Mitigate

1. **If regime change** — engage halt with a clear reason,
   re-train the model on a window that includes the new regime,
   promote via Wave 6.A `should_promote` only after Wave 4.F
   `evaluate_promotion` passes.
2. **If feature drift** — diff the feature schema between
   training-time and now (`Wave 6.B` `validate(payload, schema)`
   surfaces missing / wrongly-typed features). Re-train against
   the current schema.
3. **If random tail** — annotate the alert and watch for the
   next two cycles. If z-score returns to baseline, no action.

## Escalate

If the breach persists across three cycles, escalate to PAGE.
Operator must decide whether to halt + retrain or accept the
new normal.

---

_Last reviewed: 2026-05-01_
