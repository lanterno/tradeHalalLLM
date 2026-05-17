# Operator runbooks

Per-alert recovery procedures. The on-call operator opens the runbook
linked from the alert message; if the runbook says "no further
action", the on-call closes the alert and goes back to bed.

Every alert raised through `core/alert_router.py` carries a
`runbook_url` field. When an alert is added without a runbook here,
the renderer surfaces "(none yet — write one in docs/runbooks/)" so
the gap is visible in the alert message itself.

## Conventions

* Filenames match the alert `type` with `.` → `-`:
  `halt.engaged` → `halt-engaged.md`.
* Each runbook follows the same five-section template (see
  `_template.md`).
* Severity escalation is documented at the bottom of each runbook —
  when a procedure fails, who's the next escalation contact.
* Runbooks are reviewed quarterly. Add a "last reviewed" footer.

## Index

| Alert type | Severity | Runbook |
|---|---|---|
| `halt.engaged` | PAGE | [halt-engaged.md](halt-engaged.md) |
| `chain.backoff` | WARN | [chain-backoff.md](chain-backoff.md) |
| `drift.breach` | WARN | [drift-breach.md](drift-breach.md) |
| `broker.api.error_rate` | PAGE | [broker-api-error-rate.md](broker-api-error-rate.md) |
| `cycle.stuck` | PAGE | [cycle-stuck.md](cycle-stuck.md) |
| `llm.circuit_breaker` | PAGE | [llm-circuit-breaker.md](llm-circuit-breaker.md) |
| `db.connection_lost` | PAGE | [db-connection-lost.md](db-connection-lost.md) |
| `halal.screener.stale` | WARN | [halal-screener-stale.md](halal-screener-stale.md) |
| `snapshot.store.failure` | WARN | [snapshot-store-failure.md](snapshot-store-failure.md) |

## Operations playbooks

These are not alert-triggered runbooks but procedures the operator
runs on cadence:

| Playbook | When to run |
|---|---|
| [backups-and-pitr.md](backups-and-pitr.md) | After a DB failure (post-stabilisation), or quarterly for the restore drill |

For the alert routing model, see
[`src/halal_trader/core/alert_router.py`](../../src/halal_trader/core/alert_router.py).
