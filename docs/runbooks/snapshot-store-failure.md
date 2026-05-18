# `snapshot.store.failure` — replay snapshot couldn't be persisted

**Severity:** WARN
**Triggers when:** the cycle's post-decision snapshot writer
(`core/replay.py` storing `IndicatorSnapshot` / `LlmDecision` /
the prompt-version row) raises an exception. The trade itself
already executed; what's missing is the audit-trail row.
**Acknowledgement window:** 30 minutes.

## Likely causes

1. **Postgres write failure.** Disk full / WAL archive
   stalled / connection pool exhausted (see
   `db.connection_lost` runbook for the related PAGE-severity
   alert).
2. **JSON serialisation error.** A new field added to the
   replay payload doesn't round-trip through asyncpg's JSONB
   serialisation. The snapshot column is JSONB; an unsupported
   type (Decimal not converted, datetime without tz) raises.
3. **Schema drift.** A migration added a NOT NULL column that
   the snapshot writer doesn't populate. Wave 6.B feature
   schema migration is the long-term fix.
4. **Constraint violation.** Foreign-key violation on
   `halal_screening_id` if the screener wrote a row in a
   transaction that rolled back.

## Diagnose

```bash
just logs-tail | grep -E "snapshot|replay|IndicatorSnapshot"
```

Find the cycle that failed:

```bash
just logs-tail | grep "snapshot.store.failure" | tail -5
```

Check if the underlying trade row landed (the snapshot is
post-execution; missing snapshot but trade landed = audit gap
but no money lost):

```bash
docker exec -it $(docker compose ps -q postgres) \
  psql -U trader -d halal_trader \
  -c "SELECT id, pair, side, status, halal_screening_id
      FROM crypto_trades
      ORDER BY id DESC LIMIT 5"
```

## Mitigate

1. **If Postgres write failure** — follow the
   `db.connection_lost` runbook. Once Postgres is healthy,
   the next cycle's writes resume normally; the missed
   snapshot is **not** retroactively backfilled (the snapshot
   captures decision-time state which can't be reconstructed).
2. **If JSON serialisation error** — the failed cycle's full
   stack trace is in `logs/error.log`. The fix is code-side:
   convert Decimal/datetime/etc. before serialising. The bot
   should still trade in the meantime.
3. **If schema drift** — run `uv run halal-trader db migrate`
   to confirm the schema is at head; if it is, the missing
   column is in the writer not the migration. File an issue.
4. **If FK violation** — the related `halal_screening_id` row
   wasn't committed. Restart the bot to flush the stale
   reference; the next cycle re-screens the symbol.

## Pin

The bot's audit trail is **append-only**. A missing snapshot
row is an audit *gap*, not a *corruption* — the trade-side
records (`Trade` / `CryptoTrade`) are still consistent. A
future scholar reviewer can spot the gap (Wave 8 audit-trail
section of `SECURITY.md`) but can't fill it post-hoc; this is
intentional. Don't try to backfill missing snapshots from
later state — that would corrupt the audit promise.

## Escalate

If snapshots fail repeatedly across cycles, the bot is
running but not producing the audit data. Engage halt with
reason "snapshot store failing repeatedly; investigating"
until the cause is identified — the bot can run without
snapshots, but the operator should not let it accumulate
audit-gap rows for compliance reasons.

## Postmortem

WARN-severity normally; if the cause persists or recurs, it
becomes a PAGE-level issue worth a postmortem.

---

_Last reviewed: 2026-05-01_
