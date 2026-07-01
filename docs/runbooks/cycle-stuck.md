# `cycle.stuck` — cycle exceeded watchdog deadline

**Severity:** PAGE
**Triggers when:** the cycle's `asyncio.wait_for(interval × 2)`
watchdog cancels a cycle that didn't complete in time. The most
common cause is a hung HTTP call somewhere in the prompt-context
build pipeline.
**Acknowledgement window:** 10 minutes — a stuck cycle delays
SL/TP enforcement.

## Likely causes

1. **Stuck HTTP call** — a broker, halal screener, news feed,
   or LLM call hangs past its timeout (the upstream timeout
   was longer than the cycle's interval × 2 budget).
2. **Asyncio deadlock** — two coroutines holding each other's
   awaitable. Rare but possible when a refactor changes the
   await order in `crypto/cycle.py`.
3. **Postgres serialization conflict** — a long-running
   transaction blocks the cycle's writes. Check
   `pg_stat_activity` for stuck transactions.
4. **Remote GLM endpoint hang** — the GLM host stops responding
   without raising an exception. The 60s client timeout
   (`GLM_TIMEOUT_SECONDS`) plus `FallbackLLM` endpoint rotation
   normally handles it before the watchdog fires.

## Diagnose

```bash
just logs-tail | grep -E "cycle\.stuck|cycle\.failed"
```

Look for the *stage* the cycle was in when killed (the cycle
pipeline emits `cycle.stage.start` on every stage; the last
START without a matching END is the killer):

```bash
just logs-tail | grep "cycle.stage" | tail -50
```

For a Postgres-side hang:

```bash
docker exec -it $(docker compose ps -q postgres) \
  psql -U trader -d halal_trader \
  -c "SELECT pid, state, query_start, query
      FROM pg_stat_activity
      WHERE state != 'idle' AND query_start < now() - interval '30 seconds'"
```

## Mitigate

1. **If stuck on an HTTP stage** — engage halt and restart the
   bot. The kill will free the hung connection. If the same
   endpoint hangs again, lower its timeout
   (`GLM_TIMEOUT_SECONDS` for the LLM) or configure a fallback
   endpoint via `GLM_FALLBACK_BASE_URL`.
2. **If a Postgres transaction is stuck** — terminate it:
   ```bash
   docker exec -it $(docker compose ps -q postgres) \
     psql -U trader -d halal_trader \
     -c "SELECT pg_terminate_backend(<pid>)"
   ```
   Then resume the bot.
3. **If GLM endpoint hang** — check
   <https://status.openrouter.ai> (and <https://status.z.ai>
   when a Z.ai fallback endpoint is configured). The 60s client
   timeout plus `FallbackLLM` rotation recovers on its own; if
   the primary host is degraded for an extended period, point
   `GLM_BASE_URL` at a healthy host (or set the
   `GLM_FALLBACK_*` trio) in `.env` and restart the bot.
4. **If unclear** — engage halt with reason "cycle stuck;
   investigating", then read the per-stage timeline (Wave 5.A
   `core/cycle_timeline.py` aggregator) for the most-recent
   completed cycle to spot the pattern.

## Escalate

If the cycle gets stuck twice in a 30-min window, this is a
PAGE-level issue requiring a code-side fix. Open an issue with
the stage name + stuck query (Postgres) or stuck endpoint (HTTP)
attached.

## Postmortem

PAGE alert — file `docs/postmortems/<date>-cycle-stuck.md`
within 48h. Include: stage name, timeout values in effect,
mitigation steps, what would have caught this earlier (was the
Wave 6.F latency budget tracker amber on the offending stage?).

---

_Last reviewed: 2026-05-01_
