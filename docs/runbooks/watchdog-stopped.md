# Watchdog: "<component> stopped" — a process or a daily job stopped beating

**Severity:** PAGE for `stock.process`, `stock.monitor`; WARN for the rest
**Triggers when:** the web's watchdog (`web/watchdog.py`, every 5 minutes)
finds the component failing on two passes
in a row, judged by `core/heartbeat.py:assess`:

| Component | Stale when |
|---|---|
| `stock.process` | no beat for 3 min (the bot's run loop beats every 60 s) |
| `shadow.process` | no beat for 3 min (the shadow's event loop beats every 60 s) |
| `stock.monitor` | no tick for 5 min (stop-loss / take-profit enforcement) |
| `stock.cycle` | 10:00 ET to the close: no cycle for 45 min |
| `market.snapshot` | trading days 09:05-17:00 ET: no snapshot for 5 min |
| `recommendation.daily`, `stock.eod`, `stock.ledger`, `research.daily` | the newest run owed on a trading day (09:05, 15:50 / 12:50 on early closes, 16:30, 20:30 ET), plus its grace, left no beat |

A second message, "… is back", follows when the component beats again.

## Likely causes

1. A deploy or `docker restart` that took longer than ~8 minutes.
2. The process crashed and docker is restarting it in a loop (bad `.env`,
   a migration mismatch, a broker server that will not start).
3. The process is alive but hung (`docker ps` shows it `unhealthy`).
4. For a daily job: it raised (its own alert, `research.failed`,
   `ledger.sync_failed`, … came first), or the bot was down at its time
   and the startup catch-up judged it too late to run.
5. The database is unreachable, so no beat can be written.

## Diagnose

```bash
just health                                        # which container is not running / 503
docker ps --format '{{.Names}} {{.Status}}'        # "unhealthy" = alive but not beating
curl -s http://127.0.0.1:8082/api/health | jq .bot # each component's status and reason
just docker-logs trader-stocks                     # or trader-shadow
```

## Mitigate

1. **Restart loop (cause 2)** — read the first error in the logs; fix the
   `.env` key it names, then `just up`.
2. **Hung (cause 3)** — `docker restart trader-stocks` (outside market hours
   if possible). The bot catches up the daily jobs it missed on startup.
3. **A missed daily job (cause 4)** — a restart before its cut-off runs it
   late (end-of-day and the core only while the market is open). Otherwise
   run it by hand: `halal-trader research run`, the ledger sync, or wait for
   the next scheduled run.
4. **Database (cause 5)** — `just docker-status`, then
   `docker logs halal-trader-pg`; the bots reconnect once Postgres is back.

## Escalate

The operator is the on-call. A bot that cannot stay up through a session:
`halal-trader halt --reason "..."` and investigate outside market hours.

---

_Last reviewed: 2026-10-06_
