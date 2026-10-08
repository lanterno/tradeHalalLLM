# Operator runbooks

What to do when the bot tells you something is wrong.

## Where alerts come from

Two places, both ending in the bot's Telegram chat:

* **The processes themselves**, through `AlertSink.notify(type, details)`
  (`notifications/telegram.py`), rate-limited per type. The table below
  lists every type the code sends.
* **The watchdogs.** Inside the fleet, the web container's watchdog
  (`web/watchdog.py`) alerts when a process or daily job stops beating.
  On the server, `halabot-health.timer` alerts when a container is down or
  the bot's heartbeat is stale, and `halabot-alert@` when the nightly
  backup fails (`docs/DEPLOY.md`).

## Alerts

| Type | Means | Do |
|---|---|---|
| `cycle.failed` | A trading cycle raised | Read the bot's log (`just docker-logs`); a cycle that keeps failing is a bug to fix, not to restart away |
| `llm.failing` | Several strategy LLM calls failed in a row | [chain-backoff.md](chain-backoff.md) |
| `llm.quota_exhausted` | The LLM key is out of credits | Top up the OpenRouter key; cycles resume on their own |
| `reconcile.drift` | The bot's positions differ from the broker's | `halal-trader reconcile check` to see it; read `docs/OPERATOR_CONTEXT.md` before running `reconcile fix-drift`, which is destructive |
| `safeguards.violation` | Live-mode safeguards tripped | Stop and read the details; never override on live money |
| `ledger.sync_failed` / `ledger.fill_drift` | The broker-ledger sync failed, or the bot's fills differ from Alpaca's | `halal-trader ledger sync` / `ledger reconcile --day <day>` |
| `research.failed` | A step of the evening research run failed | The message names the step; `halal-trader books run` reruns it |
| `recommendation.failed` | The daily pick could not be made | Advisory only; it is retried, or run `halal-trader recommend` |
| `core.refused` / `core.failed` | The core portfolio refused to trade (live not armed) or its run raised | Read the message; `halal-trader core plan` shows what it would do |
| A process or job "stopped" | A heartbeat went stale | [watchdog-stopped.md](watchdog-stopped.md) |
| `UNHEALTHY` (server) | `just health` failed twice | `just docker-status`, then [watchdog-stopped.md](watchdog-stopped.md) |

## Operator procedures

| Procedure | When |
|---|---|
| [halt-engaged.md](halt-engaged.md) | The kill-switch is on and you need to decide whether to resume |
| [backups-and-pitr.md](backups-and-pitr.md) | After a database failure, and what the backups cover |
| [rotate-postgres-password.md](rotate-postgres-password.md) | After any suspected leak of the database password |

New runbooks follow `_template.md`.
