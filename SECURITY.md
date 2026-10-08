# Security policy

What this project defends, how, and what it does not. It describes the
code as it is; if the two disagree, the code wins and this file is a bug.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting ("Report a vulnerability" on
the repository's Security tab). Please do not open a public issue. Include
the commit you tested, a minimal reproducer and your assessment of the
impact. This is a one-person project: expect an acknowledgement within a
week, and no bug bounty.

## What the system is

A single-operator trading system on one host: a stock bot, a shadow
research engine that never trades, and a dashboard, as three containers
sharing one Postgres database (`infra/docker-compose.yml`). Two Alpaca
accounts:

* the **day-trader's** (an LLM strategy, retired by default:
  `DAY_TRADER_ENABLED=false`), paper unless `ALPACA_PAPER_TRADE=false`;
* the **core portfolio's** (rule-based, monthly-rebalanced strict-halal
  holdings), paper unless `CORE_PAPER=false`.

## Assets, and how each is defended

| Asset | Defence |
|---|---|
| Broker keys, LLM key, Telegram token | Only in `.env` (gitignored, excluded from the image build context, blocked by a pre-commit hook). Each container receives only the secrets it reads: compose blanks the rest (the dashboard and the shadow never see the core account's keys). Never logged: `httpx`/`httpcore` request logging is silenced because some providers take keys in the query string. The dashboard's settings schema masks secrets and never returns their defaults. The test suite never loads `.env` and blocks outbound network. |
| Real money | Both accounts default to paper. The day-trader refuses to start live without a same-day `LIVE_MODE_CONFIRMATION` token and enforces balance, order-size and loss ceilings (`core/safeguards.py`). The core goes live only when the operator puts live keys in place and sets `CORE_PAPER=false`; its readiness gate (`portfolio/readiness.py`) advises when that may happen but does not enforce it, and that switch has no daily token. A live account is meant to be a cash account (no margin); the code is long-only regardless. |
| Halal compliance | Enforced at the order boundary, not in prompts. Every BUY -- the day-trader's cycle, the news reactor's entries, the core's rebalance -- is re-checked against the strict in-house screen (`halal_screen_results`: the stricter of AAOIFI and S&P Shariah on every axis) at the moment of the order, and **fails closed** on a stale, missing or unreadable screen (`trading/executor.py`, `halal/strict.py`, `portfolio/core_executor.py`). Long-only: no short action exists and sells are clamped to the held quantity. |
| Control of the bots | A kill-switch in the database (`core/halt.py`; `halal-trader halt`, or the dashboard) that every strategy checks first, the core included; the position monitor still manages exits. |
| The dashboard's mutations | `WEB_API_TOKEN`: every non-GET request needs a matching `X-Trader-Token` header (constant-time compare); an empty token makes the dashboard read-only. Destructive operations also need a confirmation header. The dashboard binds to loopback in the home deployment and is meant to sit behind an authenticating reverse proxy; CORS is closed (the Vite dev server proxies the API). |
| The database | Published on loopback only in the home deployment. `POSTGRES_PASSWORD` in `.env` sets the password every container's `DATABASE_URL` is built from; the default in this repository is public and must be changed (`docs/runbooks/rotate-postgres-password.md`). |
| Spend | A daily and monthly LLM budget shared by every process on the key (`core/llm/spend.py`), which refuses calls once exhausted. |
| Availability | Every long-running component writes a heartbeat; the dashboard's watchdog alerts on Telegram when one goes stale, and container healthchecks mark a hung process unhealthy. |
| Supply chain | Dependencies are locked (`uv.lock`); the broker's MCP server is installed from a frozen, exact requirements file at image build time, never resolved at runtime. CI runs `pip-audit` over the locked dependencies (advisory, non-blocking). |

## Not defended

* **More than one operator.** One shared mutation token, no user
  accounts, no roles, no second factor.
* **A compromised host.** Anyone with the operator's shell, the docker
  socket or the `.env` file has everything. Secrets are plaintext at rest.
* **An exposed dashboard.** Reads are unauthenticated by design (the
  proxy's job); exposing the port directly exposes positions and history.
* **Trust in third parties.** The broker, the LLM provider, news feeds and
  SEC filings are inputs. LLM output is schema-validated and cannot place
  an order the halal and risk gates refuse, but a wrong-but-valid decision
  is a trading risk, not a security boundary.
* **Denial of service.**

## Reporting scope

In scope: reading secrets from the running system, its logs or its
database; any path that places a BUY the strict screen does not pass, a
short sale, or an order while the kill-switch is engaged; reaching a live
account while configured for paper; executing code through the dashboard
or the broker subprocess.

Out of scope: anything needing the operator's machine or `.env`; findings
in third-party services (report upstream); dependency advisories already
flagged by CI.
