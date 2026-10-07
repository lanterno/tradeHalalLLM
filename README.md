# Halal Trader

An LLM-assisted, halal-screened trading system for **US stocks** on Alpaca,
plus the research platform used to decide which strategies deserve capital.

> **Paper trading only.** Everything runs against an Alpaca paper account.
> Moving any strategy to real money is gated on a written, evidence-based
> process (a long paper-forward record, statistical tests net of costs,
> clean broker reconciliation). **Do not point this at a live account.**

## What it does

- **Trades** a halal-screened universe with one GLM-5.2 decision per
  15-minute cycle, news-driven "fast in, slow out" momentum entries, and a
  position monitor that enforces stops and trailing stops between cycles.
- **Keeps halal compliance in the mechanism**: every buy is checked against
  the screen at the order boundary and refused if the screen cannot be
  read; long-only by construction; a kill-switch the bot checks first.
- **Keeps honest books**: Alpaca's own fills and equity are the ledger; the
  bot's records are reconciled to them daily, and performance is measured
  from the broker's numbers.
- **Researches**: ten years of daily market data, an in-house AAOIFI-style
  screen from SEC filings validated against halal ETFs' holdings, and
  backtests of candidate strategies against SPUS and HLAL.
- **Shadows**: `src/halabot`, a belief-based engine, runs alongside and
  records what it would have done, so it can be compared with the live bot.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the pieces fit.

## Requirements

- Python 3.14 and [uv](https://docs.astral.sh/uv/); Docker for the deployed
  fleet; Node 22 to build the dashboard.
- An [Alpaca](https://alpaca.markets) paper account (trading and free market
  data).
- An [OpenRouter](https://openrouter.ai) key for GLM-5.2.
- Optional: Finnhub (news), FRED (macro calendar); an `EDGAR_USER_AGENT`
  with a contact e-mail for SEC data.

## Running it

```bash
cp .env.example .env          # fill in the keys; see the comments in the file
just build                    # the image (bot, shadow engine and dashboard)
just up                       # postgres, migrations, the bot, the shadow engine, the dashboard
just health                   # exits 0 only if the bot's heartbeat is fresh
just docker-logs              # follow the bot
```

On a server, [docs/DEPLOY.md](docs/DEPLOY.md) is the whole procedure:
bootstrap, secrets, off-site backups, health alerts, the dashboard over
Tailscale.

The dashboard listens on `127.0.0.1:8082`. Its only control is the
kill-switch; operations go through the CLI:

```bash
halal-trader halt --reason "..."    # stop new entries (exits keep working)
halal-trader resume
halal-trader ledger sync | reconcile | performance
halal-trader data backfill          # research market data
halal-trader compliance screen      # in-house halal screen
halal-trader compliance validate    # ... compared with SPUS/HLAL holdings
halal-trader research factor-backtest
```

## Developing

```bash
just dev            # uv sync with dev + all extras
just test           # pytest (needs the Postgres from `just pg-up`)
just lint typecheck # ruff + strict mypy
just precommit      # every hook; also runs on each commit
```

CI runs every check on every push. `tests/invariants/` pins the safety
rules where they are wired; read it before changing the order path.

## License

MIT
