# Quickstart — your first paper trade in 10 minutes

This guide takes you from a fresh clone to watching the bot place its
first paper-money trade. The whole path is paper / testnet — there is
no way to lose real money following these steps.

If you can run `git`, `docker`, and `uv`, you are ten minutes from a
running bot.

## What you will have at the end

- The Postgres + pgvector container running on `localhost:5433`.
- A trading bot connected to **Binance testnet** (free, no KYC, fake
  money) running 60-second crypto cycles.
- A live dashboard at `http://localhost:8082` showing the bot's
  decisions, positions, and halal-compliance receipts.
- Every decision the LLM makes — full prompt, response, cost — written
  to Postgres so you can replay or audit any cycle later.

## Prerequisites (≈ 2 minutes)

Make sure the following are installed *before* you start the timer.
None of them require an account.

| Tool | Version | Why | Install |
|------|---------|-----|---------|
| **Python** | 3.14+ | runtime | <https://www.python.org/downloads/> |
| **`uv`** | any recent | package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Docker** | any recent | runs Postgres | <https://docs.docker.com/get-docker/> |
| **`just`** | any recent | task runner (optional but assumed) | `cargo install just` or your package manager |
| **`git`** | any recent | clone | <https://git-scm.com/downloads> |

You also need an **OpenRouter API key** for the LLM. The bot speaks
to a single model — GLM-5.2 — over OpenAI-compatible endpoints, and
OpenRouter is the default host.

| LLM | Cost | Setup |
|-----|------|-------|
| **GLM-5.2** (via OpenRouter) | pay-per-token | grab a key at <https://openrouter.ai/keys> |

## Step 1 — clone + install (≈ 1 minute)

```bash
git clone https://github.com/your-org/halal-trader.git
cd halal-trader
just install        # uv sync — installs the core deps
```

This pulls down the runtime dependencies only. The `[ml]`,
`[sentiment]`, and `[dashboard]` extras are optional; we'll add the
dashboard one in Step 5.

## Step 2 — get a Binance testnet account (≈ 2 minutes)

1. Go to <https://testnet.binance.vision/> and click **Log In with
   GitHub**. No KYC, no email verification.
2. After login, click the key icon in the top-right → **Generate
   HMAC_SHA256 Key**. Copy the **API Key** and **Secret Key** that
   appear once — Binance will not show the secret again.
3. The testnet starts you with **2 BTC + 200,000 USDT** of fake money.
   That's plenty.

> Want stocks instead? Sign up for a free **Alpaca paper account** at
> <https://app.alpaca.markets/paper/dashboard/overview> and use the
> paper-account API key + secret. The crypto path below is the
> shorter one to a first trade because crypto markets are 24/7;
> stocks only trade during US market hours.

## Step 3 — fill in `.env` (≈ 1 minute)

```bash
cp .env.example .env
```

Open `.env` in your editor and set **only these four lines**. Every
other variable can stay at its default for the quickstart.

```bash
# Your Binance testnet keys from Step 2
BINANCE_API_KEY=<paste here>
BINANCE_SECRET_KEY=<paste here>
BINANCE_TESTNET=true

# Your OpenRouter key from https://openrouter.ai/keys
GLM_API_KEY=<paste here>
```

That's the whole LLM setup — the defaults (`GLM_BASE_URL=https://openrouter.ai/api/v1`,
`LLM_MODEL=z-ai/glm-5.2`) do the rest.

Optionally, add a **second GLM endpoint** as a fallback — the bot
rotates to it automatically if the primary goes down (e.g. Z.ai
direct; note the model id differs there):

```bash
# Optional — Z.ai-direct fallback endpoint
GLM_FALLBACK_BASE_URL=https://api.z.ai/api/paas/v4
GLM_FALLBACK_MODEL=glm-5.2
GLM_FALLBACK_API_KEY=<Z.ai key>
```

The `DATABASE_URL`, halal-screening settings, and notification webhooks
all have sensible defaults baked into `.env.example` — leave them
alone for now.

## Step 4 — start Postgres + apply migrations (≈ 1 minute)

```bash
just pg-up                 # docker compose up postgres on :5433
uv run halal-trader db migrate     # applies every Alembic revision
```

`db migrate` is idempotent — running it twice is safe. The bot
**refuses to start** if the database isn't at the head migration, so
this step is non-optional.

## Step 5 — run the bot for a single cycle (≈ 1 minute)

The fastest way to see a real trade decision is the `--once` flag:
the bot runs exactly one cycle and exits. You see every step of the
LLM's reasoning streamed to your terminal.

```bash
just crypto-once
```

Expected output (abridged):

```
[INFO] cycle.start cycle_id=2026-05-01T14:02:11Z
[INFO] halal.refresh pairs=14 source=coingecko
[INFO] indicators.computed BTCUSDT rsi=52.1 macd=+0.0014 vol_ratio=0.8x
[INFO] llm.call provider=glm model=z-ai/glm-5.2 tokens_in=4231 tokens_out=512
[INFO] decision pair=BTCUSDT side=BUY confidence=0.62 size_usd=120.00
[INFO] order.submitted symbol=BTCUSDT side=BUY qty=0.00200
[INFO] order.filled fill_price=$59,842.10 filled_at=2026-05-01T14:02:14Z
[INFO] cycle.complete cycle_id=2026-05-01T14:02:11Z duration_s=4.7
```

If you see `decision side=HOLD` for every pair on your first cycle —
that is **fine**. The LLM is being cautious. Run `just crypto-once`
again, or just go to Step 6 and let the long-running loop find an
entry.

## Step 6 — watch the bot live in the dashboard (≈ 2 minutes)

```bash
# In a new terminal:
cd dashboard && npm install && npm run build && cd ..
just dashboard            # serves the SPA + API on :8082
```

Open <http://localhost:8082> in your browser. You'll see:

- **Positions** — every open trade, current P&L, SL/TP levels.
- **Decisions** — the most recent LLM calls with their full prompt
  and response, costs, and which model produced them.
- **Halal compliance** — the AAOIFI summary tile (debt %, financial
  income %, purification outstanding) so you can verify every trade
  is Shariah-compliant.
- **Live cycle WebSocket** — the heartbeat ticks each time a cycle
  starts and ends.

## Step 7 — start the long-running loop

```bash
just crypto                # 24/7 cycles, default 60s cadence
```

That's it. The bot is now trading. Every entry / exit / halt-trip
shows up in the dashboard in real time, and every decision is
permanent in Postgres for later replay.

## What to do next

| If you want to … | Read … |
|------------------|--------|
| Understand the architecture end-to-end | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) |
| See the multi-broker / paper-broker contract | `src/halal_trader/brokers/` |
| Configure Slack / Discord / email alerts | the relevant `notifications/*.py` modules + `.env.example` |
| Trade stocks instead of crypto | swap Step 5 for `just stocks-once`; needs `ALPACA_*` keys |
| Engage the kill switch in a hurry | `halal-trader halt --reason "manual"` |
| Tune the strategy | `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, `MAX_POSITION_PCT` in `.env` |
| See the full roadmap | [`docs/round4_roadmap.md`](round4_roadmap.md) |

## Troubleshooting

**`SchemaError: db is at revision <X> but head is <Y>`** — run
`uv run halal-trader db migrate` again. If you joined the project
when it already had a non-Alembic database, run `halal-trader db stamp
head` once to adopt it.

**`OperationalError: could not connect to server` on port 5433** —
the Postgres container isn't running. `just pg-up` to start it,
`docker ps` to confirm.

**LLM call fails with `401`** — bad or missing `GLM_API_KEY`. Check
the key at <https://openrouter.ai/keys> and that `.env` was loaded.

**LLM call fails with `402`** — your OpenRouter account is out of
credits. Top up at <https://openrouter.ai/credits>.

**LLM call fails with `429`** — rate limit. Nothing to do:
`FallbackLLM` backs off automatically (60s → 30min) and rotates to
the fallback endpoint if you configured one.

**Halal screener says everything is non-compliant** — the CoinGecko
free tier is rate-limited; first-cycle requests can be throttled. The
screener caches for 6 hours, so a re-run will succeed. Set
`COINGECKO_API_KEY` for higher limits if it persists.

**Binance error code -1013** — the order quantity is below the
minimum lot size for that pair. The bot's executor catches this and
treats it as a rejection (not a circuit-breaker trip); just wait for
the next cycle.

**The bot says "halt is engaged" and refuses to trade** — run
`halal-trader halt-status` to see the reason. `halal-trader resume`
clears it. Causes include: the daily LLM USD cap was hit
(`LLM_DAILY_USD_CAP`), the LLM circuit breaker tripped, or someone
ran `halt --reason "..."` manually.

If you hit something not covered here, file an issue with the
contents of `logs/error.log` (rotated automatically by the JSON log
sink — see `just logs` for a pretty-printed view).
