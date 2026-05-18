# Halal Trader

LLM-powered halal trading bot for **stocks** (Alpaca) and **crypto** (Binance). Uses Zoya API for stock Shariah compliance and a CoinGecko-based rule engine for crypto halal screening.

> **Disclaimer**: This project uses paper/testnet accounts with simulated funds. It is an experiment to test whether LLMs can serve as a direct execution layer for trading decisions. **Do not use with real money.**

## Features

### Stock Trading
- **LLM-driven trading decisions** -- configurable to use Ollama (local), OpenAI, or Anthropic
- **Halal stock filtering** -- screens stocks via Zoya API before trading
- **Day trading strategy** -- targets 1%+ daily returns, closes all positions by market close
- **Alpaca MCP integration** -- executes trades via Alpaca's official MCP server
- **Multi-stage prompt context** -- regime detection, multi-timeframe alignment (1h/1d/1w), ML anomaly + signal classifier, portfolio risk, scheduled-release catalysts (FRED/EDGAR/Fed-speak/options-IV), and Alpaca news all flow into the LLM
- **Position monitor** -- 30-second SL/TP enforcement between 15-min cycles; trailing-stop ratchet and post-close fan-out (drift / thesis / regret / RAG / purification)
- **Shadow runner** -- optional frozen-prompt parallel strategy that observes each cycle and writes a divergence ledger
- **Ensemble + adversarial** -- optional fan-out to N additional LLMs voting on each plan, plus an attacker LLM that critiques and downsizes weak buys

### Crypto Trading
- **24/7 LLM-driven cycles** -- configurable cadence (default 60s) on Binance
- **Binance testnet support** -- develop and test risk-free, flip one flag to go live
- **Technical indicators** -- RSI, MACD, Bollinger Bands, EMA, VWAP, ATR computed per cycle and fed to the LLM
- **Real-time WebSocket data** -- streams 1-minute klines for low-latency market reads
- **Dynamic halal screening** -- CoinGecko-based rule engine inspired by Mufti Faraz Adam's Crypto Shariah Screening Framework (category, token type, legitimacy, utility filters)
- **Agentic mode** -- bounded-budget tool-calling loop where the LLM can fetch deeper analysis (RAG over past trades, regime memory, VaR) before submitting a plan
- **ML stack** -- IsolationForest anomaly detection, signal classifier, Chronos-T5 price forecaster, plus a retraining loop labeled by closed-trade outcomes
- **Sentiment + news** -- CryptoPanic feed with emergency mini-cycles on high-impact events; Reddit mention velocity for surge detection
- **Portfolio risk engine** -- correlation, heat, drawdown, ATR-baselined sizing

### Shared
- **Full audit trail** -- every LLM decision (with prompt version, token counts, cost) and trade execution lands in Postgres
- **Configurable LLM backend** -- Ollama, OpenAI, or Anthropic for both stock and crypto, with provider fallback chains
- **Halal exception queue** -- operator-managed override workflow for borderline assets, persisted to the same database
- **Live dashboard + WebSocket stream** -- React SPA on `:8082` plus a `/ws/cycle` event stream for real-time cycle observability

## In a hurry?

Read [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — it gets you from a
fresh clone to a paper-money trade running on Binance testnet in
about 10 minutes, with no real funds at risk.

Writing a custom strategy? Read
[`docs/STRATEGY_AUTHORING.md`](docs/STRATEGY_AUTHORING.md) — it
walks through the `BaseStrategy` contract, the prompt-version
registry, the four test harnesses (unit / stress / scenario /
A/B comparator), and ships a fully-worked RSI mean-reversion
example.

Reporting a security issue? See [`SECURITY.md`](SECURITY.md) for
the threat model, secure defaults, and the disclosure address.
Please don't file public issues for vulnerabilities.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) (for local LLM inference)
- Alpaca paper trading account ([sign up free](https://app.alpaca.markets/paper/dashboard/overview)) -- for stocks
- Binance testnet account ([testnet.binance.vision](https://testnet.binance.vision)) -- for crypto
- Zoya API key ([developer.zoya.finance](https://developer.zoya.finance)) -- optional, for stock halal screening

## Setup

```bash
# Clone and enter the project
cd trading

# Install dependencies
uv sync

# Copy env template and fill in your API keys
cp .env.example .env
# Edit .env with your keys (Alpaca, Binance, LLM, etc.)

# Pull an Ollama model (if using local inference)
ollama pull qwen2.5:32b

# Install Alpaca MCP server (for stock trading only)
uvx alpaca-mcp-server init
```

## Usage

### Stock Trading

```bash
# Start the stock trading bot (scheduled during market hours)
halal-trader start

# Run a single trading cycle then exit
halal-trader start --once

# Check current status (account, positions, market clock)
halal-trader status

# View stock trade history and daily P&L
halal-trader history

# Show current configuration
halal-trader config
```

### Crypto Trading

```bash
# Start the 24/7 crypto trading bot (1-minute cycles)
halal-trader crypto start

# Run a single crypto cycle then exit
halal-trader crypto start --once

# Show Binance account, balances, and live prices
halal-trader crypto status

# View crypto trade history and daily P&L
halal-trader crypto history

# Run halal screening and show compliant tokens
halal-trader crypto screen
```

## Architecture

```
                        ┌─────────────────────────────────────────────────┐
                        │               LLM Agent                        │
                        │        (Ollama / OpenAI / Anthropic)           │
                        └──────────┬────────────────────┬────────────────┘
                                   │                    │
                    ┌──────────────▼──────┐  ┌──────────▼──────────────┐
                    │   Stock Strategy    │  │   Crypto Strategy       │
                    │  (15-min cycles)    │  │  (1-min cycles + indicators)│
                    └──────────┬──────────┘  └──────────┬──────────────┘
                               │                        │
                    ┌──────────▼──────┐  ┌──────────────▼──────────────┐
                    │  Zoya Screener  │  │  CoinGecko Halal Screener  │
                    │  (stock halal)  │  │  (crypto halal)            │
                    └──────────┬──────┘  └──────────────┬──────────────┘
                               │                        │
                    ┌──────────▼──────┐  ┌──────────────▼──────────────┐
                    │  Alpaca MCP     │  │  Binance (REST + WebSocket) │
                    │  (paper trade)  │  │  (testnet / production)     │
                    └──────────┬──────┘  └──────────────┬──────────────┘
                               │                        │
                               └────────────┬───────────┘
                                            │
                                 ┌──────────▼──────────────┐
                                 │   Postgres + pgvector    │
                                 │   (trades, P&L, halal    │
                                 │    cache, LLM audit log, │
                                 │    RAG over rationales,  │
                                 │    ML artefacts, regime  │
                                 │    memory, drift state)  │
                                 └──────────────────────────┘
```

## Configuration

All settings are managed via `.env` file or environment variables. See `.env.example` for the full list.

### Shared Settings

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM backend: `ollama`, `openai`, `anthropic` | `ollama` |
| `LLM_MODEL` | Model name | `qwen2.5:32b` |
| `DATABASE_URL` | Postgres async DSN (asyncpg) | `postgresql+asyncpg://halal:halal@localhost:5433/halal_trader` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Stock Trading

| Variable | Description | Default |
|---|---|---|
| `ALPACA_API_KEY` | Alpaca API key | |
| `ALPACA_SECRET_KEY` | Alpaca secret key | |
| `ALPACA_PAPER_TRADE` | Use paper trading | `true` |
| `TRADING_INTERVAL_MINUTES` | Minutes between analysis cycles | `15` |
| `DAILY_RETURN_TARGET` | Target daily return (decimal) | `0.01` |
| `MAX_POSITION_PCT` | Max portfolio % per position | `0.20` |
| `DAILY_LOSS_LIMIT` | Max daily loss before stopping | `0.02` |

### Crypto Trading

| Variable | Description | Default |
|---|---|---|
| `BINANCE_API_KEY` | Binance API key | |
| `BINANCE_SECRET_KEY` | Binance secret key | |
| `BINANCE_TESTNET` | Use Binance testnet | `true` |
| `CRYPTO_TRADING_INTERVAL_SECONDS` | Seconds between crypto cycles | `60` |
| `CRYPTO_PAIRS` | Trading pairs to monitor | `["BTCUSDT","ETHUSDT","SOLUSDT","ADAUSDT"]` |
| `CRYPTO_DAILY_RETURN_TARGET` | Target daily return | `0.01` |
| `CRYPTO_MAX_POSITION_PCT` | Max portfolio % per position | `0.25` |
| `CRYPTO_DAILY_LOSS_LIMIT` | Max daily loss before stopping | `0.03` |
| `CRYPTO_MIN_MARKET_CAP` | Min market cap for halal screening | `1000000000` |

## License

MIT
