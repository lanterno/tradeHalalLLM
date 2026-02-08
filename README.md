# Halal Trader

LLM-powered halal day-trading bot that uses Alpaca's MCP server for stock trading execution and Zoya API for Shariah compliance screening.

> **Disclaimer**: This project uses Alpaca's paper trading account with simulated funds. It is an experiment to test whether LLMs can serve as a direct execution layer for trading decisions. **Do not use with real money.**

## Features

- **LLM-driven trading decisions** -- configurable to use Ollama (local), OpenAI, or Anthropic
- **Halal stock filtering** -- screens stocks via Zoya API before trading
- **Day trading strategy** -- targets 1%+ daily returns, closes all positions by market close
- **Alpaca MCP integration** -- executes trades via Alpaca's official MCP server
- **Full audit trail** -- logs every LLM decision and trade execution to SQLite

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) (for local LLM inference)
- Alpaca paper trading account ([sign up free](https://app.alpaca.markets/paper/dashboard/overview))
- Zoya API key ([developer.zoya.finance](https://developer.zoya.finance))

## Setup

```bash
# Clone and enter the project
cd trading

# Install dependencies
uv sync

# Copy env template and fill in your API keys
cp .env.example .env
# Edit .env with your keys

# Pull an Ollama model (if using local inference)
ollama pull qwen2.5:32b

# Install Alpaca MCP server
uvx alpaca-mcp-server init
```

## Usage

```bash
# Start the trading bot
halal-trader start

# Check current status (positions, P&L)
halal-trader status

# View trade history
halal-trader history

# Show configuration
halal-trader config
```

## Architecture

```
LLM Agent (Ollama/Cloud) --> Halal Filter (Zoya) --> Trading Decisions
                                                             |
                                                             v
                    CLI Dashboard (Rich) <-- P&L Tracker <-- Alpaca MCP Server
```

## Configuration

All settings are managed via `.env` file or environment variables:

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM backend: `ollama`, `openai`, `anthropic` | `ollama` |
| `LLM_MODEL` | Model name | `qwen2.5:32b` |
| `TRADING_INTERVAL_MINUTES` | Minutes between analysis cycles | `15` |
| `DAILY_RETURN_TARGET` | Target daily return (decimal) | `0.01` |
| `MAX_POSITION_PCT` | Max portfolio % per position | `0.20` |
| `DAILY_LOSS_LIMIT` | Max daily loss before stopping (decimal) | `0.02` |

## License

MIT
