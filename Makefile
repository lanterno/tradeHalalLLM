.PHONY: help install dev crypto crypto-once crypto-status crypto-history \
       stocks stocks-once status config logs logs-tail logs-errors \
       test lint format clean db-reset

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────

install: ## Install dependencies
	uv sync

dev: ## Install with dev + all optional deps
	uv sync --extra dev --extra all

# ── Crypto bot ────────────────────────────────────────────

crypto: ## Start 24/7 crypto trading bot
	uv run halal-trader crypto start

crypto-once: ## Run a single crypto trading cycle
	uv run halal-trader crypto start --once

crypto-status: ## Show Binance account and balances
	uv run halal-trader crypto status

crypto-history: ## Show crypto trade history
	uv run halal-trader crypto history

crypto-stats: ## Show crypto performance metrics
	uv run halal-trader crypto stats

crypto-screen: ## Show halal-screened crypto pairs
	uv run halal-trader crypto screen

# ── Stock bot ─────────────────────────────────────────────

stocks: ## Start stock trading bot (market hours)
	uv run halal-trader start

stocks-once: ## Run a single stock trading cycle
	uv run halal-trader start --once

status: ## Show Alpaca account and positions
	uv run halal-trader status

# ── Info ──────────────────────────────────────────────────

config: ## Show current configuration
	uv run halal-trader config

dashboard: ## Start web dashboard on :8082
	uv run halal-trader dashboard

# ── Logs ──────────────────────────────────────────────────

logs: ## Show last 50 JSON log entries (app only)
	@tail -50 logs/halal_trader.log 2>/dev/null | \
		python3 -c "import sys,json; [print(f'{d[\"timestamp\"][-12:]} {d[\"level\"]:7s} {d[\"message\"][:120]}') for line in sys.stdin if (d:=json.loads(line)) and d['name'].startswith('halal_trader')]" \
		2>/dev/null || echo "No log file found"

logs-tail: ## Follow the log file live (app messages only)
	@tail -f logs/halal_trader.log 2>/dev/null | \
		python3 -u -c "import sys,json; [print(f'{d[\"timestamp\"][-12:]} {d[\"level\"]:7s} {d[\"message\"][:120]}', flush=True) for line in sys.stdin if (d:=json.loads(line)) and d['name'].startswith('halal_trader')]" \
		2>/dev/null || echo "No log file found"

logs-errors: ## Show recent errors
	@tail -100 logs/error.log 2>/dev/null | \
		python3 -c "import sys,json; [print(f'{d[\"timestamp\"][-12:]} {d[\"message\"][:140]}') for line in sys.stdin if (d:=json.loads(line))]" \
		2>/dev/null || echo "No error log found"

# ── Development ───────────────────────────────────────────

test: ## Run tests
	uv run pytest

lint: ## Run linter
	uv run ruff check src/ tests/

format: ## Auto-format code
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

# ── Maintenance ───────────────────────────────────────────

clean: ## Remove caches and compiled files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

db-reset: ## Delete the SQLite database (⚠ destroys all trade history)
	@echo "This will DELETE halal_trader.db. Press Ctrl+C to cancel."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -f halal_trader.db && echo "Database deleted." || echo "Cancelled."
