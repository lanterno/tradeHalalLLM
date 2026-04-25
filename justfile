_default:
    @just --list

# ── Setup ─────────────────────────────────────────────────

# Install dependencies
install:
    uv sync

# Install with dev + all optional deps
dev:
    uv sync --extra dev --extra all

# ── Crypto bot ────────────────────────────────────────────

# Start 24/7 crypto trading bot
crypto:
    uv run halal-trader crypto start

# Run a single crypto trading cycle
crypto-once:
    uv run halal-trader crypto start --once

# Show Binance account and balances
crypto-status:
    uv run halal-trader crypto status

# Show crypto trade history
crypto-history:
    uv run halal-trader crypto history

# Show crypto performance metrics
crypto-stats:
    uv run halal-trader crypto stats

# Show halal-screened crypto pairs
crypto-screen:
    uv run halal-trader crypto screen

# ── Stock bot ─────────────────────────────────────────────

# Start stock trading bot (market hours)
stocks:
    uv run halal-trader start

# Run a single stock trading cycle
stocks-once:
    uv run halal-trader start --once

# Show Alpaca account and positions
status:
    uv run halal-trader status

# ── Info ──────────────────────────────────────────────────

# Show current configuration
config:
    uv run halal-trader config

# Start web dashboard on :8082
dashboard:
    uv run halal-trader dashboard

# ── Logs ──────────────────────────────────────────────────

# Show last 50 JSON log entries (app only)
logs:
    @tail -50 logs/halal_trader.log 2>/dev/null | \
        python3 -c "import sys,json; [print(f'{d[\"timestamp\"][-12:]} {d[\"level\"]:7s} {d[\"message\"][:120]}') for line in sys.stdin if (d:=json.loads(line)) and d['name'].startswith('halal_trader')]" \
        2>/dev/null || echo "No log file found"

# Follow the log file live (app messages only)
logs-tail:
    @tail -f logs/halal_trader.log 2>/dev/null | \
        python3 -u -c "import sys,json; [print(f'{d[\"timestamp\"][-12:]} {d[\"level\"]:7s} {d[\"message\"][:120]}', flush=True) for line in sys.stdin if (d:=json.loads(line)) and d['name'].startswith('halal_trader')]" \
        2>/dev/null || echo "No log file found"

# Show recent errors
logs-errors:
    @tail -100 logs/error.log 2>/dev/null | \
        python3 -c "import sys,json; [print(f'{d[\"timestamp\"][-12:]} {d[\"message\"][:140]}') for line in sys.stdin if (d:=json.loads(line))]" \
        2>/dev/null || echo "No error log found"

# ── Development ───────────────────────────────────────────

# Run tests
test:
    uv run pytest

# Run linter
lint:
    uv run ruff check src/ tests/

# Auto-format code
format:
    uv run ruff format src/ tests/
    uv run ruff check --fix src/ tests/

# Type-check (domain + core, strict)
typecheck:
    uv run mypy

# Run all pre-commit hooks against every tracked file
precommit:
    uv run pre-commit run --all-files

# Install pre-commit's git hooks into .git/hooks/
precommit-install:
    uv run pre-commit install

# ── Maintenance ───────────────────────────────────────────

# Remove caches and compiled files
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache .ruff_cache

# Delete the SQLite database (⚠ destroys all trade history)
db-reset:
    @echo "This will DELETE halal_trader.db. Press Ctrl+C to cancel."
    @read -p "Are you sure? [y/N] " confirm && [ "$confirm" = "y" ] && rm -f halal_trader.db && echo "Database deleted." || echo "Cancelled."
