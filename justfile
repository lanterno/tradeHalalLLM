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

# Start 24/7 crypto trading bot (caffeinate -i = no idle sleep / App Nap)
crypto:
    caffeinate -dimsu uv run halal-trader crypto start

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

# Start stock trading bot (caffeinate -i = no idle sleep / App Nap)
stocks:
    caffeinate -dimsu uv run halal-trader start

# Run a single stock trading cycle
stocks-once:
    uv run halal-trader start --once

# Show Alpaca account and positions
status:
    uv run halal-trader status

# ── launchd (macOS auto-start + auto-restart) ─────────────

# Install stocks + watchdog only (default; enable crypto separately when ready)
launchd-install:
    @mkdir -p ~/Library/LaunchAgents logs
    cp infra/launchd/com.halabot.stocks.plist ~/Library/LaunchAgents/
    cp infra/launchd/com.halabot.watchdog.plist ~/Library/LaunchAgents/
    -launchctl bootout "gui/$(id -u)/com.halabot.stocks" 2>/dev/null
    -launchctl bootout "gui/$(id -u)/com.halabot.watchdog" 2>/dev/null
    launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.halabot.stocks.plist
    launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.halabot.watchdog.plist
    @echo "Installed stocks + watchdog. (Crypto stays disabled — run \`just launchd-enable-crypto\` to turn it on.)"

# Install ALL three agents (stocks + crypto + watchdog) — needs Binance creds in .env
launchd-install-all:
    @mkdir -p ~/Library/LaunchAgents logs
    cp infra/launchd/com.halabot.stocks.plist ~/Library/LaunchAgents/
    cp infra/launchd/com.halabot.crypto.plist ~/Library/LaunchAgents/
    cp infra/launchd/com.halabot.watchdog.plist ~/Library/LaunchAgents/
    -launchctl bootout "gui/$(id -u)/com.halabot.stocks" 2>/dev/null
    -launchctl bootout "gui/$(id -u)/com.halabot.crypto" 2>/dev/null
    -launchctl bootout "gui/$(id -u)/com.halabot.watchdog" 2>/dev/null
    launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.halabot.stocks.plist
    launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.halabot.crypto.plist
    launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.halabot.watchdog.plist
    @echo "Installed stocks + crypto + watchdog."

# Enable + start the crypto agent (needs Binance creds in .env)
launchd-enable-crypto:
    cp infra/launchd/com.halabot.crypto.plist ~/Library/LaunchAgents/
    -launchctl bootout "gui/$(id -u)/com.halabot.crypto" 2>/dev/null
    launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.halabot.crypto.plist
    @echo "Crypto agent enabled."

# Bootout the crypto agent (keeps plist on disk — re-enable with launchd-enable-crypto)
launchd-disable-crypto:
    -launchctl bootout "gui/$(id -u)/com.halabot.crypto"
    @echo "Crypto agent disabled. Plist stays at ~/Library/LaunchAgents/com.halabot.crypto.plist."

# Remove the launchd agents
launchd-uninstall:
    -launchctl bootout "gui/$(id -u)/com.halabot.stocks"
    -launchctl bootout "gui/$(id -u)/com.halabot.crypto"
    -launchctl bootout "gui/$(id -u)/com.halabot.watchdog"
    rm -f ~/Library/LaunchAgents/com.halabot.stocks.plist
    rm -f ~/Library/LaunchAgents/com.halabot.crypto.plist
    rm -f ~/Library/LaunchAgents/com.halabot.watchdog.plist
    @echo "Removed."

# Restart just the stocks agent
launchd-restart-stocks:
    launchctl kickstart -k "gui/$(id -u)/com.halabot.stocks"

# Restart just the crypto agent
launchd-restart-crypto:
    launchctl kickstart -k "gui/$(id -u)/com.halabot.crypto"

# Show launchd agent status + pids
launchd-status:
    @launchctl print "gui/$(id -u)/com.halabot.stocks" 2>/dev/null | grep -E 'state|pid|last exit' || echo "stocks: not loaded"
    @launchctl print "gui/$(id -u)/com.halabot.crypto" 2>/dev/null | grep -E 'state|pid|last exit' || echo "crypto: not loaded"
    @launchctl print "gui/$(id -u)/com.halabot.watchdog" 2>/dev/null | grep -E 'state|pid|last exit' || echo "watchdog: not loaded"

# Run the dead-man-switch watchdog once (smoke test)
watchdog:
    uv run halal-trader watchdog --any-time --dry-run

# ── Info ──────────────────────────────────────────────────

# Show current configuration
config:
    uv run halal-trader config

# Start web dashboard on :8082
dashboard:
    uv run halal-trader dashboard

# Install dashboard frontend deps (one-time, before -build / -dev / -lint)
dashboard-install:
    cd dashboard && npm install

# Build the React SPA (production)
dashboard-build:
    cd dashboard && npm run build

# Start the Vite dev server with HMR
dashboard-dev:
    cd dashboard && npm run dev

# Lint the React SPA
dashboard-lint:
    cd dashboard && npm run lint

# ── Logs ──────────────────────────────────────────────────

# Show last 50 JSON log entries (app only)
logs:
    @tail -50 logs/halal_trader.log 2>/dev/null | python3 scripts/format_logs.py 2>/dev/null \
        || echo "No log file found"

# Follow the log file live (app messages only)
logs-tail:
    @tail -f logs/halal_trader.log 2>/dev/null | python3 -u scripts/format_logs.py 2>/dev/null \
        || echo "No log file found"

# Show recent errors
logs-errors:
    @tail -100 logs/error.log 2>/dev/null | python3 scripts/format_logs.py --errors 2>/dev/null \
        || echo "No error log found"

# ── Development ───────────────────────────────────────────

# Run tests
test:
    uv run pytest

# Pre-deploy gate — stress harness over the standard scenarios
stress:
    uv run halal-trader insights stress

# Run the full pre-deploy gate suite (lint + tests + stress harness)
predeploy: lint test stress

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

# Drop + recreate the Postgres database (⚠ destroys all trade history)
db-reset:
    @echo "This will DROP DATABASE halal_trader. Press Ctrl+C to cancel."
    @read -p "Are you sure? [y/N] " confirm && [ "$confirm" = "y" ] && \
        docker exec halal-trader-pg psql -U trader -d postgres -c 'DROP DATABASE IF EXISTS halal_trader' && \
        docker exec halal-trader-pg psql -U trader -d postgres -c 'CREATE DATABASE halal_trader' && \
        uv run halal-trader db migrate && \
        echo "Database reset and migrated." || echo "Cancelled."

# Bring up the Postgres + pgvector container (localhost:5433)
pg-up:
    cd infra && docker compose up -d postgres

# Stop the Postgres container (data persists in the named volume)
pg-down:
    cd infra && docker compose stop postgres

# Drop + recreate the test database (run before pytest if it gets corrupted)
test-db-reset:
    docker exec halal-trader-pg psql -U trader -d postgres -c 'DROP DATABASE IF EXISTS halal_trader_test'

# ── Full Docker stack (postgres + bots + web in containers) ──

# Build the bot image (multi-stage: deps + venv → slim runtime)
docker-build:
    docker compose -f infra/docker-compose.yml build

# Start everything (postgres + crypto + stocks + web) in the background
docker-up:
    docker compose -f infra/docker-compose.yml up -d

# Stop and remove all containers (data volumes persist)
docker-down:
    docker compose -f infra/docker-compose.yml down

# Rebuild from scratch and recreate all containers (picks up .env + code changes)
docker-rebuild:
    docker compose -f infra/docker-compose.yml build
    docker compose -f infra/docker-compose.yml up -d --force-recreate

# Tail logs from a single service (default: crypto). Usage: just docker-logs [service]
docker-logs service="trader-crypto":
    docker compose -f infra/docker-compose.yml logs -f --tail=50 {{service}}

# Tail logs from every service interleaved
docker-logs-all:
    docker compose -f infra/docker-compose.yml logs -f --tail=20

# Apply Alembic migrations inside the running stack
docker-migrate:
    docker compose -f infra/docker-compose.yml run --rm trader-crypto halal-trader db migrate

# Open a psql shell against the containerised Postgres
docker-psql:
    docker compose -f infra/docker-compose.yml exec postgres psql -U trader halal_trader

# Quick health check on every service + the web API
docker-status:
    @docker compose -f infra/docker-compose.yml ps
    @echo "---"
    @curl -s -o /dev/null -w "Web /api/health → HTTP %{http_code}\n" http://localhost:8082/api/health
