_default:
    @just --list

# ── Setup ─────────────────────────────────────────────────

# Install dependencies
install:
    uv sync

# Install with dev + all optional deps
dev:
    uv sync --extra dev --extra all

# ── Stock bot ─────────────────────────────────────────────

# Start the stock trading bot in the foreground (the deployed bot runs in
# docker via home-up; use this only with the fleet's trader-stocks stopped --
# two bots on one account will fight)
stocks:
    uv run halal-trader start

# Show Alpaca account, positions and market clock
status:
    uv run halal-trader status

# Run one pre-market check and one trading cycle, then exit (refuses to run
# while another bot holds the lock)
stocks-once:
    uv run halal-trader start --once

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

# Every compose recipe in this file goes through this one command. The
# project name is pinned to `infra` (that is what keeps the live
# infra_pg-data volume attached) and infra/compose.home.yml is always layered
# on: dashboard on 127.0.0.1:6010, Postgres on 127.0.0.1:5433. A recipe that
# used the base file alone would recreate the running containers with both
# ports published on every interface.
home_compose := "docker compose -p infra -f infra/docker-compose.yml -f infra/compose.home.yml"

# Bring up the Postgres + pgvector container (127.0.0.1:5433)
pg-up:
    {{home_compose}} up -d postgres

# Stop the Postgres container (data persists in the named volume)
pg-down:
    {{home_compose}} stop postgres

# Dump the whole DB (schema + data + alembic version) to ./halabot-db.dump
# for moving to another machine. Custom format, compressed. The file is
# gitignored — copy it to the new machine, then `just db-restore`.
db-dump file="halabot-db.dump":
    docker exec halal-trader-pg pg_dump -U trader -d halal_trader -Fc -f /tmp/halabot-db.dump
    docker cp halal-trader-pg:/tmp/halabot-db.dump {{file}}
    docker exec halal-trader-pg rm -f /tmp/halabot-db.dump
    @echo "Wrote {{file}} ($(du -h {{file}} | cut -f1)). Copy it to the new machine, then: just db-restore {{file}}"

# Restore a db-dump into the local Postgres (run `just pg-up` FIRST; do NOT
# run migrate — the dump carries the schema + alembic head). --clean makes
# it safe to re-run over an already-migrated DB. ⚠ overwrites local data.
db-restore file="halabot-db.dump":
    docker cp {{file}} halal-trader-pg:/tmp/halabot-db.dump
    docker exec halal-trader-pg pg_restore -U trader -d halal_trader --clean --if-exists --no-owner /tmp/halabot-db.dump
    docker exec halal-trader-pg rm -f /tmp/halabot-db.dump
    @echo "Restored {{file}}. Verify: halal-trader db current  (should show head)"

# Drop + recreate the test database (run before pytest if it gets corrupted)
test-db-reset:
    docker exec halal-trader-pg psql -U trader -d postgres -c 'DROP DATABASE IF EXISTS halal_trader_test'

# ── Full Docker stack (postgres + bots + web in containers) ──
# The same fleet the home stack runs (see home-* below); these are the
# hands-on verbs. All go through {{home_compose}}.

# Build the bot image (multi-stage: deps + venv → slim runtime)
docker-build:
    {{home_compose}} build

# Start the fleet (postgres + migrate + stocks + shadow + web) in the background
docker-up:
    {{home_compose}} up -d

# Stop and remove all containers (data volumes persist)
docker-down:
    {{home_compose}} down

# Rebuild and recreate all containers (picks up .env + code changes)
docker-rebuild:
    {{home_compose}} build
    {{home_compose}} up -d --force-recreate

# Follow logs from one service (default: stocks). Usage: just docker-logs [service]
docker-logs service="trader-stocks":
    {{home_compose}} logs -f --tail=50 {{service}}

# Follow logs from every service interleaved
docker-logs-all:
    {{home_compose}} logs -f --tail=20

# Apply Alembic migrations inside the running stack
docker-migrate:
    {{home_compose}} run --rm trader-migrate

# Open a psql shell against the containerised Postgres
docker-psql:
    {{home_compose}} exec postgres psql -U trader halal_trader

# Quick health check on every service + the web API
docker-status:
    @{{home_compose}} ps
    @echo "---"
    @curl -s -o /dev/null -w "Web /api/health → HTTP %{http_code}\n" http://127.0.0.1:6010/api/health

# ── Home stack (~/lab/home services.toml) ─────────────────
# The verbs ~/lab/home's services.toml names. Same fleet as docker-up, plus
# infra/compose.home.yml: dashboard on 127.0.0.1:6010, Postgres on loopback.
# home_compose is defined once, above pg-up, and every compose recipe uses it.

# Start the fleet in the background (postgres + migrate + stocks + shadow + web)
home-up:
    {{home_compose}} up -d

# Stop the fleet (containers removed, volumes kept)
home-down:
    {{home_compose}} down

# Rebuild the image the fleet runs
home-build:
    {{home_compose}} build

# Exit 0 only if every long-running container is running AND the API answers
home-health:
    #!/usr/bin/env sh
    for c in halal-trader-pg trader-stocks trader-shadow trader-web; do
        [ "$(docker inspect -f '{{{{.State.Status}}' "$c" 2>/dev/null)" = running ] \
            || { echo "$c not running"; exit 1; }
    done
    # /api/health/bot is 503 unless the bot's process heartbeat (written to
    # the DB every 60 s) is fresh -- a hung or crash-looping bot fails this
    # even while its container reads "running".
    curl -fsS --max-time 3 http://127.0.0.1:6010/api/health/bot >/dev/null

# Last 100 lines from every service, no --follow (for callers that expect it to finish)
home-logs:
    {{home_compose}} logs --tail=100 --no-color

# pg_dump the database into <dest>/halal_trader.dump (restore: just db-restore <file>).
# Rows that can be rebuilt from free sources are left out (their schema is kept):
#   daily_bars, market_assets, monthly_bars, minute_bars -> `halal-trader data backfill`,
#     `data pit-universe`, `events intraday`
#   events, event_facts, event_labels                    -> `events backfill all`,
#     `events extract`, the evening run's labelling
#   eps_facts, annual_fundamentals, etf_holdings         -> `events backfill eps`,
#     `data fundamentals`, `compliance etf-history`
# event_scores goes with events (it references them). What cannot be rebuilt --
# the live reactor's headlines with the time it saw them, and every score
# (the post-cutoff LLM evidence) -- is exported to live_events.jsonl.gz, keyed by
# (source, source_id, symbol), which a rebuilt event store shares.
# Every dump is checked with pg_restore --list; on the 1st of each month it is
# also fully restored into a scratch database (restore-drill). Outcomes land in
# the heartbeats table (backup.nightly, backup.restore_drill) for the digest.
home-backup dest:
    #!/usr/bin/env bash
    set -euo pipefail
    pg() { docker exec halal-trader-pg psql -U trader -d halal_trader -tAq "$@"; }
    excluded=""
    for t in daily_bars market_assets monthly_bars minute_bars events event_facts event_labels event_scores eps_facts annual_fundamentals etf_holdings; do
        excluded="$excluded --exclude-table-data=$t"
    done
    docker exec halal-trader-pg pg_dump -U trader -d halal_trader -Fc $excluded -f /tmp/home-backup.dump
    docker exec halal-trader-pg pg_restore --list /tmp/home-backup.dump > /dev/null
    docker cp halal-trader-pg:/tmp/home-backup.dump "{{dest}}/halal_trader.dump"
    docker exec halal-trader-pg rm -f /tmp/home-backup.dump
    pg -c "SELECT row_to_json(t) FROM (
             SELECT e.source, e.source_id, e.symbol, e.kind, e.published_at, e.seen_at, e.payload,
                    coalesce(json_agg(json_build_object('scorer', s.scorer, 'score', s.score,
                        'tag', s.tag, 'rationale', s.rationale, 'scored_at', s.scored_at))
                        FILTER (WHERE s.id IS NOT NULL), '[]') AS scores
             FROM events e LEFT JOIN event_scores s ON s.event_id = e.id
             WHERE e.payload->>'backfill' IS NULL OR s.id IS NOT NULL
             GROUP BY e.id) t" | gzip > "{{dest}}/live_events.jsonl.gz"
    size=$(du -m "{{dest}}/halal_trader.dump" | cut -f1)
    live=$(du -k "{{dest}}/live_events.jsonl.gz" | cut -f1)
    pg -c "INSERT INTO heartbeats (component, beat_at, detail) VALUES ('backup.nightly', now(), '{\"dump_mb\": $size, \"live_events_kb\": $live}') ON CONFLICT (component) DO UPDATE SET beat_at = EXCLUDED.beat_at, detail = EXCLUDED.detail"
    echo "halal_trader.dump: ${size} MB, live_events.jsonl.gz: ${live} kB"
    if [ "$(date -u +%d)" = "01" ]; then
        just restore-drill "{{dest}}/halal_trader.dump"
        pg -c "INSERT INTO heartbeats (component, beat_at, detail) VALUES ('backup.restore_drill', now(), '{\"ok\": true}') ON CONFLICT (component) DO UPDATE SET beat_at = EXCLUDED.beat_at, detail = EXCLUDED.detail"
    fi

# Restore a home-backup dump into a scratch database, compare it with the live
# one table by table, then drop it. Never touches halal_trader itself.
# Usage: just restore-drill /path/to/halal_trader.dump
restore-drill file:
    #!/usr/bin/env bash
    set -euo pipefail
    pg() { docker exec halal-trader-pg psql -U trader -tAq "$@"; }
    pg -d postgres -c 'DROP DATABASE IF EXISTS restore_drill' -c 'CREATE DATABASE restore_drill'
    trap 'pg -d postgres -c "DROP DATABASE IF EXISTS restore_drill" >/dev/null; docker exec halal-trader-pg rm -f /tmp/restore-drill.dump' EXIT
    docker cp "{{file}}" halal-trader-pg:/tmp/restore-drill.dump
    start=$(date +%s)
    docker exec halal-trader-pg pg_restore -U trader -d restore_drill --no-owner --exit-on-error /tmp/restore-drill.dump
    echo "restored in $(( $(date +%s) - start ))s"
    printf '%-22s %12s %12s\n' table restored live
    status=0
    for t in alembic_version trades daily_pnl llm_decisions halal_cache broker_activities broker_equity heartbeats hb_outcome hb_belief_state core_orders core_runs purification_accruals zakat_assessments quant_trials forward_book_days halal_screen_results llm_spend; do
        r=$(pg -d restore_drill -c "SELECT count(*) FROM $t" 2>/dev/null || echo missing)
        l=$(pg -d halal_trader -c "SELECT count(*) FROM $t" 2>/dev/null || echo missing)
        printf '%-22s %12s %12s\n' "$t" "$r" "$l"
        # A table the live DB has must come back. Counts may differ: the live
        # bot keeps writing after the dump was taken.
        if [ "$r" = missing ] && [ "$l" != missing ]; then status=1; fi
    done
    rv=$(pg -d restore_drill -c 'SELECT version_num FROM alembic_version')
    lv=$(pg -d halal_trader -c 'SELECT version_num FROM alembic_version')
    echo "schema: restored $rv, live $lv"
    [ "$rv" = "$lv" ] || { echo "SCHEMA MISMATCH"; status=1; }
    exit $status
