# Infrastructure

Single-host Docker Compose stack for running the bot in production
mode. All four services share one Postgres + pgvector instance; the
two bots write concurrently to the same database.

## Services

| Service | Image | Purpose |
|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | Canonical DB (trades, P&L, halal cache, LLM audit, RAG vectors, …). Listens on `127.0.0.1:5433`. |
| `trader-crypto` | `halal-trader:latest` | 24/7 crypto bot (`halal-trader crypto start`). Built with `[ml,sentiment]` extras. |
| `trader-stocks` | `halal-trader:latest` | Market-hours stock bot (`halal-trader start`). |
| `trader-web` | `halal-trader:latest` | FastAPI dashboard + Prometheus `/metrics`, exposed on `8082`. |

The Prometheus + Grafana sidecars are commented out — uncomment when
the operator actually wants observability dashboards. The bot's web
service exposes `/metrics` regardless, so a separately-deployed
Prometheus can scrape directly.

## Local dev

For day-to-day development you only need the `postgres` service; run
the bot via `uv run halal-trader …` from the host so you get fast
edit-test cycles:

```bash
just pg-up                       # bring up postgres only
uv run halal-trader db migrate   # run migrations
uv run halal-trader crypto start --once   # one cycle, then exit
```

`just pg-down` stops Postgres but keeps the data volume, so a
restart picks up where you left off.

## Production deploy

```bash
cp .env.example .env             # fill in real keys
cd infra
docker compose up -d --build     # build the image + start everything
```

The bots auto-restart on crash (`restart: unless-stopped`). Logs land
under the `trader-logs` named volume; the `trader-data` volume holds
HuggingFace caches (Chronos weights) and any operator-managed JSON
that hasn't moved into the DB yet.

## Observability

`prometheus.yml` is the scrape config that targets the web service's
`/metrics` endpoint when Prometheus is enabled. `grafana/` ships a
prebuilt dashboard JSON (`halal-trader.json`) covering:

* per-stage cycle latency (p50 / p95)
* portfolio drawdown % (`max` across markets) and portfolio heat per
  market — `halal_trader_drawdown_pct{market="crypto"|"stocks"}` and
  `halal_trader_portfolio_heat_pct{market="…"}` are emitted with a
  market discriminator so co-hosted bots show as separate series.
* LLM call cost rate per provider
* broker error rate
* kill-switch state
* daily P&L curve
* drift state, regime distribution, halal cache hit rate

Import it via the Grafana UI once the sidecar is up.

## Database upgrades

Migrations live in `alembic/versions/`. The bot refuses to start
unless the DB is at the Alembic head, so any schema change must be
applied before the bots restart:

```bash
docker compose run --rm trader-crypto halal-trader db migrate
docker compose restart trader-crypto trader-stocks trader-web
```

`just db-reset` drops + recreates the database (destroys all trade
history); `just test-db-reset` does the same for the test database.

## Resetting

```bash
docker compose down              # keeps volumes
docker compose down -v           # WIPES Postgres + caches + logs
```
