# Cleanup Roadmap

Remaining cleanup work after the Postgres-only / sidecar-purge sweep
shipped in commits `0e563ad..4c4207a`. Each item below is independently
shippable and reverts cleanly via the feature it's wrapped in.

The waves are ordered by risk × payoff: do them in order, but you can
stop after any one and the codebase is strictly cleaner than before.

---

## Wave A — SQLModel `session.execute` → `session.exec`

**Why**: every test run emits ~14 `DeprecationWarning`s pointing at
`db/repository.py:479` and a handful of other sites. SQLModel will
remove the `session.execute()` shim eventually; surfacing the warnings
forever erodes the signal value of pytest's warning summary, and a
future SQLModel bump will break the suite outright.

**Scope**: ~6–8 call sites in `db/repository.py` (`get_open_trades`,
`get_today_trades`, the dividend-purification reads), plus 2–3
incidental sites in `core/llm/budget.py` and `core/halt.py`. Each is a
mechanical rewrite from `await session.execute(select(X))…scalars().all()`
to `await session.exec(select(X)).all()`.

**Cross-cutting concerns**:
- `session.exec()` returns `ScalarResult`-shaped iterables directly, so
  call sites that currently call `.scalars().all()` collapse one method
  call. Tuples-of-columns selects (rare) keep `.execute()` since
  `exec()` returns scalars only.
- Make sure typing stays correct — SQLModel's `exec()` is typed via
  `select` overloads.

**Acceptance**:
- pytest run emits zero `session.execute()` deprecation warnings.
- mypy strict still green.
- One commit, one PR — mechanical.

**Estimated effort**: 30 min.

---

## Wave B — Parallel-runner safety on the test fixture

**Why**: the session-scoped `_pg_test_db_ready` fixture does
`pg_terminate_backend` + `DROP DATABASE halal_trader_test` + `CREATE
DATABASE`. If two pytest invocations race (CI shard, an operator
running the suite while the dashboard is also pointed at the test DB,
or — as I hit during this session — two background tabs in the same
shell), the second runner can land on `ObjectInUse: database is being
accessed by other users` and a single terminate isn't enough.

**Scope**: `tests/conftest.py`. Acquire a Postgres advisory lock at
the top of `_pg_test_db_ready`, hold it for the lifetime of the
session. Second runner blocks on lock acquisition (or fails fast with
a clear "another runner is using halal_trader_test" message under a
short statement_timeout) instead of corrupting the DB mid-recreate.

**Sketch**:
```python
@pytest.fixture(scope="session")
def _pg_test_db_ready() -> Iterator[str]:
    lock_conn = psycopg.connect(_ADMIN_DSN_SYNC, autocommit=True)
    lock_conn.execute("SET statement_timeout = '5s'")
    try:
        lock_conn.execute("SELECT pg_advisory_lock(8675309)")
    except psycopg.errors.QueryCanceled:
        pytest.exit("Another pytest session holds the test-DB lock.", 1)
    try:
        _terminate_and_drop(PG_DBNAME)
        # … create + migrate …
        yield PG_TEST_URL_ASYNC
    finally:
        lock_conn.execute("SELECT pg_advisory_unlock(8675309)")
        lock_conn.close()
```

**Acceptance**:
- Two `pytest -q` invocations started within 1s of each other don't
  corrupt `halal_trader_test`; second one either waits or exits with
  a useful message.
- Single-runner suite stays at 1190 passed.

**Estimated effort**: 30 min.

---

## Wave C — Promote `ReplayStore` to a DB table

**Why**: the cycle's snapshot store is the last large JSON-on-disk
writer in the production write path. Each cycle writes one
`<cycle_id>.json` file under `data/replay/`; on a 60s crypto cadence
that's ~1500 files/day. Postgres handles the load trivially and the
dashboard's `/api/insights/replay` route gets to do `LIMIT 50` SQL
instead of a directory listing.

**Scope**:
- New `ReplaySnapshot` SQLModel: `cycle_id` (PK), `created_at`,
  `payload jsonb`, `schema_version`, `prompt_version`,
  `cost_usd_total`. The full snapshot lives in the JSONB column —
  treating it as opaque keeps the migration churn low.
- Alembic migration. Use `JSONB` (Postgres-native) with a
  `created_at desc` index for the recent-listing query.
- Rewrite `core/replay.py`:
  - `ReplayStore.__init__(engine: AsyncEngine)`.
  - `record_snapshot()` becomes async; the cycle path that calls it is
    already async, so the call site change is one `await`.
  - `replay_cycle(cycle_id, callable)` reads the row by PK.
  - `list_cycle_ids()` does `ORDER BY created_at DESC LIMIT N`.
- Update `web/routes/insights.py:api_replay` to use the engine and
  return the same JSON shape.
- Update `cli/insights.py:replay_cmd` accordingly.
- Tests: rewrite `tests/test_replay.py` against the `engine` fixture.

**Cross-cutting concerns**:
- `CycleSnapshot` dataclass stays as the in-memory shape; the JSONB
  column stores its `dataclasses.asdict()` form — same on-disk format
  as today, just under a different storage backend.
- The dashboard frontend hits `/api/insights/replay` for cycle-id
  listing; the response shape is unchanged.

**Acceptance**:
- `data/replay/` no longer exists in production after one cycle.
- `pytest -q` green; replay tests use `engine` fixture not `tmp_path`.
- Manual smoke: `halal-trader insights replay` lists recent cycles.

**Estimated effort**: 2–3 hours.

---

## Wave D — Promote `RegimeMemory` to a DB table

**Why**: the second-largest JSON sidecar in production writes. Daily
regime snapshots accumulate up to the configured `capacity=730`
rows; today the whole file is rewritten on every snapshot. A real
table with an index on `date` makes the embedding-similarity query
explicit SQL and unblocks future pgvector adoption (Wave 2.4 of the
v2 → v5 roadmap).

**Scope**:
- New `RegimeSnapshotRow` SQLModel: `date` (PK), `features jsonb`,
  `outcome_pnl_pct`, `outcome_win_rate`, `outcome_n_trades`,
  `note`, plus a `vector_json` column (stay portable; pgvector lands
  in a follow-up).
- Alembic migration.
- `ml/regime_memory.py` rewrite — `RegimeMemory(engine: AsyncEngine,
  capacity=730)`. Make `record_snapshot()` async; `query()` returns
  top-K by cosine over JSON-encoded vectors (same primitive shape as
  `DBRationaleStore`).
- Cycle wiring in `crypto/cycle.py:_build_regime_text` becomes async
  (already inside an async fn, so just add `await`).
- Update `web/routes/insights.py:api_regime` for the new shape.

**Cross-cutting concerns**:
- The cycle reads `insights_hub.regime` via the in-process hub; that
  hub now holds the DB-backed memory and queries are async. The
  prompt builder needs to await.
- pgvector promotion lands as a separate one-line migration once this
  Wave is in.

**Acceptance**:
- `data/analytics/regime_memory.json` is no longer written.
- Daily-snapshot cron path still produces a row.
- Cycle prompt continues to surface analogous regimes.

**Estimated effort**: 2–3 hours.

---

## Wave E — `[fingpt]` extra: wire it or drop it

**Why**: the `[fingpt]` extra ships ~3 GB of `torch` + `transformers`
+ `peft` weights. `trading/sentiment.py` imports them lazily and the
`SENTIMENT_USE_FINBERT=true` flag is the gate. As of today nothing in
the cycle path actually flips that flag — the FinBERT analyzer is
constructed only if the operator manually sets it. Either:

**Option E1 (wire it)**: thread a `use_finbert` config knob into the
crypto and stock prompt builders so a sentiment score lands in the
prompt context. ~half a day, with backtest comparison.

**Option E2 (drop it)**: delete the `[fingpt]` extra and the
`trading/sentiment.py:FinBERTSentimentAnalyzer` class. Drop the
`use_finbert` setting. ~30 min.

The roadmap (Wave 3.3) wanted Fed-speak FinBERT scoring; that landed
under `trading/fed_speak.py` with a lexicon, not transformers, so the
trading-side need is met. The crypto side has `sentiment/manager.py`
covering CryptoPanic.

**Recommendation**: E2. The extra is dead weight; FinBERT can return
when there's a concrete edge hypothesis to test.

**Acceptance**:
- `pyproject.toml` no longer carries `[fingpt]`.
- `uv sync --extra all` is ~3 GB lighter.
- Suite green.

**Estimated effort**: 30 min for E2.

---

## Wave F — `web_actions` retention

**Why**: the mutation-audit table grows unbounded. Every dashboard
PATCH/POST/DELETE writes a row. On a single-user paper-trading bot
that's a slow leak (low write rate), but on a real deployment with
GUI-driven manual interventions it accumulates fast and there's no
DELETE path.

**Scope**:
- Add a daily prune in `core/scheduler.py:_daily_end()` that deletes
  rows older than `WEB_AUDIT_RETENTION_DAYS` (default 90).
- Add the setting to `WebSettings`.
- Document in CLAUDE.md.

**Acceptance**:
- New `delete_old_web_actions(engine, older_than: timedelta)` helper.
- One unit test against the conftest engine fixture.

**Estimated effort**: 45 min.

---

## Wave G — `ExceptionQueue` to DB (optional)

**Why**: the Sharia exception queue is the last JSON-on-disk
production writer. Operator-paced (low rate, single-writer), so the
JSON form is genuinely fine — but if Waves C + D land, this is the
last sidecar standing and finishing it makes the "Postgres-only"
property exact.

**Scope**: same shape as Wave C / D — model, migration, async store
with the existing public API, route + CLI updates.

**Decision criteria**: do this only if the user wants a clean
"Postgres-only" property in marketing copy. The functional value over
JSON is near zero (single-writer, low rate).

**Estimated effort**: 1.5 hours.

---

## Cross-cutting principles

- **No new abstractions**. Each Wave swaps an implementation behind
  the same public dataclass / public method names. The cycle code
  changes by ≤5 lines per Wave.
- **One migration per Wave**. If you find yourself touching two
  unrelated tables, split.
- **Suite green at every commit**. Each Wave gets a single PR / commit
  that passes `just precommit` end-to-end.
- **No production data migration**. Greenfield-aggressive remains in
  effect: drop the JSON sidecars, don't bother backfilling. The
  current suite recreates state from trades anyway.

---

## Suggested order

1. **A (deprecation cleanup)** — pure mechanical, no risk, eliminates
   suite-noise that masks future regressions. Do first.
2. **B (parallel-runner safety)** — independent of any production
   code, prevents a flake class. Quick win.
3. **C (replay → DB)** — biggest functional payoff (1500 files/day
   collapse to 1500 rows/day with proper indexing).
4. **D (regime → DB)** — enables future pgvector adoption.
5. **E2 (drop [fingpt])** — frees ~3 GB of unused deps.
6. **F (web_actions retention)** — quality-of-life for long-running
   deployments.
7. **G (exception queue → DB)** — only if the "all storage in
   Postgres" property is a goal.

Total estimated effort if everything ships: **~8–10 hours of focused
work**, broken into 7 commits.
