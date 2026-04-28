# Cleanup Roadmap — Round 2

After the Postgres-only sweep (commits `08358f1..14a4849`) the
codebase is in good shape. The items below are the ones I'd reach for
next, ordered by risk × payoff. Each is independently shippable as a
single PR / commit. Stop after any one and the codebase is strictly
cleaner than before.

---

## Wave A — Reconcile alembic on timezone-aware datetimes

**Why**: every `db revision --autogenerate` run prints ~25 spurious
`Detected type change from TIMESTAMP() to DateTime(timezone=True)`
notices. The initial migration created the columns as bare
`TIMESTAMP` (Postgres `timestamp without time zone`); the SQLModel
declarations all use `sa.DateTime(timezone=True)`. Autogenerate
drifts them every time, and contributors learn to ignore the warning
and hand-edit the diff. That instinct will eventually delete a real
schema change.

**Scope**: one migration that does the `ALTER COLUMN ... TYPE
TIMESTAMPTZ USING (...)` for every drifted column in one shot. After
that, autogenerate stays quiet on a clean head and any future drift
is signal again.

**Cross-cutting concerns**:
- The ALTERs need a `USING (column AT TIME ZONE 'UTC')` clause so
  Postgres knows how to interpret the existing naïve timestamps.
  Every row we've written is already UTC by convention (`datetime.now(UTC)`),
  so the cast is lossless.
- Touches ~25 columns across ~15 tables. Run `EXPLAIN` on a populated
  dev DB first to make sure none of them rewrites a huge table.
  Worst case the migration takes a few seconds on the volumes this
  bot produces.

**Acceptance**:
- `halal-trader db revision -m "noop" --autogenerate` produces an
  empty migration body (or, equivalently, prints zero
  `Detected type change` lines).
- `pytest -q` green; the test DB recreate path applies the migration
  cleanly.

**Estimated effort**: 1 hour.

---

## Wave B — JSON → JSONB for the remaining text-encoded JSON columns

**Why**: only `replay_snapshots.payload` is `JSONB` today. Every
other "JSON in a string column" we have (`research_jobs.params`,
`research_jobs.result`, `runtime_config.value`,
`web_actions.payload`, `halal_screenings.criteria`,
`crypto_halal_cache.screening_criteria`,
`llm_decisions.parsed_action`, `rag_rationales.vector`,
`regime_snapshots.features_json`, `regime_snapshots.vector_json`)
sits in `AutoString` (TEXT). Postgres can't index into them, the
dashboard can't query JSON paths, and the bot pays a parse-on-read
tax for nothing.

**Scope**: one migration that ALTERs each column from `TEXT` to
`JSONB USING column::jsonb`. SQLModel field types switch from `str`
to `dict | list[float]` as appropriate; call sites that
`json.dumps` / `json.loads` collapse to direct dict access.

**Cross-cutting concerns**:
- Some callers cast the JSON to typed dataclasses on read; those
  stay (the cast is the contract). Most just want a dict — they
  simplify.
- `rag_rationales.vector` and `regime_snapshots.vector_json` are
  `list[float]`. Storing them as JSONB unblocks the pgvector
  promotion in Wave C without changing storage twice.

**Acceptance**:
- `\\d` in psql shows `jsonb` for every column above.
- `core/llm/rag_db.py`, `core/regret_db.py`, `core/thesis_db.py`,
  `ml/regime_memory.py`, `db/repository.py:begin_web_action` all
  drop their `json.dumps` / `json.loads` boilerplate.
- Suite green.

**Estimated effort**: 2 hours.

---

## Wave C — pgvector for rag + regime similarity

**Why**: `core/llm/rag_db.py:DBRationaleStore.query` and
`ml/regime_memory.py:RegimeMemory.query` both load every row into
Python and compute cosine in a list comprehension. RAG is bounded
to ~10k rows by the schema's retention; regime is bounded to 730.
That's not a performance problem *yet*, but it makes the whole "use
embeddings" story smaller than it should be — there's no path to
HNSW, no path to filter+rerank, no SQL composability.

The `pgvector` extension is already installed in
`infra/docker-compose.yml`. The vectors are stable shapes
(`hashing_embedder.dim` for RAG, 10 for regime).

**Scope**:
- Add a `Vector(N)` column next to the existing `vector_json` /
  `vector` columns. Migration backfills from JSON.
- Switch `query()` to `ORDER BY vector <=> :query LIMIT k`.
- Drop the JSON column once the new path is verified.
- Add an HNSW index on the RAG table — `~1ms` queries even at 100k
  rows.

**Cross-cutting concerns**:
- The pgvector dialect plugs into SQLAlchemy via the `pgvector`
  Python package (already in `pyproject.toml`).
- Embedding dimension is set at table-create time; if RAG's
  `hashing_embedder.dim` changes, that's a schema migration. Pin
  the dim in a constant.

**Acceptance**:
- `RegimeMemory.query` and `DBRationaleStore.query` execute a single
  `SELECT ... ORDER BY embedding <=> $1 LIMIT k` (`EXPLAIN` shows
  the index when present).
- Behavior is unchanged: the existing tests still pass, plus a new
  test that asserts the SQL plan uses the index for ≥1k rows.

**Estimated effort**: 3 hours.

---

## Wave D — Decompose `crypto/cycle.py:_run_cycle_impl`

**Why**: the function is ~340 lines today and growing every time we
add a prompt-context source. Adding sentiment/whale-flow/regret
data each grew it; adding a new source means scrolling past nine
unrelated blocks to find the right insertion point. Each block is
also independently testable but can't be tested in isolation.

**Scope**: peel each `try/except` "augmentation" block out into its
own private async method (`_augment_with_rag`,
`_augment_with_regime_memory`, `_augment_with_news`,
`_augment_with_whale_flows`, `_augment_with_basis` etc.) that
returns its prompt fragment. The main cycle becomes a sequence of
~20 awaited helpers. No behavior change.

**Cross-cutting concerns**:
- Several blocks share inputs (`indicators_cache`, `account`,
  `today_pnl`). Pass them positionally; don't introduce a context
  bag.
- Don't touch `_record_cycle_analytics` — it's already factored.

**Acceptance**:
- `_run_cycle_impl` is under ~120 lines.
- Each new helper has at least one focused test (most are
  swallowed-exception-safe today; verify that explicitly).
- `pytest -q` green.

**Estimated effort**: 2–3 hours.

---

## Wave E — Audit `getattr(self, "_x", None)` dead paths

**Why**: I found one during Wave C — `_replay_store` was read by
`_record_cycle_analytics` via `getattr(self, "_replay_store", None)`
but never assigned anywhere. The whole "snapshot every cycle" flow
was dead code. There may be others.

**Scope**: grep for `getattr(self, "_` across `src/`, audit each
hit. Three outcomes:
- It's a soft-optional dependency (legitimate, leave alone).
- It's set somewhere but the assignment was deleted later (wire it
  back or delete the read).
- It's never set (delete the read, possibly the whole code path).

**Acceptance**:
- Each remaining `getattr(self, "_x", None)` corresponds to an
  actual `self._x = …` assignment somewhere.
- A short PR description listing what was found.

**Estimated effort**: 45 min.

---

## Wave F — Promote mypy strict to `db/repository.py`

**Why**: `just typecheck` only checks `domain/` + `core/` strictly.
Running `mypy src/halal_trader/db/repository.py` directly today
prints ~62 errors, mostly the SQLModel column-attribute typing
issue (`Trade.timestamp.desc()` is typed as
`datetime | None`-with-no-`desc()`). The same pattern is throughout
the codebase, but `repository.py` is the file most likely to grow
new SQL — having it strict-clean would pay off every time we add a
helper.

**Scope**: typed `col(Model.field).desc()` instead of bare
`Model.field.desc()`, add `# type: ignore[attr-defined]` only where
SQLModel's typing genuinely lies (the `is_(None)` / `is_not(None)`
cases). Add `repository.py` to the `[[tool.mypy.overrides]]` strict
list.

**Cross-cutting concerns**:
- Same pattern crops up in `cli/insights.py` and a few other DB
  helpers. Don't fix those in this wave — keep the diff small.
- The existing `from sqlmodel import col` import I added in Wave C
  is the right primitive.

**Acceptance**:
- `uv run mypy src/halal_trader/db/repository.py --strict` clean.
- The `just typecheck` recipe passes including this file.

**Estimated effort**: 1.5 hours.

---

## Wave G — Enable pytest parallelism with pytest-xdist

**Why**: the full suite is 2:11 today and growing every wave.
After Wave B (advisory lock) the test DB is no longer the
bottleneck for parallel runs — only the single-DB
TRUNCATE-per-test fixture is. Switching each test to a per-worker
schema or a per-test transaction-rollback pattern unlocks 4–8x
speedup on the dev workstation and CI.

**Scope**: add `pytest-xdist` to `[dev]`, change `conftest.py` so
each xdist worker gets its own DB (e.g. `halal_trader_test_gw0`,
`halal_trader_test_gw1`, …). The advisory lock from Wave B
generalises to a per-DB lock keyed by the worker id.

**Alternative**: run all tests in a SAVEPOINT, rollback on
teardown. Faster but the test code that opens its own session
breaks the rollback isolation — would need every helper to share
the test's session, which is invasive.

**Decision**: per-worker DB. Simpler, no per-test code changes.

**Acceptance**:
- `pytest -n auto -q` runs the full suite green in ≤45s on the dev
  workstation (8 cores).
- The advisory lock still prevents two sessions from clobbering
  the same worker DB.

**Estimated effort**: 1.5 hours.

---

## Wave H — Drop the websockets deprecation warnings

**Why**: every test run prints two deprecation warnings from
`python-binance` (`websockets.WebSocketClientProtocol` and
`websockets.legacy`). They're not our bug, but they erode the
signal value of the warning summary the same way the SQLModel
deprecations did pre-Wave-A.

**Scope**: bump `python-binance` to the latest, or fork the WS
client locally if upstream still uses the deprecated API. Verify
the WS reconnect / heartbeat tests still pass.

**Cross-cutting concerns**:
- python-binance had a v2 rewrite that fixed this; check
  release notes for breaking changes around the spot/futures
  client signatures we use.

**Acceptance**:
- pytest run emits zero `websockets` deprecation warnings.
- `crypto/exchange.py` and `crypto/ws_manager.py` work against
  testnet in a manual smoke run.

**Estimated effort**: 1–2 hours (depends on python-binance
breaking changes).

---

## Wave I — Replace `insights_hub` singleton with explicit DI

**Why**: now that `regime`, `rag`, `shadow`, `drift` etc. are all
either DB-backed or single-process state, the module-level
`insights_hub.hub` is doing very little work. Cycles read from it,
the dashboard reads `to_app_state()` from it, and tests
`reset_hub()` constantly. It's the kind of singleton that looks
free until you try to test the cycle in isolation and find half a
dozen module-state writes you have to rollback.

**Scope**: pass the hub (or its members directly) through the
constructors that need them. The wiring chain is short:
`CryptoComponents` builds it, the bot holds it, the cycle / monitor
/ web app receive it. `reset_hub()` becomes a no-op (or just
deleted).

**Cross-cutting concerns**:
- This is the largest item on the list and could break tests in
  surprising places. Recommend doing it after Wave G so test
  parallelism is in place to keep iteration fast.
- `web/app.py` reads `insights_hub.to_app_state()` once at startup
  — that already takes a hub-like object via
  `app_state["insights"]`. The work is mostly in the cycle.

**Acceptance**:
- `from halal_trader.core.insights_hub import hub` deleted from all
  source files; `reset_hub()` deleted.
- `test_insights_hub.py` either deleted or rewritten against the
  injected types.
- Suite green.

**Estimated effort**: 4–5 hours.

---

## Cross-cutting principles

Same rules as the last roadmap:
- One migration per Wave. Don't bundle.
- Suite green at every commit.
- No new abstractions unless the Wave is explicitly about
  introducing one (Wave I is the only one that qualifies).
- Greenfield-aggressive: if a code path is dead, delete it; don't
  preserve it for "compatibility" the user has never asked for.

---

## Suggested order

1. **A (timezone reconcile)** — pure noise reduction, prevents a
   future deletion-by-confusion. Do first.
2. **B (JSONB)** — unblocks Wave C and removes parse-on-read tax.
3. **C (pgvector)** — biggest functional payoff once the JSONB shape
   is in.
4. **D (cycle decomposition)** — quality-of-life, no dependencies.
5. **E (dead-getattr audit)** — quick win, removes confusion.
6. **F (mypy strict on repository)** — catches future bugs at
   compile time.
7. **G (pytest-xdist)** — pays for itself across every subsequent
   commit.
8. **H (websockets)** — depends on upstream; may stall.
9. **I (drop insights_hub)** — bigger refactor; do last when the
   smaller items have settled.

Total estimated effort: **~17–20 hours of focused work**, broken
into 9 commits. Like the last roadmap, every Wave is independently
shippable — stop after any one and the bar's been raised.
