# Notebooks

Operator-runnable Jupyter notebooks for research over the bot's own
data. Each notebook is self-contained — clone the repo, install the
`[dashboard]` extra, point the connection string at your own
Postgres, and run. No sample data ships in the repo (the bot's
audit trail is per-operator) so notebooks operate on whatever the
operator has accumulated.

## Setup

```bash
# One-time: install Jupyter alongside the dashboard extras.
uv sync --extra dashboard
uv pip install jupyterlab

# Per-session: run the bot's Postgres + start Jupyter.
just pg-up
uv run jupyter lab notebooks/
```

The notebooks read `DATABASE_URL` from `.env` via the same
`Settings.database_url` path the bot uses, so no separate
configuration is needed.

## Available notebooks

| Notebook | Purpose |
|---|---|
| [explore-replay-store.ipynb](explore-replay-store.ipynb) | Inspect the recent cycle / decision / trade history with summary stats and operator-tunable filters. |
| [test-custom-prompt.ipynb](test-custom-prompt.ipynb) | Take a candidate LLM prompt template and run it against a sample of historical decisions to compare its outputs vs the actual recorded decisions. |
| [sentiment-vs-returns.ipynb](sentiment-vs-returns.ipynb) | Correlate the bot's recorded sentiment scores against per-trade returns. Builds the bullish / bearish / neutral buckets and a per-bucket return distribution. |

## Halal alignment

These notebooks are research only — they read from the operator's
audit trail and produce plots / tables. **None of them open a
trade, screen an asset, or modify any persisted row.** The
`commit=False` invariant applies to every Postgres connection
opened in these notebooks (a regression check operators can run
to verify is in the per-notebook header).

## Conventions

* Every notebook starts with a "What this does / What it
  doesn't do" markdown cell so an operator scanning the notebook
  list knows whether it's the right tool.
* Every SQL query in a code cell uses parameterised execution
  (no f-strings into raw SQL) — operators editing for their own
  research should keep that pattern.
* Notebooks output **anonymised aggregates by default**. When a
  cell shows operator-identifying data (e.g., per-trade
  rationale), the cell's preceding markdown explicitly notes it.

## Contributing

Operators who write a research notebook they think is useful can
PR it under `notebooks/`. The maintainer will:

1. Run the notebook end-to-end on the contributor's sample data
   (or a synthetic stand-in) to verify it doesn't crash.
2. Confirm the read-only invariant (no `INSERT` / `UPDATE` /
   `DELETE` SQL).
3. Add it to the index above.
