"""Read-first API (REARCHITECTURE L9) — understanding before action.

The dashboard's job is to answer "what does the bot think, and why" at a glance:
the belief board, the decision stream (a causal chain replayable by
correlation_id, INV-5), risk state, system health, and the operator kill-switch.
Query logic lives in ``queries.py`` (pure, DB-tested); ``app.py`` is a thin
FastAPI surface over it.
"""
