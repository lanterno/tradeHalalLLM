"""analysis — read-only reporting over the engine's durable log + legacy trades.

The Phase-3 acceptance gate lives here: compare the shadow engine's proposed
trades against the live cycle's actual trades over real sessions, to decide
whether the conviction engine churns materially less (REARCHITECTURE Part IV).
"""
