"""policy — conviction → target weights → trade *deltas* (REARCHITECTURE L5).

The deterministic bridge from understanding to action. Trades are emitted only
as the difference between the portfolio held and the one conviction implies, so
a stable belief produces no trade — this is what kills churn. In Phase 3 the
policy runs in **shadow** (log-only): it emits ``policy.target_changed`` /
``policy.trade_proposed`` events but never executes.
"""
