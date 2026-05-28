"""belief — the world model: the per-asset ``BeliefState``, its versioned store,
and the incremental, deterministic-first updater that turns evidence into beliefs.

This is the heart of the engine (REARCHITECTURE.md L3). Everything upstream
(perception, cognition) feeds evidence in; everything downstream (conviction,
policy) reads beliefs out. The updater is LLM-free except for one guarded
thesis-refresh call, so beliefs and risk-management survive an LLM outage (INV-1).
"""
