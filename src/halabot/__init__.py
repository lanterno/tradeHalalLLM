"""halabot — the market-understanding engine (re-architecture).

A continuously-maintained model of the market (the ``BeliefState``) from
which trades fall out as the *delta* between the portfolio we hold and the
one our conviction implies. See ``docs/REARCHITECTURE.md`` for the full
spec; this package is built strangler-fig alongside the legacy
``halal_trader`` package and shares its Postgres + config until the
legacy transactional pipeline is decommissioned (migration Phase 6).

Constraints (unchanged): paper/testnet only — never real money; halal
compliance is non-negotiable.
"""
