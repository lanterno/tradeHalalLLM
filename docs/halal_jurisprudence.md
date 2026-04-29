# Halal Jurisprudence Reference

This document is the project's rulebook for which financial products
are permissible (halal) under Sharia law and the criteria the
screeners use to decide. The bot's `halal/explainer.py` cites
section numbers from this file when it explains a trade decision.

## Section 1: Utility tokens

A crypto asset is permissible when:

1. It represents a utility (network fee token, governance token,
   storage credit, …) — not a debt instrument.
2. It is not interest-bearing (no automatic yield on holding).
3. The issuer's revenue is primarily from halal activities.
4. Market cap above ~$1B (proxies for liquidity + counterparty
   solvency on the exchange).

## Section 2: Equities

Stock screening follows AAOIFI standards via Zoya:

- Sector exclusions: alcohol, gambling, conventional finance,
  pork, weapons, adult entertainment, tobacco.
- Financial ratios: interest-bearing debt < 30% of market cap;
  cash + receivables < 70%; non-permissible income < 5%.

## Section 3: Prohibited activities

The screener's hard `not_halal` decisions ground out here:

- **Riba (interest)**: any product that pays or charges interest.
- **Maysir (gambling)**: leveraged perp futures with no underlying
  delivery, prediction markets without a real-world reference.
- **Gharar (excessive uncertainty)**: derivatives whose value
  depends on future events outside the underlying.

## Section 4: Doubtful and overrides

Decisions tagged "doubtful" go to the exception queue
(`halal/exception_queue.py`). The operator can:

1. Approve — typical for newly-listed tokens with insufficient
   data but a halal sector (e.g. a new layer-1 with no DeFi
   features).
2. Reject — for borderline cases the operator wants to wait on.
3. Defer — explicit "ask a scholar before acting".

Approved overrides are logged with `decided_by` so a future
scholar challenge has the audit trail.

## Section 5: Purification

Capital gains accrued on a permissible-on-balance asset that has
some impure revenue stream are purified by donating the
proportionate share. The bot's `halal/round_trip_purification.py`
records each closed-trade gain × the asset's `impure_ratio`; the
operator marks them paid via the dashboard.
