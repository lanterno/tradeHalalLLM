"""learning — close the loop: outcomes → recalibration (REARCHITECTURE L8).

In Phase 3 (shadow), the :class:`ShadowOutcomeTracker` marks the engine's
hypothetical fills to price and records closed-position outcomes — giving the
A/B a P&L dimension (not just churn) and the conviction calibrator its training
corpus (entry-time features only — no mid-trade leakage).
"""
