"""risk — portfolio-level guardrails the policy consults (REARCHITECTURE L7).

Computes three independent halt conditions (unrealized heat, peak-to-trough
drawdown, and the *realized* intraday daily-loss floor — R-10) plus per-asset
correlation/volatility size multipliers and total gross exposure. A risk halt
overrides any conviction; exits are always allowed.
"""
