"""Idle-cash treasury policy — halal yield on parked capital.

The bot is flat overnight, on weekends, and during cooldowns. That
capital is doing nothing — at scale a few percent of APY on idle cash
moves the needle and is straightforward to set up via halal money-
market or short-duration sukuk ETFs (``SPSK``, ``WISE``, etc.).

This module is the *policy* layer:

* :class:`TreasuryPolicy` — fraction of cash to keep liquid vs. parked,
  trigger thresholds, target instrument, halal allow-list.
* :class:`IdleCashPlan` — what the policy says to do *right now*: deploy
  $X into instrument Y, redeem $Z, etc.
* :func:`plan_idle_cash` — pure function: given (cash_balance,
  positions_value, current_treasury_position, policy) → :class:`IdleCashPlan`.

The actual broker call to buy/sell the treasury instrument is *not*
this module's job — the cycle / executor wires it. Keeping the policy
pure means we can backtest treasury behaviour independently of any
broker integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Policy ───────────────────────────────────────────────────────


# Allow-list of halal cash-equivalent instruments. Operator can extend
# via the ``policy.halal_instruments`` constructor argument.
DEFAULT_HALAL_INSTRUMENTS: tuple[str, ...] = (
    "SPSK",  # SP Funds Dow Jones Global Sukuk ETF
    "WISE",  # Wahed Investing's halal money-market product
    "HLAL",  # Wahed FTSE USA Shariah ETF (cash equiv on weekends)
)


@dataclass(frozen=True)
class TreasuryPolicy:
    """Per-account knobs for idle-cash deployment.

    * ``min_idle_pct`` — keep at least this fraction of total equity in
      strictly liquid cash for trading.
    * ``deploy_threshold_usd`` — only re-deploy when there's at least
      this much *new* idle cash (avoid 6 sub-dollar trades a day).
    * ``redeem_threshold_usd`` — only redeem (sell treasury) when the
      shortfall in liquid cash is at least this much.
    * ``target_instrument`` — what to deploy into. Must be in
      ``halal_instruments``.
    """

    min_idle_pct: float = 0.10
    deploy_threshold_usd: float = 250.0
    redeem_threshold_usd: float = 100.0
    target_instrument: str = "SPSK"
    halal_instruments: tuple[str, ...] = DEFAULT_HALAL_INSTRUMENTS

    def is_halal(self, instrument: str) -> bool:
        return instrument.upper() in {s.upper() for s in self.halal_instruments}

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_idle_pct < 1.0:
            raise ValueError(f"min_idle_pct out of range: {self.min_idle_pct}")
        if not self.is_halal(self.target_instrument):
            raise ValueError(
                f"target_instrument {self.target_instrument!r} not in halal_instruments"
            )


# ── Plan ─────────────────────────────────────────────────────────


PlanAction = Literal["hold", "deploy", "redeem"]


@dataclass
class IdleCashPlan:
    """What the policy wants to do right this moment."""

    action: PlanAction
    instrument: str
    amount_usd: float = 0.0
    reason: str = ""
    # Pre-/post-snapshot for the dashboard:
    cash_before: float = 0.0
    treasury_before: float = 0.0
    cash_target: float = 0.0
    treasury_target: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return self.action == "hold" or self.amount_usd <= 0


# ── Pure compute ─────────────────────────────────────────────────


def plan_idle_cash(
    *,
    cash_balance: float,
    positions_value: float,
    current_treasury_value: float,
    policy: TreasuryPolicy,
) -> IdleCashPlan:
    """Decide whether to deploy/redeem and how much.

    Total equity = cash + positions + treasury. We keep
    ``min_idle_pct × equity`` in strictly liquid cash. Anything beyond
    that floor is eligible for treasury deployment; if cash is below
    the floor, we redeem from treasury until we're back at it.

    The ``deploy_threshold_usd`` / ``redeem_threshold_usd`` knobs
    suppress small actions so the bot isn't constantly nibbling.
    """
    equity = cash_balance + positions_value + current_treasury_value
    floor = policy.min_idle_pct * equity
    notes: list[str] = []

    if equity <= 0:
        return IdleCashPlan(
            action="hold",
            instrument=policy.target_instrument,
            reason="no equity",
            cash_before=cash_balance,
            treasury_before=current_treasury_value,
            cash_target=cash_balance,
            treasury_target=current_treasury_value,
        )

    if cash_balance < floor:
        shortfall = floor - cash_balance
        if shortfall < policy.redeem_threshold_usd or current_treasury_value <= 0:
            notes.append(
                f"shortfall ${shortfall:.2f} below redeem threshold ${policy.redeem_threshold_usd}"
            )
            return IdleCashPlan(
                action="hold",
                instrument=policy.target_instrument,
                reason="below floor but under redeem threshold",
                cash_before=cash_balance,
                treasury_before=current_treasury_value,
                cash_target=cash_balance,
                treasury_target=current_treasury_value,
                notes=notes,
            )
        amount = min(shortfall, current_treasury_value)
        return IdleCashPlan(
            action="redeem",
            instrument=policy.target_instrument,
            amount_usd=amount,
            reason=f"refill liquid cash to floor ({policy.min_idle_pct:.0%})",
            cash_before=cash_balance,
            treasury_before=current_treasury_value,
            cash_target=cash_balance + amount,
            treasury_target=current_treasury_value - amount,
            notes=notes,
        )

    excess = cash_balance - floor
    if excess < policy.deploy_threshold_usd:
        notes.append(f"excess ${excess:.2f} below deploy threshold ${policy.deploy_threshold_usd}")
        return IdleCashPlan(
            action="hold",
            instrument=policy.target_instrument,
            reason="excess under deploy threshold",
            cash_before=cash_balance,
            treasury_before=current_treasury_value,
            cash_target=cash_balance,
            treasury_target=current_treasury_value,
            notes=notes,
        )

    return IdleCashPlan(
        action="deploy",
        instrument=policy.target_instrument,
        amount_usd=excess,
        reason=f"deploy excess above {policy.min_idle_pct:.0%} liquid floor",
        cash_before=cash_balance,
        treasury_before=current_treasury_value,
        cash_target=cash_balance - excess,
        treasury_target=current_treasury_value + excess,
        notes=notes,
    )


# ── Yield estimator ──────────────────────────────────────────────


def estimate_annual_yield_usd(
    treasury_value: float, *, apy: float = 0.04, days: int = 365
) -> float:
    """Linear estimate of treasury yield over ``days`` at ``apy``.

    Sukuk ETFs distribute monthly so this is a coarse approximation —
    fine for a dashboard tile. ``apy`` defaults to a conservative 4%.
    """
    if treasury_value <= 0 or apy <= 0 or days <= 0:
        return 0.0
    return treasury_value * apy * (days / 365.0)
