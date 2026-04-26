"""Shadow strategy runtime — frozen-prompt parallel runner.

The ``ShadowLedger`` (``halal_trader.core.shadow``) gives us the
bookkeeping side of Wave 4.6 — comparing two equity curves and alerting
on divergence. The missing piece is *what produces the shadow curve*:
a strategy whose prompt and model are locked to a known-good baseline,
running alongside the live strategy on the same per-cycle inputs.

This module provides that piece without doubling the trading footprint.
The shadow doesn't actually execute orders — it produces a plan, then a
small simulator estimates what *would* have happened given the live
account and last-bar prices. That estimated equity is what we compare
against the live curve.

Two pieces:

* :class:`FrozenPromptStrategy` — wraps any strategy whose ``analyze``
  returns a plan-shaped object and pins the prompt-version label so
  later prompt changes don't bleed into the shadow.
* :class:`ShadowRunner` — runs both strategies on the same inputs each
  cycle, simulates fills for the shadow's plan against the latest bar
  price, and writes one row to a :class:`ShadowLedger`.

The simulator is deliberately simple: each shadow buy adds notional at
the latest close; sells release the same. There's no slippage / fee
model — equality with the live curve is what matters, not absolute
realism. If you need fidelity, drive the shadow through
:mod:`halal_trader.crypto.backtest` instead.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from halal_trader.core.shadow import ShadowLedger

logger = logging.getLogger(__name__)


# ── Frozen-prompt strategy wrapper ────────────────────────────────


@dataclass
class FrozenPromptStrategy:
    """Records the *baseline* prompt-version label at construction time.

    The wrapped strategy can have its prompts evolve underneath us; this
    class doesn't try to enforce immutability of the underlying prompt
    text (that would require deep-copying the prompt registry). Instead
    it tags every decision with ``frozen_prompt_version`` so divergence
    can be attributed: "shadow uses v0, live uses v3".
    """

    inner: Any  # any strategy with an async analyze(...) method
    frozen_prompt_version: str

    async def analyze(self, *args: Any, **kwargs: Any) -> Any:
        plan = await self.inner.analyze(*args, **kwargs)
        # Tag the plan for traceability — best-effort, no hard schema.
        try:
            existing = getattr(plan, "risk_notes", "") or ""
            tag = f"frozen_prompt={self.frozen_prompt_version}"
            if tag not in existing:
                if hasattr(plan, "model_copy"):
                    plan = plan.model_copy(
                        update={"risk_notes": (existing + " | " + tag if existing else tag)}
                    )
        except Exception:  # noqa: BLE001
            pass
        return plan


# ── Plan simulator ────────────────────────────────────────────────


@dataclass
class SimulatedShadowAccount:
    """Tracks a paper-only account driven by shadow plans.

    The state is a ``cash + dict[symbol, qty]`` ledger; equity is
    cash + sum(qty × latest_price). All-paper, all-deterministic.
    """

    cash: float = 1000.0
    positions: dict[str, float] = field(default_factory=dict)

    def equity(self, prices: dict[str, float]) -> float:
        e = self.cash
        for sym, qty in self.positions.items():
            p = prices.get(sym)
            if p is not None:
                e += qty * p
        return e

    def apply_decision(self, decision: Any, prices: dict[str, float]) -> None:
        action = getattr(decision, "action", "")
        action = (action.value if hasattr(action, "value") else str(action)).lower()
        symbol = getattr(decision, "symbol", "")
        qty = float(getattr(decision, "quantity", 0) or 0)
        price = prices.get(symbol)
        if price is None or qty <= 0:
            return
        notional = qty * price
        if action == "buy":
            if notional > self.cash:
                qty = self.cash / price
                notional = self.cash
            self.cash -= notional
            self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        elif action == "sell":
            held = self.positions.get(symbol, 0.0)
            sell_qty = min(qty, held)
            if sell_qty <= 0:
                return
            self.cash += sell_qty * price
            self.positions[symbol] = held - sell_qty


# ── Runner ────────────────────────────────────────────────────────


@dataclass
class ShadowRunner:
    """Drives a shadow strategy alongside the live cycle.

    Wire ``runner.observe_cycle(...)`` from the cycle's per-cycle hook
    after the live plan resolves. The runner builds the shadow plan
    from the same inputs, simulates fills against the latest prices,
    and writes one row to its ledger.
    """

    shadow_strategy: Any
    ledger: ShadowLedger
    starting_cash: float = 1000.0
    account: SimulatedShadowAccount | None = None

    def __post_init__(self) -> None:
        if self.account is None:
            self.account = SimulatedShadowAccount(cash=self.starting_cash)

    async def observe_cycle(
        self,
        *,
        cycle_id: str,
        live_equity: float,
        latest_prices: dict[str, float],
        analyze_kwargs: dict[str, Any],
        plan_filter: Callable[[Any], Awaitable[Any] | Any] | None = None,
    ) -> float:
        """Run the shadow plan and apply it to the simulated account.

        Returns the shadow equity after this cycle, or the previous
        equity if any step fails. Writes one row to ``ledger`` either
        way so the live/shadow series have the same length.
        """
        assert self.account is not None
        try:
            plan = await self.shadow_strategy.analyze(**analyze_kwargs)
            if plan_filter is not None:
                maybe = plan_filter(plan)
                if hasattr(maybe, "__await__"):
                    plan = await maybe  # type: ignore[assignment]
                else:
                    plan = maybe
            for d in getattr(plan, "decisions", []) or []:
                self.account.apply_decision(d, latest_prices)
        except Exception as exc:  # noqa: BLE001
            logger.warning("shadow plan/simulation failed: %s", exc)

        equity = self.account.equity(latest_prices)
        self.ledger.record(
            cycle_id=cycle_id,
            live_equity=live_equity,
            shadow_equity=equity,
        )
        return equity
