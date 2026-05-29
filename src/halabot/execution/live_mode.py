"""Live-mode arming gate (REARCHITECTURE §4, INV-9). FLAG-OFF BY DEFAULT.

Live execution is armed ONLY when an operator deliberately turns it on:

* ``ENGINE_LIVE`` names a market (else shadow-only — the default), AND
* ``ENGINE_LIVE_TOKEN`` is a DATED token matching today (``LIVE-YYYY-MM-DD``),
  so it expires every day and a stale env var can't keep the engine live.

The SAFEGUARD floors are **un-loosenable ceilings** (INV-9): the *effective*
per-order and per-account caps are ``min(requested config, SAFEGUARD)``, so no
amount of other config can push the engine above them. The decision carries the
clamped limits; the live bridge enforces them on every order.

This module is built + tested but, with the defaults, ALWAYS returns
``armed=False`` — the engine never trades until an operator arms it AND the
Phase-3 statistical gate (``analysis.significance.promotion_gate``) has passed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Final

from halabot.platform.config import HalabotSettings

_TOKEN_RE = re.compile(r"^LIVE-(\d{4})-(\d{2})-(\d{2})$")

# Absolute, CODE-LEVEL hard floors (INV-9). The SafeguardSettings config can only
# TIGHTEN below these — it can never raise the engine above them, because a config
# value is reachable from env (a typo / careless operator could otherwise blow past
# the intended cap in one env change). The effective cap is min(config, ABS_MAX).
ABS_MAX_ORDER_USD: Final[float] = 1_000.0
ABS_MAX_ACCOUNT_USD: Final[float] = 10_000.0
ABS_MAX_DAILY_LOSS_FLOOR_PCT: Final[float] = 0.05


@dataclass(frozen=True)
class LiveModeDecision:
    armed: bool
    reason: str
    market: str = ""
    # Effective (clamped) caps the bridge enforces — never exceed the SAFEGUARD.
    max_order_usd: float = 0.0
    max_account_usd: float = 0.0
    daily_loss_floor_pct: float = 0.0


def expected_token(today: date) -> str:
    """The dated token an operator must supply to arm live mode today."""
    return f"LIVE-{today.isoformat()}"


class LiveModeChecker:
    """Decides whether live execution may arm, enforcing INV-9."""

    def check(
        self,
        settings: HalabotSettings,
        today: date,
        *,
        requested_max_order_usd: float | None = None,
    ) -> LiveModeDecision:
        sg = settings.safeguard
        # Effective caps = min(requested, config SAFEGUARD, ABSOLUTE code floor).
        # Config can only TIGHTEN; the ABS_MAX_* constants are the real ceiling no
        # env can raise (INV-9, the un-loosenable property).
        req_order = (
            settings.execution.min_notional_usd
            if requested_max_order_usd is None
            else requested_max_order_usd
        )
        config_order = (
            sg.live_max_order_usd if req_order <= 0 else min(req_order, sg.live_max_order_usd)
        )
        eff_order = min(config_order, ABS_MAX_ORDER_USD)
        eff_account = min(sg.live_max_account_usd, ABS_MAX_ACCOUNT_USD)
        eff_daily_loss = min(sg.live_daily_loss_floor_pct, ABS_MAX_DAILY_LOSS_FLOOR_PCT)
        clamp = LiveModeDecision(
            armed=False,
            reason="",
            market=settings.live.strip(),
            max_order_usd=eff_order,
            max_account_usd=eff_account,
            daily_loss_floor_pct=eff_daily_loss,
        )

        if not settings.live_enabled:
            return _with(clamp, armed=False, reason="shadow only (ENGINE_LIVE unset)")

        token = settings.live_token.strip()
        m = _TOKEN_RE.match(token)
        if not m:
            return _with(clamp, armed=False, reason="missing/malformed ENGINE_LIVE_TOKEN")
        try:
            tok_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return _with(clamp, armed=False, reason="invalid token date")
        if tok_date != today:
            return _with(
                clamp,
                armed=False,
                reason=f"stale token ({token}); expected {expected_token(today)}",
            )

        # Sanity on the EFFECTIVE (clamped) floors — refuse to arm on nonsense.
        if not (0 < eff_order <= eff_account) or not (0 < eff_daily_loss <= 1):
            return _with(clamp, armed=False, reason="invalid SAFEGUARD floors")

        return _with(clamp, armed=True, reason=f"ARMED for {settings.live.strip()}")


def _with(base: LiveModeDecision, *, armed: bool, reason: str) -> LiveModeDecision:
    return LiveModeDecision(
        armed=armed,
        reason=reason,
        market=base.market,
        max_order_usd=base.max_order_usd,
        max_account_usd=base.max_account_usd,
        daily_loss_floor_pct=base.daily_loss_floor_pct,
    )
