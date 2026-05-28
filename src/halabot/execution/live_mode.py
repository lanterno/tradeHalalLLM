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

from halabot.platform.config import HalabotSettings

_TOKEN_RE = re.compile(r"^LIVE-(\d{4})-(\d{2})-(\d{2})$")


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
        # Effective caps are clamped to the SAFEGUARD ceilings regardless of any
        # other config that asks for more (un-loosenable, INV-9).
        req_order = (
            settings.execution.min_notional_usd
            if requested_max_order_usd is None
            else requested_max_order_usd
        )
        # Clamp to the SAFEGUARD ceiling regardless of what config requested.
        eff_order = (
            sg.live_max_order_usd if req_order <= 0 else min(req_order, sg.live_max_order_usd)
        )
        clamp = LiveModeDecision(
            armed=False,
            reason="",
            market=settings.live.strip(),
            max_order_usd=eff_order,
            max_account_usd=sg.live_max_account_usd,
            daily_loss_floor_pct=sg.live_daily_loss_floor_pct,
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

        # Sanity on the floors themselves — refuse to arm on a nonsensical config.
        if not (0 < eff_order <= sg.live_max_account_usd) or not (
            0 < sg.live_daily_loss_floor_pct <= 1
        ):
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
