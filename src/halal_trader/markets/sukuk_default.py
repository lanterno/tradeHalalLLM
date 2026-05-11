"""Sukuk default + recovery model — Round-5 Wave 3.F.

Default modeling for sukuk diverges from conventional bonds in two
material ways:

1. **Recovery profiles depend on the underlying contract.** Pure-debt
   sukuk (Murabaha / Salam) recover like senior unsecured debt — the
   issuer owes a fixed amount; restructuring math is bond-style.
   Asset-backed sukuk (Ijara / Istisna) recover via the underlying
   asset — recovery tracks asset liquidation rather than balance-sheet
   waterfall, so the curve is bimodal (full recovery if the asset
   sells, near-zero if it doesn't). Partnership sukuk (Mudarabah /
   Musharakah / Wakalah) recover *equity-like* — the holder is a
   profit-share counterparty, not a creditor; recovery follows the
   residual after senior creditors are paid.

2. **Regional regimes diverge.** GCC restructuring (Saudi, UAE, Bahrain,
   Qatar, Kuwait) tends to favour negotiated workouts where the
   originator's balance sheet absorbs partial losses; historical
   recovery on Ijara is ~75-85¢. Asia-Pacific (Malaysia, Indonesia,
   Pakistan) trends lower — ~60-75¢ on Ijara, ~30-50¢ on partnership
   structures, reflecting weaker creditor protections and a more
   pure-equity treatment of Mudarabah/Musharakah holders. The model
   pins these as separate parameter blocks so the operator can swap
   regional assumptions transparently.

This module is the pure-Python primitive. No I/O, no DB. The live
default-feed adapter (S&P / Fitch / GCC central-bank issuer monitor)
is a follow-up; this module exercises the math in isolation.

Pinned semantics:

- **Recovery curve = (mean, std) by SukukType × Region.** The
  numbers are calibrated against published academic surveys (S&P
  Sukuk Defaults 2008-2023; IIFM annual report). Operator-tunable
  via `RecoveryParams` overrides — the defaults are conservative.
- **PD curve uses Vasicek-style hazard with sukuk-type multiplier.**
  Asset-backed sukuk get a 0.7× multiplier (asset cushion); pure-
  debt sukuk get 1.0×; partnership sukuk get 1.3× (equity-like
  default risk). Calibration proxy.
- **Expected loss = PD × (1 - mean_recovery).** Variance of loss
  uses the recovery distribution std; the Loss-Given-Default LGD
  is `(1 - mean_recovery)`.
- **`expected_loss` and `loss_at_quantile` are pure functions** —
  no clock side-effects, no random draws inside the engine. The
  Monte-Carlo simulator is a separate function `simulate_losses`
  that accepts an explicit RNG seed so operators can pin reports.
- **No-secret-leak pin** on render output — issuer / amount only.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from halal_trader.halal.aaoifi_standard_17 import SukukType


class Region(str, Enum):
    """Closed-set regional regime for default + recovery calibration."""

    GCC = "gcc"
    ASIA_PACIFIC = "asia_pacific"
    EUROPE = "europe"
    AFRICA = "africa"


# Per (sukuk_type, region) recovery params: (mean, std). Calibrated
# against published surveys; conservative where data is sparse.
_RECOVERY_TABLE: dict[tuple[SukukType, Region], tuple[float, float]] = {
    # GCC — strongest creditor regime; asset-backed especially robust.
    (SukukType.IJARA, Region.GCC): (0.80, 0.12),
    (SukukType.ISTISNA, Region.GCC): (0.70, 0.15),
    (SukukType.MURABAHA, Region.GCC): (0.65, 0.15),
    (SukukType.SALAM, Region.GCC): (0.60, 0.18),
    (SukukType.MUDARABAH, Region.GCC): (0.45, 0.22),
    (SukukType.MUSHARAKAH, Region.GCC): (0.45, 0.22),
    (SukukType.WAKALAH, Region.GCC): (0.55, 0.20),
    # Asia-Pacific — weaker creditor regime; partnership sukuk treated
    # closer to equity in restructurings.
    (SukukType.IJARA, Region.ASIA_PACIFIC): (0.65, 0.15),
    (SukukType.ISTISNA, Region.ASIA_PACIFIC): (0.55, 0.18),
    (SukukType.MURABAHA, Region.ASIA_PACIFIC): (0.50, 0.18),
    (SukukType.SALAM, Region.ASIA_PACIFIC): (0.45, 0.20),
    (SukukType.MUDARABAH, Region.ASIA_PACIFIC): (0.30, 0.22),
    (SukukType.MUSHARAKAH, Region.ASIA_PACIFIC): (0.30, 0.22),
    (SukukType.WAKALAH, Region.ASIA_PACIFIC): (0.40, 0.22),
    # Europe — sparse data; treat as GCC-light.
    (SukukType.IJARA, Region.EUROPE): (0.70, 0.15),
    (SukukType.ISTISNA, Region.EUROPE): (0.60, 0.18),
    (SukukType.MURABAHA, Region.EUROPE): (0.55, 0.18),
    (SukukType.SALAM, Region.EUROPE): (0.50, 0.20),
    (SukukType.MUDARABAH, Region.EUROPE): (0.35, 0.22),
    (SukukType.MUSHARAKAH, Region.EUROPE): (0.35, 0.22),
    (SukukType.WAKALAH, Region.EUROPE): (0.45, 0.20),
    # Africa — least creditor protection; conservative across the board.
    (SukukType.IJARA, Region.AFRICA): (0.55, 0.18),
    (SukukType.ISTISNA, Region.AFRICA): (0.45, 0.20),
    (SukukType.MURABAHA, Region.AFRICA): (0.40, 0.20),
    (SukukType.SALAM, Region.AFRICA): (0.35, 0.22),
    (SukukType.MUDARABAH, Region.AFRICA): (0.20, 0.22),
    (SukukType.MUSHARAKAH, Region.AFRICA): (0.20, 0.22),
    (SukukType.WAKALAH, Region.AFRICA): (0.30, 0.22),
}


# Multiplier on the base hazard rate by sukuk type. Asset-backed get a
# discount; partnership sukuk get a premium.
_HAZARD_MULTIPLIER: dict[SukukType, float] = {
    SukukType.IJARA: 0.7,
    SukukType.ISTISNA: 0.85,
    SukukType.MURABAHA: 1.0,
    SukukType.SALAM: 1.0,
    SukukType.MUDARABAH: 1.3,
    SukukType.MUSHARAKAH: 1.3,
    SukukType.WAKALAH: 1.15,
}


@dataclass(frozen=True)
class RecoveryParams:
    """Recovery distribution for a sukuk on default.

    `mean` and `std` are in [0, 1]; both must be set. Operators can
    override the table defaults via `lookup(..., overrides=...)`.
    """

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.mean <= 1.0:
            raise ValueError("mean recovery must be in [0, 1]")
        if not 0.0 <= self.std <= 0.5:
            raise ValueError("std must be in [0, 0.5]")


def lookup_recovery(
    sukuk_type: SukukType,
    region: Region,
    *,
    overrides: dict[tuple[SukukType, Region], tuple[float, float]] | None = None,
) -> RecoveryParams:
    """Look up the calibrated recovery params for (type, region)."""
    if overrides and (sukuk_type, region) in overrides:
        mean, std = overrides[(sukuk_type, region)]
    else:
        mean, std = _RECOVERY_TABLE[(sukuk_type, region)]
    return RecoveryParams(mean=mean, std=std)


def hazard_multiplier(sukuk_type: SukukType) -> float:
    """The default-hazard multiplier for the sukuk type."""
    return _HAZARD_MULTIPLIER[sukuk_type]


def cumulative_pd(
    *,
    base_hazard: float,
    sukuk_type: SukukType,
    horizon_years: float,
) -> float:
    """Cumulative probability of default over the horizon.

    Uses Vasicek-style constant-hazard: PD(t) = 1 - exp(-h × t)
    with h = base_hazard × hazard_multiplier(sukuk_type).
    """
    if base_hazard < 0:
        raise ValueError("base_hazard must be non-negative")
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    h = base_hazard * _HAZARD_MULTIPLIER[sukuk_type]
    return 1.0 - math.exp(-h * horizon_years)


@dataclass(frozen=True)
class SukukExposure:
    """A single sukuk holding exposed to default risk."""

    issuer: str
    sukuk_type: SukukType
    region: Region
    face_value: float

    def __post_init__(self) -> None:
        if not self.issuer or not self.issuer.strip():
            raise ValueError("issuer must be non-empty")
        if self.face_value <= 0:
            raise ValueError("face_value must be positive")


@dataclass(frozen=True)
class LossEstimate:
    """Output of `expected_loss`."""

    pd: float
    mean_recovery: float
    expected_loss: float
    loss_std: float

    def lgd(self) -> float:
        """Loss-given-default = 1 - mean_recovery."""
        return 1.0 - self.mean_recovery


def expected_loss(
    exposure: SukukExposure,
    *,
    base_hazard: float,
    horizon_years: float,
    overrides: dict[tuple[SukukType, Region], tuple[float, float]] | None = None,
) -> LossEstimate:
    """Compute the expected loss + std for one exposure."""
    pd = cumulative_pd(
        base_hazard=base_hazard,
        sukuk_type=exposure.sukuk_type,
        horizon_years=horizon_years,
    )
    rec = lookup_recovery(exposure.sukuk_type, exposure.region, overrides=overrides)
    lgd = 1.0 - rec.mean
    el = pd * lgd * exposure.face_value
    # Variance approximation: pd * (1-pd) * (lgd * face)**2  +  pd * (rec.std * face)**2
    var = (
        pd * (1 - pd) * (lgd * exposure.face_value) ** 2 + pd * (rec.std * exposure.face_value) ** 2
    )
    return LossEstimate(
        pd=pd,
        mean_recovery=rec.mean,
        expected_loss=el,
        loss_std=math.sqrt(var),
    )


def loss_at_quantile(
    exposure: SukukExposure,
    *,
    base_hazard: float,
    horizon_years: float,
    quantile: float,
    overrides: dict[tuple[SukukType, Region], tuple[float, float]] | None = None,
) -> float:
    """Approximate loss at the given quantile using gaussian inverse-CDF.

    Treats the loss distribution as normal(EL, loss_std). For tail
    risk reporting prefer `simulate_losses` which captures the
    bimodal shape on small numbers of exposures.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    est = expected_loss(
        exposure,
        base_hazard=base_hazard,
        horizon_years=horizon_years,
        overrides=overrides,
    )
    z = _norm_inv(quantile)
    return max(0.0, est.expected_loss + z * est.loss_std)


def simulate_losses(
    exposures: Iterable[SukukExposure],
    *,
    base_hazard: float,
    horizon_years: float,
    n_paths: int = 10_000,
    seed: int = 0,
    overrides: dict[tuple[SukukType, Region], tuple[float, float]] | None = None,
) -> tuple[float, ...]:
    """Monte-Carlo simulation of total portfolio loss.

    Each exposure independently:
    - defaults with probability `cumulative_pd(...)`
    - given default, recovery is drawn from a clipped normal
      (mean, std) and loss = (1 - recovery) × face_value.

    Returns the sorted tuple of total-loss outcomes (length n_paths).
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    rng = random.Random(seed)
    exp_tuple = tuple(exposures)
    if not exp_tuple:
        return tuple()
    pds_recs: list[tuple[float, RecoveryParams, float]] = []
    for e in exp_tuple:
        pd = cumulative_pd(
            base_hazard=base_hazard,
            sukuk_type=e.sukuk_type,
            horizon_years=horizon_years,
        )
        rec = lookup_recovery(e.sukuk_type, e.region, overrides=overrides)
        pds_recs.append((pd, rec, e.face_value))
    paths: list[float] = []
    for _ in range(n_paths):
        total_loss = 0.0
        for pd, rec, face in pds_recs:
            if rng.random() < pd:
                draw = rng.gauss(rec.mean, rec.std)
                recovery = max(0.0, min(1.0, draw))
                total_loss += (1.0 - recovery) * face
        paths.append(total_loss)
    paths.sort()
    return tuple(paths)


def quantile(paths: tuple[float, ...], q: float) -> float:
    """Return the q-quantile of the simulated loss paths."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if not paths:
        raise ValueError("paths must be non-empty")
    n = len(paths)
    idx = min(n - 1, max(0, int(round(q * (n - 1)))))
    return paths[idx]


def _norm_inv(p: float) -> float:
    """Beasley-Springer-Moro inverse normal CDF.

    Sufficient accuracy for tail-loss reporting; deterministic + pure.
    """
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


@dataclass(frozen=True)
class PortfolioLossReport:
    """Aggregate loss report across multiple sukuk exposures."""

    n_exposures: int
    total_face: float
    expected_loss: float
    var_95: float
    var_99: float
    cvar_95: float


def portfolio_loss_report(
    exposures: Iterable[SukukExposure],
    *,
    base_hazard: float,
    horizon_years: float,
    n_paths: int = 10_000,
    seed: int = 0,
    overrides: dict[tuple[SukukType, Region], tuple[float, float]] | None = None,
) -> PortfolioLossReport:
    """Simulate + summarise the loss distribution for a sukuk portfolio."""
    exp_tuple = tuple(exposures)
    if not exp_tuple:
        raise ValueError("exposures must be non-empty")
    paths = simulate_losses(
        exp_tuple,
        base_hazard=base_hazard,
        horizon_years=horizon_years,
        n_paths=n_paths,
        seed=seed,
        overrides=overrides,
    )
    total_face = sum(e.face_value for e in exp_tuple)
    el = sum(paths) / len(paths)
    var_95 = quantile(paths, 0.95)
    var_99 = quantile(paths, 0.99)
    # CVaR_95 = mean of losses above the 95% threshold.
    threshold_idx = int(0.95 * (len(paths) - 1))
    tail = paths[threshold_idx:]
    cvar_95 = sum(tail) / len(tail) if tail else var_95
    return PortfolioLossReport(
        n_exposures=len(exp_tuple),
        total_face=total_face,
        expected_loss=el,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
    )


def render_report(report: PortfolioLossReport, *, currency: str = "USD") -> str:
    """Operator-readable summary of the loss report."""
    return (
        f"📉 Sukuk default report: {report.n_exposures} exposure(s), "
        f"face={report.total_face:.2f} {currency}\n"
        f"  • Expected loss: {report.expected_loss:.2f} "
        f"({report.expected_loss / report.total_face * 100:.2f}%)\n"
        f"  • VaR 95%:       {report.var_95:.2f}\n"
        f"  • VaR 99%:       {report.var_99:.2f}\n"
        f"  • CVaR 95%:      {report.cvar_95:.2f}"
    )
