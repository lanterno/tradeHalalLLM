"""Breach-rate test for price bands (from Phase 0 of the quant roadmap).

Whether a band's *breach rate* is statistically consistent with its nominal
rate is tested with the Kupiec proportion-of-failures likelihood-ratio test
(``kupiec_pof``); the conformal band maintenance (``quant/conformal.py``)
uses it to flag coverage drift. Band models themselves are scored by
``quant/band_compare.py``.

Pure numpy + stdlib by design: no scipy. The chi-square survival function
has closed forms at 1 and 2 degrees of freedom (``erfc(sqrt(x/2))`` and
``exp(-x/2)``) — see ``_chi2_sf``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt

FloatArray = Sequence[float] | npt.NDArray[Any]
"""Any 1-d float-coercible input; normalized internally via ``np.asarray``."""

BreachArray = Sequence[bool] | Sequence[int] | Sequence[float] | npt.NDArray[Any]
"""A 1-d breach-indicator series; entries must be boolean or 0/1."""


@dataclass(frozen=True, slots=True)
class LRTestResult:
    """Outcome of a likelihood-ratio test: the LR statistic and its p-value."""

    lr_stat: float
    p_value: float


def _xlogy(x: float, y: float) -> float:
    """``x * log(y)`` with the MLE convention ``0 * log(0) == 0``.

    This is the exact limit of the binomial log-likelihood terms as an
    empirical rate hits 0 or 1, so the LR tests below stay finite at the
    edges (0 breaches, all breaches) without arbitrary epsilon clamping.
    """
    if x == 0.0:
        return 0.0
    return x * math.log(y)


def _chi2_sf(x: float, dof: int) -> float:
    """Chi-square survival function ``P(X > x)`` for ``dof`` in {1, 2}.

    Closed forms, no scipy: for 1 dof, X = Z² with Z standard normal, so
    ``P(X > x) = 2·P(Z > √x) = erfc(√(x/2))``; for 2 dof, chi-square is
    Exponential(rate=1/2), so ``P(X > x) = exp(-x/2)``. Non-positive ``x``
    returns 1.0.
    """
    if dof not in (1, 2):
        raise ValueError(f"_chi2_sf supports dof 1 or 2 only, got {dof}")
    if x <= 0.0:
        return 1.0
    if dof == 1:
        return math.erfc(math.sqrt(x / 2.0))
    return math.exp(-x / 2.0)


def kupiec_pof(n_breaches: int, n_obs: int, expected_rate: float) -> LRTestResult:
    """Kupiec proportion-of-failures test: is the breach *rate* as advertised?

    With ``n1 = n_breaches``, ``n0 = n_obs - n1``, ``p = expected_rate`` and
    the observed rate ``π̂ = n1/n_obs``::

        LR_pof = -2·ln[ (1-p)^n0 · p^n1 / ((1-π̂)^n0 · π̂^n1) ]  ~  χ²(1)

    under H0 (true breach probability = ``p``). Small p-value ⇒ the observed
    breach count is inconsistent with the nominal rate (either direction:
    too many breaches *or* suspiciously few, i.e. over-wide bands). The
    ``n1 = 0`` and ``n1 = n_obs`` edges are exact via the ``0·log(0) = 0``
    convention (see ``_xlogy``) — no epsilon fudging.

    Raises ``ValueError`` unless ``n_obs > 0``, ``0 <= n_breaches <= n_obs``,
    and ``0 < expected_rate < 1``.
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}")
    if not 0 <= n_breaches <= n_obs:
        raise ValueError(f"n_breaches must be in [0, n_obs], got {n_breaches} of {n_obs}")
    if not 0.0 < expected_rate < 1.0:
        raise ValueError(f"expected_rate must be in (0, 1), got {expected_rate}")
    n1 = n_breaches
    n0 = n_obs - n_breaches
    pi_hat = n1 / n_obs
    ll_null = _xlogy(n0, 1.0 - expected_rate) + _xlogy(n1, expected_rate)
    ll_alt = _xlogy(n0, 1.0 - pi_hat) + _xlogy(n1, pi_hat)
    lr = max(0.0, 2.0 * (ll_alt - ll_null))
    return LRTestResult(lr_stat=lr, p_value=_chi2_sf(lr, 1))
