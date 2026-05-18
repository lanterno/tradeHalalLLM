"""Hierarchical Risk Parity (HRP) allocator.

Round-4 wave 4.G: replaces the flat ``max_position_pct`` cap with a
covariance-aware allocation across the active candidate set. Based on
Marcos López de Prado's *Building Diversified Portfolios that
Outperform Out-of-Sample* (Journal of Portfolio Management, 2016).

Why HRP rather than mean-variance / Black-Litterman:

* **Covariance-only.** No expected-return inputs. The bot already has
  a separate per-pair conviction signal (LLM confidence × edge × vol)
  in ``core/sizing.py`` — we don't want a second one. HRP just
  asks "given how these assets co-move, how should I split a fixed
  budget across them so no single cluster dominates risk?"
* **Numerically stable.** Mean-variance requires inverting the
  covariance matrix, which is ill-conditioned when assets are highly
  correlated (BTC + ETH + every alt that hugs them). HRP never
  inverts; it uses hierarchical clustering + recursive bisection.
* **Interpretable.** The output cluster tree mirrors what an operator
  would draw by eye: "majors", "DeFi blue chips", "alt L1s". Easy to
  audit; easy to gate on halal sub-cluster constraints.

Halal constraints baked in:

* Weights are **always non-negative** (no shorts).
* Weights sum to ``≤ 1.0`` (no leverage). Caller can scale by total
  available equity afterwards.
* The allocator is symbol-agnostic — it operates on a returns matrix.
  The caller is responsible for filtering the universe to halal
  assets before invoking ``allocate``.

The implementation uses pure NumPy with no SciPy / sklearn dependency
so it imports cleanly without the ``[ml]`` extra. Hierarchical
clustering uses the textbook single-linkage agglomerative algorithm
(O(n²·log n) — fine for ≤ 100 assets which is our universe).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HRPAllocation:
    """Result of an HRP run.

    ``weights`` is a dict of symbol → fractional allocation in [0, 1].
    Sums to ≤ 1.0 (will equal 1.0 unless the caller passed
    ``cash_buffer_pct``). Order of insertion mirrors the cluster
    ordering, which is useful for log readability ("majors first,
    then alts").

    ``cluster_order`` exposes the symbol order produced by the
    quasi-diagonalisation step — handy for explanatory dashboards
    that want to render the dendrogram-like ordering.
    """

    weights: dict[str, float]
    cluster_order: list[str]


def _correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """López de Prado's distance metric: ``d_ij = sqrt(0.5 * (1 -
    ρ_ij))``. Range [0, 1]; obeys the triangle inequality so the
    clustering output is well-defined.

    Clamps correlations to [-1, 1] before the transform so floating-
    point drift on near-degenerate matrices doesn't yield ``nan``.
    """
    corr_clamped = np.clip(corr, -1.0, 1.0)
    return np.sqrt(0.5 * (1.0 - corr_clamped))


def _single_linkage_order(distance: np.ndarray) -> list[int]:
    """Single-linkage hierarchical clustering → leaf ordering.

    We don't need the full linkage matrix; HRP only needs the *order*
    of leaves once the tree is built, because the recursive bisection
    walks that order. Implementing the textbook nearest-neighbour
    chain gives O(n²) memory and O(n²·log n) time which is fine for
    our universe size.

    Returns the list of original indices in cluster-traversal order.
    """
    n = distance.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Each "active cluster" is represented as a list of original
    # indices in its in-cluster order.
    clusters: list[list[int]] = [[i] for i in range(n)]
    # Pairwise inter-cluster distances; updated under single linkage
    # (min over pairs).
    dist = distance.copy()
    np.fill_diagonal(dist, np.inf)

    while len(clusters) > 1:
        # Find the closest pair of clusters.
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = int(idx[0]), int(idx[1])
        if i > j:
            i, j = j, i
        # Merge cluster j into cluster i (concatenate orderings).
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)
        # Update distance row/col i with single-linkage min, then
        # delete row/col j.
        new_row = np.minimum(dist[i, :], dist[j, :])
        new_row[i] = np.inf
        dist[i, :] = new_row
        dist[:, i] = new_row
        dist = np.delete(dist, j, axis=0)
        dist = np.delete(dist, j, axis=1)

    return clusters[0]


def _inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Within-cluster weighting: inversely proportional to each
    asset's variance, then normalised. This is the leaf-level
    allocation HRP uses inside each recursion step."""
    inv_var = 1.0 / np.diag(cov)
    return inv_var / inv_var.sum()


def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    """Variance of the inverse-variance-weighted sub-portfolio over
    the asset indices passed in. Matches the López de Prado paper's
    ``getClusterVar`` helper."""
    sub = cov[np.ix_(indices, indices)]
    w = _inverse_variance_weights(sub)
    return float(w @ sub @ w)


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Allocate 100% across ``order`` by repeatedly splitting it in
    half and dividing the parent's weight between the two children
    inversely-proportional to their cluster variances.

    Returns a weight vector indexed by *original* asset index (zeros
    for positions not in ``order``, though callers always pass the
    full order).
    """
    n = cov.shape[0]
    weights = np.zeros(n)
    weights[order] = 1.0

    # Each work item is a sub-list of the cluster order to bisect.
    queue: list[list[int]] = [order]
    while queue:
        cluster = queue.pop()
        if len(cluster) <= 1:
            continue
        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]
        var_left = _cluster_variance(cov, left)
        var_right = _cluster_variance(cov, right)
        # Allocation factor: more weight to the *lower-variance*
        # child. Falls out of the inverse-variance formula.
        if var_left + var_right == 0:
            alpha = 0.5
        else:
            alpha = 1.0 - var_left / (var_left + var_right)
        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= 1.0 - alpha
        queue.append(left)
        queue.append(right)

    return weights


def allocate(
    returns: np.ndarray,
    symbols: list[str],
    *,
    cash_buffer_pct: float = 0.0,
    min_history: int = 30,
) -> HRPAllocation:
    """Compute HRP weights for ``symbols`` from a returns matrix.

    ``returns`` is shape ``(T, N)`` — rows are time periods, columns
    are assets in the same order as ``symbols``. Use simple percent
    returns (or log returns, the algorithm doesn't care).

    ``cash_buffer_pct`` is reserved as cash (returned as the implicit
    leftover; weights sum to ``1 - cash_buffer_pct``). Useful to
    leave room for rebalancing slippage.

    ``min_history`` rejects allocations on too-thin a window; raises
    ``ValueError`` rather than producing meaningless weights from
    a 5-row covariance estimate.
    """
    if not 0.0 <= cash_buffer_pct < 1.0:
        raise ValueError(f"cash_buffer_pct must be in [0, 1); got {cash_buffer_pct}")
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2D; got shape {returns.shape}")
    n_periods, n_assets = returns.shape
    if n_assets != len(symbols):
        raise ValueError(f"returns has {n_assets} columns but {len(symbols)} symbols supplied")
    if n_periods < min_history:
        raise ValueError(
            f"returns has only {n_periods} rows; need at least {min_history} for stable cov"
        )
    if n_assets == 0:
        return HRPAllocation(weights={}, cluster_order=[])
    if n_assets == 1:
        return HRPAllocation(
            weights={symbols[0]: 1.0 - cash_buffer_pct},
            cluster_order=[symbols[0]],
        )

    # Drop assets with zero variance — they break the correlation
    # matrix and the inverse-variance step. Treat as "skip", not
    # "fail"; the caller can re-invoke after refreshing prices.
    variances = returns.var(axis=0)
    keep_mask = variances > 0
    if not keep_mask.all():
        skipped = [s for s, k in zip(symbols, keep_mask) if not k]
        logger.info("hrp.skip_zero_variance assets=%s", skipped)
    keep_indices = np.where(keep_mask)[0]
    if len(keep_indices) == 0:
        return HRPAllocation(weights={}, cluster_order=[])
    if len(keep_indices) == 1:
        only = symbols[int(keep_indices[0])]
        return HRPAllocation(
            weights={only: 1.0 - cash_buffer_pct},
            cluster_order=[only],
        )

    sub_returns = returns[:, keep_indices]
    sub_symbols = [symbols[i] for i in keep_indices]

    # 1. Covariance + correlation.
    cov = np.cov(sub_returns, rowvar=False)
    std = np.sqrt(np.diag(cov))
    # Guard against zero std (theoretically excluded above, but
    # defend against numerical edges).
    std = np.where(std == 0, 1.0, std)
    corr = cov / np.outer(std, std)

    # 2. Distance + clustering → leaf order.
    distance = _correlation_to_distance(corr)
    order = _single_linkage_order(distance)

    # 3. Recursive bisection → weights.
    raw_weights = _recursive_bisection(cov, order)

    # 4. Apply cash buffer + assemble result in cluster order.
    scale = 1.0 - cash_buffer_pct
    weights: dict[str, float] = {}
    for idx in order:
        weights[sub_symbols[idx]] = float(raw_weights[idx] * scale)

    return HRPAllocation(
        weights=weights,
        cluster_order=[sub_symbols[i] for i in order],
    )
