"""K-nearest-neighbor past-setup retrieval for the LLM prompt.

The ML pipeline already labels every closed trade's entry-indicator
snapshot with its realised return. This module turns that labelled
history into a *prior* for the LLM: given the current pair's indicator
vector, fetch the K most-similar past setups and format their
realised PnL as a compact "you've seen this before" prompt section.

This is the highest-leverage ML use right now because:

* It composes with prompt caching — the cached static prefix never
  changes; the per-cycle KNN block sits in the dynamic suffix.
* It teaches the LLM from its own track record without us having to
  fine-tune anything.
* Failures degrade gracefully — when there aren't enough labeled
  snapshots, the function returns ``""`` and the prompt section is
  simply elided (already supported by the optional-section logic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from halal_trader.ml.features import FEATURE_KEYS

_FEATURES = list(FEATURE_KEYS)


@dataclass(frozen=True)
class PastSetup:
    """One labeled past trade with its features + realized return."""

    pair: str
    features: tuple[float, ...]
    return_pct: float
    label: int  # 1 = profitable, 0 = unprofitable
    distance: float = 0.0


def _to_vector(payload: dict) -> tuple[float, ...] | None:
    """Pull a fixed-order feature tuple from any dict that exposes ``FEATURE_KEYS``.

    Returns ``None`` when any key is missing — KNN must compare apples to
    apples, so an incomplete sample is dropped rather than imputed.
    """
    out = []
    for key in _FEATURES:
        v = payload.get(key)
        if v is None:
            return None
        try:
            out.append(float(v))
        except Exception:
            return None
    return tuple(out)


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    return float(np.linalg.norm(arr_a - arr_b))


def find_nearest_setups(
    current: dict,
    past: Sequence[dict],
    *,
    k: int = 5,
) -> list[PastSetup]:
    """Return the ``k`` past setups with the smallest Euclidean distance.

    ``past`` is a sequence of indicator-snapshot dicts (typically pulled
    from ``Repository.get_labeled_snapshots``). Distance is computed in
    raw feature space — that's biased toward features with larger
    magnitudes, but the production features are already on roughly
    comparable scales (RSI 0-100, ATR fractions, MACD histogram). A
    follow-up can normalise via z-score once we have enough samples to
    fit it without bias.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    cur = _to_vector(current)
    if cur is None:
        return []

    candidates: list[PastSetup] = []
    for snap in past:
        feats = _to_vector(snap)
        if feats is None:
            continue
        if snap.get("return_pct") is None or snap.get("label") is None:
            continue
        candidates.append(
            PastSetup(
                pair=snap.get("pair", ""),
                features=feats,
                return_pct=float(snap["return_pct"]),
                label=int(snap["label"]),
                distance=_euclidean(cur, feats),
            )
        )

    candidates.sort(key=lambda s: s.distance)
    return candidates[:k]


def format_setups_for_prompt(setups: Sequence[PastSetup]) -> str:
    """Render a KNN result list as a compact bullet block.

    Empty when there are no setups so the prompt template can elide the
    section. Each line is ``pair: realised return (W/L)`` so the LLM
    sees both magnitude and direction.
    """
    if not setups:
        return ""
    wins = sum(1 for s in setups if s.label == 1)
    avg_return = float(np.mean([s.return_pct for s in setups]))
    header = (
        f"  Nearest {len(setups)} past setups: avg return {avg_return * 100:+.2f}%, "
        f"{wins}/{len(setups)} profitable"
    )
    lines = [header]
    for s in setups:
        outcome = "W" if s.label == 1 else "L"
        lines.append(
            f"    - {s.pair}: {s.return_pct * 100:+.2f}% ({outcome}, dist={s.distance:.3f})"
        )
    return "\n".join(lines)
