"""Wave F fitness functions for the prompt-evolution GA.

The runner in ``core/llm/prompt_evo_runner.py`` is callable-based: each
``Evaluator`` returns a float fitness for one ``(genome, snapshot)``
pair. This module ships the concrete evaluators the bot's nightly
job uses.

Design constraints (deliberately conservative):

* **No live LLM calls.** Replaying a real LLM against every
  ``(genome, snapshot)`` pair would cost ~12 × 200 × N_generations
  calls per nightly run — economically unviable. Today's evaluators
  are *cheap signals* over snapshot metadata and prompt-shape features
  the GA can still rank-order against. Future work can swap in a
  small fine-tuned distilled model as a per-pair scorer.
* **Stable under partial data.** Snapshots predating Wave F won't have
  every prompt-context field populated. The evaluators degrade
  gracefully — NaN / unscored pairs are dropped by the runner.
* **Halal-safe.** Evaluators never bias toward more-frequent trading
  or larger sizing — the GA can't optimise the bot into worse risk
  posture by gaming the fitness curve.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from halal_trader.core.llm.prompt_evo import PromptGenome
    from halal_trader.core.replay import CycleSnapshot

logger = logging.getLogger(__name__)


# Soft regulariser: prefer concise alleles (token cost is real).
# 1 char ≈ 0.25 tokens; the LLM's per-input-token cost dominates
# nightly spend, so a 1k-char-longer prompt × 24h × 1440 cycles =
# ~10M extra tokens / day. The regulariser is small enough not to
# dominate the P&L signal but large enough that wasteful padding
# (e.g. a 500-char filler allele) loses to a tight one.
_LEN_PENALTY_PER_CHAR = 1.0e-6


async def replay_pnl_fitness(
    genome: "PromptGenome",
    snapshot: "CycleSnapshot",
) -> float:
    """Fitness = signed today_pnl from the snapshot minus a length penalty.

    Returns the cycle's realised today_pnl_pct (computed from
    ``snapshot.today_pnl`` and the snapshot's recorded equity) as a raw
    fraction, regularised by the total length of the genome's allele
    text. Positive scores beat negative scores; among genomes with
    similar P&L the GA prefers the more concise prompt.

    This is intentionally a *weak* signal — the GA explores the slot
    space but doesn't claim to predict P&L. A useful run still produces
    a ranking of candidate prompts the operator can inspect via
    ``/api/prompts/candidates`` and promote with one click.
    """
    equity = float((snapshot.account or {}).get("total_balance_usdt") or 0.0)
    pnl_pct = (snapshot.today_pnl / equity) if equity > 0 else 0.0
    length_bytes = sum(len(s) for s in genome.slots.values())
    return float(pnl_pct - _LEN_PENALTY_PER_CHAR * length_bytes)


async def confidence_proxy_fitness(
    genome: "PromptGenome",
    snapshot: "CycleSnapshot",
) -> float:
    """Fitness = aggregate ML-signal confidence the snapshot saw.

    A lightweight proxy when ``today_pnl`` is too noisy (early-history
    cycles, low-volume regimes): pulls the ``ml_signals_text`` mean
    confidence the LLM would have seen, biases the GA toward genomes
    paired with cycles where the ML detector spotted *something*. No
    causal claim — useful as a sanity-check evaluator alongside
    :func:`replay_pnl_fitness`.
    """
    text = snapshot.ml_signals_text or ""
    if not text:
        return 0.0
    # The signals_text format includes lines like
    # ``BTCUSDT: anomaly_score=0.72, confidence=0.81``. Pull mean
    # confidence cheaply — a single regex would be tighter but we
    # want to tolerate format drift without crashing the nightly job.
    scores: list[float] = []
    for token in text.split():
        if token.startswith("confidence="):
            try:
                scores.append(float(token.removeprefix("confidence=").rstrip(",;")))
            except ValueError:
                continue
    if not scores:
        return 0.0
    length_bytes = sum(len(s) for s in genome.slots.values())
    return float(sum(scores) / len(scores) - _LEN_PENALTY_PER_CHAR * length_bytes)
