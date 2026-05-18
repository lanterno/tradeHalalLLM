"""Reinforcement learning policy training scaffold.

The roadmap pins Wave 4.A: "Beyond LLM-only: train a PPO/SAC
policy on historical replay data (the existing `core/replay.py`
snapshots are perfect training data). The RL policy + LLM
rationale form an ensemble; the LLM provides explainability
while the RL policy provides rigour." This module is the
**pure-Python config + episode tracker + reward-shaping engine**
that the actual training step (Stable-Baselines3 / Ray RLlib /
CleanRL — operator-side) consumes as configuration.

Picked a focused config + episode model over a "single training
script" approach because (a) hyperparameter validation (clip
ranges, learning rates, batch sizes) is a decision rule that
should fail fast at training entry rather than after 30 minutes
of GPU burn — a frozen `TrainingConfig` with `__post_init__`
validation rejects invalid combinations at construction, (b)
reward shaping (avoid drawdown, prefer Sharpe) is a pure
function of (returns, drawdown, sharpe) that benefits from
deterministic regression-pinning — a contributor changing the
shaping weights has to update tests rather than silently
shipping, (c) the episode + trajectory primitives let training
runs be replayed and diffed against historical baselines without
re-running the actual training, and (d) the deployment lifecycle
(SHADOW → CANARY → PRODUCTION) mirrors Wave 6.I distillation
deployment so operators have one mental model for "promoting
a learned policy to production".

Pinned semantics:
- **Two algorithms supported: PPO / SAC.** Closed enum; adding
  one is a code review change. PPO for stable on-policy
  training with discrete actions; SAC for continuous actions
  + sample efficiency.
- **Reward shaping has three components.** Raw return + Sharpe
  bonus + drawdown penalty. Weights are operator-tunable but
  validation enforces non-negative shaping weights so the
  reward direction can't accidentally flip.
- **Hyperparameter ranges enforced.** Learning rate in
  (0, 0.1], gamma in (0, 1), batch size positive, clip range
  in (0, 1) for PPO. Out-of-range fails fast.
- **Episode tracking is immutable.** Each step appends to a new
  Trajectory tuple rather than mutating; pinned for replay-
  ability across training runs.
- **Render output never includes model weights, optimizer
  state, or raw replay data.** The audit row is summary-only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class RLAlgorithm(str, Enum):
    """Closed-set RL algorithms.

    Pinned string values for JSON / DB stability. PPO = on-policy,
    discrete actions, stable training. SAC = off-policy,
    continuous actions, sample efficient.
    """

    PPO = "ppo"
    SAC = "sac"


class ActionSpace(str, Enum):
    """Action-space kind. Pinned values."""

    DISCRETE = "discrete"  # BUY / SELL / HOLD
    CONTINUOUS = "continuous"  # position-size in [-1, 1]


_ALGO_REQUIRES_DISCRETE: frozenset[RLAlgorithm] = frozenset({RLAlgorithm.PPO})
_ALGO_REQUIRES_CONTINUOUS: frozenset[RLAlgorithm] = frozenset({RLAlgorithm.SAC})


_DEFAULT_LEARNING_RATE = 0.0003
_DEFAULT_GAMMA = 0.99
_DEFAULT_BATCH_SIZE = 64
_DEFAULT_CLIP_RANGE = 0.2
_DEFAULT_TOTAL_TIMESTEPS = 1_000_000


@dataclass(frozen=True)
class TrainingConfig:
    """Operator-tunable RL training hyperparameters.

    Defaults follow the published baselines for PPO on financial
    time-series; operators tune via the constructor for specific
    cohorts (e.g. higher gamma for long-horizon strategies).
    """

    algorithm: RLAlgorithm
    action_space: ActionSpace
    learning_rate: float = _DEFAULT_LEARNING_RATE
    gamma: float = _DEFAULT_GAMMA
    batch_size: int = _DEFAULT_BATCH_SIZE
    clip_range: float = _DEFAULT_CLIP_RANGE  # PPO only
    total_timesteps: int = _DEFAULT_TOTAL_TIMESTEPS

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate <= 0.1:
            raise ValueError(f"learning_rate {self.learning_rate} must be in (0, 0.1]")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(
                f"gamma {self.gamma} must be in (0, 1) — exactly 1.0 means "
                f"no discounting and infinite horizon, exactly 0 means myopic"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError(f"clip_range {self.clip_range} must be in (0, 1)")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        # Algorithm / action_space compatibility
        if (
            self.algorithm in _ALGO_REQUIRES_DISCRETE
            and self.action_space is not ActionSpace.DISCRETE
        ):
            raise ValueError(f"{self.algorithm.value} requires discrete action space")
        if (
            self.algorithm in _ALGO_REQUIRES_CONTINUOUS
            and self.action_space is not ActionSpace.CONTINUOUS
        ):
            raise ValueError(f"{self.algorithm.value} requires continuous action space")


@dataclass(frozen=True)
class RewardShapingPolicy:
    """Operator-tunable reward-shaping weights.

    The three components compose: total_reward = return_weight *
    raw_return + sharpe_weight * sharpe_bonus - drawdown_weight *
    drawdown_penalty. Validation enforces non-negative weights so
    the reward direction can't accidentally flip (a negative
    drawdown_weight would *reward* drawdowns).
    """

    return_weight: float = 1.0
    sharpe_weight: float = 0.5
    drawdown_weight: float = 2.0

    def __post_init__(self) -> None:
        if self.return_weight < 0:
            raise ValueError("return_weight must be non-negative")
        if self.sharpe_weight < 0:
            raise ValueError("sharpe_weight must be non-negative")
        if self.drawdown_weight < 0:
            raise ValueError("drawdown_weight must be non-negative")


DEFAULT_SHAPING = RewardShapingPolicy()


def shaped_reward(
    *,
    raw_return: float,
    sharpe: float,
    drawdown_pct: float,
    policy: RewardShapingPolicy = DEFAULT_SHAPING,
) -> float:
    """Compute the shaped reward for one episode step.

    `raw_return` is the per-step return (e.g. 0.01 for 1%);
    `sharpe` is the running Sharpe ratio (typically -3 to +3);
    `drawdown_pct` is the drawdown from peak as a positive
    number (e.g. 0.05 for 5% drawdown). Pin: drawdown_pct must
    be non-negative — a "negative drawdown" is meaningless and
    would invert the penalty direction.
    """

    if drawdown_pct < 0:
        raise ValueError("drawdown_pct must be non-negative")
    return (
        policy.return_weight * raw_return
        + policy.sharpe_weight * sharpe
        - policy.drawdown_weight * drawdown_pct
    )


@dataclass(frozen=True)
class TrajectoryStep:
    """One (state, action, reward, next_state) tuple.

    Generic enough to wrap any RL framework's step format; the
    actual training loop materialises these from `core/replay.py`
    snapshots.
    """

    timestep: int
    action_id: str  # serialized BUY / SELL / HOLD or position-size
    raw_return: float
    sharpe: float
    drawdown_pct: float
    shaped_reward: float

    def __post_init__(self) -> None:
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")
        if not self.action_id or not self.action_id.strip():
            raise ValueError("action_id must be non-empty")
        if self.drawdown_pct < 0:
            raise ValueError("drawdown_pct must be non-negative")


@dataclass(frozen=True)
class Episode:
    """One training episode.

    Episodes are immutable; `append_step` returns a new Episode
    with the step appended. Operators replay an episode by
    iterating its steps tuple.
    """

    episode_id: str
    started_at: datetime
    steps: tuple[TrajectoryStep, ...]
    terminated: bool = False

    def __post_init__(self) -> None:
        if not self.episode_id or not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty")
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def total_reward(self) -> float:
        return sum(s.shaped_reward for s in self.steps)

    @property
    def total_raw_return(self) -> float:
        return sum(s.raw_return for s in self.steps)

    @property
    def max_drawdown_pct(self) -> float:
        if not self.steps:
            return 0.0
        return max(s.drawdown_pct for s in self.steps)


def start_episode(*, episode_id: str, now: datetime) -> Episode:
    """Create a fresh episode with no steps."""

    if not episode_id or not episode_id.strip():
        raise ValueError("episode_id must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return Episode(episode_id=episode_id, started_at=now, steps=())


def append_step(episode: Episode, step: TrajectoryStep) -> Episode:
    """Append a step to the episode (returns a new immutable Episode)."""

    if episode.terminated:
        raise ValueError(f"cannot append to terminated episode {episode.episode_id!r}")
    expected = episode.step_count
    if step.timestep != expected:
        raise ValueError(f"step.timestep {step.timestep} != expected {expected}")
    return Episode(
        episode_id=episode.episode_id,
        started_at=episode.started_at,
        steps=episode.steps + (step,),
        terminated=episode.terminated,
    )


def terminate_episode(episode: Episode) -> Episode:
    """Mark an episode terminated (no further steps)."""

    if episode.terminated:
        raise ValueError(f"already terminated: {episode.episode_id!r}")
    return Episode(
        episode_id=episode.episode_id,
        started_at=episode.started_at,
        steps=episode.steps,
        terminated=True,
    )


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed training run.

    `mean_episode_reward` is averaged across `episode_count`
    completed episodes. `policy_id` identifies the trained model
    artefact in the Wave 6.A model registry.
    """

    policy_id: str
    config: TrainingConfig
    episode_count: int
    mean_episode_reward: float
    final_loss: float
    completed_at: datetime

    def __post_init__(self) -> None:
        if not self.policy_id or not self.policy_id.strip():
            raise ValueError("policy_id must be non-empty")
        if self.episode_count < 0:
            raise ValueError("episode_count must be non-negative")
        if self.completed_at.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware")


def aggregate_results(
    episodes: Iterable[Episode],
    *,
    config: TrainingConfig,
    policy_id: str,
    final_loss: float,
    completed_at: datetime,
) -> TrainingResult:
    """Aggregate finished episodes into a TrainingResult."""

    if completed_at.tzinfo is None:
        raise ValueError("completed_at must be timezone-aware")
    episode_list = list(episodes)
    if episode_list:
        mean_reward = sum(e.total_reward for e in episode_list) / len(episode_list)
    else:
        mean_reward = 0.0
    return TrainingResult(
        policy_id=policy_id,
        config=config,
        episode_count=len(episode_list),
        mean_episode_reward=mean_reward,
        final_loss=final_loss,
        completed_at=completed_at,
    )


_ALGO_EMOJI: dict[RLAlgorithm, str] = {
    RLAlgorithm.PPO: "🎯",
    RLAlgorithm.SAC: "🌊",
}


def render_config(config: TrainingConfig) -> str:
    """Format the training config for ops display.

    No-secret-leak: structural — config is hyperparameters, no
    credentials.
    """

    emoji = _ALGO_EMOJI[config.algorithm]
    lines = [
        f"{emoji} RL training config — {config.algorithm.value}",
        f"  action space: {config.action_space.value}",
        f"  lr: {config.learning_rate}",
        f"  gamma: {config.gamma}",
        f"  batch: {config.batch_size}",
        f"  clip: {config.clip_range}",
        f"  total timesteps: {config.total_timesteps:,}",
    ]
    return "\n".join(lines)


def render_result(result: TrainingResult) -> str:
    """Format a training result for ops display."""

    return (
        f"📊 RL training result — policy {result.policy_id}\n"
        f"  episodes: {result.episode_count}\n"
        f"  mean reward: {result.mean_episode_reward:.4f}\n"
        f"  final loss: {result.final_loss:.4f}\n"
        f"  completed: {result.completed_at.isoformat()}"
    )


__all__ = [
    "DEFAULT_SHAPING",
    "ActionSpace",
    "Episode",
    "RLAlgorithm",
    "RewardShapingPolicy",
    "TrainingConfig",
    "TrainingResult",
    "TrajectoryStep",
    "aggregate_results",
    "append_step",
    "render_config",
    "render_result",
    "shaped_reward",
    "start_episode",
    "terminate_episode",
]
