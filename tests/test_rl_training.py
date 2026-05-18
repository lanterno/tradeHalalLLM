"""Tests for `halal_trader.ml.rl_training` (Wave 4.A).

Covers: TrainingConfig hyperparameter validation, algorithm/action-
space compatibility, reward shaping math, episode tracking
immutability, no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.ml.rl_training import (
    DEFAULT_SHAPING,
    ActionSpace,
    Episode,
    RewardShapingPolicy,
    RLAlgorithm,
    TrainingConfig,
    TrainingResult,
    TrajectoryStep,
    aggregate_results,
    append_step,
    render_config,
    render_result,
    shaped_reward,
    start_episode,
    terminate_episode,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_rl_algorithm_string_values_pinned() -> None:
    assert RLAlgorithm.PPO.value == "ppo"
    assert RLAlgorithm.SAC.value == "sac"


def test_action_space_string_values_pinned() -> None:
    assert ActionSpace.DISCRETE.value == "discrete"
    assert ActionSpace.CONTINUOUS.value == "continuous"


# --------------------------- TrainingConfig ----------------------------------


def _config(**overrides: object) -> TrainingConfig:
    base: dict[str, object] = {
        "algorithm": RLAlgorithm.PPO,
        "action_space": ActionSpace.DISCRETE,
    }
    base.update(overrides)
    return TrainingConfig(**base)  # type: ignore[arg-type]


def test_default_config_is_valid() -> None:
    config = _config()
    assert config.learning_rate == 0.0003
    assert config.gamma == 0.99
    assert config.batch_size == 64
    assert config.clip_range == 0.2
    assert config.total_timesteps == 1_000_000


def test_config_rejects_zero_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        _config(learning_rate=0.0)


def test_config_rejects_negative_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        _config(learning_rate=-0.001)


def test_config_rejects_learning_rate_above_max() -> None:
    """Pin: 0.1 inclusive; 0.11 fails."""

    with pytest.raises(ValueError, match="learning_rate"):
        _config(learning_rate=0.11)


def test_config_accepts_learning_rate_at_upper_boundary() -> None:
    config = _config(learning_rate=0.1)
    assert config.learning_rate == 0.1


def test_config_rejects_gamma_at_zero() -> None:
    with pytest.raises(ValueError, match="gamma"):
        _config(gamma=0.0)


def test_config_rejects_gamma_at_one() -> None:
    """Pin: gamma=1.0 means infinite horizon, rejected."""

    with pytest.raises(ValueError, match="gamma"):
        _config(gamma=1.0)


def test_config_rejects_zero_batch() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _config(batch_size=0)


def test_config_rejects_negative_batch() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _config(batch_size=-1)


def test_config_rejects_clip_range_at_zero() -> None:
    with pytest.raises(ValueError, match="clip_range"):
        _config(clip_range=0.0)


def test_config_rejects_clip_range_at_one() -> None:
    with pytest.raises(ValueError, match="clip_range"):
        _config(clip_range=1.0)


def test_config_rejects_zero_total_timesteps() -> None:
    with pytest.raises(ValueError, match="total_timesteps"):
        _config(total_timesteps=0)


def test_ppo_requires_discrete_action_space() -> None:
    """Pin: PPO is for discrete actions."""

    with pytest.raises(ValueError, match="discrete"):
        _config(algorithm=RLAlgorithm.PPO, action_space=ActionSpace.CONTINUOUS)


def test_sac_requires_continuous_action_space() -> None:
    """Pin: SAC is for continuous actions."""

    with pytest.raises(ValueError, match="continuous"):
        _config(algorithm=RLAlgorithm.SAC, action_space=ActionSpace.DISCRETE)


def test_sac_with_continuous_works() -> None:
    config = _config(algorithm=RLAlgorithm.SAC, action_space=ActionSpace.CONTINUOUS)
    assert config.algorithm is RLAlgorithm.SAC


def test_config_is_frozen() -> None:
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.learning_rate = 0.001  # type: ignore[misc]


# --------------------------- RewardShapingPolicy -----------------------------


def test_default_shaping_weights() -> None:
    """Pin: default emphasises drawdown penalty (2x return weight)."""

    assert DEFAULT_SHAPING.return_weight == 1.0
    assert DEFAULT_SHAPING.sharpe_weight == 0.5
    assert DEFAULT_SHAPING.drawdown_weight == 2.0


def test_shaping_rejects_negative_return_weight() -> None:
    with pytest.raises(ValueError, match="return_weight"):
        RewardShapingPolicy(return_weight=-0.1)


def test_shaping_rejects_negative_sharpe_weight() -> None:
    with pytest.raises(ValueError, match="sharpe_weight"):
        RewardShapingPolicy(sharpe_weight=-0.1)


def test_shaping_rejects_negative_drawdown_weight() -> None:
    """Pin: drawdown_weight must be non-negative — a negative weight
    would *reward* drawdowns, inverting the safety guarantee."""

    with pytest.raises(ValueError, match="drawdown_weight"):
        RewardShapingPolicy(drawdown_weight=-0.1)


def test_shaping_accepts_zero_weight() -> None:
    """Pin: zero weight is valid (operator wants to disable that
    component) — only negative is rejected."""

    p = RewardShapingPolicy(return_weight=0.0, sharpe_weight=0.0, drawdown_weight=0.0)
    assert p.return_weight == 0.0


def test_shaping_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_SHAPING.return_weight = 99.0  # type: ignore[misc]


# --------------------------- shaped_reward -----------------------------------


def test_shaped_reward_basic() -> None:
    """1.0 * 0.01 + 0.5 * 1.0 - 2.0 * 0.05 = 0.01 + 0.5 - 0.10 = 0.41"""

    reward = shaped_reward(raw_return=0.01, sharpe=1.0, drawdown_pct=0.05)
    assert reward == pytest.approx(0.41)


def test_shaped_reward_zero_drawdown() -> None:
    reward = shaped_reward(raw_return=0.01, sharpe=1.0, drawdown_pct=0.0)
    assert reward == pytest.approx(0.51)


def test_shaped_reward_negative_return_with_drawdown() -> None:
    """Pin: a losing step with drawdown gets a doubly-negative reward."""

    reward = shaped_reward(raw_return=-0.02, sharpe=-0.5, drawdown_pct=0.10)
    # 1.0 * -0.02 + 0.5 * -0.5 - 2.0 * 0.10 = -0.02 - 0.25 - 0.20 = -0.47
    assert reward == pytest.approx(-0.47)


def test_shaped_reward_rejects_negative_drawdown() -> None:
    """Pin: drawdown_pct is non-negative by construction."""

    with pytest.raises(ValueError, match="drawdown_pct"):
        shaped_reward(raw_return=0.0, sharpe=0.0, drawdown_pct=-0.01)


def test_shaped_reward_custom_policy() -> None:
    """Operator can disable the drawdown penalty entirely."""

    no_dd = RewardShapingPolicy(drawdown_weight=0.0)
    reward = shaped_reward(raw_return=0.01, sharpe=1.0, drawdown_pct=0.5, policy=no_dd)
    # Should equal 0.01 + 0.5 = 0.51 (drawdown ignored)
    assert reward == pytest.approx(0.51)


# --------------------------- TrajectoryStep ----------------------------------


def test_trajectory_step_rejects_negative_timestep() -> None:
    with pytest.raises(ValueError, match="timestep"):
        TrajectoryStep(
            timestep=-1,
            action_id="BUY",
            raw_return=0.0,
            sharpe=0.0,
            drawdown_pct=0.0,
            shaped_reward=0.0,
        )


def test_trajectory_step_rejects_empty_action_id() -> None:
    with pytest.raises(ValueError, match="action_id"):
        TrajectoryStep(
            timestep=0,
            action_id="",
            raw_return=0.0,
            sharpe=0.0,
            drawdown_pct=0.0,
            shaped_reward=0.0,
        )


def test_trajectory_step_rejects_negative_drawdown() -> None:
    with pytest.raises(ValueError, match="drawdown_pct"):
        TrajectoryStep(
            timestep=0,
            action_id="BUY",
            raw_return=0.0,
            sharpe=0.0,
            drawdown_pct=-0.01,
            shaped_reward=0.0,
        )


def test_trajectory_step_is_frozen() -> None:
    step = TrajectoryStep(
        timestep=0,
        action_id="BUY",
        raw_return=0.01,
        sharpe=1.0,
        drawdown_pct=0.05,
        shaped_reward=0.41,
    )
    with pytest.raises(FrozenInstanceError):
        step.shaped_reward = 99.0  # type: ignore[misc]


# --------------------------- Episode -----------------------------------------


def test_episode_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="episode_id"):
        Episode(episode_id="", started_at=T0, steps=())


def test_episode_rejects_naive_started_at() -> None:
    with pytest.raises(ValueError, match="started_at"):
        Episode(
            episode_id="ep",
            started_at=datetime(2026, 5, 1),
            steps=(),
        )


def test_episode_is_frozen() -> None:
    ep = start_episode(episode_id="ep", now=T0)
    with pytest.raises(FrozenInstanceError):
        ep.episode_id = "other"  # type: ignore[misc]


def test_start_episode_basic() -> None:
    ep = start_episode(episode_id="ep_1", now=T0)
    assert ep.steps == ()
    assert ep.terminated is False
    assert ep.step_count == 0
    assert ep.total_reward == 0.0


def test_start_episode_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="episode_id"):
        start_episode(episode_id="", now=T0)


def test_start_episode_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        start_episode(episode_id="ep", now=datetime(2026, 5, 1))


def _step(timestep: int = 0, **overrides: object) -> TrajectoryStep:
    base: dict[str, object] = {
        "timestep": timestep,
        "action_id": "BUY",
        "raw_return": 0.01,
        "sharpe": 1.0,
        "drawdown_pct": 0.05,
        "shaped_reward": 0.41,
    }
    base.update(overrides)
    return TrajectoryStep(**base)  # type: ignore[arg-type]


def test_append_step_adds_to_episode() -> None:
    ep = start_episode(episode_id="ep_1", now=T0)
    ep = append_step(ep, _step(timestep=0))
    assert ep.step_count == 1
    assert ep.steps[0].action_id == "BUY"


def test_append_step_returns_new_state() -> None:
    """Pin: episodes are immutable."""

    original = start_episode(episode_id="ep_1", now=T0)
    new_ep = append_step(original, _step(timestep=0))
    assert original.step_count == 0
    assert new_ep.step_count == 1


def test_append_step_rejects_wrong_timestep() -> None:
    """Pin: steps must come in order — step.timestep equals current
    step_count."""

    ep = start_episode(episode_id="ep_1", now=T0)
    with pytest.raises(ValueError, match="timestep"):
        append_step(ep, _step(timestep=5))


def test_append_step_rejects_terminated_episode() -> None:
    ep = start_episode(episode_id="ep_1", now=T0)
    ep = terminate_episode(ep)
    with pytest.raises(ValueError, match="terminated"):
        append_step(ep, _step(timestep=0))


def test_terminate_episode_sets_flag() -> None:
    ep = start_episode(episode_id="ep_1", now=T0)
    ep = append_step(ep, _step(timestep=0))
    ep = terminate_episode(ep)
    assert ep.terminated is True


def test_terminate_already_terminated_rejected() -> None:
    ep = start_episode(episode_id="ep_1", now=T0)
    ep = terminate_episode(ep)
    with pytest.raises(ValueError, match="already terminated"):
        terminate_episode(ep)


def test_episode_total_reward_aggregates() -> None:
    ep = start_episode(episode_id="ep_1", now=T0)
    ep = append_step(ep, _step(timestep=0, shaped_reward=0.3))
    ep = append_step(ep, _step(timestep=1, shaped_reward=0.5))
    ep = append_step(ep, _step(timestep=2, shaped_reward=-0.1))
    assert ep.total_reward == pytest.approx(0.7)


def test_episode_total_raw_return_aggregates() -> None:
    ep = start_episode(episode_id="ep_1", now=T0)
    ep = append_step(ep, _step(timestep=0, raw_return=0.02))
    ep = append_step(ep, _step(timestep=1, raw_return=-0.01))
    assert ep.total_raw_return == pytest.approx(0.01)


def test_episode_max_drawdown_is_max() -> None:
    ep = start_episode(episode_id="ep_1", now=T0)
    ep = append_step(ep, _step(timestep=0, drawdown_pct=0.02))
    ep = append_step(ep, _step(timestep=1, drawdown_pct=0.10))
    ep = append_step(ep, _step(timestep=2, drawdown_pct=0.05))
    assert ep.max_drawdown_pct == pytest.approx(0.10)


def test_episode_max_drawdown_zero_when_empty() -> None:
    ep = start_episode(episode_id="ep_1", now=T0)
    assert ep.max_drawdown_pct == 0.0


# --------------------------- TrainingResult ----------------------------------


def test_training_result_rejects_empty_policy_id() -> None:
    with pytest.raises(ValueError, match="policy_id"):
        TrainingResult(
            policy_id="",
            config=_config(),
            episode_count=10,
            mean_episode_reward=0.5,
            final_loss=0.01,
            completed_at=T0,
        )


def test_training_result_rejects_negative_episode_count() -> None:
    with pytest.raises(ValueError, match="episode_count"):
        TrainingResult(
            policy_id="p",
            config=_config(),
            episode_count=-1,
            mean_episode_reward=0.5,
            final_loss=0.01,
            completed_at=T0,
        )


def test_training_result_rejects_naive_completed_at() -> None:
    with pytest.raises(ValueError, match="completed_at"):
        TrainingResult(
            policy_id="p",
            config=_config(),
            episode_count=10,
            mean_episode_reward=0.5,
            final_loss=0.01,
            completed_at=datetime(2026, 5, 1),
        )


def test_training_result_is_frozen() -> None:
    result = TrainingResult(
        policy_id="p",
        config=_config(),
        episode_count=10,
        mean_episode_reward=0.5,
        final_loss=0.01,
        completed_at=T0,
    )
    with pytest.raises(FrozenInstanceError):
        result.policy_id = "other"  # type: ignore[misc]


# --------------------------- aggregate_results -------------------------------


def test_aggregate_basic() -> None:
    eps = []
    for i in range(3):
        ep = start_episode(episode_id=f"ep_{i}", now=T0 + timedelta(hours=i))
        ep = append_step(ep, _step(timestep=0, shaped_reward=0.5))
        eps.append(ep)

    result = aggregate_results(
        eps,
        config=_config(),
        policy_id="policy_v1",
        final_loss=0.01,
        completed_at=T0 + timedelta(hours=4),
    )
    assert result.episode_count == 3
    assert result.mean_episode_reward == pytest.approx(0.5)


def test_aggregate_empty_episodes() -> None:
    result = aggregate_results(
        [],
        config=_config(),
        policy_id="p",
        final_loss=0.0,
        completed_at=T0,
    )
    assert result.episode_count == 0
    assert result.mean_episode_reward == 0.0


def test_aggregate_rejects_naive_completed_at() -> None:
    with pytest.raises(ValueError, match="completed_at"):
        aggregate_results(
            [],
            config=_config(),
            policy_id="p",
            final_loss=0.0,
            completed_at=datetime(2026, 5, 1),
        )


# --------------------------- render ------------------------------------------


def test_render_config_includes_algo_and_hyperparams() -> None:
    config = _config()
    out = render_config(config)
    assert "ppo" in out.lower()
    assert "0.0003" in out
    assert "0.99" in out
    assert "1,000,000" in out


def test_render_config_shows_emoji() -> None:
    """Pin: PPO has 🎯 emoji; SAC has 🌊."""

    ppo_out = render_config(_config(algorithm=RLAlgorithm.PPO))
    sac_out = render_config(_config(algorithm=RLAlgorithm.SAC, action_space=ActionSpace.CONTINUOUS))
    assert "🎯" in ppo_out
    assert "🌊" in sac_out


def test_render_result_includes_policy_id_and_metrics() -> None:
    result = TrainingResult(
        policy_id="rl_v1_2026_05",
        config=_config(),
        episode_count=100,
        mean_episode_reward=0.42,
        final_loss=0.001,
        completed_at=T0,
    )
    out = render_result(result)
    assert "rl_v1_2026_05" in out
    assert "100" in out
    assert "0.42" in out


def test_render_no_secret_leak() -> None:
    """Pin: render is summary metadata only."""

    config = _config()
    result = TrainingResult(
        policy_id="p",
        config=config,
        episode_count=10,
        mean_episode_reward=0.5,
        final_loss=0.01,
        completed_at=T0,
    )
    out = render_config(config) + render_result(result)
    assert "weights" not in out.lower()
    assert "optimizer" not in out.lower()
    assert "checkpoint" not in out.lower()
    assert "api_key" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_full_training_episode_lifecycle() -> None:
    """Real-world: build a 10-step episode, terminate, aggregate."""

    ep = start_episode(episode_id="ep_realistic", now=T0)
    for t in range(10):
        reward = shaped_reward(
            raw_return=0.005,
            sharpe=0.5,
            drawdown_pct=0.01 if t > 5 else 0.0,
        )
        step = TrajectoryStep(
            timestep=t,
            action_id="BUY" if t % 3 == 0 else "HOLD",
            raw_return=0.005,
            sharpe=0.5,
            drawdown_pct=0.01 if t > 5 else 0.0,
            shaped_reward=reward,
        )
        ep = append_step(ep, step)
    ep = terminate_episode(ep)

    assert ep.step_count == 10
    assert ep.terminated is True
    assert ep.max_drawdown_pct == pytest.approx(0.01)


def test_e2e_aggregate_realistic_training_run() -> None:
    """100 episodes; aggregate produces a reasonable result."""

    eps = []
    for i in range(100):
        ep = start_episode(episode_id=f"ep_{i}", now=T0 + timedelta(minutes=i))
        for t in range(5):
            reward = shaped_reward(
                raw_return=0.01,
                sharpe=1.0,
                drawdown_pct=0.0,
            )
            ep = append_step(
                ep,
                TrajectoryStep(
                    timestep=t,
                    action_id="BUY",
                    raw_return=0.01,
                    sharpe=1.0,
                    drawdown_pct=0.0,
                    shaped_reward=reward,
                ),
            )
        ep = terminate_episode(ep)
        eps.append(ep)

    result = aggregate_results(
        eps,
        config=_config(),
        policy_id="policy_v1",
        final_loss=0.001,
        completed_at=T0 + timedelta(hours=2),
    )
    assert result.episode_count == 100
    # 5 steps × (1.0 * 0.01 + 0.5 * 1.0 - 2.0 * 0.0) = 5 × 0.51 = 2.55
    assert result.mean_episode_reward == pytest.approx(2.55)


def test_e2e_replay_consistency() -> None:
    """Same operations produce equal episodes."""

    def build() -> Episode:
        ep = start_episode(episode_id="ep_1", now=T0)
        ep = append_step(ep, _step(timestep=0, shaped_reward=0.3))
        ep = append_step(ep, _step(timestep=1, shaped_reward=0.5))
        return ep

    a = build()
    b = build()
    assert a == b
