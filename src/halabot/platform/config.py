"""Narrowed engine settings (REARCHITECTURE §7).

A small, typed settings tree for the new engine. Reads from env with the
``HALABOT_`` prefix and ``__`` nesting (e.g. ``HALABOT_BELIEF__CONVICTION_ENTRY_BAND``).
The shared Postgres URL comes from ``DATABASE_URL`` (same env the legacy bot
uses) so the two systems share one database during migration.

Safety: ``live`` defaults to empty (shadow only) and the ``safeguard`` floors are
hard ceilings config cannot loosen in live mode (INV-9). Execution settings are
present but inert until ``ENGINE_LIVE`` is explicitly set (Phase-4 gate).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EngineSettings(BaseModel):
    heartbeat_interval_s: float = Field(default=900.0, gt=0)
    max_open_positions: int = Field(default=8, ge=1)
    # When set, restricts the universe; empty = use the halal screener's list.
    universe: list[str] = Field(default_factory=list)


class BeliefSettings(BaseModel):
    evidence_decay_halflife_min: float = Field(default=240.0, gt=0)
    evidence_decay_trading_time: bool = True
    bootstrap_window_min: float = Field(default=7200.0, ge=0)  # 5 days of trading-ish
    thesis_max_age_h: float = Field(default=4.0, gt=0)
    catalyst_impact_threshold: float = Field(default=0.7, ge=0, le=1)
    long_threshold: float = Field(default=0.05, ge=0, lt=1)


class CognitionSettings(BaseModel):
    llm_thesis_enabled: bool = False  # sparse LLM thesis off by default (cheap shadow)
    # The built-in forecaster is a cheap deterministic OLS-slope projection (no
    # [ml] extra). Off by default; a richer model (Chronos) can replace it behind
    # the same interpreter seam.
    forecaster_enabled: bool = False
    multiframe_enabled: bool = True
    # Cheap, always-on structural signals the engine otherwise ignores (B3):
    # volume-confirmed moves and swing support/resistance proximity.
    volume_enabled: bool = True
    structure_enabled: bool = True
    # Relative strength vs a market benchmark (alpha-vs-beta). Needs the benchmark
    # symbol's bars in the buffer (fed but never traded).
    relstrength_enabled: bool = True
    benchmark_symbol: str = "SPY"
    # Sparse LLM headline scoring: only fires when the cheap lexicon abstains.
    # Off by default (per-headline LLM cost); the lexicon path always runs.
    news_llm_enabled: bool = False


class ConvictionSettings(BaseModel):
    min_samples_to_calibrate: int = Field(default=50, ge=1)
    win_threshold_pct: float = Field(default=0.002, ge=0)


class PolicySettings(BaseModel):
    # Cold-start bands tuned to the observed raw-conviction scale (B.2 note);
    # the fitted calibrator replaces these once outcomes accumulate.
    # Re-tuned to 0.35 after the merge-dedup fix activated the FULL signal stack
    # (conviction is richer/higher with all interpreters live, so a higher bar is
    # right): backtest sweep (15d, 5bps) showed 0.35 dominates 0.25 on profit
    # factor (3.45 vs 2.46), total return, AND drawdown. Single-window — revisit
    # as outcomes accumulate and the fitted calibrator takes over.
    conviction_entry_band: float = Field(default=0.35, ge=0, lt=1)
    conviction_exit_band: float = Field(default=0.15, ge=0, lt=1)
    max_weight_per_asset: float = Field(default=0.20, gt=0, le=1)
    max_gross_exposure: float = Field(default=1.0, gt=0, le=1)
    target_rebalance_threshold: float = Field(default=0.03, gt=0, le=1)
    # Veto buys lagging the benchmark by this relstrength magnitude; 0 = off.
    relstrength_gate: float = Field(default=0.5, ge=0, le=1)
    # Market-regime ("don't fight the tape") gate: block BUYs while the benchmark
    # sits below its SMA. Default ON — a controlled same-bars backtest A/B
    # (2026-05-29, 5 windows, 1H, 5bps) showed it improves profit factor and
    # ~halves drawdown in every window where it fires (20d/45d/90d) and is
    # identical where the market stays risk-on (30d/60d) — it never hurts.
    # Needs the benchmark fed (relstrength_enabled); inert otherwise.
    market_gate_enabled: bool = True
    market_sma_window: int = Field(default=50, ge=2)


class RiskSettings(BaseModel):
    max_portfolio_heat_pct: float = Field(default=0.05, gt=0, le=1)
    max_drawdown_pct: float = Field(default=0.08, gt=0, le=1)
    daily_loss_limit: float = Field(default=0.02, gt=0, le=1)


class ExecSettings(BaseModel):
    venue: str = Field(default="alpaca")
    min_notional_usd: float = Field(default=50.0, ge=0)
    reconcile_interval_s: float = Field(default=300.0, gt=0)
    per_asset_breaker_threshold: int = Field(default=3, ge=1)
    per_asset_breaker_cooldown_s: float = Field(default=900.0, gt=0)


class HalalSettings(BaseModel):
    cache_ttl_h: float = Field(default=24.0, gt=0)


class SafeguardSettings(BaseModel):
    # Un-loosenable hard floors in live mode (INV-9). These are CEILINGS the
    # engine never exceeds regardless of other config; only an operator with the
    # dated token even arms live mode.
    live_max_account_usd: float = Field(default=10_000.0, gt=0)
    live_max_order_usd: float = Field(default=1_000.0, gt=0)
    live_daily_loss_floor_pct: float = Field(default=0.05, gt=0, le=1)


class HalabotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HALABOT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Shared Postgres — same DB as the legacy bot (read straight from DATABASE_URL).
    database_url: str = Field(
        default="postgresql+asyncpg://trader:trader-dev-only@localhost:5433/halal_trader",
        validation_alias="DATABASE_URL",
    )

    # Phase-4 gate: which markets trade LIVE. Empty = shadow only (default).
    # The execution path is not even instantiated unless this names a market.
    live: str = Field(default="", validation_alias="ENGINE_LIVE")
    # Operator arming token (INV-9). Live mode also requires a DATED token that
    # expires daily (format "LIVE-YYYY-MM-DD"), forcing a deliberate daily re-arm
    # — a stale env var can't silently keep the engine live.
    live_token: str = Field(default="", validation_alias="ENGINE_LIVE_TOKEN")

    engine: EngineSettings = Field(default_factory=EngineSettings)
    belief: BeliefSettings = Field(default_factory=BeliefSettings)
    cognition: CognitionSettings = Field(default_factory=CognitionSettings)
    conviction: ConvictionSettings = Field(default_factory=ConvictionSettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    # Named ``execution`` (not ``exec``) to avoid shadowing the builtin while
    # keeping a clean, prefix-consistent env name: HALABOT_EXECUTION__*.
    execution: ExecSettings = Field(default_factory=ExecSettings)
    halal: HalalSettings = Field(default_factory=HalalSettings)
    safeguard: SafeguardSettings = Field(default_factory=SafeguardSettings)

    @property
    def live_enabled(self) -> bool:
        """True only when an operator explicitly armed a live market."""
        return bool(self.live.strip())


@lru_cache(maxsize=1)
def get_settings() -> HalabotSettings:
    return HalabotSettings()
