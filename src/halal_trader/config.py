"""Application configuration via nested Pydantic Settings sub-models.

The top-level ``Settings`` exposes domain-grouped sub-models (``settings.binance``,
``settings.crypto``, ``settings.llm.openai``, …). Each sub-model is its own
``BaseSettings`` class with an ``env_prefix`` chosen to match the existing
``.env`` variable names so operators don't have to migrate their config.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


_BASE_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)


# ── Brokers ────────────────────────────────────────────────────


class AlpacaSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="ALPACA_")
    api_key: str = Field(default="")
    secret_key: str = Field(default="")
    paper_trade: bool = Field(default=True)


class BinanceSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="BINANCE_")
    api_key: str = Field(default="")
    secret_key: str = Field(default="")
    testnet: bool = Field(default=True)


# ── Halal Screening ────────────────────────────────────────────


class ZoyaSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="ZOYA_")
    api_key: str = Field(default="")
    # Default to sandbox so a fresh checkout doesn't burn the operator's
    # paid quota on first run. Flip to ``false`` once a prod key is wired.
    use_sandbox: bool = Field(default=True)


class CoinGeckoSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="COINGECKO_")
    api_key: str = Field(default="")


class FREDSettings(BaseSettings):
    """St. Louis Fed economic-data API.

    Drives the macro-catalyst calendar (CPI, FOMC, NFP, GDP release
    dates) so the stock cycle can shrink position sizing in the 4h
    window before a high-impact release. ``api_key=""`` disables the
    feed cleanly — the catalyst module degrades to whatever other
    sources are configured.
    """

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="FRED_")
    api_key: str = Field(default="")


class EDGARSettings(BaseSettings):
    """SEC EDGAR filings (free, no key — just an identifying user-agent).

    The SEC requires every request to carry a ``User-Agent`` with a
    real contact (per their ``accessing-edgar-data`` policy); empty
    disables the feed. Drives the 8-K material-event stream for the
    stock catalyst feed.
    """

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="EDGAR_")
    user_agent: str = Field(default="")


class EtherscanSettings(BaseSettings):
    """Etherscan (free) — used for on-chain whale-flow signals.

    Drives a crypto-side feature that watches large stablecoin /
    token transfers to/from major exchanges. Free tier is 5 req/sec,
    enough to poll the top halal pairs each cycle. Empty key disables
    the feed cleanly.
    """

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="ETHERSCAN_")
    api_key: str = Field(default="")


class HalalSettings(BaseSettings):
    """Cross-cutting halal-screening cadence + safety knobs."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="HALAL_")

    # Cache TTL — how long a cached screening decision is trusted before
    # we ask the upstream provider again. Tightened from the legacy 24h
    # to 6h so a screening provider that flips a symbol from halal to
    # not_halal mid-day doesn't leave us trading the stale verdict.
    cache_max_age_hours: int = Field(default=6)
    # Mid-cycle refresh threshold — if the cache is older than this when
    # the cycle starts, refresh it inline before screening any symbols.
    midcycle_refresh_hours: int = Field(default=4)


# ── LLM Providers ──────────────────────────────────────────────


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="OLLAMA_")
    host: str = Field(default="http://localhost:11434")
    fallback_model: str = Field(default="")


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="OPENAI_")
    api_key: str | None = Field(default=None)
    fallback_model: str = Field(default="gpt-4o-mini")


class AnthropicSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="ANTHROPIC_")
    api_key: str | None = Field(default=None)
    fallback_model: str = Field(default="claude-sonnet-4-20250514")


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LLM_")
    provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    model: str = Field(default="qwen2.5:32b")
    fallback_providers: list[str] = Field(default_factory=list)
    # Hard ceiling on per-UTC-day cumulative spend across all providers.
    # 0 disables the cap (useful in tests / local Ollama runs). When the
    # cap trips it engages the kill-switch so both bots stop entering
    # new positions until the operator clears it.
    daily_usd_cap: float = Field(default=0.0)
    # Adversarial co-bot — runs a cheap follow-up LLM call that critiques
    # each plan and downsizes/skips buys when it surfaces a strong
    # counter-thesis. Off by default (extra cost; opt-in).
    adversarial_enabled: bool = Field(default=False)
    # Ensemble fan-out — number of additional LLM variants that vote
    # alongside the primary. Median quantity / confidence wins; agreement
    # score scales sizing in [0.5, 1.0]. 0 disables.
    ensemble_size: int = Field(default=0, ge=0, le=5)
    # Shadow strategy — runs an analyze() pass per cycle on the same
    # inputs and simulates fills against latest prices. Used to detect
    # decay between live (mutating) and shadow (frozen-prompt) curves.
    shadow_enabled: bool = Field(default=False)
    shadow_starting_cash: float = Field(default=1000.0, gt=0)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)


# ── Trading parameters ─────────────────────────────────────────


class StockSettings(BaseSettings):
    """Stock-side trading parameters; legacy unprefixed env names preserved."""

    model_config = SettingsConfigDict(**_BASE_CONFIG)
    trading_interval_minutes: int = Field(default=15)
    daily_return_target: float = Field(default=0.01, gt=0, le=0.5)
    max_position_pct: float = Field(default=0.20, gt=0, le=1.0)
    daily_loss_limit: float = Field(default=0.02, ge=0, le=0.5)
    max_simultaneous_positions: int = Field(default=5, ge=1)


class CryptoSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="CRYPTO_")
    trading_interval_seconds: int = Field(default=60, ge=5)
    pairs: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"])
    max_position_pct: float = Field(default=0.25, gt=0, le=1.0)
    daily_loss_limit: float = Field(default=0.03, ge=0, le=0.5)
    daily_return_target: float = Field(default=0.01, gt=0, le=0.5)
    max_simultaneous_positions: int = Field(default=4, ge=1)
    min_market_cap: float = Field(default=1_000_000_000, ge=0)
    max_pairs_per_cycle: int = Field(default=10, ge=1)

    # Portfolio risk
    max_portfolio_heat_pct: float = Field(default=0.05, ge=0.01, le=0.5)
    max_drawdown_pct: float = Field(default=0.08, ge=0.01, le=0.5)
    high_correlation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    correlation_reduction_factor: float = Field(default=0.5, ge=0.1, le=1.0)
    atr_baseline: float = Field(default=0.02, gt=0)

    # Flat-market skip
    flat_price_threshold: float = Field(default=0.03, ge=0)
    flat_rsi_lower: float = Field(default=40.0, ge=0, le=50)
    flat_rsi_upper: float = Field(default=60.0, ge=50, le=100)
    flat_vol_threshold: float = Field(default=1.2, ge=1.0)
    max_consecutive_flat_skips: int = Field(default=5, ge=1)

    # Trailing stop / monitor
    trailing_stop_activation_pct: float = Field(default=0.005, ge=0)
    trailing_stop_distance_pct: float = Field(default=0.003, gt=0)
    monitor_interval: float = Field(default=2.0, gt=0)

    # Per-pair circuit breaker
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_window: int = Field(default=600, ge=60)
    circuit_breaker_cooldown: int = Field(default=1800, ge=60)

    # LLM circuit breaker
    llm_failure_threshold: int = Field(default=5, ge=1)
    llm_cooldown_seconds: int = Field(default=600, ge=60)


# ── Sentiment ──────────────────────────────────────────────────


class RedditSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="REDDIT_")
    client_id: str = Field(default="")
    client_secret: str = Field(default="")


class CryptoPanicSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="CRYPTOPANIC_")
    api_key: str = Field(default="")


class SentimentSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="SENTIMENT_")
    update_interval_seconds: int = Field(default=300)
    reddit: RedditSettings = Field(default_factory=RedditSettings)
    cryptopanic: CryptoPanicSettings = Field(default_factory=CryptoPanicSettings)


# ── ML / Notifications / Live-mode / Logging ──────────────────


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="ML_")
    enabled: bool = Field(default=False)
    device: str = Field(default="cpu")
    models_dir: Path = Field(default=Path("models"))


class TelegramSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="TELEGRAM_")
    bot_token: str = Field(default="")
    chat_id: str = Field(default="")


class LiveModeSettings(BaseSettings):
    """Live-mode safeguards — all knobs use the ``LIVE_MODE_`` prefix."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LIVE_MODE_")
    confirmation: str = Field(default="")
    max_daily_loss_pct: float = Field(default=0.02, ge=0, le=0.5)
    max_account_balance_usd: float = Field(default=500.0, gt=0)
    max_single_order_usd: float = Field(default=100.0, gt=0)


class LogSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LOG_")
    level: str = Field(default="INFO")
    dir: Path = Field(default=Path("logs"))
    file_level: str = Field(default="DEBUG")
    max_bytes: int = Field(default=10_485_760)
    backup_count: int = Field(default=5)


# ── Web dashboard ──────────────────────────────────────────────


class WebSettings(BaseSettings):
    """Dashboard control-surface knobs.

    The dashboard binds to localhost by default; ``api_token`` is the
    shared-secret header gate for any state-changing endpoint. Empty
    token means *mutations are disabled* — read-only mode for safety
    on a fresh deployment until the operator explicitly opts in.
    """

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="WEB_")

    api_token: str = Field(default="")
    # Forced confirmation for destructive ops can be turned off in tests
    # so the runner doesn't have to forge headers — never disable in prod.
    require_confirmation: bool = Field(default=True)
    # Days to keep mutation-audit rows in ``web_actions``. The daily
    # bot-end hook prunes anything older. 0 disables the prune.
    audit_retention_days: int = Field(default=90)


# ── Top-level Settings ─────────────────────────────────────────


class Settings(BaseSettings):
    """All application settings, grouped by domain.

    Each sub-model loads its own slice of ``.env`` independently, so
    individual sub-models can be constructed in tests without bringing
    the whole tree along.
    """

    model_config = SettingsConfigDict(**_BASE_CONFIG)

    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    zoya: ZoyaSettings = Field(default_factory=ZoyaSettings)
    coingecko: CoinGeckoSettings = Field(default_factory=CoinGeckoSettings)
    fred: FREDSettings = Field(default_factory=FREDSettings)
    edgar: EDGARSettings = Field(default_factory=EDGARSettings)
    etherscan: EtherscanSettings = Field(default_factory=EtherscanSettings)
    halal: HalalSettings = Field(default_factory=HalalSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    stocks: StockSettings = Field(default_factory=StockSettings)
    crypto: CryptoSettings = Field(default_factory=CryptoSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    live_mode: LiveModeSettings = Field(default_factory=LiveModeSettings)
    log: LogSettings = Field(default_factory=LogSettings)

    # Postgres baseline — see docker-compose for the matching service.
    # Override via DATABASE_URL in .env. SQLAlchemy / alembic both
    # read this directly. ``+asyncpg`` and ``+psycopg`` drivers are
    # interchangeable; we ship asyncpg for the runtime and psycopg
    # for sync paths (alembic, db/admin.py).
    database_url: str = Field(
        default="postgresql+asyncpg://trader:trader-dev-only@localhost:5433/halal_trader",
        description="SQLAlchemy connection URL for the Postgres database",
    )
    # Sidecar / replay / analytics dir on the filesystem.
    data_dir: Path = Field(default=Path("data"))

    def resolve_data_dir(self) -> Path:
        """Return an absolute data-dir path, resolving relative paths from project root."""
        if self.data_dir.is_absolute():
            return self.data_dir
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / self.data_dir

    def database_url_sync(self) -> str:
        """Return a synchronous-driver URL for alembic + admin scripts.

        SQLAlchemy URLs encode the driver as ``+asyncpg`` for the runtime
        and ``+psycopg`` for sync. We map the runtime URL to its sync
        cousin so alembic reads from the same DB without needing a
        second env var.
        """
        return self.database_url.replace("+asyncpg", "+psycopg").replace(
            "postgresql://", "postgresql+psycopg://"
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
