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
    use_sandbox: bool = Field(default=False)


class CoinGeckoSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="COINGECKO_")
    api_key: str = Field(default="")


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
    use_finbert: bool = Field(default=False)
    reddit: RedditSettings = Field(default_factory=RedditSettings)
    cryptopanic: CryptoPanicSettings = Field(default_factory=CryptoPanicSettings)


# ── ML / Notifications / Live-mode / Backup / Logging ─────────


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
    """Live-mode safeguards. ``confirmation`` and ``max_daily_loss_pct`` use
    the ``LIVE_MODE_`` prefix; the two ``MAX_*_USD`` ceilings predate that
    prefix and are read via explicit aliases for backward compatibility."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LIVE_MODE_")
    confirmation: str = Field(default="")
    max_daily_loss_pct: float = Field(default=0.02, ge=0, le=0.5)
    max_account_balance_usd: float = Field(
        default=500.0, gt=0, validation_alias="MAX_ACCOUNT_BALANCE_USD"
    )
    max_single_order_usd: float = Field(
        default=100.0, gt=0, validation_alias="MAX_SINGLE_ORDER_USD"
    )


class BackupSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="BACKUP_")
    dir: Path = Field(default=Path("backups"))
    retention_days: int = Field(default=30, ge=1)
    weekly_count: int = Field(default=12, ge=0)


class LogSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LOG_")
    level: str = Field(default="INFO")
    dir: Path = Field(default=Path("logs"))
    file_level: str = Field(default="DEBUG")
    max_bytes: int = Field(default=10_485_760)
    backup_count: int = Field(default=5)


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
    llm: LLMSettings = Field(default_factory=LLMSettings)
    stocks: StockSettings = Field(default_factory=StockSettings)
    crypto: CryptoSettings = Field(default_factory=CryptoSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    live_mode: LiveModeSettings = Field(default_factory=LiveModeSettings)
    backup: BackupSettings = Field(default_factory=BackupSettings)
    log: LogSettings = Field(default_factory=LogSettings)

    db_path: Path = Field(default=Path("halal_trader.db"))

    def resolve_db_path(self) -> Path:
        """Return an absolute db_path, resolving relative paths from the project root."""
        if self.db_path.is_absolute():
            return self.db_path
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / self.db_path


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
