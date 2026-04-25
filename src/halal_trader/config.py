"""Application configuration via Pydantic Settings."""

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """All application settings, loaded from .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Alpaca API ──────────────────────────────────────────────
    alpaca_api_key: str = Field(default="", description="Alpaca Trading API key")
    alpaca_secret_key: str = Field(default="", description="Alpaca Trading API secret")
    alpaca_paper_trade: bool = Field(
        default=True, description="Use paper trading (True) or live (False)"
    )

    # ── Binance API ────────────────────────────────────────────
    binance_api_key: str = Field(default="", description="Binance API key")
    binance_secret_key: str = Field(default="", description="Binance API secret")
    binance_testnet: bool = Field(
        default=True, description="Use Binance testnet (True) or production (False)"
    )

    # ── Zoya API ───────────────────────────────────────────────
    zoya_api_key: str = Field(default="", description="Zoya API key for halal screening")
    zoya_use_sandbox: bool = Field(
        default=False, description="Use Zoya sandbox environment (free, randomized data)"
    )

    # ── CoinGecko API ──────────────────────────────────────────
    coingecko_api_key: str = Field(
        default="", description="CoinGecko API key (optional, for higher rate limits)"
    )

    # ── LLM ─────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA, description="LLM backend")
    llm_model: str = Field(
        default="qwen2.5:32b", description="Model name for the selected provider"
    )
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    llm_fallback_providers: list[str] = Field(
        default=[],
        description=(
            "Ordered list of fallback LLM providers (e.g. ['openai', 'anthropic']). "
            "Empty = no fallbacks."
        ),
    )
    ollama_fallback_model: str = Field(
        default="",
        description="Model name for Ollama when used as fallback (empty = same as llm_model)",
    )
    openai_fallback_model: str = Field(
        default="gpt-4o-mini", description="Model name for OpenAI when used as fallback"
    )
    anthropic_fallback_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model name for Anthropic when used as fallback",
    )

    # ── Stock Trading Parameters ────────────────────────────────
    trading_interval_minutes: int = Field(default=15, description="Minutes between analysis cycles")
    daily_return_target: float = Field(
        default=0.01, gt=0, le=0.5, description="Target daily return (1% = 0.01)"
    )
    max_position_pct: float = Field(
        default=0.20, gt=0, le=1.0, description="Max portfolio % per position"
    )
    daily_loss_limit: float = Field(
        default=0.02, ge=0, le=0.5, description="Max daily loss before halting (2% = 0.02)"
    )
    max_simultaneous_positions: int = Field(
        default=5, ge=1, description="Max number of open positions"
    )

    # ── Crypto Trading Parameters ──────────────────────────────
    crypto_trading_interval_seconds: int = Field(
        default=60, ge=5, description="Seconds between crypto analysis cycles"
    )
    crypto_pairs: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"],
        description="Crypto trading pairs to monitor",
    )
    crypto_max_position_pct: float = Field(
        default=0.25, gt=0, le=1.0, description="Max portfolio % per crypto position"
    )
    crypto_daily_loss_limit: float = Field(
        default=0.03, ge=0, le=0.5, description="Max daily crypto loss before halting (3% = 0.03)"
    )
    crypto_daily_return_target: float = Field(
        default=0.01, gt=0, le=0.5, description="Target daily crypto return (1% = 0.01)"
    )
    crypto_max_simultaneous_positions: int = Field(
        default=4, ge=1, description="Max number of open crypto positions"
    )
    crypto_min_market_cap: float = Field(
        default=1_000_000_000, ge=0, description="Minimum market cap for halal screening ($1B)"
    )
    crypto_max_pairs_per_cycle: int = Field(
        default=10, ge=1, description="Max trading pairs per cycle"
    )

    # ── Portfolio-level risk ─────────────────────────────────
    crypto_max_portfolio_heat_pct: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Max unrealized loss before blocking entries",
    )
    crypto_max_drawdown_pct: float = Field(
        default=0.08,
        ge=0.01,
        le=0.5,
        description="Max peak-to-trough drawdown before halt",
    )
    crypto_high_correlation_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Correlation threshold for size reduction",
    )
    crypto_correlation_reduction_factor: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Size multiplier when correlated with open",
    )
    crypto_atr_baseline: float = Field(
        default=0.02, gt=0, description="ATR baseline for volatility-adjusted sizing"
    )

    # ── Flat-market skip thresholds ───────────────────────────
    crypto_flat_price_threshold: float = Field(
        default=0.03, ge=0, description="Min 5m price change (%) to consider non-flat"
    )
    crypto_flat_rsi_lower: float = Field(
        default=40.0, ge=0, le=50, description="RSI below this = non-flat (oversold)"
    )
    crypto_flat_rsi_upper: float = Field(
        default=60.0, ge=50, le=100, description="RSI above this = non-flat (overbought)"
    )
    crypto_flat_vol_threshold: float = Field(
        default=1.2, ge=1.0, description="Volume ratio above this = non-flat"
    )
    crypto_max_consecutive_flat_skips: int = Field(
        default=5, ge=1, description="Max consecutive flat-market skips before forcing LLM"
    )

    # ── Trailing stop / monitor ───────────────────────────────
    crypto_trailing_stop_activation_pct: float = Field(
        default=0.005, ge=0, description="% above entry to activate trailing stop"
    )
    crypto_trailing_stop_distance_pct: float = Field(
        default=0.003, gt=0, description="Trailing stop distance from high water mark"
    )
    crypto_monitor_interval: float = Field(
        default=2.0, gt=0, description="Seconds between position monitor checks"
    )

    # ── Circuit breaker (per-pair) ──────────────────────────────
    crypto_circuit_breaker_threshold: int = Field(
        default=5, ge=1, description="Errors before blocking a pair"
    )
    crypto_circuit_breaker_window: int = Field(
        default=600, ge=60, description="Window in seconds for counting errors"
    )
    crypto_circuit_breaker_cooldown: int = Field(
        default=1800, ge=60, description="Cooldown in seconds after circuit break"
    )

    # ── LLM circuit breaker ───────────────────────────────────
    crypto_llm_failure_threshold: int = Field(
        default=5, ge=1, description="Consecutive LLM failures before cooldown"
    )
    crypto_llm_cooldown_seconds: int = Field(
        default=600, ge=60, description="Seconds to pause LLM calls after threshold failures"
    )

    # ── Sentiment ──────────────────────────────────────────────
    reddit_client_id: str = Field(default="", description="Reddit API client ID (for sentiment)")
    reddit_client_secret: str = Field(
        default="", description="Reddit API client secret (for sentiment)"
    )
    cryptopanic_api_key: str = Field(
        default="", description="CryptoPanic API key (for news sentiment)"
    )
    sentiment_update_interval_seconds: int = Field(
        default=300, description="Seconds between sentiment updates"
    )
    sentiment_use_finbert: bool = Field(
        default=False, description="Use FinBERT model for sentiment scoring"
    )

    # ── ML Models ──────────────────────────────────────────────
    ml_enabled: bool = Field(default=False, description="Enable HuggingFace ML models")
    ml_device: str = Field(default="cpu", description="Device for ML models (cpu/cuda/mps)")
    ml_models_dir: Path = Field(
        default=Path("models"), description="Directory for cached ML model files"
    )
    # ── Telegram Notifications ─────────────────────────────────
    telegram_bot_token: str = Field(default="", description="Telegram bot API token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID for alerts")

    # ── Database ────────────────────────────────────────────────
    db_path: Path = Field(default=Path("halal_trader.db"), description="SQLite database path")

    def resolve_db_path(self) -> Path:
        """Return an absolute db_path, resolving relative paths from the project root."""
        if self.db_path.is_absolute():
            return self.db_path
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / self.db_path

    # ── Logging ─────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Console logging level")
    log_dir: Path = Field(default=Path("logs"), description="Directory for log files")
    log_file_level: str = Field(default="DEBUG", description="File logging level")
    log_max_bytes: int = Field(default=10_485_760, description="Max log file size in bytes (10 MB)")
    log_backup_count: int = Field(default=5, description="Number of rotated log files to keep")


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
