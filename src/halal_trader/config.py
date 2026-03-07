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

    # ── Stock Trading Parameters ────────────────────────────────
    trading_interval_minutes: int = Field(default=15, description="Minutes between analysis cycles")
    daily_return_target: float = Field(default=0.01, description="Target daily return (1% = 0.01)")
    max_position_pct: float = Field(default=0.20, description="Max portfolio % per position")
    daily_loss_limit: float = Field(
        default=0.02, description="Max daily loss before halting (2% = 0.02)"
    )
    max_simultaneous_positions: int = Field(default=5, description="Max number of open positions")

    # ── Crypto Trading Parameters ──────────────────────────────
    crypto_trading_interval_seconds: int = Field(
        default=60, description="Seconds between crypto analysis cycles"
    )
    crypto_pairs: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"],
        description="Crypto trading pairs to monitor",
    )
    crypto_max_position_pct: float = Field(
        default=0.25, description="Max portfolio % per crypto position"
    )
    crypto_daily_loss_limit: float = Field(
        default=0.03, description="Max daily crypto loss before halting (3% = 0.03)"
    )
    crypto_daily_return_target: float = Field(
        default=0.01, description="Target daily crypto return (1% = 0.01)"
    )
    crypto_max_simultaneous_positions: int = Field(
        default=4, description="Max number of open crypto positions"
    )
    crypto_min_market_cap: float = Field(
        default=1_000_000_000, description="Minimum market cap for halal screening ($1B)"
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
    ml_retrain_interval_hours: int = Field(
        default=24, description="Hours between ML model retraining"
    )

    # ── Telegram Notifications ─────────────────────────────────
    telegram_bot_token: str = Field(default="", description="Telegram bot API token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID for alerts")

    # ── Database ────────────────────────────────────────────────
    db_path: Path = Field(default=Path("halal_trader.db"), description="SQLite database path")

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
