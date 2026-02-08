"""Application configuration via Pydantic Settings."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

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

    # ── Zoya API ───────────────────────────────────────────────
    zoya_api_key: str = Field(default="", description="Zoya API key for halal screening")
    zoya_use_sandbox: bool = Field(
        default=False, description="Use Zoya sandbox environment (free, randomized data)"
    )

    # ── LLM ─────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA, description="LLM backend")
    llm_model: str = Field(
        default="qwen2.5:32b", description="Model name for the selected provider"
    )
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    # ── Trading Parameters ──────────────────────────────────────
    trading_interval_minutes: int = Field(default=15, description="Minutes between analysis cycles")
    daily_return_target: float = Field(default=0.01, description="Target daily return (1% = 0.01)")
    max_position_pct: float = Field(default=0.20, description="Max portfolio % per position")
    daily_loss_limit: float = Field(
        default=0.02, description="Max daily loss before halting (2% = 0.02)"
    )
    max_simultaneous_positions: int = Field(default=5, description="Max number of open positions")

    # ── Database ────────────────────────────────────────────────
    db_path: Path = Field(default=Path("halal_trader.db"), description="SQLite database path")

    # ── Logging ─────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Logging level")


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
