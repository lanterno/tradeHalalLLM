"""Tests for configuration management."""

from typing import Any

from halal_trader.config import (
    AlpacaSettings,
    AnthropicSettings,
    BinanceSettings,
    CoinGeckoSettings,
    CryptoPanicSettings,
    CryptoSettings,
    LiveModeSettings,
    LLMProvider,
    LLMSettings,
    LogSettings,
    MLSettings,
    OllamaSettings,
    OpenAISettings,
    RedditSettings,
    SentimentSettings,
    Settings,
    StockSettings,
    TelegramSettings,
    ZoyaSettings,
)


def _isolated_settings(**overrides: Any) -> Settings:
    """Build a Settings tree with ``.env`` reading disabled at every level.

    Each nested ``BaseSettings`` would otherwise reread ``.env`` because
    its ``model_config`` declares ``env_file=".env"``. Passing
    ``_env_file=None`` on construction overrides that for one instance.
    """
    defaults = {
        "alpaca": AlpacaSettings(_env_file=None),
        "binance": BinanceSettings(_env_file=None),
        "zoya": ZoyaSettings(_env_file=None),
        "coingecko": CoinGeckoSettings(_env_file=None),
        "llm": LLMSettings(
            _env_file=None,
            ollama=OllamaSettings(_env_file=None),
            openai=OpenAISettings(_env_file=None),
            anthropic=AnthropicSettings(_env_file=None),
        ),
        "stocks": StockSettings(_env_file=None),
        "crypto": CryptoSettings(_env_file=None),
        "sentiment": SentimentSettings(
            _env_file=None,
            reddit=RedditSettings(_env_file=None),
            cryptopanic=CryptoPanicSettings(_env_file=None),
        ),
        "ml": MLSettings(_env_file=None),
        "telegram": TelegramSettings(_env_file=None),
        "live_mode": LiveModeSettings(_env_file=None),
        "log": LogSettings(_env_file=None),
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


class TestSettings:
    def test_default_values(self):
        settings = _isolated_settings()
        assert settings.llm.provider == LLMProvider.OLLAMA
        assert settings.llm.model == "qwen2.5:32b"
        assert settings.alpaca.paper_trade is True
        assert settings.stocks.daily_return_target == 0.01
        assert settings.stocks.max_position_pct == 0.20
        assert settings.stocks.daily_loss_limit == 0.02
        assert settings.stocks.trading_interval_minutes == 15
        assert settings.stocks.max_simultaneous_positions == 5
        # Re-entry cooldown raised 30→60 (2026-06-17) to curb round-trip churn.
        assert settings.stocks.recent_close_cooldown_minutes == 60

    def test_custom_values(self):
        settings = _isolated_settings(
            llm=LLMSettings(
                _env_file=None,
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
                ollama=OllamaSettings(_env_file=None),
                openai=OpenAISettings(_env_file=None),
                anthropic=AnthropicSettings(_env_file=None),
            ),
            stocks=StockSettings(
                _env_file=None,
                daily_return_target=0.02,
                max_position_pct=0.10,
            ),
        )
        assert settings.llm.provider == LLMProvider.OPENAI
        assert settings.llm.model == "gpt-4o"
        assert settings.stocks.daily_return_target == 0.02
        assert settings.stocks.max_position_pct == 0.10

    def test_provider_enum(self):
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
