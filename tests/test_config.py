"""Tests for configuration management."""

from halal_trader.config import LLMProvider, Settings


class TestSettings:
    def test_default_values(self):
        settings = Settings(
            _env_file=None,  # Don't read .env in tests
        )
        assert settings.llm_provider == LLMProvider.OLLAMA
        assert settings.llm_model == "qwen2.5:32b"
        assert settings.alpaca_paper_trade is True
        assert settings.daily_return_target == 0.01
        assert settings.max_position_pct == 0.20
        assert settings.daily_loss_limit == 0.02
        assert settings.trading_interval_minutes == 15
        assert settings.max_simultaneous_positions == 5

    def test_custom_values(self):
        settings = Settings(
            _env_file=None,
            llm_provider=LLMProvider.OPENAI,
            llm_model="gpt-4o",
            daily_return_target=0.02,
            max_position_pct=0.10,
        )
        assert settings.llm_provider == LLMProvider.OPENAI
        assert settings.llm_model == "gpt-4o"
        assert settings.daily_return_target == 0.02
        assert settings.max_position_pct == 0.10

    def test_provider_enum(self):
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
