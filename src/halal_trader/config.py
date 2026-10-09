"""Application configuration: domain-grouped settings under one ``Settings``.

Each group (``settings.alpaca``, ``settings.stocks``, ``settings.llm.glm``, …)
names in ``ENV`` the few fields the environment or ``.env`` may set: secrets,
endpoints and the operator's switches. Every other field is a decided value,
fixed here; code (tests) can still pass it to the constructor, but the
environment cannot change it. ``.env.example`` lists exactly the ``ENV``
fields (tests/test_settings_parity.py).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# HALAL_TRADER_ENV_FILE exists so the test suite can point settings at a file
# that does not exist: with a bare ".env" every pytest run from the repo root
# loaded the operator's live API keys.
_BASE_CONFIG = SettingsConfigDict(
    env_file=os.environ.get("HALAL_TRADER_ENV_FILE", ".env"),
    env_file_encoding="utf-8",
    extra="ignore",
)


class _OnlyEnvFields(PydanticBaseSettingsSource):
    """An env / .env source that passes on only a group's ``ENV`` fields."""

    def __init__(self, source: PydanticBaseSettingsSource, names: frozenset[str]) -> None:
        super().__init__(source.settings_cls)
        self._source = source
        self._names = names

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._source.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in self._source().items() if k in self._names}


class _Group(BaseSettings):
    """A settings group: only the fields in ``ENV`` come from the environment."""

    ENV: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _OnlyEnvFields(env_settings, cls.ENV),
            _OnlyEnvFields(dotenv_settings, cls.ENV),
        )


# ── Brokers and data ───────────────────────────────────────────


class AlpacaSettings(_Group):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="ALPACA_")
    ENV = frozenset({"api_key", "secret_key", "paper_trade"})

    api_key: str = ""
    secret_key: str = ""
    paper_trade: bool = True
    # The alpaca-mcp-server release the bot runs. The image installs it from
    # the frozen closure in infra/alpaca-mcp-server.txt (a test keeps the two
    # in step); a host run launches `uvx alpaca-mcp-server@<this>`. An
    # unpinned server, or one with floating dependencies, has broken the bot
    # at startup: bump only after re-freezing and the contract tests.
    mcp_server_version: str = Field(default="2.3.2", pattern=r"^\d+\.\d+\.\d+$")


class FinnhubSettings(_Group):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="FINNHUB_")
    ENV = frozenset({"api_key"})

    api_key: str = ""


class FREDSettings(_Group):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="FRED_")
    ENV = frozenset({"api_key"})

    api_key: str = ""


class EDGARSettings(_Group):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="EDGAR_")
    ENV = frozenset({"user_agent"})

    # SEC requires a User-Agent with a real contact; empty disables SEC data.
    user_agent: str = ""


class HalalSettings(_Group):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="HALAL_")

    # How long a cached verdict is trusted, and the age at which a cycle
    # refreshes it before screening.
    cache_max_age_hours: int = 6
    midcycle_refresh_hours: int = 4
    # The day-trader's and the reactor's universe: the largest names the
    # strict in-house screen passes.
    universe_size: int = Field(default=20, ge=1, le=500)


# ── LLM (GLM-5.2 only) ─────────────────────────────────────────


class GLMSettings(_Group):
    """GLM-5.2 over an OpenAI-compatible API; OpenRouter by default.

    Another host (e.g. Z.ai direct, https://api.z.ai/api/paas/v4) needs its
    own model id in ``LLM_MODEL`` (``glm-5.2`` there).
    """

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="GLM_")
    ENV = frozenset(
        {"api_key", "base_url", "fallback_base_url", "fallback_model", "fallback_api_key"}
    )

    api_key: str = ""  # required: create_llm() refuses to start without it
    base_url: str = "https://openrouter.ai/api/v1"
    # Optional second endpoint for the same model, tried when the primary
    # fails; the key defaults to the primary's.
    fallback_base_url: str = ""
    fallback_model: str = ""
    fallback_api_key: str = ""
    timeout_seconds: int = Field(default=60, gt=0)
    # Thinking off: cycle latency and cost.
    thinking: bool = False
    # OpenRouter: only hosts that honour JSON mode and tool calls.
    require_parameters: bool = True


class LLMSettings(_Group):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LLM_")
    ENV = frozenset({"model", "monthly_live_usd", "monthly_research_usd"})

    model: str = "z-ai/glm-5.2"
    # Spend caps, always enforced (core/llm/spend.py): a daily total over every
    # process on the key, and monthly pools: "live" (the bot and the shadow)
    # and "research". They sit under the key's own $50 limit.
    daily_usd_cap: float = 3.0
    monthly_live_usd: float = 25.0
    monthly_research_usd: float = 15.0
    glm: GLMSettings = Field(default_factory=GLMSettings)


# ── Strategies ─────────────────────────────────────────────────


class CoreSettings(_Group):
    """The strict-halal core portfolio, on its own Alpaca account."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="CORE_")
    ENV = frozenset(
        {
            "alpaca_api_key",
            "alpaca_secret_key",
            "paper",
            "live_confirmation",
            "live_max_notional",
        }
    )

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    paper: bool = True
    # Live money also needs today's "I-UNDERSTAND-REAL-MONEY-CORE-<UTC date>".
    live_confirmation: str = ""
    # The most the live core holds invested, in dollars.
    live_max_notional: float = Field(default=1_000.0, gt=0)
    top_n: int = Field(default=100, ge=10, le=500)

    @property
    def enabled(self) -> bool:
        """The core trades whenever its own account's keys are set."""
        return bool(self.alpaca_api_key and self.alpaca_secret_key)


class ZakatSettings(_Group):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="ZAKAT_")
    ENV = frozenset({"hawl_hijri"})

    # The Hijri day the zakat year ends, "MM-DD" (e.g. "09-01" for 1 Ramadan).
    hawl_hijri: str = ""


class StockSettings(_Group):
    """The LLM day-trader and its news reactor, on the ALPACA_* account.

    Both always run beside the core (operator, 2026-10-08); nothing here is
    read from the environment.
    """

    model_config = SettingsConfigDict(**_BASE_CONFIG)

    trading_interval_minutes: int = 15
    daily_return_target: float = Field(default=0.01, gt=0, le=0.5)
    max_position_pct: float = Field(default=0.20, gt=0, le=1.0)
    daily_loss_limit: float = Field(default=0.02, ge=0, le=0.5)
    max_simultaneous_positions: int = Field(default=5, ge=1)
    monitor_interval_seconds: float = Field(default=30.0, gt=0)
    trailing_stop_activation_pct: float | None = None
    trailing_stop_distance_pct: float = Field(default=0.005, gt=0)

    # Portfolio risk (trading/risk.py), tuned for daily equity bars.
    max_portfolio_heat_pct: float = Field(default=0.05, ge=0.01, le=0.5)
    max_drawdown_pct: float = Field(default=0.08, ge=0.01, le=0.5)
    high_correlation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    correlation_reduction_factor: float = Field(default=0.5, ge=0.1, le=1.0)
    atr_baseline: float = Field(default=0.02, gt=0)

    # Reactor: classifier calls per UTC day (1500 covers a day's distinct
    # headlines), the age past which a headline is recorded but not scored,
    # and the observe-only research list with its own allowance.
    reactor_daily_classify_cap: int = Field(default=1500, ge=0)
    reactor_max_headline_age_s: float = Field(default=1800.0, ge=0)
    reactor_observe_size: int = Field(default=300, ge=0)
    reactor_observe_daily_classify_cap: int = Field(default=600, ge=0)
    # Reactor entries ("fast in"): half the normal position, only with the
    # stock also up on the session; then "slow out" on a wide trailing stop
    # and a trend-break exit (close below the 20-bar hourly SMA).
    reactor_entry_size_fraction: float = Field(default=0.5, gt=0, le=1.0)
    reactor_entry_min_intraday_change_pct: float = Field(default=0.002, ge=0.0, le=0.5)
    reactor_trailing_stop_distance_pct: float = Field(default=0.08, gt=0, le=0.5)
    # Decided 2026-10-09: the reactor's entries run in shadow. Its
    # pre-registered minute-bar test failed (buying 60 s after positively
    # scored headlines lost 0.30% the same day, net of cost), and so did three
    # rebuilt variants. It keeps every gate and logs what it would buy
    # (reactor_decisions) without placing it, until a variant passes.
    reactor_places_orders: bool = False
    trend_break_ma_period: int = Field(default=20, ge=2, le=200)
    trend_break_timeframe: str = "1Hour"
    # Re-entry blocks after any close, and longer after a stop-out.
    recent_close_cooldown_minutes: int = Field(default=60, ge=0)
    stop_loss_reentry_cooldown_minutes: int = Field(default=120, ge=0)


# ── Notifications / live mode / logging / web ─────────────────


class TelegramSettings(_Group):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="TELEGRAM_")
    ENV = frozenset({"bot_token", "chat_id"})

    bot_token: str = ""
    chat_id: str = ""


class LiveModeSettings(_Group):
    """The day-trader on real money (ALPACA_PAPER_TRADE=false)."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LIVE_MODE_")
    ENV = frozenset({"confirmation"})

    # Today's "I-UNDERSTAND-REAL-MONEY-<UTC date>"; without it the bot refuses.
    confirmation: str = ""
    max_daily_loss_pct: float = Field(default=0.02, ge=0, le=0.5)
    max_account_balance_usd: float = Field(default=500.0, gt=0)
    max_single_order_usd: float = Field(default=100.0, gt=0)


class LogSettings(_Group):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LOG_")
    ENV = frozenset({"level", "dir"})

    level: str = "INFO"
    dir: Path = Path("logs")
    file_level: str = "INFO"
    max_bytes: int = 10_485_760
    backup_count: int = 5


class WebSettings(_Group):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="WEB_")
    ENV = frozenset({"api_token"})

    # Shared secret for the dashboard's state-changing endpoints; empty
    # keeps the dashboard read-only.
    api_token: str = ""
    # Destructive actions need a confirmation header (tests turn it off).
    require_confirmation: bool = True
    audit_retention_days: int = 90  # web_actions rows kept
    watchdog_interval_seconds: int = Field(default=300, ge=0)


# ── Top-level Settings ─────────────────────────────────────────


class Settings(_Group):
    """All application settings, grouped by domain."""

    model_config = SettingsConfigDict(**_BASE_CONFIG)
    ENV = frozenset({"database_url", "data_dir"})

    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    fred: FREDSettings = Field(default_factory=FREDSettings)
    finnhub: FinnhubSettings = Field(default_factory=FinnhubSettings)
    edgar: EDGARSettings = Field(default_factory=EDGARSettings)
    halal: HalalSettings = Field(default_factory=HalalSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    zakat: ZakatSettings = Field(default_factory=ZakatSettings)
    core: CoreSettings = Field(default_factory=CoreSettings)
    stocks: StockSettings = Field(default_factory=StockSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    live_mode: LiveModeSettings = Field(default_factory=LiveModeSettings)
    log: LogSettings = Field(default_factory=LogSettings)

    # asyncpg at runtime; database_url_sync() gives alembic its psycopg twin.
    database_url: str = Field(
        default="postgresql+asyncpg://trader:trader-dev-only@localhost:5433/halal_trader",
        description="SQLAlchemy connection URL for the Postgres database",
    )
    data_dir: Path = Path("data")  # signing keys, sidecar files

    def resolve_data_dir(self) -> Path:
        """Return an absolute data-dir path, resolving relative paths from project root."""
        if self.data_dir.is_absolute():
            return self.data_dir
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / self.data_dir

    def database_url_sync(self) -> str:
        """The same database with the synchronous driver, for alembic and admin scripts."""
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
