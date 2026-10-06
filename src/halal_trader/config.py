"""Application configuration via nested Pydantic Settings sub-models.

The top-level ``Settings`` exposes domain-grouped sub-models (``settings.alpaca``,
``settings.stocks``, ``settings.llm.glm``, …). Each sub-model is its own
``BaseSettings`` class with an ``env_prefix`` chosen to match the existing
``.env`` variable names so operators don't have to migrate their config.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# HALAL_TRADER_ENV_FILE exists so the test suite can point settings at a file
# that does not exist: with a bare ".env" every pytest run from the repo root
# loaded the operator's live API keys.
_BASE_CONFIG = SettingsConfigDict(
    env_file=os.environ.get("HALAL_TRADER_ENV_FILE", ".env"),
    env_file_encoding="utf-8",
    extra="ignore",
)


# ── Brokers ────────────────────────────────────────────────────


class AlpacaSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="ALPACA_")
    api_key: str = Field(default="")
    secret_key: str = Field(default="")
    paper_trade: bool = Field(default=True)
    # The broker adapter is a subprocess fetched by `uvx` at startup. Unpinned,
    # every container restart silently took whatever PyPI had that day: five
    # upstream schema changes each caused a silent no-trade outage, and the
    # 2026-10-01 restart jumped 2.1.1 -> 2.3.2 unannounced.
    #
    # Pinning the top-level release is necessary but NOT sufficient: uvx still
    # resolves the server's own dependencies fresh. Proven the same day --
    # 2.1.1 (July's version) no longer starts, because it imports
    # fastmcp.tools.tool and today's fastmcp has removed it. 2.3.2 is what the
    # live bot connected with on 2026-10-01. The durable fix is to stop
    # resolving at runtime at all (bake the server into the image with a locked
    # closure, then replace MCP with alpaca-py: plan 1.6 / 3.2). Bump this only
    # after the recorded-payload contract tests pass.
    mcp_server_version: str = Field(default="2.3.2", pattern=r"^\d+\.\d+\.\d+$")
    # Explicit server executable to launch instead of the default resolution:
    # the image's build-time install of infra/alpaca-mcp-server.txt at
    # /opt/alpaca-mcp (used automatically when present), else
    # "uvx alpaca-mcp-server@<mcp_server_version>" on a host. Normally empty.
    mcp_server_command: str = Field(default="")
    # How the bot reaches Alpaca: "mcp" (the frozen alpaca-mcp-server
    # subprocess) or "rest" (execution/alpaca_broker.py, direct and typed).
    # Stays "mcp" until `halal-trader broker compare` has agreed for a while.
    broker_adapter: str = Field(default="mcp", pattern=r"^(mcp|rest)$")


# ── Halal Screening ────────────────────────────────────────────


class ZoyaSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="ZOYA_")
    api_key: str = Field(default="")
    # Default to sandbox so a fresh checkout doesn't burn the operator's
    # paid quota on first run. Flip to ``false`` once a prod key is wired.
    use_sandbox: bool = Field(default=True)


class FinnhubSettings(BaseSettings):
    """Finnhub.io free-tier API key for stocks news. Drop-in
    replacement for the Yahoo Finance search endpoint which started
    hitting per-IP 429 rate limits within minutes of the morning
    cycle on 2026-05-21. Empty ``api_key`` falls back to Yahoo with
    its existing circuit breaker.
    """

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="FINNHUB_")
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
    # The day-trader's and the reactor's universe (and the shadow's, which
    # reads the same cache): this many of the largest names the strict
    # in-house screen passes, by market cap.
    universe_size: int = Field(default=20, ge=1, le=500)


# ── LLM (GLM-5.2 only) ─────────────────────────────────────────


class GLMSettings(BaseSettings):
    """GLM-5.2 endpoint configuration (OpenAI-compatible API).

    The bot speaks to exactly one model family: GLM-5.2. The default
    endpoint is OpenRouter — its multi-host routing for the same
    MIT-licensed weights is the structural fix for single-provider
    outages. Point ``base_url`` at Z.ai direct
    (https://api.z.ai/api/paas/v4) or any other OpenAI-compatible host
    to switch; remember the model id naming differs per host
    (OpenRouter ``z-ai/glm-5.2`` vs Z.ai ``glm-5.2``).
    """

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="GLM_")
    # REQUIRED — create_llm() refuses to start without it.
    api_key: str = Field(default="")
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    # Optional second endpoint tried by FallbackLLM when the primary
    # fails (e.g. Z.ai direct as a backstop behind OpenRouter). All
    # three must describe the SAME underlying model; fallback_api_key
    # defaults to the primary key when empty.
    fallback_base_url: str = Field(default="")
    fallback_model: str = Field(default="")
    fallback_api_key: str = Field(default="")
    # Client-side ceiling per call. GLM-5.2 with thinking disabled
    # answers our prompts well inside this, leaving most of the 15-min
    # stock cycle for everything else.
    timeout_seconds: int = Field(default=60, gt=0)
    # GLM-5.2 thinks by default upstream — the bot turns it off for
    # cycle latency and cost. Flip on for offline research runs only.
    thinking: bool = Field(default=False)
    # OpenRouter-only: route exclusively to hosts that honour every
    # request param (response_format + tools). Without it a request can
    # land on a host that silently drops JSON mode or tool calls.
    require_parameters: bool = Field(default=True)


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="CORE_")
    # The strict-halal core portfolio (portfolio/core_executor.py) trades on its
    # own Alpaca paper account: the day-trader flattens its account every day.
    # Nothing is ordered unless enabled AND both keys are set.
    enabled: bool = Field(default=False)
    alpaca_api_key: str = Field(default="")
    alpaca_secret_key: str = Field(default="")
    # Which Alpaca environment the core's keys belong to. True (paper) until the
    # operator deliberately switches it; live money needs the plan's G1 sign-off.
    paper: bool = Field(default=True)
    top_n: int = Field(default=100, ge=10, le=500)
    # Live money (paper=false) also needs, at bot start and on `core run`, this
    # dated token: "I-UNDERSTAND-REAL-MONEY-CORE-<UTC date>" (core/safeguards.py).
    live_confirmation: str = Field(default="")
    # The first live stage's ceiling: the most the core may hold invested on
    # the live account, in dollars. Buys stop there; nothing is sold to meet it.
    live_max_notional: float = Field(default=1_000.0, gt=0)


class ZakatSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="ZAKAT_")
    # The Hijri day the operator's zakat year ends, "MM-DD" (e.g. "09-01" for
    # 1 Ramadan). Empty: `halal-trader zakat assess` needs --as-of.
    hawl_hijri: str = Field(default="")


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LLM_")
    model: str = Field(default="z-ai/glm-5.2")
    # Per-UTC-day LLM spend cap, summed over every process on the key
    # (core/llm/spend.py). 0 disables it. In observe mode crossing it alerts;
    # with budget_enforce it also refuses further LLM calls until the next
    # UTC day (exits keep working: the position monitor uses no LLM).
    daily_usd_cap: float = Field(default=0.0)
    budget_enforce: bool = Field(default=False)
    # Monthly pools (core/llm/spend.py), summed per UTC calendar month:
    # "live" = the stock bot + the shadow engine, "research" = research
    # scoring. 0 disables a pool's cap. They sit under the key's own limit.
    monthly_live_usd: float = Field(default=25.0)
    monthly_research_usd: float = Field(default=15.0)
    glm: GLMSettings = Field(default_factory=GLMSettings)


# ── Trading parameters ─────────────────────────────────────────


class StockSettings(BaseSettings):
    """Stock-side trading parameters; legacy unprefixed env names preserved."""

    model_config = SettingsConfigDict(**_BASE_CONFIG)
    trading_interval_minutes: int = Field(default=15)
    daily_return_target: float = Field(default=0.01, gt=0, le=0.5)
    max_position_pct: float = Field(default=0.20, gt=0, le=1.0)
    daily_loss_limit: float = Field(default=0.02, ge=0, le=0.5)
    max_simultaneous_positions: int = Field(default=5, ge=1)

    # Position monitor — polls open trades against SL/TP between LLM
    # cycles (cycle is 15min; the monitor fills the gap). Stocks aren't
    # 24/7 and spreads are tight, so a 30s loop is plenty.
    monitor_interval_seconds: float = Field(default=30.0, gt=0)
    trailing_stop_activation_pct: float | None = Field(default=None)
    trailing_stop_distance_pct: float = Field(default=0.005, gt=0)

    # Portfolio-risk knobs (used by ``trading/risk.py``). Default values
    # are tuned for daily equity bars; the operator can override per
    # deployment.
    max_portfolio_heat_pct: float = Field(default=0.05, ge=0.01, le=0.5)
    max_drawdown_pct: float = Field(default=0.08, ge=0.01, le=0.5)
    high_correlation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    correlation_reduction_factor: float = Field(default=0.5, ge=0.1, le=1.0)
    atr_baseline: float = Field(default=0.02, gt=0)

    # Advisory options-implied expected move on the daily recommendation
    # (quant/expected_move.py). One option-chain fetch per candidate on the
    # 09:05 job — degrades silently on any failure. Turn off if the ~20
    # extra fetches make the job too slow.
    recommendation_expected_move: bool = Field(default=True)

    # News-momentum reactor: hard ceiling on classifier API calls per
    # UTC day. Trip is silent (score=0.0) with one warning log per day.
    # 0 disables the cap. Pure cost backstop against runaway polling spend.
    # Raised 250 -> 1500 on 2026-06-09: 250 silently dropped ~65-75% of a
    # day's ~700-1000 DISTINCT headlines once dedup stopped re-scoring
    # duplicates, starving the "fast in" edge for trivial savings
    # (~$0.0001/call -> ~$0.15/day at the new ceiling). The quota-exhaustion
    # call-spam that originally motivated 250 (2026-05-22: 3,736 wasted 429s)
    # is now owned by the classifier's half-open quota breaker, not this cap.
    reactor_daily_classify_cap: int = Field(default=1500, ge=0)
    # Which headline classifier the reactor uses: "llm" (GLM-5.2, default)
    # or "finbert" (local ProsusAI/finbert — free, no API dependency, resilient
    # to LLM outages, but sentiment-only vs the LLM's event-typed scoring).
    headline_classifier: str = Field(default="llm")
    # Where the reactor's news comes from: "alpaca" (Benzinga, the feed event
    # research is backtested on; one request for every symbol) or "finnhub"
    # (one request per symbol). With "alpaca", Finnhub (if keyed) is the
    # fallback for the trading watchlist when Alpaca fails.
    reactor_news_source: str = Field(default="alpaca", pattern="^(alpaca|finnhub)$")
    # Headlines older than this are recorded but never scored or traded: the
    # fetch window reaches back hours, and a stale catalyst is in the price.
    reactor_max_headline_age_s: float = Field(default=1800.0, ge=0)
    # Observe-only list (docs/EVENT_DRIVEN_ROADMAP.md, Phase 0): the largest
    # halal names of the in-house screen, scored and recorded for research,
    # never traded. 0 disables. Its own daily classify allowance keeps it from
    # starving the trading watchlist of the shared daily cap.
    reactor_observe_size: int = Field(default=300, ge=0)
    reactor_observe_daily_classify_cap: int = Field(default=600, ge=0)

    # News-momentum reactor: entry execution ("fast in").
    # When enabled, a high-confidence scored catalyst places a real
    # paper BUY — gated on news+price-up confluence — instead of just
    # logging/notifying. Off by default since the day-trader's retirement
    # (2026-10-04): the reactor still scores and records news.
    reactor_entries_enabled: bool = Field(default=False)
    # The LLM day-trader's 15-minute cycles. False retires the strategy without
    # the kill-switch (which stops every strategy, the core included): no cycle
    # is scheduled, while the position monitor still manages open positions'
    # exits and the end-of-day flatten still runs. Off by default: retired
    # 2026-10-04 (+2.5%/yr against SPUS's 18.6%).
    day_trader_enabled: bool = Field(default=False)
    # Reactor entries are reactive / higher-variance than scheduled
    # cycle entries, so they're sized at a FRACTION of the normal
    # per-position cap (0.5 = half of ``max_position_pct``) to cap
    # blast radius while the signal is validated in production.
    reactor_entry_size_fraction: float = Field(default=0.5, gt=0, le=1.0)
    # Price-confirmation gate: a scored catalyst only fires an entry
    # when the stock is also up at least this fraction on the session
    # (latest vs prior close / today's open). Expresses the operator's
    # "news + price-up confluence" rule — we don't chase a bullish
    # headline a falling tape is already rejecting. 0 = up-or-flat only.
    reactor_entry_min_intraday_change_pct: float = Field(default=0.002, ge=0.0, le=0.5)
    # Slow-out: reactor (news-momentum) positions are locked from LLM
    # exits, so the position monitor's wide trailing stop is their main
    # exit. ~8% lets a winner run through normal intraday volatility and
    # hold overnight, only stopping out on a real structural reversal.
    # This doubles as the initial hard-stop distance set at entry.
    reactor_trailing_stop_distance_pct: float = Field(default=0.08, gt=0, le=0.5)
    # Slow-out: reactor positions are exempt from the EOD flatten so
    # winners run across days until the trailing stop / trend-break
    # exits them. Set False to force intraday-only (flatten at EOD).
    reactor_hold_overnight: bool = Field(default=True)
    # Slow-out trend-break exit: in addition to the wide trailing stop,
    # exit a *winning* reactor position when its price structure breaks
    # (closes below an SMA of recent bars) — locks in gains on a real
    # reversal instead of giving the full ~8% back to the trailing stop.
    trend_break_enabled: bool = Field(default=True)
    trend_break_ma_period: int = Field(default=20, ge=2, le=200)
    trend_break_timeframe: str = Field(default="1Hour")
    # Reason-agnostic re-entry gate: refuse a BUY for any symbol closed
    # within this window. Raised 30 → 60 on 2026-06-17 after observing
    # round-trip churn — the bot sold INTU then re-bought it exactly 30 min
    # (2 cycles) later, paying slippage both ways, which fights the
    # operator's "slow out" direction. 60 min = 4 cycles: blocks the
    # immediate flip-flop while still allowing same-day re-entry. 0 disables.
    recent_close_cooldown_minutes: int = Field(default=60, ge=0)
    # Stop-loss re-entry gate: a position the monitor STOPPED OUT is a
    # stronger "stay away" signal than an LLM-chosen sell, so it gets a
    # longer re-entry block than the reason-agnostic recent-close
    # cooldown. Stops the falling-knife loop of re-buying a downtrending
    # stock each time the cooldown elapses (observed 2026-05-27:
    # MSFT stopped out twice with an LLM re-buy between). 0 disables.
    stop_loss_reentry_cooldown_minutes: int = Field(default=120, ge=0)


# ── Notifications / Live-mode / Logging ───────────────────────


class TelegramSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="TELEGRAM_")
    bot_token: str = Field(default="")
    chat_id: str = Field(default="")


class SlackSettings(BaseSettings):
    """Slack webhook settings. Nothing reads them: no Slack notifier exists."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="SLACK_")
    webhook_url: str = Field(default="")
    channel: str = Field(default="")


class DiscordSettings(BaseSettings):
    """Discord webhook settings. Nothing reads them: no Discord notifier exists."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="DISCORD_")
    webhook_url: str = Field(default="")
    username: str = Field(default="halal-trader")


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
    # INFO: the bot at DEBUG rotated its 10 MB log ~5x a session, pushing
    # the last useful lines out of reach. LOG_FILE_LEVEL=DEBUG to dig.
    file_level: str = Field(default="INFO")
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
    # Days to keep mutation-audit rows in ``web_actions``. The bot's
    # end-of-day job prunes anything older. 0 disables the prune.
    audit_retention_days: int = Field(default=90)
    # How often the web's watchdog (web/watchdog.py) checks every process's
    # and daily job's heartbeat and alerts on Telegram. 0 disables it.
    watchdog_interval_seconds: int = Field(default=300, ge=0)
    # Allow the Vite dev server's origins (localhost:5173) through CORS.
    # Only for `npm run dev` against a local API; the built SPA is same-origin.
    cors_dev_origins: bool = Field(default=False)


# ── Top-level Settings ─────────────────────────────────────────


class Settings(BaseSettings):
    """All application settings, grouped by domain.

    Each sub-model loads its own slice of ``.env`` independently, so
    individual sub-models can be constructed in tests without bringing
    the whole tree along.
    """

    model_config = SettingsConfigDict(**_BASE_CONFIG)

    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    zoya: ZoyaSettings = Field(default_factory=ZoyaSettings)
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
    slack: SlackSettings = Field(default_factory=SlackSettings)
    discord: DiscordSettings = Field(default_factory=DiscordSettings)
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
