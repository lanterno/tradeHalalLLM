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

    # Position monitor — polls open trades against SL/TP between LLM
    # cycles (cycle is 15min; the monitor fills the gap). Mirrors the
    # crypto-side knobs but at coarser cadence: stocks aren't 24/7 and
    # spreads are tighter, so a 30s loop is plenty.
    monitor_interval_seconds: float = Field(default=30.0, gt=0)
    trailing_stop_activation_pct: float | None = Field(default=None)
    trailing_stop_distance_pct: float = Field(default=0.005, gt=0)

    # Portfolio-risk knobs (used by ``trading/risk.py``). Default values
    # are tuned for daily equity bars; the operator can override per
    # deployment. Round-4 wave 0.C moved these from CryptoSettings so
    # stocks + crypto have independent volatility regimes.
    max_portfolio_heat_pct: float = Field(default=0.05, ge=0.01, le=0.5)
    max_drawdown_pct: float = Field(default=0.08, ge=0.01, le=0.5)
    high_correlation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    correlation_reduction_factor: float = Field(default=0.5, ge=0.1, le=1.0)
    atr_baseline: float = Field(default=0.02, gt=0)

    # Wave H stocks-side agentic mode. Off by default; opt in to drive
    # the LLM through a bounded tool-calling loop (query_rag,
    # query_regime_memory, submit_decisions) before each cycle's
    # decision. Mirrors the crypto knobs.
    agentic_enabled: bool = Field(default=False)
    agentic_max_turns: int = Field(default=5, ge=1, le=20)
    agentic_max_seconds: float = Field(default=30.0, gt=0, le=120.0)

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
    # Which headline classifier the reactor uses: "llm" (GPT-4o-mini, default)
    # or "finbert" (local ProsusAI/finbert — free, no API dependency, resilient
    # to LLM outages, but sentiment-only vs the LLM's event-typed scoring).
    headline_classifier: str = Field(default="llm")

    # News-momentum reactor: entry execution (Phase 2B / "fast in").
    # When enabled, a high-confidence scored catalyst places a real
    # paper BUY — gated on news+price-up confluence — instead of just
    # logging/notifying. Enabled by default on paper; flip to False to
    # return the reactor to observation-only.
    reactor_entries_enabled: bool = Field(default=True)
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

    # Agentic mode (Wave H) — when on, the LLM gets a toolbelt and
    # decides whether to fetch more context before submitting its
    # plan. Per-cycle budget caps cost.
    agentic_enabled: bool = Field(default=False)
    agentic_max_turns: int = Field(default=5, ge=1, le=20)
    agentic_max_seconds: float = Field(default=30.0, gt=0, le=120.0)

    # Prompt evolution (Wave F) — once-per-day GA sweep that scores
    # candidate prompts against recent replay snapshots and persists
    # them to ``prompt_genomes`` for one-click promotion. The bot
    # never auto-promotes; the operator is always in the loop.
    prompt_evo_generations: int = Field(default=8, ge=1, le=50)
    prompt_evo_population: int = Field(default=12, ge=4, le=64)
    prompt_evo_snapshots: int = Field(default=200, ge=20, le=1000)


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


class SlackSettings(BaseSettings):
    """Round-4 wave 5.G — Slack webhook notifier."""

    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="SLACK_")
    webhook_url: str = Field(default="")
    channel: str = Field(default="")


class DiscordSettings(BaseSettings):
    """Round-4 wave 5.G — Discord webhook notifier."""

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
    finnhub: FinnhubSettings = Field(default_factory=FinnhubSettings)
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
