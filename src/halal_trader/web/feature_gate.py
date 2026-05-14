"""Open-source / paid hosting feature gate.

The roadmap pins Wave 10.F: "The core trading engine + screener
stays open source (MIT). Hosted multi-user is the paid product.
Aligns incentives — community contributes to a free core;
commercial users fund development." This module is the
**pure-Python feature-gating engine** that decides which features
are available in the open-source build vs the hosted product, and
within the hosted product, which tier (free / pro / enterprise)
unlocks them.

Picked a focused gate over scattered `if settings.edition == ...`
checks because (a) the OSS / HOSTED boundary is the *commercial*
boundary that funds development; mistakes here either leak paid
features into the free build (revenue loss) or wall off OSS features
behind a paywall (community pain), so the policy lives in one
inspectable matrix rather than 30 scattered conditionals; (b) the
tier requirements within HOSTED need to compose with Wave 3.C
quotas and Wave 3.F billing — they all key on the same `Tier` enum,
so adding a new feature is a single registry entry, not a
cross-module change; (c) the *render* of the feature matrix is
what marketing pages and operator-facing docs show; pinning it as
a pure function of the registry means the docs can't drift from
the code; (d) the core trading + screener features must stay
edition-agnostic — pinned via a "core feature available in OSS"
regression test against silent regression that walls off the
trading engine itself.

Pinned semantics:
- **OSS edition is the strict subset.** Every feature available in
  OSS is also available in HOSTED at every tier. The reverse is
  not true: hosted-only features (multi-user, billing, admin
  console) are unavailable in OSS regardless of tier.
- **Tier requirement is independent of edition.** The OSS edition
  has no tier (it's all-or-nothing per feature); HOSTED edition
  has a per-feature minimum tier requirement. Asking "is feature X
  available?" requires both edition and tier.
- **Closed-set features.** Features are listed as a frozen registry
  at module load; runtime mutation is impossible. Adding a feature
  is a code review change, mirroring Wave 3.C quota system.
- **Core trading + screener stays in OSS.** The CYCLE_RUN,
  STOCK_TRADING, CRYPTO_TRADING, HALAL_SCREENER features are
  pinned available in OSS via regression test — a future PR that
  walls these off behind a paywall fails CI rather than ships.
- **Render output never names internal config keys / env vars /
  Stripe IDs.** Mirrors the no-secret patterns of upstream waves.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from halal_trader.web.quotas import Tier


class Edition(str, Enum):
    """Build edition. Pinned string values for JSON / DB stability.

    `OSS` = the open-source MIT-licensed core (run on operator's own
    laptop / VPS). `HOSTED` = the paid multi-user hosted product
    (run on the project's infrastructure with billing / admin /
    multi-tenant primitives).
    """

    OSS = "oss"
    HOSTED = "hosted"


class Feature(str, Enum):
    """Catalogue of gateable features.

    Pinned string values. Adding a feature is a code review change.
    """

    # Core trading (pinned available in OSS)
    CYCLE_RUN = "cycle_run"
    STOCK_TRADING = "stock_trading"
    CRYPTO_TRADING = "crypto_trading"
    HALAL_SCREENER = "halal_screener"

    # Single-user dashboards (OSS) and operator tooling
    LOCAL_DASHBOARD = "local_dashboard"
    LOCAL_BACKTEST = "local_backtest"
    PURIFICATION_LEDGER = "purification_ledger"

    # Hosted-only (multi-user, billing, admin)
    MULTI_USER_AUTH = "multi_user_auth"
    PER_USER_VAULT = "per_user_vault"
    PER_USER_QUOTAS = "per_user_quotas"
    BILLING = "billing"
    ADMIN_CONSOLE = "admin_console"
    LEADERBOARD = "leaderboard"
    ONBOARDING_FLOW = "onboarding_flow"

    # Hosted, tier-gated
    LIVE_LLM_CYCLES = "live_llm_cycles"
    PREMIUM_DATASETS = "premium_datasets"
    SCHOLAR_REVIEW_QUEUE = "scholar_review_queue"
    PUBLIC_RESEARCH_API = "public_research_api"


# Tier ordering (FREE < PRO < ENTERPRISE) for the "meets minimum tier" check.
_TIER_ORDER: dict[Tier, int] = {
    Tier.FREE: 0,
    Tier.PRO: 1,
    Tier.ENTERPRISE: 2,
}


@dataclass(frozen=True)
class FeatureSpec:
    """One feature's edition + tier availability.

    `oss_available` is the OSS edition gate (True = available in OSS;
    False = HOSTED-only). `min_tier` is the minimum HOSTED tier; FREE
    means available to all hosted users; ENTERPRISE means only the
    ENTERPRISE tier sees it. Ignored entirely in OSS edition (where
    there's no tier).
    """

    feature: Feature
    oss_available: bool
    min_tier: Tier
    description: str

    def __post_init__(self) -> None:
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")


# The canonical feature registry. Frozen at module load.
_FEATURE_REGISTRY: dict[Feature, FeatureSpec] = {
    # ---- core trading: OSS-available, all hosted tiers
    Feature.CYCLE_RUN: FeatureSpec(
        feature=Feature.CYCLE_RUN,
        oss_available=True,
        min_tier=Tier.FREE,
        description="Run trading cycles (the core engine)",
    ),
    Feature.STOCK_TRADING: FeatureSpec(
        feature=Feature.STOCK_TRADING,
        oss_available=True,
        min_tier=Tier.FREE,
        description="Stock paper-trading via the broker plugin",
    ),
    Feature.CRYPTO_TRADING: FeatureSpec(
        feature=Feature.CRYPTO_TRADING,
        oss_available=True,
        min_tier=Tier.FREE,
        description="Crypto paper-trading on Binance testnet",
    ),
    Feature.HALAL_SCREENER: FeatureSpec(
        feature=Feature.HALAL_SCREENER,
        oss_available=True,
        min_tier=Tier.FREE,
        description="Halal compliance screening (Zoya + crypto screener)",
    ),
    Feature.LOCAL_DASHBOARD: FeatureSpec(
        feature=Feature.LOCAL_DASHBOARD,
        oss_available=True,
        min_tier=Tier.FREE,
        description="Local-only single-user dashboard",
    ),
    Feature.LOCAL_BACKTEST: FeatureSpec(
        feature=Feature.LOCAL_BACKTEST,
        oss_available=True,
        min_tier=Tier.FREE,
        description="Local backtest runner",
    ),
    Feature.PURIFICATION_LEDGER: FeatureSpec(
        feature=Feature.PURIFICATION_LEDGER,
        oss_available=True,
        min_tier=Tier.FREE,
        description="Per-trade purification ledger + disbursement scheduler",
    ),
    # ---- hosted-only multi-tenant primitives
    Feature.MULTI_USER_AUTH: FeatureSpec(
        feature=Feature.MULTI_USER_AUTH,
        oss_available=False,
        min_tier=Tier.FREE,
        description="Multi-user accounts + sessions",
    ),
    Feature.PER_USER_VAULT: FeatureSpec(
        feature=Feature.PER_USER_VAULT,
        oss_available=False,
        min_tier=Tier.FREE,
        description="Encrypted per-user secrets vault",
    ),
    Feature.PER_USER_QUOTAS: FeatureSpec(
        feature=Feature.PER_USER_QUOTAS,
        oss_available=False,
        min_tier=Tier.FREE,
        description="Per-user resource quotas",
    ),
    Feature.BILLING: FeatureSpec(
        feature=Feature.BILLING,
        oss_available=False,
        min_tier=Tier.FREE,
        description="Stripe billing + tier management",
    ),
    Feature.ADMIN_CONSOLE: FeatureSpec(
        feature=Feature.ADMIN_CONSOLE,
        oss_available=False,
        min_tier=Tier.FREE,
        description="Admin console for hosted-tenant management",
    ),
    Feature.LEADERBOARD: FeatureSpec(
        feature=Feature.LEADERBOARD,
        oss_available=False,
        min_tier=Tier.FREE,
        description="Public anonymised strategy leaderboard",
    ),
    Feature.ONBOARDING_FLOW: FeatureSpec(
        feature=Feature.ONBOARDING_FLOW,
        oss_available=False,
        min_tier=Tier.FREE,
        description="Self-service onboarding wizard",
    ),
    # ---- hosted, tier-gated value-adds
    Feature.LIVE_LLM_CYCLES: FeatureSpec(
        feature=Feature.LIVE_LLM_CYCLES,
        oss_available=False,
        min_tier=Tier.PRO,
        description="LLM-driven live cycles (vs free-tier rule-based)",
    ),
    Feature.PREMIUM_DATASETS: FeatureSpec(
        feature=Feature.PREMIUM_DATASETS,
        oss_available=False,
        min_tier=Tier.PRO,
        description="Premium alt-data feeds (sentiment, filings)",
    ),
    Feature.SCHOLAR_REVIEW_QUEUE: FeatureSpec(
        feature=Feature.SCHOLAR_REVIEW_QUEUE,
        oss_available=False,
        min_tier=Tier.PRO,
        description="Scholar-review queue routing for halal exceptions",
    ),
    Feature.PUBLIC_RESEARCH_API: FeatureSpec(
        feature=Feature.PUBLIC_RESEARCH_API,
        oss_available=False,
        min_tier=Tier.ENTERPRISE,
        description="Public research API access",
    ),
}


def feature_spec(feature: Feature) -> FeatureSpec:
    """Return the spec for a feature."""

    return _FEATURE_REGISTRY[feature]


def all_features() -> tuple[FeatureSpec, ...]:
    """Return all features in canonical order."""

    return tuple(_FEATURE_REGISTRY[f] for f in Feature)


class FeatureNotAvailableError(Exception):
    """Raised when a feature is requested but not available in the
    current edition / tier context. Carries the feature + edition +
    tier so the caller's handler can render an actionable message."""

    def __init__(
        self,
        feature: Feature,
        edition: Edition,
        tier: Tier | None,
    ) -> None:
        tier_str = tier.value if tier is not None else "n/a"
        super().__init__(
            f"feature {feature.value!r} not available in edition={edition.value} tier={tier_str}"
        )
        self.feature = feature
        self.edition = edition
        self.tier = tier


def is_feature_available(
    feature: Feature,
    *,
    edition: Edition,
    tier: Tier | None = None,
) -> bool:
    """Return True if `feature` is available in the given context.

    For OSS edition, `tier` is ignored; the OSS gate alone determines
    availability. For HOSTED edition, `tier` must be provided and the
    user's tier must meet the feature's `min_tier` (FREE < PRO <
    ENTERPRISE).

    Raises ValueError if HOSTED edition is queried without a tier.
    """

    spec = feature_spec(feature)
    if edition is Edition.OSS:
        return spec.oss_available
    if tier is None:
        raise ValueError("tier is required for HOSTED edition")
    return _TIER_ORDER[tier] >= _TIER_ORDER[spec.min_tier]


def require_feature(
    feature: Feature,
    *,
    edition: Edition,
    tier: Tier | None = None,
) -> None:
    """Raise FeatureNotAvailableError if `feature` is not available."""

    if not is_feature_available(feature, edition=edition, tier=tier):
        raise FeatureNotAvailableError(feature, edition, tier)


def features_available(
    *,
    edition: Edition,
    tier: Tier | None = None,
) -> tuple[FeatureSpec, ...]:
    """Return the features available in the given context.

    Same input → same output (deterministic order from `Feature`
    enum).
    """

    if edition is Edition.HOSTED and tier is None:
        raise ValueError("tier is required for HOSTED edition")
    return tuple(
        spec
        for spec in all_features()
        if is_feature_available(spec.feature, edition=edition, tier=tier)
    )


def render_matrix() -> str:
    """Render the full feature × edition × tier availability matrix.

    For marketing pages + docs. Shows every feature with checkmarks
    per (edition, tier) cell. No-secret-leak by construction: the
    function takes no arguments and emits only feature catalogue
    contents, which contain only documented descriptions.
    """

    headers = ("Feature", "OSS", "Free", "Pro", "Enterprise")
    rows: list[tuple[str, ...]] = []
    for spec in all_features():
        cells = (
            spec.feature.value,
            "✅" if spec.oss_available else "—",
            "✅"
            if is_feature_available(spec.feature, edition=Edition.HOSTED, tier=Tier.FREE)
            else "—",
            "✅"
            if is_feature_available(spec.feature, edition=Edition.HOSTED, tier=Tier.PRO)
            else "—",
            "✅"
            if is_feature_available(spec.feature, edition=Edition.HOSTED, tier=Tier.ENTERPRISE)
            else "—",
        )
        rows.append(cells)

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    body_lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def render_context_summary(
    *,
    edition: Edition,
    tier: Tier | None = None,
) -> str:
    """Render a short summary of what's available in the given context."""

    available = features_available(edition=edition, tier=tier)
    tier_label = tier.value if tier is not None else "n/a"
    lines = [
        f"Edition: {edition.value} | Tier: {tier_label} | "
        f"{len(available)}/{len(_FEATURE_REGISTRY)} features available"
    ]
    for spec in available:
        lines.append(f"  ✅ {spec.feature.value} — {spec.description}")
    return "\n".join(lines)


__all__ = [
    "Edition",
    "Feature",
    "FeatureNotAvailableError",
    "FeatureSpec",
    "all_features",
    "feature_spec",
    "features_available",
    "is_feature_available",
    "render_context_summary",
    "render_matrix",
    "require_feature",
]
