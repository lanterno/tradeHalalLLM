"""Per-scholar Sharia-compliance policy profiles.

Round-4 wave 2.C: scholars don't all agree. AAOIFI (the most-cited
international standard) sets debt-to-market-cap at 30% and
non-permissible-income at 5%. Mufti Taqi Usmani — chairman of the
AAOIFI Sharia Board for years — has at times argued for stricter
thresholds in specific contexts. Sheikh Yusuf Talal DeLorenzo's
DJIM-era methodology used 33%. Practitioners live with the
disagreement; the bot should let the operator name which methodology
they're following so the audit trail records the choice.

A `ScholarProfile` bundles three concerns:

* **Per-provider weights** for the WEIGHTED consensus path. Some
  scholars trust certain providers more than others (e.g. operators
  in the Hanafi tradition often prefer Musaffa's ruling on
  borderline tech sector cases over Zoya's broader-net approach).
* **Threshold overrides** for the debt-to-market-cap and
  non-permissible-income ratios. The defaults match AAOIFI's
  published 30% / 5% / 33% triplet (debt / non-permissible-income
  / cash-and-receivables). Profiles can tighten or relax
  individually.
* **Default consensus policy** so the operator's profile choice
  also picks the resolution rule (STRICT / MAJORITY / WEIGHTED).

Why a registry rather than a class hierarchy: profiles are *data*,
not behaviour. New profiles ship as dict literals, not subclasses,
and the dashboard / audit trail can serialise them straight to JSON.
The data shape mirrors what we'd persist in a `ScholarProfile`
table later if profiles need to be operator-editable at runtime.

Halal alignment: the operator must explicitly choose a profile —
defaulting to AAOIFI is informative, not consent. The audit row
records which profile gated each trade so a future scholar can
review consistency.

Pure-Python; no DB, no async. The registry is module-level so a
profile can be looked up by name without any I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from halal_trader.halal.consensus import (
    ConsensusDecision,
    ConsensusPolicy,
    ScreeningOpinion,
    consensus,
)

# ── Threshold defaults (AAOIFI 2.4) ───────────────────────


# AAOIFI Standard 21 + 30: financial screens cap interest-bearing
# debt and non-permissible income against market cap. The numbers
# below are the published defaults; profiles may tighten them.
_AAOIFI_DEBT_RATIO_MAX = 0.30
_AAOIFI_NON_PERMISSIBLE_INCOME_MAX = 0.05
_AAOIFI_CASH_RECEIVABLES_MAX = 0.33


# ── Profile dataclass ─────────────────────────────────────


@dataclass(frozen=True)
class ScreeningThresholds:
    """Three-ratio cap set used by every mainstream screen.

    All values are fractions in [0, 1]. ``debt_to_marketcap_max``
    and ``non_permissible_income_max`` rejecting → ``not_halal``;
    ``cash_and_receivables_max`` is informational on most profiles
    (different schools weigh it differently — included so a future
    profile can flip it to a hard reject).
    """

    debt_to_marketcap_max: float = _AAOIFI_DEBT_RATIO_MAX
    non_permissible_income_max: float = _AAOIFI_NON_PERMISSIBLE_INCOME_MAX
    cash_and_receivables_max: float = _AAOIFI_CASH_RECEIVABLES_MAX


@dataclass(frozen=True)
class ScholarProfile:
    """One scholar / methodology's set of compliance preferences.

    Frozen so the registry can hand out shared instances without
    risk of accidental mutation. Profiles serialize cleanly to JSON
    via :func:`profile_to_dict`."""

    name: str
    description: str
    thresholds: ScreeningThresholds = field(default_factory=ScreeningThresholds)
    default_policy: ConsensusPolicy = ConsensusPolicy.STRICT
    provider_weights: Mapping[str, float] = field(default_factory=dict)
    rulings_doc: str | None = None  # path to / inline markdown summary


# ── Helpers ───────────────────────────────────────────────


def evaluate_thresholds(
    *,
    profile: ScholarProfile,
    debt_to_marketcap: float | None = None,
    non_permissible_income: float | None = None,
    cash_and_receivables: float | None = None,
) -> tuple[bool, list[str]]:
    """Apply the profile's threshold caps to a set of ratios.

    Returns ``(passed, violations)`` where ``violations`` lists each
    failing check by name. Missing inputs are *skipped* (treated as
    "not measured", not "passes") — pin so a partial filing doesn't
    silently approve.
    """
    violations: list[str] = []
    t = profile.thresholds
    if debt_to_marketcap is not None and debt_to_marketcap > t.debt_to_marketcap_max:
        violations.append(f"debt {debt_to_marketcap:.2%} > cap {t.debt_to_marketcap_max:.2%}")
    if non_permissible_income is not None and non_permissible_income > t.non_permissible_income_max:
        violations.append(
            f"non_permissible_income {non_permissible_income:.2%} > "
            f"cap {t.non_permissible_income_max:.2%}"
        )
    if cash_and_receivables is not None and cash_and_receivables > t.cash_and_receivables_max:
        violations.append(
            f"cash_and_receivables {cash_and_receivables:.2%} > "
            f"cap {t.cash_and_receivables_max:.2%}"
        )
    return len(violations) == 0, violations


def apply_profile_weights(
    profile: ScholarProfile, opinions: list[ScreeningOpinion]
) -> list[ScreeningOpinion]:
    """Re-weight a list of opinions using the profile's
    provider trust map.

    Sources not listed in the profile's ``provider_weights`` keep
    their existing weight (default 1.0). A weight of 0 in the
    profile effectively *abstains* that provider — the WEIGHTED
    resolver clamps it at 0 contribution.

    Returns a *new* list of frozen opinions; never mutates the
    input.
    """
    if not profile.provider_weights:
        return list(opinions)
    weighted: list[ScreeningOpinion] = []
    for op in opinions:
        new_weight = profile.provider_weights.get(op.source, op.weight)
        weighted.append(
            ScreeningOpinion(
                source=op.source,
                decision=op.decision,
                weight=new_weight,
                criteria=op.criteria,
            )
        )
    return weighted


def consensus_with_profile(
    profile: ScholarProfile,
    opinions: list[ScreeningOpinion],
    *,
    policy: ConsensusPolicy | None = None,
) -> ConsensusDecision:
    """Run the consensus aggregator using the profile's defaults.

    Convenience over composing :func:`apply_profile_weights` and
    :func:`halal.consensus.consensus` — does the right thing in
    one call: re-weights opinions if the profile carries
    provider weights AND the policy is WEIGHTED, otherwise leaves
    weights untouched. ``policy`` overrides ``profile.default_policy``
    when supplied — useful for the dashboard's "show me the
    conservative read" toggle.
    """
    chosen_policy = policy or profile.default_policy
    if chosen_policy == ConsensusPolicy.WEIGHTED and profile.provider_weights:
        opinions = apply_profile_weights(profile, opinions)
    return consensus(opinions, policy=chosen_policy)


def profile_to_dict(profile: ScholarProfile) -> dict:
    """Audit-trail-ready dict serialisation. Pin so the dashboard
    and the screening receipt see the same shape."""
    return {
        "name": profile.name,
        "description": profile.description,
        "default_policy": profile.default_policy.value,
        "thresholds": {
            "debt_to_marketcap_max": profile.thresholds.debt_to_marketcap_max,
            "non_permissible_income_max": profile.thresholds.non_permissible_income_max,
            "cash_and_receivables_max": profile.thresholds.cash_and_receivables_max,
        },
        "provider_weights": dict(profile.provider_weights),
        "rulings_doc": profile.rulings_doc,
    }


# ── Built-in profiles ─────────────────────────────────────


AAOIFI_DEFAULT = ScholarProfile(
    name="aaoifi_default",
    description=(
        "AAOIFI Sharia Standards 21 + 30 (the international default). "
        "Debt ≤ 30% market cap; non-permissible income ≤ 5%; "
        "cash-and-receivables ≤ 33%. Strict consensus across providers."
    ),
    thresholds=ScreeningThresholds(),
    default_policy=ConsensusPolicy.STRICT,
    provider_weights={},  # equal-weight under STRICT (weights ignored)
    rulings_doc="docs/halal_jurisprudence.md",
)


TAQI_USMANI = ScholarProfile(
    name="taqi_usmani",
    description=(
        "Stricter than AAOIFI default. Tightens debt cap to 25% and "
        "non-permissible income to 3% on the rationale that the "
        "AAOIFI thresholds were a starting concession, not a target. "
        "Aligns with Mufti Taqi Usmani's published guidance for "
        "Islamic financial institutions."
    ),
    thresholds=ScreeningThresholds(
        debt_to_marketcap_max=0.25,
        non_permissible_income_max=0.03,
        cash_and_receivables_max=0.33,
    ),
    default_policy=ConsensusPolicy.STRICT,
    provider_weights={
        # Operators in this tradition often weight the more
        # conservative providers heavier when in WEIGHTED mode.
        "musaffa": 1.5,
        "idealratings": 1.5,
        "zoya": 1.0,
    },
    rulings_doc="docs/halal_jurisprudence.md#mufti-taqi-usmani-profile",
)


DELORENZO_DJIM = ScholarProfile(
    name="delorenzo_djim",
    description=(
        "Sheikh Yusuf Talal DeLorenzo's DJIM-era methodology. "
        "Marginally more permissive than AAOIFI on debt (33% cap) "
        "but otherwise aligned. Useful for operators following the "
        "older Dow Jones Islamic Market screening tradition."
    ),
    thresholds=ScreeningThresholds(
        debt_to_marketcap_max=0.33,
        non_permissible_income_max=0.05,
        cash_and_receivables_max=0.33,
    ),
    default_policy=ConsensusPolicy.MAJORITY,
    provider_weights={},
    rulings_doc="docs/halal_jurisprudence.md#delorenzo-djim-profile",
)


# ── Registry ──────────────────────────────────────────────


_PROFILES: dict[str, ScholarProfile] = {
    AAOIFI_DEFAULT.name: AAOIFI_DEFAULT,
    TAQI_USMANI.name: TAQI_USMANI,
    DELORENZO_DJIM.name: DELORENZO_DJIM,
}


def get_profile(name: str) -> ScholarProfile:
    """Look up a profile by name. Raises ``KeyError`` with the list
    of available names if the name isn't registered — pin so a
    typo'd config surfaces immediately."""
    profile = _PROFILES.get(name.lower())
    if profile is None:
        available = sorted(_PROFILES)
        raise KeyError(f"unknown scholar profile {name!r}; available: {available}")
    return profile


def list_profiles() -> list[ScholarProfile]:
    """All registered profiles (used by the CLI / dashboard
    profile-picker)."""
    return list(_PROFILES.values())


def register_profile(profile: ScholarProfile) -> None:
    """Add a new profile to the registry. Lets operators ship custom
    profiles without forking the codebase. Names are
    case-insensitive but must be unique — re-registering an
    existing name overwrites silently (matches `_PROFILES` dict
    semantics; pin tests will catch it if a re-registration is
    ever a bug)."""
    if not profile.name:
        raise ValueError("profile name must not be empty")
    _PROFILES[profile.name.lower()] = profile


__all__ = [
    "AAOIFI_DEFAULT",
    "DELORENZO_DJIM",
    "ScholarProfile",
    "ScreeningThresholds",
    "TAQI_USMANI",
    "apply_profile_weights",
    "consensus_with_profile",
    "evaluate_thresholds",
    "get_profile",
    "list_profiles",
    "profile_to_dict",
    "register_profile",
]
