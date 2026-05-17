"""Halal NFT screening — Round-5 Wave 22.E.

NFTs are halal-permissible if (a) the underlying subject matter is
itself halal (no haram imagery, music, gambling associations, etc.),
(b) the provenance chain is verifiable + free of theft / fraud, and
(c) the NFT is not used as a wrapper for prohibited contracts (e.g.
fractional ownership of haram assets, lending-based rentals).

This module is the **screen**. NFT marketplaces + on-chain verification
sit above; here we encode the halal/haram subject taxonomy + the
provenance + utility-purpose checks.

Pinned semantics:

- **Closed-set SubjectMatter ladder** — 14 categories spanning
  permitted (calligraphy, abstract, nature, sukuk, IP licenses) and
  prohibited (idolatry, gambling, music_instruments_haram_view,
  haram_animal_imagery, etc.).
- **Closed-set NftIssue ladder** — explicit reasons for blocking.
- **`screen_nft`** is pure — caller passes inputs → assessment.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SubjectMatter(str, Enum):
    """Closed-set NFT subject-matter categories."""

    CALLIGRAPHY = "calligraphy"
    ABSTRACT_ART = "abstract_art"
    NATURE = "nature"
    ARCHITECTURE = "architecture"
    UTILITY_TOKEN = "utility_token"  # access pass / membership
    SUKUK_REPRESENTATION = "sukuk_representation"
    IP_LICENSE = "ip_license"
    PORTRAIT_HALAL_FIGURE = "portrait_halal_figure"
    # Prohibited
    IDOLATRY = "idolatry"
    GAMBLING_THEME = "gambling_theme"
    MUSIC_INSTRUMENT_HARAM = "music_instrument_haram"
    HARAM_ANIMAL = "haram_animal"
    NUDITY = "nudity"
    PROHIBITED_BEVERAGE = "prohibited_beverage"


PROHIBITED_SUBJECTS: frozenset[SubjectMatter] = frozenset(
    {
        SubjectMatter.IDOLATRY,
        SubjectMatter.GAMBLING_THEME,
        SubjectMatter.MUSIC_INSTRUMENT_HARAM,
        SubjectMatter.HARAM_ANIMAL,
        SubjectMatter.NUDITY,
        SubjectMatter.PROHIBITED_BEVERAGE,
    }
)


class NftIssue(str, Enum):
    """Closed-set issues an NFT can carry."""

    PROHIBITED_SUBJECT = "prohibited_subject"
    PROVENANCE_UNVERIFIED = "provenance_unverified"
    PROVENANCE_HAS_THEFT = "provenance_has_theft"
    UTILITY_IS_PROHIBITED = "utility_is_prohibited"
    EMBEDDED_FINANCING = "embedded_financing"
    FRACTIONAL_HARAM_ASSET = "fractional_haram_asset"
    NO_CREATOR_DISCLOSED = "no_creator_disclosed"


@dataclass(frozen=True)
class NftInputs:
    """Inputs for an NFT screen."""

    nft_id: str
    title: str
    subject_matter: SubjectMatter
    creator_handle: str
    provenance_verified: bool
    has_known_theft_in_chain: bool
    utility_purpose: str  # "art" / "access" / "fractional" / etc.
    embeds_financing_contract: bool
    represents_haram_underlying_asset: bool

    def __post_init__(self) -> None:
        if not self.nft_id or not self.nft_id.strip():
            raise ValueError("nft_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.utility_purpose or not self.utility_purpose.strip():
            raise ValueError("utility_purpose must be non-empty")


@dataclass(frozen=True)
class NftAssessment:
    """Result of screening an NFT."""

    nft_id: str
    subject_matter: SubjectMatter
    issues: frozenset[NftIssue]
    is_compliant: bool

    def __post_init__(self) -> None:
        if self.is_compliant and self.issues:
            raise ValueError("is_compliant=True but issues non-empty")
        if (not self.is_compliant) and not self.issues:
            raise ValueError("is_compliant=False but issues empty")


_PROHIBITED_UTILITY_PURPOSES: frozenset[str] = frozenset(
    {"gambling", "lottery", "haram_lending", "interest_bearing"}
)


def screen_nft(inputs: NftInputs) -> NftAssessment:
    """Screen an NFT for halal compliance."""
    issues: set[NftIssue] = set()

    if inputs.subject_matter in PROHIBITED_SUBJECTS:
        issues.add(NftIssue.PROHIBITED_SUBJECT)
    if not inputs.provenance_verified:
        issues.add(NftIssue.PROVENANCE_UNVERIFIED)
    if inputs.has_known_theft_in_chain:
        issues.add(NftIssue.PROVENANCE_HAS_THEFT)
    if inputs.utility_purpose.lower() in _PROHIBITED_UTILITY_PURPOSES:
        issues.add(NftIssue.UTILITY_IS_PROHIBITED)
    if inputs.embeds_financing_contract:
        issues.add(NftIssue.EMBEDDED_FINANCING)
    if inputs.represents_haram_underlying_asset:
        issues.add(NftIssue.FRACTIONAL_HARAM_ASSET)
    if not inputs.creator_handle.strip():
        issues.add(NftIssue.NO_CREATOR_DISCLOSED)

    return NftAssessment(
        nft_id=inputs.nft_id,
        subject_matter=inputs.subject_matter,
        issues=frozenset(issues),
        is_compliant=len(issues) == 0,
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "wallet_address",
    "private_key",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_assessment(inputs: NftInputs, assessment: NftAssessment) -> str:
    emoji = "✅" if assessment.is_compliant else "❌"
    head = f"{emoji} {inputs.nft_id}: {inputs.title} ({inputs.subject_matter.value})"
    lines = [head]
    for issue in sorted(assessment.issues, key=lambda x: x.value):
        lines.append(f"  • {issue.value}")
    return _scrub("\n".join(lines))
