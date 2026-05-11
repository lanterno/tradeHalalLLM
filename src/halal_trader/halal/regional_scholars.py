"""Regional scholar network routing — Round-5 Wave 23.J.

Indonesian users should be routed to Indonesian scholars; Saudi users
to Saudi scholars; etc. This module is the **region → scholar router +
preference ranker**.

Pinned semantics:

- **Closed-set Region ladder** — GULF / LEVANT / SOUTH_ASIA /
  SOUTHEAST_ASIA / WEST_AFRICA / EAST_AFRICA / MAGHREB / TURKEY / IRAN /
  EUROPE / NORTH_AMERICA / SOUTH_AMERICA.
- **Closed-set Madhhab ladder** — HANAFI / MALIKI / SHAFII / HANBALI /
  JAAFARI / ZAHIRI.
- **Per-scholar primary + alternative regions + madhhabs**.
- **Routing**: exact-region scholars first, then madhhab-compatible,
  then "global" scholars; deterministic ranking on tie.
- **Scholar status FSM** — ACTIVE → INACTIVE → DECEASED. Routing
  excludes INACTIVE + DECEASED.
- **Pure-Python deterministic.**
- **No-secret-leak pin** — scholar contact info never exposed in
  rendering.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum


class Region(str, Enum):
    """Closed-set region ladder."""

    GULF = "gulf"
    LEVANT = "levant"
    SOUTH_ASIA = "south_asia"
    SOUTHEAST_ASIA = "southeast_asia"
    WEST_AFRICA = "west_africa"
    EAST_AFRICA = "east_africa"
    MAGHREB = "maghreb"
    TURKEY = "turkey"
    IRAN = "iran"
    EUROPE = "europe"
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"


class Madhhab(str, Enum):
    """Closed-set madhhab ladder."""

    HANAFI = "hanafi"
    MALIKI = "maliki"
    SHAFII = "shafii"
    HANBALI = "hanbali"
    JAAFARI = "jaafari"
    ZAHIRI = "zahiri"


class ScholarStatus(str, Enum):
    """Closed-set scholar-status FSM."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DECEASED = "deceased"


@dataclass(frozen=True)
class Scholar:
    """One scholar of record."""

    scholar_id: str
    display_name: str
    primary_region: Region
    """Region the scholar primarily serves."""
    alternative_regions: tuple[Region, ...] = ()
    madhhabs: tuple[Madhhab, ...] = ()
    """Madhhabs the scholar issues fatwa under. Empty = mixed."""
    is_global: bool = False
    """Globally-recognised scholars (e.g. cross-region fatwa councils)
    are returned as a last-resort fallback."""
    status: ScholarStatus = ScholarStatus.ACTIVE
    bio_summary: str = ""

    def __post_init__(self) -> None:
        if not self.scholar_id or not self.scholar_id.strip():
            raise ValueError("scholar_id must be non-empty")
        if not self.display_name.strip():
            raise ValueError("display_name must be non-empty")
        if len(self.display_name) > 200:
            raise ValueError("display_name must be ≤ 200 chars")
        if self.primary_region in self.alternative_regions:
            raise ValueError("primary_region cannot also appear in alternative_regions")
        if len(set(self.alternative_regions)) != len(self.alternative_regions):
            raise ValueError("alternative_regions must be unique")
        if len(set(self.madhhabs)) != len(self.madhhabs):
            raise ValueError("madhhabs must be unique")
        if len(self.bio_summary) > 1000:
            raise ValueError("bio_summary must be ≤ 1000 chars")

    def serves(self, region: Region) -> bool:
        """True iff scholar's primary or alternative regions include `region`."""
        return self.primary_region is region or region in self.alternative_regions


@dataclass(frozen=True)
class UserProfile:
    """Operator-supplied user profile snippet used for routing."""

    user_id: str
    region: Region
    preferred_madhhab: Madhhab | None = None

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")


def transition_status(scholar: Scholar, *, new_status: ScholarStatus) -> Scholar:
    """Move the scholar through the status FSM.

    Pinned: ACTIVE → INACTIVE → DECEASED; ACTIVE → DECEASED also legal;
    DECEASED is terminal; INACTIVE → ACTIVE allowed (sabbatical).
    """
    legal: dict[ScholarStatus, set[ScholarStatus]] = {
        ScholarStatus.ACTIVE: {ScholarStatus.INACTIVE, ScholarStatus.DECEASED},
        ScholarStatus.INACTIVE: {ScholarStatus.ACTIVE, ScholarStatus.DECEASED},
        ScholarStatus.DECEASED: set(),
    }
    if new_status not in legal[scholar.status]:
        raise ValueError(f"illegal transition {scholar.status.value} → {new_status.value}")
    return replace(scholar, status=new_status)


def _score_for_user(scholar: Scholar, user: UserProfile) -> tuple[int, int, int, str]:
    """Sort key for routing — lower is better.

    Tiers (most-preferred first):
      0 — primary_region match
      1 — alternative_regions match
      2 — global scholar
      3 — no region match (not routed)

    Within a tier, madhhab match (when user has a preference) breaks ties:
      0 — madhhab matches
      1 — no madhhab match

    Within sub-tier, scholar_id provides deterministic tie-break.
    """
    region_score: int
    if scholar.primary_region is user.region:
        region_score = 0
    elif user.region in scholar.alternative_regions:
        region_score = 1
    elif scholar.is_global:
        region_score = 2
    else:
        region_score = 3
    madhhab_score: int
    if user.preferred_madhhab is None or not scholar.madhhabs:
        madhhab_score = 1
    elif user.preferred_madhhab in scholar.madhhabs:
        madhhab_score = 0
    else:
        madhhab_score = 1
    return (region_score, madhhab_score, 0, scholar.scholar_id)


def route_user(
    user: UserProfile,
    scholars: Iterable[Scholar],
    *,
    top_n: int = 5,
) -> tuple[Scholar, ...]:
    """Return the top-N scholars for this user, ranked per `_score_for_user`.

    Pinned: INACTIVE + DECEASED scholars are excluded; scholars with
    region_score=3 (no region match, not global) are excluded entirely.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    active = [s for s in scholars if s.status is ScholarStatus.ACTIVE]
    scored = [(_score_for_user(s, user), s) for s in active if _score_for_user(s, user)[0] < 3]
    scored.sort(key=lambda kv: kv[0])
    return tuple(s for _, s in scored[:top_n])


def scholars_for_region(region: Region, scholars: Iterable[Scholar]) -> tuple[Scholar, ...]:
    """All ACTIVE scholars who serve `region` (primary or alternative).
    Deterministic ordering by scholar_id."""
    out = [s for s in scholars if s.status is ScholarStatus.ACTIVE and s.serves(region)]
    out.sort(key=lambda s: s.scholar_id)
    return tuple(out)


def has_madhhab_compatible(
    region: Region,
    madhhab: Madhhab,
    scholars: Iterable[Scholar],
) -> bool:
    """True iff ≥ 1 ACTIVE scholar in `region` covers `madhhab`."""
    for s in scholars:
        if s.status is not ScholarStatus.ACTIVE:
            continue
        if not s.serves(region):
            continue
        if madhhab in s.madhhabs:
            return True
    return False


_REGION_EMOJI: dict[Region, str] = {
    Region.GULF: "🕌",
    Region.LEVANT: "🕌",
    Region.SOUTH_ASIA: "🇵🇰",
    Region.SOUTHEAST_ASIA: "🇮🇩",
    Region.WEST_AFRICA: "🇳🇬",
    Region.EAST_AFRICA: "🇰🇪",
    Region.MAGHREB: "🇲🇦",
    Region.TURKEY: "🇹🇷",
    Region.IRAN: "🇮🇷",
    Region.EUROPE: "🇪🇺",
    Region.NORTH_AMERICA: "🇺🇸",
    Region.SOUTH_AMERICA: "🌎",
}


def render_scholar(scholar: Scholar) -> str:
    """Operator-readable scholar summary. Pin: bio_summary truncated;
    scholar_id rendered as-is (operator-side identifier)."""
    emoji = _REGION_EMOJI.get(scholar.primary_region, "🌐")
    status = scholar.status.value
    flag = " 🌐" if scholar.is_global else ""
    madhhabs = " [" + "/".join(m.value for m in scholar.madhhabs) + "]" if scholar.madhhabs else ""
    return (
        f"{emoji} {scholar.display_name} [{scholar.primary_region.value}/{status}]{flag}{madhhabs}"
    )


def render_routing(
    user: UserProfile,
    matches: Iterable[Scholar],
) -> str:
    """Operator-readable routing-result summary."""
    rows = tuple(matches)
    head = (
        f"🧭 Route for {user.user_id} (region={user.region.value}"
        + (f", madhhab={user.preferred_madhhab.value}" if user.preferred_madhhab else "")
        + f"): {len(rows)} scholar(s)"
    )
    if not rows:
        return head
    lines = [head]
    for s in rows:
        lines.append(f"  • {render_scholar(s)}")
    return "\n".join(lines)
