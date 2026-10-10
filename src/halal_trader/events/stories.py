"""The news engine's stories: one symbol's admitted news and 8-Ks of one reaction session.

A **story** is every admitted item of one symbol that shares a reaction
session S (spec §A). The builder is deterministic and, given its inputs,
pure (:func:`build`); :func:`build_range` reads the event store, builds and
persists ``news_stories`` rows.

**Admission** (spec §A.2), in order, each step counted in the run log:

1. *Kind.* News, 8-K and 8-K/A only: insider rows and 10-Q/10-K are context.
2. *Roundup.* News tagging more than :data:`ROUNDUP_MAX_SYMBOLS` symbols is
   dropped. A live row has no ``payload.symbols``: it counts as one and is
   flagged ``nsym_unknown``.
3. *Entity.* News is kept only when its headline names the company
   (``aliases.AliasMatcher``). An analyst headline (``ANALYST_ACTION``) with
   a company slot in some clause (``ANALYST_SLOT``) is read clause by clause
   (:func:`analyst_clause`): the clause whose slot names the company, with
   the slot-less clauses after it, is what the taxonomy classifies; one with
   no such clause is dropped. An analyst headline with no slot at all ("Vetr
   Issues Downgrade To Hold On Costco", "FactSet Cuts FY23 Revenue Guidance,
   Reiterates Adj EPS") is checked and classified whole, as any other news.
4. *Time.* News is public at ``published_at``; an 8-K at
   :func:`filing_public_at` of its acceptance (EDGAR disseminates a filing
   accepted after 17:30 ET, or on a day EDGAR is closed, at 06:00 ET of its
   next business day). Every item is usable :data:`NEWS_LAG` later
   (``available_at``).

Before admission, :func:`load_items` drops every news row that is not its
symbol's own (``renames.owner``: a ticker another company held that day, or
an old ticker's row whose copy the renamed-news backfill stored under the
current symbol). Filings are kept as they are.

**Grouping.** The reaction session of an item is the session of its
``available_at`` when that is no later than :data:`REACT_CUTOFF` before the
session's effective close, else the next session (:func:`reaction_session`).

**Inside a story** (spec §A.4): an item whose headline shingles have a
Jaccard of at least :data:`DUP_JACCARD` with an earlier item's is a
duplicate (``dup_of`` its first copy). A "CORRECTION"/"CORRECTED" wire
that is substantive (not noise, a law-firm alert or a mover) supersedes the
earlier items sharing a fact key (kind, period, basis, metric) or a Jaccard
of at least :data:`CORRECTION_JACCARD`: from its own ``available_at`` on,
their facts no longer count.

**Time-indexed labels.** A story never knows an item before the item's
``available_at``: :meth:`Story.card_at` is ``taxonomy.resolve`` over the
items available by ``t``, with ``follower=``:meth:`Story.follower_at` at the
same ``t``. A story *follows* its parent -- the symbol's latest story, among
the :data:`FOLLOW_SESSIONS` sessions before S, that was not itself a
follower at its own close and is more than noise -- when the parent closed
structural, when every item it has so far is reactive (an analyst
price-target or other note, a mover, unparsed guidance), or when its first
substantive item re-runs a parent item (Jaccard at least
:data:`REPOST_JACCARD`). The reactive rule turns false as soon as a
non-reactive item arrives. :meth:`Story.nsn_at` re-reads the card at every
item until its cutoff.

Noise, law-firm and mover items (``taxonomy.NOISE_TYPES``) stay in the story
for display: they never trigger it (no ``detect_at``; they supersede
nothing, so no card turns NSN at their arrival), and a story of nothing
else is persisted as ``noise_only`` and is nobody's parent.

**Inputs a build checks** (:func:`build_range`, unless forced): every month
of the renamed tickers' news fetched, the story aliases stored, and every
news event it reads parsed by the current earnings extractor
(``earnings_parse.EXTRACTOR``, read when the build runs). A build replaces
its range one symbol batch at a time, each batch in one transaction, and
records the range complete in ``backfill_progress`` (task :data:`TASK`,
unit :func:`build_unit`) only once every batch is written. The mark names
the inputs the build read (:func:`inputs_sha`: every pin, the stored
aliases' among them, and the extractor); counts refuse a range no complete
build from today's inputs covers, so a range built before the aliases, a
pin or the extractor changed is rebuilt before it is counted. A change to
the builder's rules that no pin sees must bump ``BUILDER_VERSION``.

The builder's own constants are pinned by :data:`STORIES_SHA`;
:func:`pins` gathers it with the other pins of the pre-registration.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from halal_trader.data.minutes import session_bounds
from halal_trader.db.models import NewsStory
from halal_trader.events import earnings_parse, headline_patterns, renames
from halal_trader.events.aliases import (
    BUILDER_VERSION,
    AliasMatcher,
    alias_sha,
    load_aliases,
    matcher_for,
)
from halal_trader.events.earnings_parse import EarningsFacts
from halal_trader.events.headline_patterns import (
    ANALYST_ACTION,
    ANALYST_SLOT,
    CORRECTION,
    REISSUE_PREFIX,
)
from halal_trader.events.history import mark_units
from halal_trader.events.taxonomy import (
    FAMILY,
    NOISE_TYPES,
    REACTIVE,
    STRUCTURAL,
    TAXONOMY_SHA,
    StoryCard,
    classify_item,
    resolve,
)
from halal_trader.market_hours import (
    MARKET_TZ,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    trading_day_end_utc,
    trading_day_start_utc,
)

if TYPE_CHECKING:
    from halal_trader.events.context import PitContext

logger = logging.getLogger(__name__)

__all__ = [
    "BUILDER_VERSION",
    "HEADLINE_PATTERNS_SHA",
    "STORIES_SHA",
    "RawItem",
    "Story",
    "StoryCounts",
    "StoryItem",
    "StoriesNotReady",
    "TASK",
    "admit",
    "analyst_clause",
    "build",
    "build_range",
    "build_unit",
    "built_ranges",
    "count_stories",
    "counts_table",
    "edgar_business_day",
    "eligibility_at",
    "fact_keys",
    "federal_holidays",
    "filing_public_at",
    "has_analyst_slot",
    "inputs_sha",
    "jaccard",
    "load_items",
    "missing_facts",
    "next_edgar_business_day",
    "persist",
    "pins",
    "reaction_session",
    "shingles",
    "story_row",
    "uncovered_sessions",
    "window_of",
]

# ── constants (spec §A.1) ──────────────────────────────────────

ROUNDUP_MAX_SYMBOLS: Final = 3
NEWS_LAG: Final = timedelta(seconds=600)  # every item usable at public time + 600 s
REACT_CUTOFF: Final = timedelta(minutes=90)  # before the session's effective close
DUP_JACCARD: Final = 0.60  # the same item repackaged, inside a story
CORRECTION_JACCARD: Final = 0.50  # a correction's text match against the item it corrects
REPOST_JACCARD: Final = 0.50  # a follower's text match against a parent item
FOLLOW_SESSIONS: Final = 3
SHINGLE_K: Final = 3
FILING_OPEN: Final = time(6, 0)
FILING_CUTOFF: Final = time(17, 30)
STORY_KINDS: Final = frozenset({"news", "8-k", "8-k/a"})  # insider, 10-Q, 10-K: context only

# A parent that closed with one of these makes every story in its shadow a
# follower (spec §A.5(a); guidance_cut is among the structural types).
FOLLOW_STRUCTURAL: Final = STRUCTURAL | {"guidance_cut"}
# Items a story reads nothing from (taxonomy.resolve ignores them too).
_IGNORED: Final = frozenset({"noise", "law_firm"})
NOISE_ONLY: Final = "noise_only"

# Days EDGAR was closed on a weekday that is no federal holiday: executive
# orders closing the federal government (24 December 2018-2020, 2024, 2025;
# 26 December 2025) and national days of mourning (George H. W. Bush,
# 2018-12-05; Jimmy Carter, 2025-01-09). The event store holds no 8-K, 10-Q
# or 10-K accepted during business hours on any of them.
EDGAR_CLOSURES: Final = frozenset(
    {
        date(2018, 12, 5),
        date(2018, 12, 24),
        date(2019, 12, 24),
        date(2020, 12, 24),
        date(2024, 12, 24),
        date(2025, 1, 9),
        date(2025, 12, 24),
        date(2025, 12, 26),
    }
)
JUNETEENTH_FROM: Final = 2021

# Shingles (spec §A.4): lowercase, no re-issue prefix, numbers as '#', no stopwords.
_TOKEN: Final = re.compile(r"[a-z][a-z0-9&'\-]+|\d+(?:\.\d+)?")
STOPWORDS: Final = frozenset(
    "the a an of to in on for and or with by at as is are from after vs est estimate "
    "inc corp co company shares stock says said".split()
)
# An analyst headline's clauses: commas and semicolons, and "and" before a new action.
CLAUSE_SPLIT: Final = re.compile(
    r"[;,]\s*|\s+and\s+(?=(?:Downgrades|Upgrades|Initiates|Assumes|Resumes|Reinstates|"
    r"Maintains|Reiterates)\b)"
)

# Items are read from here whatever the range asked for, so a story's parent
# (and its parent's) is the same in any split of the build: the news
# backfill's first day.
HISTORY_FROM: Final = date(2016, 1, 1)
# A story of S holds items from a few days before S (a weekend, a holiday).
_LEAD_DAYS: Final = 10
BATCH_SYMBOLS: Final = 100
_INSERT_CHUNK: Final = 500
# A complete build of [start, end] is one backfill_progress row (task TASK,
# unit build_unit(start, end, inputs_sha)), written after its last batch.
TASK: Final = "stories"

# The research windows `events stories counts` splits by (spec §G.5).
TRAIN: Final = (date(2016, 10, 3), date(2021, 12, 31))
VALIDATION: Final = (date(2022, 1, 3), date(2024, 12, 31))

Window = Literal["train", "validation", "other"]
Universe = Literal["all", "primary", "tech"]
UNIVERSES: Final[tuple[Universe, ...]] = ("all", "primary", "tech")


class StoriesNotReady(RuntimeError):
    """A build's inputs are incomplete (renamed-ticker news, story aliases, the
    current extractor's facts), or counts were asked of a range no complete
    build covers."""


# ── types (spec §A.1) ──────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RawItem:
    """One events row as the builder reads it."""

    event_id: int
    source_id: str
    kind: str  # 'news' | '8-k' | '8-k/a'
    symbol: str
    published_at: datetime  # news: Alpaca's created_at; a filing: EDGAR's acceptance
    seen_at: datetime
    headline: str  # '' for filings
    n_symbols: int | None  # len(payload.symbols); None for live rows
    items_8k: tuple[str, ...] = ()
    facts: tuple[EarningsFacts, ...] = ()  # extractor benzinga-earnings-v4


@dataclass(frozen=True, slots=True)
class StoryItem:
    """An admitted item: when it was public and usable, what it is, what it repeats.

    Satisfies ``taxonomy.ItemLike``.
    """

    raw: RawItem
    at: datetime  # public time: news published_at; a filing filing_public_at(accepted)
    available_at: datetime  # at + NEWS_LAG
    itype: str  # taxonomy.classify_item
    shingles: frozenset[str]
    dup_of: int | None = None  # event_id of the first copy of the item it repeats
    supersedes: tuple[int, ...] = ()  # event_ids whose facts it replaces (corrections)
    entity_ok: bool | None = None  # True: news that named the company; None: not checked

    @property
    def event_id(self) -> int:
        return self.raw.event_id

    @property
    def facts(self) -> tuple[EarningsFacts, ...]:
        return self.raw.facts


@dataclass(slots=True, eq=False)
class Story:
    """Every admitted item of one symbol that shares a reaction session S.

    Satisfies ``halabot.playbooks.interfaces.StoryView``. ``items`` are in
    (``available_at``, ``event_id``) order and must not change after the
    story is made: the cards are cached per number of known items.
    """

    story_id: str  # f"{symbol}:{session.isoformat()}"
    symbol: str
    session: date  # S, the reaction session
    items: list[StoryItem]
    parent: str | None = None  # the symbol's latest non-follower story in S-3..S-1
    parent_type_close: str | None = None
    parent_shingles: tuple[frozenset[str], ...] = ()  # the parent's items' shingles
    _times: list[datetime] = field(init=False, repr=False)
    _cards: dict[int, StoryCard] = field(init=False, repr=False)
    _followers: dict[int, bool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._times = [i.available_at for i in self.items]
        if self._times != sorted(self._times):
            raise ValueError(f"{self.story_id}: items are not in available_at order")
        self._cards = {}
        self._followers = {}

    # ── time-indexed labels ─────────────────────────────────

    def _known(self, t: datetime) -> int:
        """How many items are available at ``t``."""
        return bisect_right(self._times, t)

    def follower_at(self, t: datetime) -> bool:
        """Whether the story follows its parent, judged on the items available by ``t``.

        (a) the parent closed structural; (b) every item known so far that is
        not noise or a law-firm alert is reactive (true while only noise is
        known, and false for good once an item is not reactive); (c) the
        first substantive item (not noise, law firm or mover) re-runs a parent
        item. Before its first item the story does not exist yet: False.
        """
        k = self._known(t)
        if self.parent is None or k == 0:
            return False
        cached = self._followers.get(k)
        if cached is not None:
            return cached
        known = self.items[:k]
        first = next((i for i in known if i.itype not in NOISE_TYPES), None)
        follower = (
            self.parent_type_close in FOLLOW_STRUCTURAL
            or all(i.itype in REACTIVE for i in known if i.itype not in _IGNORED)
            or (
                first is not None
                and any(jaccard(first.shingles, s) >= REPOST_JACCARD for s in self.parent_shingles)
            )
        )
        self._followers[k] = follower
        return follower

    def card_at(self, t: datetime) -> StoryCard:
        """``taxonomy.resolve`` over the items available by ``t``, with ``follower_at(t)``."""
        k = self._known(t)
        card = self._cards.get(k)
        if card is None:
            card = resolve(self.items[:k], t, follower=self.follower_at(t))
            self._cards[k] = card
        return card

    def nsn_at(self, cutoff: datetime) -> datetime | None:
        """The first item ``available_at <= cutoff`` at which the card is
        ``NSN_CORE``; None when it never is by ``cutoff``.

        The card is re-read at every item, so a story that turns NSN late (a
        downgrade after reactive notes) is detected then. A noise, law-firm or
        mover item supersedes nothing and is no type the family reads, so the
        card never turns NSN at one; it is re-read there all the same.
        """
        for item in self.items:
            if item.available_at > cutoff:
                break
            if self.card_at(item.available_at).family == FAMILY:
                return item.available_at
        return None

    def _trigger(self) -> StoryItem | None:
        """The item at whose arrival the story first turned NSN: of the items
        arriving at that moment, the first substantive one (any, if none is)."""
        if not self.items:
            return None
        t = self.nsn_at(self._times[-1])
        if t is None:
            return None
        arrived = [i for i in self.items if i.available_at == t]
        return next((i for i in arrived if i.itype not in NOISE_TYPES), arrived[0])

    def at_news(self) -> datetime | None:
        """``at`` (public time) of the item that first made the story NSN_CORE."""
        item = self._trigger()
        return item.at if item is not None else None

    def _first_real(self) -> StoryItem | None:
        return next((i for i in self.items if i.itype not in NOISE_TYPES), None)

    def detect_at(self) -> datetime | None:
        """``available_at`` of the first substantive item (not noise, law firm or mover)."""
        first = self._first_real()
        return first.available_at if first is not None else None

    def start_case(self) -> Literal["in", "out"]:
        """``"in"`` when the news came inside S's session ([open, effective close)).

        The news is the item that made the story NSN (``at_news``); for a
        story that never turns NSN, its first substantive item. A story with
        neither is ``"out"``.
        """
        at = self.at_news()
        if at is None:
            first = self._first_real()
            at = first.at if first is not None else None
        if at is None:
            return "out"
        open_, close = session_bounds(self.session)
        return "in" if open_ <= at < close else "out"

    def news_times(self) -> list[datetime]:
        """``available_at`` of every item, in order (the simulator's ``NewsIn`` times)."""
        return list(self._times)

    # ── the story at S's close ──────────────────────────────

    @property
    def close(self) -> datetime:
        """S's effective close: every item of the story is available by then."""
        return session_bounds(self.session)[1]

    @property
    def noise_only(self) -> bool:
        """Only noise, law-firm and mover items: not a candidate for anything."""
        return all(i.itype in NOISE_TYPES for i in self.items)

    def type_close(self) -> str:
        """The type at S's close; ``noise_only`` for a story of noise items only."""
        return NOISE_ONLY if self.noise_only else self.card_at(self.close).type

    def type_detect(self) -> str:
        """The type when the story was detected (its first substantive item)."""
        at = self.detect_at()
        return NOISE_ONLY if at is None else self.card_at(at).type


# ── time (spec §A.2 step 4, §A.3) ──────────────────────────────


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7, weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    """A holiday on a Saturday is observed the Friday before, on a Sunday the Monday after."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@functools.cache
def federal_holidays(year: int) -> frozenset[date]:
    """The observed US federal holidays of ``year`` (5 U.S.C. 6103).

    New Year's Day falling on a Saturday is observed on 31 December of the
    year before, which this set (for ``year``) then holds.
    """
    mon, thu = 0, 3
    days = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, mon, 3),  # Martin Luther King Jr.
        _nth_weekday(year, 2, mon, 3),  # Washington's Birthday
        _last_weekday(year, 5, mon),  # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, mon, 1),  # Labor Day
        _nth_weekday(year, 10, mon, 2),  # Columbus Day
        _observed(date(year, 11, 11)),  # Veterans Day
        _nth_weekday(year, 11, thu, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    if year >= JUNETEENTH_FROM:
        days.add(_observed(date(year, 6, 19)))
    return frozenset(days)


def edgar_business_day(day: date) -> bool:
    """A weekday that is no federal holiday (observed) and no other EDGAR closure."""
    return (
        day.weekday() < 5
        and day not in federal_holidays(day.year)
        and day not in federal_holidays(day.year + 1)  # New Year's observed on 31 Dec
        and day not in EDGAR_CLOSURES
    )


def next_edgar_business_day(after: date) -> date:
    day = after + timedelta(days=1)
    while not edgar_business_day(day):
        day += timedelta(days=1)
    return day


def filing_public_at(accepted: datetime) -> datetime:
    """When a filing accepted at ``accepted`` became public, in UTC.

    Accepted on an EDGAR business day in [06:00, 17:30) ET: then. Before
    06:00 on a business day: 06:00 that day. Otherwise (after 17:30, or on a
    day EDGAR is closed): 06:00 ET on the next EDGAR business day.
    """
    if accepted.tzinfo is None:
        raise ValueError("accepted must be timezone-aware")
    local = accepted.astimezone(MARKET_TZ)
    day = local.date()
    if edgar_business_day(day):
        if FILING_OPEN <= local.time() < FILING_CUTOFF:
            return accepted.astimezone(UTC)
        if local.time() < FILING_OPEN:
            return datetime.combine(day, FILING_OPEN, MARKET_TZ).astimezone(UTC)
    day = next_edgar_business_day(day)
    return datetime.combine(day, FILING_OPEN, MARKET_TZ).astimezone(UTC)


def reaction_session(available_at: datetime) -> date:
    """The session an item available at ``available_at`` reacts in.

    Its own New York day when that is a session and ``available_at`` is no
    later than the effective close less :data:`REACT_CUTOFF` (14:30, or
    11:30 on early closes); otherwise the next session.
    """
    if available_at.tzinfo is None:
        raise ValueError("available_at must be timezone-aware")
    day = available_at.astimezone(MARKET_TZ).date()
    if is_trading_day(day) and available_at <= session_bounds(day)[1] - REACT_CUTOFF:
        return day
    return next_trading_day(day)


# ── text (spec §A.4) ───────────────────────────────────────────


def _tokens(headline: str) -> list[str]:
    stripped = REISSUE_PREFIX.sub("", headline, count=1).lower()
    tokens = ("#" if t[0].isdigit() else t for t in _TOKEN.findall(stripped))
    return [t for t in tokens if t not in STOPWORDS]


def shingles(headline: str, k: int = SHINGLE_K) -> frozenset[str]:
    """The headline's word ``k``-grams after normalising (spec §A.4).

    A headline of fewer than ``k`` tokens is one shingle of all of them (as
    the prototype); one with none (a filing) has no shingles, so it is never
    anything's duplicate.
    """
    tokens = _tokens(headline)
    if not tokens:
        return frozenset()
    if len(tokens) < k:
        return frozenset({" ".join(tokens)})
    return frozenset(" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|a ∩ b| / |a ∪ b|; 0 when either is empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _clauses(headline: str) -> list[tuple[int, int]]:
    """The (start, end) spans of an analyst headline's clauses."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for sep in CLAUSE_SPLIT.finditer(headline):
        spans.append((pos, sep.start()))
        pos = sep.end()
    spans.append((pos, len(headline)))
    return [(a, b) for a, b in spans if headline[a:b].strip()]


def _slot(clause: str) -> str | None:
    """The clause's ``ANALYST_SLOT`` company (the first alternative that matched)."""
    m = ANALYST_SLOT.search(clause)
    if m is None:
        return None
    return next((g for g in m.groups() if g), None)


def _slots(headline: str) -> tuple[list[tuple[int, int]], list[str | None]]:
    spans = _clauses(headline)
    return spans, [_slot(headline[a:b]) for a, b in spans]


def has_analyst_slot(headline: str) -> bool:
    """Whether some clause of ``headline`` names a company in an ``ANALYST_SLOT``
    slot: then, and only then, the clause rule decides the entity check."""
    return any(slot is not None for slot in _slots(headline)[1])


def analyst_clause(headline: str, matcher: AliasMatcher) -> str | None:
    """The clause group of an analyst headline that is about ``matcher``'s company.

    The headline is split on commas, semicolons and an "and" before a new
    action. The group is the first clause whose ``ANALYST_SLOT`` slot names
    the company, with the clauses after it that have no slot of their own
    ("Baird Downgrades Acuity Brands to Neutral, Lowers Target to $265.00");
    None when no clause names it. "X Downgrades Intel, Upgrades AMD" is a
    downgrade for Intel and an upgrade for AMD.
    """
    spans, slots = _slots(headline)
    for i, slot in enumerate(slots):
        if slot is None or not matcher.matches(slot):
            continue
        j = i + 1
        while j < len(spans) and slots[j] is None:
            j += 1
        return headline[spans[i][0] : spans[j - 1][1]]
    return None


FactKey = tuple[str, str | None, str | None, str]


def fact_keys(facts: Iterable[EarningsFacts]) -> frozenset[FactKey]:
    """(kind, period, basis, metric) of every statement among ``facts``.

    A result states EPS, sales or both; a guidance fact names its metric.
    """
    keys: set[FactKey] = set()
    for f in facts:
        period, basis = f.fields.get("period"), f.fields.get("basis")
        p = None if period is None else str(period)
        b = None if basis is None else str(basis)
        if f.kind == "guidance":
            keys.add(("guidance", p, b, str(f.fields.get("metric"))))
        elif f.kind == "result":
            for m in ("eps", "sales"):
                if f.fields.get(m) is not None or f.fields.get(f"{m}_verdict") is not None:
                    keys.add(("result", p, b, m))
    return frozenset(keys)


# ── admission and building (spec §A.2-§A.5) ────────────────────


def admit(
    raw: RawItem, aliases: Mapping[str, AliasMatcher], counters: Counter[str]
) -> StoryItem | None:
    """``raw`` as a story item, or None when a rule of spec §A.2 drops it (counted)."""
    counters["rows"] += 1
    if raw.kind not in STORY_KINDS:
        counters["kind"] += 1
        return None
    clause: str | None = None
    entity_ok: bool | None = None
    if raw.kind == "news":
        if raw.n_symbols is None:
            counters["nsym_unknown"] += 1  # counts as one symbol
        elif raw.n_symbols > ROUNDUP_MAX_SYMBOLS:
            counters["roundup"] += 1
            return None
        matcher = matcher_for(raw.symbol, aliases)
        analyst = ANALYST_ACTION.search(raw.headline) is not None
        if analyst and has_analyst_slot(raw.headline):
            clause = analyst_clause(raw.headline, matcher)
            if clause is None:
                counters["entity_analyst"] += 1
                return None
        else:
            if analyst:
                counters["analyst_no_slot"] += 1  # checked and classified whole
            if not matcher.matches(raw.headline):
                counters["entity"] += 1
                return None
        entity_ok = None if matcher.empty else True
        at = raw.published_at.astimezone(UTC)
    else:
        at = filing_public_at(raw.published_at)
    counters["admitted"] += 1
    itype = classify_item(raw.kind, raw.headline, raw.items_8k, raw.facts, clause)
    return StoryItem(raw, at, at + NEWS_LAG, itype, shingles(raw.headline), entity_ok=entity_ok)


def _link(items: Sequence[StoryItem]) -> list[StoryItem]:
    """Duplicates and corrections inside one story (items in story order).

    Only a substantive correction supersedes: a noise, law-firm or mover wire
    headed "CORRECTION:" is never what the story reads its facts from, so it
    cannot take them away either.
    """
    out: list[StoryItem] = []
    for item in items:
        dup_of: int | None = None
        for prev in out:
            if jaccard(item.shingles, prev.shingles) >= DUP_JACCARD:
                dup_of = prev.dup_of if prev.dup_of is not None else prev.event_id
                break
        supersedes: tuple[int, ...] = ()
        if (
            item.raw.kind == "news"
            and item.itype not in NOISE_TYPES
            and CORRECTION.search(item.raw.headline)
        ):
            keys = fact_keys(item.facts)
            supersedes = tuple(
                prev.event_id
                for prev in out
                if keys & fact_keys(prev.facts)
                or jaccard(item.shingles, prev.shingles) >= CORRECTION_JACCARD
            )
        out.append(replace(item, dup_of=dup_of, supersedes=supersedes))
    return out


def _window_start(session: date) -> date:
    """The first of the :data:`FOLLOW_SESSIONS` sessions before ``session``."""
    day = session
    for _ in range(FOLLOW_SESSIONS):
        day = previous_trading_day(day)
    return day


def _parent(earlier: Sequence[Story], session: date) -> Story | None:
    """The latest story in S-3..S-1 that was no follower at its own close and is
    more than noise (``earlier`` in session order)."""
    lo = _window_start(session)
    for story in reversed(earlier):
        if story.session >= session:
            continue
        if story.session < lo:
            return None
        if not story.noise_only and not story.follower_at(story.close):
            return story
    return None


def _symbol_stories(symbol: str, items: Sequence[StoryItem], earlier: list[Story]) -> list[Story]:
    """One symbol's stories, sessions in order; ``earlier`` (its stories so far) grows."""
    groups: dict[date, list[StoryItem]] = defaultdict(list)
    for item in items:
        groups[reaction_session(item.available_at)].append(item)
    out: list[Story] = []
    for session in sorted(groups):
        linked = _link(sorted(groups[session], key=lambda i: (i.available_at, i.event_id)))
        parent = _parent(earlier, session)
        story = Story(
            story_id=f"{symbol}:{session.isoformat()}",
            symbol=symbol,
            session=session,
            items=linked,
            parent=parent.story_id if parent is not None else None,
            parent_type_close=parent.type_close() if parent is not None else None,
            parent_shingles=tuple(i.shingles for i in parent.items) if parent is not None else (),
        )
        earlier.append(story)
        out.append(story)
    return out


def build(
    items: Iterable[RawItem],
    aliases: Mapping[str, AliasMatcher],
    *,
    history: Mapping[str, Sequence[Story]] | None = None,
    counters: Counter[str] | None = None,
) -> list[Story]:
    """Stories of ``items``: per symbol, sessions in order; pure.

    ``history`` holds each symbol's stories built before these items (in
    session order), where a story's parent may be. ``counters`` collects
    the admission counts (spec §A.2). The result does not depend on the
    order of ``items``.
    """
    c = counters if counters is not None else Counter()
    admitted: dict[str, list[StoryItem]] = defaultdict(list)
    for raw in items:
        item = admit(raw, aliases, c)
        if item is not None:
            admitted[raw.symbol].append(item)
    out: list[Story] = []
    for symbol in sorted(admitted):
        earlier = sorted((history or {}).get(symbol, ()), key=lambda s: s.session)
        out += _symbol_stories(symbol, admitted[symbol], earlier)
    return out


# ── persistence (spec §A.7) ────────────────────────────────────


def _iso(t: datetime | None) -> str | None:
    return t.astimezone(UTC).isoformat() if t is not None else None


def story_row(story: Story) -> dict[str, Any]:
    """The ``news_stories`` row of ``story`` (this builder version)."""
    nsn = story.nsn_at(story.close)
    return {
        "builder_version": BUILDER_VERSION,
        "story_id": story.story_id,
        "symbol": story.symbol,
        "session": story.session,
        "start_case": story.start_case(),
        "detect_at": story.detect_at(),
        "nsn_at": nsn,
        "at_news": story.at_news(),
        "type_detect": story.type_detect(),
        "type_close": story.type_close(),
        "family_ever": FAMILY if nsn is not None else None,
        "follower_close": story.follower_at(story.close),
        "parent": story.parent,
        "n_items": len(story.items),
        "n_distinct": sum(1 for i in story.items if i.dup_of is None),
        "items": [
            {
                "event_id": i.event_id,
                "at": _iso(i.at),
                "available_at": _iso(i.available_at),
                "itype": i.itype,
                "dup_of": i.dup_of,
                "supersedes": list(i.supersedes),
                "entity_ok": i.entity_ok,
            }
            for i in story.items
        ],
        "flags": {
            "nsym_unknown": sum(
                1 for i in story.items if i.raw.kind == "news" and i.raw.n_symbols is None
            ),
            "corrections": sum(1 for i in story.items if i.supersedes),
            "superseded": len({e for i in story.items for e in i.supersedes}),
            "noise_items": sum(1 for i in story.items if i.itype in NOISE_TYPES),
            "parent_type_close": story.parent_type_close,
        },
    }


async def _upsert(conn: AsyncConnection, stories: Sequence[Story]) -> int:
    """Upsert the stories' rows on ``conn`` (its transaction); returns rows written."""
    rows = [story_row(s) for s in stories]
    table = NewsStory.__table__  # type: ignore[attr-defined]
    for i in range(0, len(rows), _INSERT_CHUNK):
        stmt = insert(table).values(rows[i : i + _INSERT_CHUNK])
        stmt = stmt.on_conflict_do_update(
            index_elements=["builder_version", "story_id"],
            set_={
                c.name: stmt.excluded[c.name]
                for c in table.columns
                if c.name not in ("builder_version", "story_id")
            },
        )
        await conn.execute(stmt)
    return len(rows)


async def persist(engine: AsyncEngine, stories: Sequence[Story]) -> int:
    """Upsert the stories' rows (this builder version) in one transaction;
    returns rows written. Marks no range complete (that is :func:`build_range`'s)."""
    async with engine.begin() as conn:
        return await _upsert(conn, stories)


# ── reading the event store ────────────────────────────────────

_ITEMS_SQL: Final = (
    "SELECT e.id, e.source_id, e.kind, e.symbol, e.published_at, e.seen_at, "
    "coalesce(e.payload->>'headline', '') AS headline, e.payload->'symbols' AS tagged, "
    "e.payload->'items' AS items_8k, f.facts "
    "FROM events e LEFT JOIN LATERAL ("
    "  SELECT jsonb_agg(jsonb_build_array(x.kind, x.fields) ORDER BY x.id) AS facts "
    "  FROM event_facts x WHERE e.kind = 'news' AND x.event_id = e.id "
    "  AND x.extractor = :x AND x.kind IN ('result', 'guidance')"
    ") f ON true "
    "WHERE e.symbol = ANY(:s) AND e.kind IN ('news', '8-k', '8-k/a') "
    "AND e.published_at >= :lo AND e.published_at < :hi "
    "ORDER BY e.symbol, e.published_at, e.id"
)


async def load_items(
    engine: AsyncEngine,
    *,
    start: date,
    end: date,
    symbols: Collection[str],
    extractor: str | None = None,
    counters: Counter[str] | None = None,
) -> AsyncIterator[RawItem]:
    """Stream the news, 8-K and 8-K/A rows of ``symbols`` published on New York
    days [start, end], by symbol then time, with their ``extractor`` facts
    (default: ``earnings_parse.EXTRACTOR`` as it is when called).

    A news row that is not its symbol's own (``renames.owner``) is dropped and
    counted (``owner``); filings are kept as they are.
    """
    history = renames.ticker_history()
    c = counters if counters is not None else Counter()
    params = {
        "s": sorted(symbols),
        "x": extractor if extractor is not None else earnings_parse.EXTRACTOR,
        "lo": trading_day_start_utc(start),
        "hi": trading_day_end_utc(end),
    }
    async with engine.connect() as conn:
        rows = await conn.stream(text(_ITEMS_SQL), params)
        async for r in rows:
            news = r.kind == "news"
            tagged = r.tagged if isinstance(r.tagged, list) else None
            if news:
                day = r.published_at.astimezone(MARKET_TZ).date()
                if history.owner(r.symbol, day, tagged) != r.symbol:
                    c["owner"] += 1
                    continue
            yield RawItem(
                event_id=int(r.id),
                source_id=str(r.source_id),
                kind=str(r.kind),
                symbol=str(r.symbol),
                published_at=r.published_at,
                seen_at=r.seen_at,
                headline=str(r.headline) if news else "",
                n_symbols=len(tagged) if tagged is not None else None,
                items_8k=tuple(str(x) for x in (r.items_8k or ())) if not news else (),
                facts=tuple(EarningsFacts(str(k), dict(f)) for k, f in (r.facts or ())),
            )


async def _symbols(engine: AsyncEngine, lo: date, hi: date) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT DISTINCT symbol FROM events WHERE kind IN ('news', '8-k', '8-k/a') "
                "AND published_at >= :lo AND published_at < :hi ORDER BY symbol"
            ),
            {"lo": trading_day_start_utc(lo), "hi": trading_day_end_utc(hi)},
        )
        return [str(r.symbol) for r in rows]


async def _other_kinds(engine: AsyncEngine, symbols: Sequence[str], lo: date, hi: date) -> int:
    """Rows of ``symbols`` in the range that are not story items (admission step 1)."""
    async with engine.connect() as conn:
        n = await conn.scalar(
            text(
                "SELECT count(*) FROM events WHERE symbol = ANY(:s) "
                "AND kind NOT IN ('news', '8-k', '8-k/a') "
                "AND published_at >= :lo AND published_at < :hi"
            ),
            {"s": list(symbols), "lo": trading_day_start_utc(lo), "hi": trading_day_end_utc(hi)},
        )
    return int(n or 0)


async def _has_aliases(engine: AsyncEngine) -> bool:
    async with engine.connect() as conn:
        n = await conn.scalar(
            text("SELECT count(*) FROM story_aliases WHERE builder_version = :v"),
            {"v": BUILDER_VERSION},
        )
    return bool(n)


async def missing_facts(engine: AsyncEngine, *, start: date, end: date) -> tuple[int, int | None]:
    """News events published on New York days [start, end] that the current
    extractor (``earnings_parse.EXTRACTOR``, read now) has not parsed: (how
    many, the lowest event id).

    ``extract_all`` writes a ``none`` row for a headline with no earnings
    statement, so a parsed event always has a row; one without was never
    read, and its facts would be silently missing from the stories.
    """
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT count(*) AS n, min(e.id) AS first FROM events e "
                    "WHERE e.kind = 'news' AND e.published_at >= :lo AND e.published_at < :hi "
                    "AND NOT EXISTS (SELECT 1 FROM event_facts f "
                    "WHERE f.event_id = e.id AND f.extractor = :x)"
                ),
                {
                    "x": earnings_parse.EXTRACTOR,
                    "lo": trading_day_start_utc(start),
                    "hi": trading_day_end_utc(end),
                },
            )
        ).one()
    return int(row.n), (int(row.first) if row.first is not None else None)


async def _per_symbol(raws: AsyncIterator[RawItem]) -> AsyncIterator[list[RawItem]]:
    """Consecutive items of one symbol (the stream is ordered by symbol)."""
    batch: list[RawItem] = []
    async for raw in raws:
        if batch and raw.symbol != batch[0].symbol:
            yield batch
            batch = []
        batch.append(raw)
    if batch:
        yield batch


# ── complete builds (backfill_progress) ────────────────────────

_DELETE_RANGE: Final = (
    "DELETE FROM news_stories WHERE builder_version = :v AND session >= :a AND session <= :b"
)


def build_unit(start: date, end: date, inputs: str) -> str:
    """The ``backfill_progress`` unit (task :data:`TASK`) of a complete build of
    [start, end] from the inputs hashed ``inputs`` (:func:`inputs_sha`)."""
    return f"{BUILDER_VERSION}:{start.isoformat()}:{end.isoformat()}:{inputs}"


@dataclass(frozen=True, slots=True)
class _Mark:
    """A complete build's mark: [start, end] built from the inputs hashed ``inputs``."""

    unit: str
    start: date
    end: date
    inputs: str  # '' for a mark that recorded no inputs: it matches none


def _mark_of(unit: str) -> _Mark | None:
    """This builder version's unit as a mark; None for another version's."""
    prefix = f"{BUILDER_VERSION}:"
    if not unit.startswith(prefix):
        return None
    lo, _, rest = unit.removeprefix(prefix).partition(":")
    hi, _, inputs = rest.partition(":")
    try:
        return _Mark(unit, date.fromisoformat(lo), date.fromisoformat(hi), inputs)
    except ValueError:
        return None


async def _marks(conn: AsyncConnection) -> list[_Mark]:
    rows = await conn.execute(
        text("SELECT unit FROM backfill_progress WHERE task = :t ORDER BY unit"), {"t": TASK}
    )
    return [m for u in rows if (m := _mark_of(str(u.unit))) is not None]


async def built_ranges(engine: AsyncEngine) -> list[tuple[date, date]]:
    """The ranges [start, end] this builder version has completely built from the
    current inputs (:func:`inputs_sha`): the ones :func:`count_stories` accepts."""
    current = await inputs_sha(engine)
    async with engine.connect() as conn:
        return [(m.start, m.end) for m in await _marks(conn) if m.inputs == current]


def uncovered_sessions(
    ranges: Sequence[tuple[date, date]], *, start: date, end: date
) -> list[date]:
    """The sessions in [start, end] no range covers (a story's S is always a session)."""
    out: list[date] = []
    day = start if is_trading_day(start) else next_trading_day(start)
    while day <= end:
        if not any(a <= day <= b for a, b in ranges):
            out.append(day)
        day = next_trading_day(day)
    return out


async def _unmark(engine: AsyncEngine, start: date, end: date) -> None:
    """Withdraw the complete marks over [start, end] before it is rebuilt.

    A mark reaching past the range keeps its parts outside it (with their
    stories counted, and the inputs they were built from), so rebuilding a
    month leaves the rest of a complete build complete -- or, when the inputs
    have changed since, still refused by the counts; a build that stops
    before its own mark leaves [start, end] unmarked.
    """
    async with engine.begin() as conn:
        keep: dict[str, int] = {}
        for m in await _marks(conn):
            if m.end < start or m.start > end:
                continue
            await conn.execute(
                text("DELETE FROM backfill_progress WHERE task = :t AND unit = :u"),
                {"t": TASK, "u": m.unit},
            )
            parts = [(m.start, start - timedelta(days=1))] if m.start < start else []
            parts += [(end + timedelta(days=1), m.end)] if m.end > end else []
            for lo, hi in parts:
                n = await conn.scalar(
                    text(
                        "SELECT count(*) FROM news_stories WHERE builder_version = :v "
                        "AND session >= :a AND session <= :b"
                    ),
                    {"v": BUILDER_VERSION, "a": lo, "b": hi},
                )
                keep[build_unit(lo, hi, m.inputs)] = int(n or 0)
        await mark_units(engine, TASK, keep, conn=conn)


async def build_range(
    engine: AsyncEngine,
    *,
    start: date,
    end: date,
    force: bool = False,
    counters: Counter[str] | None = None,
) -> int:
    """Build every story with S in [start, end] and replace this version's rows there.

    Refuses (:class:`StoriesNotReady`) while a month of the renamed tickers'
    news is missing (``owner`` relies on it), no story alias is stored, or a
    news event the build reads has no row of the current extractor
    (:func:`missing_facts` over every day items are read from), unless
    ``force``. Items are read from :data:`HISTORY_FROM` (or a few days before
    ``start``, if earlier) so each story's parent comes from the same history
    however the build is split.

    Symbols go in batches of :data:`BATCH_SYMBOLS`, one symbol at a time,
    sessions in order; a batch's old rows in the range are deleted and its
    new ones written in one transaction. The range's complete mark
    (:func:`build_unit`) is withdrawn first and written last, in the
    transaction that deletes the rows of symbols with no story there any
    more, so a build that stops part-way leaves its range unmarked and
    :func:`count_stories` refuses it. A forced build is never marked: its
    inputs were not checked. Returns the stories written; ``counters``
    collects the admission counts over every row read.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    lead = start - timedelta(days=_LEAD_DAYS)
    lo = min(HISTORY_FROM, lead)
    if not force:
        missing = await renames.missing_units(engine)
        if missing:
            raise StoriesNotReady(
                f"{len(missing)} month(s) of renamed-ticker news not fetched yet "
                f"(first: {missing[0]}); run `halal-trader events renames backfill` first"
            )
        if not await _has_aliases(engine):
            raise StoriesNotReady(
                f"no story aliases stored for {BUILDER_VERSION}; "
                "run `halal-trader events aliases build` first"
            )
        unparsed, first = await missing_facts(engine, start=lo, end=end)
        if unparsed:
            raise StoriesNotReady(
                f"{unparsed} news event(s) published {lo}..{end} have no "
                f"{earnings_parse.EXTRACTOR} facts row (first: event {first}); "
                "run `halal-trader events extract` first"
            )
    c = counters if counters is not None else Counter()
    # Hashed before the aliases are read: if they change in between, the mark
    # names the older inputs and the counts refuse the range (never the reverse).
    inputs = await inputs_sha(engine)
    aliases = await load_aliases(engine)
    symbols = await _symbols(engine, lead, end)
    await _unmark(engine, start, end)
    span = {"v": BUILDER_VERSION, "a": start, "b": end}
    written = 0
    for i in range(0, len(symbols), BATCH_SYMBOLS):
        batch = symbols[i : i + BATCH_SYMBOLS]
        c["kind"] += await _other_kinds(engine, batch, lo, end)
        async with engine.begin() as conn:
            await conn.execute(text(_DELETE_RANGE + " AND symbol = ANY(:s)"), span | {"s": batch})
            raws = load_items(engine, start=lo, end=end, symbols=batch, counters=c)
            async for rows in _per_symbol(raws):
                stories = build(rows, aliases, counters=c)
                written += await _upsert(conn, [s for s in stories if start <= s.session <= end])
        logger.info("stories: %d of %d symbols, %d stories", i + len(batch), len(symbols), written)
    async with engine.begin() as conn:
        await conn.execute(text(_DELETE_RANGE + " AND symbol <> ALL(:s)"), span | {"s": symbols})
        if not force:
            await mark_units(engine, TASK, {build_unit(start, end, inputs): written}, conn=conn)
    if force:
        logger.warning(
            "stories %s..%s: a forced build, left unmarked (counts refuse it)", start, end
        )
    logger.info(
        "stories %s..%s (%s): %d written; items read from %s: %s",
        start,
        end,
        BUILDER_VERSION,
        written,
        lo,
        dict(sorted(c.items())),
    )
    return written


# ── counts (spec E.0 D9) ───────────────────────────────────────


def window_of(session: date) -> Window:
    if TRAIN[0] <= session <= TRAIN[1]:
        return "train"
    if VALIDATION[0] <= session <= VALIDATION[1]:
        return "validation"
    return "other"


@dataclass(slots=True)
class StoryCounts:
    """Persisted stories counted per universe, year and window (no prices read).

    ``all`` is every story; ``primary`` the ones whose symbol is PRIMARY at S
    (``PitContext.eligibility``); ``tech`` the PRIMARY ones in Technology.
    """

    types: Counter[tuple[Universe, str, int, Window]] = field(default_factory=Counter)
    nsn: Counter[tuple[Universe, int, Window]] = field(default_factory=Counter)
    reasons: Counter[tuple[str, str]] = field(default_factory=Counter)  # ("all"|"nsn", reason)

    def add(
        self, *, session: date, type_close: str, nsn: bool, universes: Sequence[Universe]
    ) -> None:
        year, window = session.year, window_of(session)
        for u in universes:
            self.types[(u, type_close, year, window)] += 1
            if nsn:
                self.nsn[(u, year, window)] += 1


def eligibility_at(
    nsn_at: datetime | None, at_news: datetime | None, detect_at: datetime | None, session: date
) -> datetime:
    """The news time a story's eligibility is judged at: its NSN item's ``at``,
    else its first substantive item's (``detect_at`` less the lag), else S's open."""
    if nsn_at is not None and at_news is not None:
        return at_news
    if detect_at is not None:
        return detect_at - NEWS_LAG
    return session_bounds(session)[0]


async def _context(engine: AsyncEngine, symbols: Collection[str], lo: date, hi: date) -> PitContext:
    from halal_trader.events.context import PitContext

    return await PitContext.load(engine, symbols=symbols, start=lo, end=hi)


async def count_stories(
    engine: AsyncEngine, *, start: date = TRAIN[0], end: date = VALIDATION[1]
) -> StoryCounts:
    """Count the persisted stories with S in [start, end] by close type and by NSN.

    A story counts as NSN when ``nsn_at`` is no later than S's entry cutoff
    (the simulator's ``Session.entry_cutoff``). Eligibility is read one year
    at a time, each year's context loaded for that year's symbols only.

    Refuses (:class:`StoriesNotReady`) when a session in [start, end] lies in
    no range completely built from the current inputs (:func:`built_ranges`):
    a build that stopped part-way, or none at all, would be counted as if it
    held every story, and one built from other aliases, pins or extractor
    (:func:`inputs_sha`) as if it were today's.
    """
    from halabot.playbooks.types import Session

    current = await inputs_sha(engine)
    async with engine.connect() as conn:
        marks = await _marks(conn)
    gaps = uncovered_sessions(
        [(m.start, m.end) for m in marks if m.inputs == current], start=start, end=end
    )
    if gaps:
        stale = [d for d in gaps if any(m.start <= d <= m.end for m in marks)]
        other = (
            f"; {len(stale)} of them built from other inputs (first: {stale[0]}), "
            "since changed: aliases, pins or extractor"
            if stale
            else ""
        )
        raise StoriesNotReady(
            f"{len(gaps)} session(s) in {start}..{end} have no complete {BUILDER_VERSION} "
            f"build from the current inputs {current} (first: {gaps[0]}){other}; "
            "run `halal-trader events stories build` over them"
        )
    counts = StoryCounts()
    for year in range(start.year, end.year + 1):
        lo, hi = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT symbol, session, type_close, nsn_at, at_news, detect_at "
                        "FROM news_stories WHERE builder_version = :v "
                        "AND session >= :a AND session <= :b ORDER BY symbol, session"
                    ),
                    {"v": BUILDER_VERSION, "a": lo, "b": hi},
                )
            ).all()
        if not rows:
            continue
        ctx = await _context(engine, {r.symbol for r in rows}, lo, hi)
        for r in rows:
            nsn = r.nsn_at is not None and r.nsn_at <= Session.of(r.session).entry_cutoff
            at = eligibility_at(r.nsn_at, r.at_news, r.detect_at, r.session)
            elig = ctx.eligibility(r.symbol, r.session, at_news=at)
            counts.reasons[("all", elig.reason)] += 1
            if nsn:
                counts.reasons[("nsn", elig.reason)] += 1
            universes: list[Universe] = ["all"]
            if elig.eligible:
                universes.append("primary")
                if elig.tech:
                    universes.append("tech")
            counts.add(session=r.session, type_close=r.type_close, nsn=nsn, universes=universes)
        del ctx
    return counts


_TITLES: Final[dict[Universe, str]] = {
    "all": "all stories",
    "primary": "PRIMARY (PIT halal, rank < 1000, price >= 5, one share class, sigma)",
    "tech": "Technology (PRIMARY and in Technology)",
}
_WINDOW_HEAD: Final[dict[Window, str]] = {"train": "train", "validation": "valid", "other": "other"}


def counts_table(counts: StoryCounts, *, start: date, end: date) -> list[str]:
    """The counts as text: per universe, close type x year with window totals,
    then the NSN row (``nsn_at`` by the entry cutoff) and every story; then
    the eligibility reasons."""
    years = list(range(start.year, end.year + 1))
    windows: list[Window] = ["train", "validation"]
    if any(k[3] == "other" for k in counts.types):
        windows.append("other")
    head = (
        f"{'type':<28}"
        + "".join(f"{y:>7}" for y in years)
        + "".join(f"{_WINDOW_HEAD[w]:>8}" for w in windows)
        + f"{'total':>8}"
    )

    def line(label: str, cell: Mapping[tuple[int, Window], int]) -> str:
        by_year = [sum(cell.get((y, w), 0) for w in windows) for y in years]
        by_window = [sum(cell.get((y, w), 0) for y in years) for w in windows]
        return (
            f"{label:<28}"
            + "".join(f"{n:>7}" for n in by_year)
            + "".join(f"{n:>8}" for n in by_window)
            + f"{sum(by_window):>8}"
        )

    lines: list[str] = []
    for u in UNIVERSES:
        cells: dict[str, Counter[tuple[int, Window]]] = defaultdict(Counter)
        for (universe, kind, year, window), n in counts.types.items():
            if universe == u:
                cells[kind][(year, window)] += n
        every: Counter[tuple[int, Window]] = Counter()
        for cell in cells.values():
            every.update(cell)
        nsn: Counter[tuple[int, Window]] = Counter(
            {(y, w): n for (universe, y, w), n in counts.nsn.items() if universe == u}
        )
        lines += ["", f"== {_TITLES[u]} ==", head]
        for kind in sorted(cells, key=lambda k: (-sum(cells[k].values()), k)):
            lines.append(line(kind, cells[kind]))
        lines.append(line(f"{FAMILY} (by entry cutoff)", nsn))
        lines.append(line("ALL", every))
    for scope in ("all", "nsn"):
        reasons = sorted(
            ((r, n) for (s, r), n in counts.reasons.items() if s == scope),
            key=lambda x: (-x[1], x[0]),
        )
        label = "every story" if scope == "all" else f"{FAMILY} stories"
        lines.append(f"eligibility ({label}): " + ", ".join(f"{r} {n}" for r, n in reasons))
    return lines


# ── pins (spec §G.14) ──────────────────────────────────────────


def sources() -> dict[str, str]:
    """The builder's own constants, for :data:`STORIES_SHA`."""
    tables: dict[str, object] = {
        "BUILDER_VERSION": BUILDER_VERSION,
        "ROUNDUP_MAX_SYMBOLS": ROUNDUP_MAX_SYMBOLS,
        "NEWS_LAG_S": NEWS_LAG.total_seconds(),
        "REACT_CUTOFF_S": REACT_CUTOFF.total_seconds(),
        "DUP_JACCARD": DUP_JACCARD,
        "CORRECTION_JACCARD": CORRECTION_JACCARD,
        "REPOST_JACCARD": REPOST_JACCARD,
        "FOLLOW_SESSIONS": FOLLOW_SESSIONS,
        "SHINGLE_K": SHINGLE_K,
        "FILING_OPEN": FILING_OPEN.isoformat(),
        "FILING_CUTOFF": FILING_CUTOFF.isoformat(),
        "STORY_KINDS": sorted(STORY_KINDS),
        "FOLLOW_STRUCTURAL": sorted(FOLLOW_STRUCTURAL),
        "EDGAR_CLOSURES": sorted(d.isoformat() for d in EDGAR_CLOSURES),
        "JUNETEENTH_FROM": JUNETEENTH_FROM,
        "STOPWORDS": sorted(STOPWORDS),
        "HISTORY_FROM": HISTORY_FROM.isoformat(),
    }
    return {"_TOKEN": _TOKEN.pattern, "CLAUSE_SPLIT": CLAUSE_SPLIT.pattern} | {
        name: json.dumps(value, sort_keys=True) for name, value in tables.items()
    }


def _sha(blob: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]


# The builder's pin, and the shared headline patterns' (headline_patterns.py).
STORIES_SHA: Final = _sha(sources())
HEADLINE_PATTERNS_SHA: Final = _sha(headline_patterns.sources())


async def pins(engine: AsyncEngine) -> dict[str, str]:
    """Every pin the stories depend on, for the pre-registration's ``metrics``."""
    return {
        "builder_version": BUILDER_VERSION,
        "stories_sha": STORIES_SHA,
        "alias_sha": await alias_sha(engine),
        "renames_sha": renames.renames_sha(),
        "taxonomy_sha": TAXONOMY_SHA,
        "parser_sha": earnings_parse.PARSER_SHA,
        "headline_patterns_sha": HEADLINE_PATTERNS_SHA,
    }


async def inputs_sha(engine: AsyncEngine) -> str:
    """A short hash of what a build's stories depend on: every pin (:func:`pins`:
    the builder's constants, the stored aliases, the renames, the taxonomy, the
    parser, the headline patterns) and the extractor whose facts it reads
    (``earnings_parse.EXTRACTOR``, read now).

    A complete build's mark records it (:func:`build_unit`), and
    :func:`count_stories` accepts only the marks recording today's. A change to
    the builder's rules that no pin sees bumps ``BUILDER_VERSION``.
    """
    return _sha(await pins(engine) | {"extractor": earnings_parse.EXTRACTOR})
