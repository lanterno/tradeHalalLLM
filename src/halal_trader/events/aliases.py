"""The names a headline may call each company by: the story builder's entity check.

Benzinga tags an article with every symbol it touches, so an article about
one company often carries its peers, customers and index-mates. The story
builder keeps an (article, symbol) row only when the headline names the
company (spec §A.2 step 3). The names come from four sources, persisted per
builder version in ``story_aliases`` so the set the pre-registration pins
(``alias_sha``) can be read back unchanged:

* **name** -- ``market_assets.name`` and Alpaca's inactive-asset names, with
  legal suffixes stripped (``SUFFIX``): the cleaned name, its first word when
  it is distinctive (four letters, or three in capitals, and not ``GENERIC``)
  and its first two words when they make six characters;
* **learned** -- Benzinga's own name for the company: the company slot of its
  analyst, earnings and guidance templates in articles tagging only that
  symbol, kept when it is at least ``LEARN_MIN_COUNT`` of them and
  ``LEARN_MIN_SHARE`` of the symbol's slots (with its first word, under the
  same rule). This recovers renamed and abbreviated names ("Priceline",
  "Square", "JB Hunt Transport Servs");
* **override** -- ``ALIAS_OVERRIDES``: the brands a headline uses for a few
  large companies (Google, YouTube, AWS, Instagram, ...);
* **ticker** -- the symbol and its old tickers (``renames.TICKER_RENAMES``),
  matched case-sensitively and optionally after a ``$``. Only tickers of two
  characters or more: "F" or "T" would match words.

Names come from today's asset list, not from the date of each article: a
renamed company's old name survives only where Benzinga's slot learned it.
Slots are learned only from rows that are their symbol's own
(``renames.owner``): Pandora's articles under P teach P nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final, Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.headline_patterns import ANALYST_SLOT, EARN_CO, GUIDE_CO
from halal_trader.events.renames import TICKER_RENAMES, old_tickers, ticker_history
from halal_trader.market_hours import MARKET_TZ, trading_day_end_utc, trading_day_start_utc

logger = logging.getLogger(__name__)

# The story builder's version (events/stories.py): every alias row carries it.
BUILDER_VERSION: Final = "stories-v1"

# Learned slots: single-symbol Benzinga articles published in [LEARN_FROM, LEARN_TO].
LEARN_FROM: Final = date(2016, 1, 1)
LEARN_TO: Final = date(2026, 10, 9)
LEARN_MIN_COUNT: Final = 3
LEARN_MIN_SHARE: Final = 0.10
_SLOT_LEN = (2, 40)  # a slot outside this length is not a company name
_INSERT_CHUNK = 5_000

Source = Literal["name", "learned", "override", "ticker"]

# Legal and generic suffixes stripped from an asset name (the scratchpad
# prototype's table, verbatim; only the line breaks differ).
SUFFIX: Final = re.compile(
    r"\b(?:Common Stock|Class [A-C]|Ordinary Shares?|American Depositary Shares?|"
    r"Depositary Shares?|ADS|ADR|Units?|"
    r"Incorporated|Inc|Corporation|Corp|Company|Co|Holdings?|Group|Ltd|Limited|plc|"
    r"N\.?V|S\.?A|L\.?P|LLC|AG|SE|"
    r"Technologies|Technology|Systems|International|Enterprises|Industries|Worldwide|"
    r"Platforms|Brands|Solutions|"
    r"Communications|Laboratories|Labs|Pharmaceuticals|Therapeutics|Entertainment|"
    r"Interactive|Software|Networks|"
    r"New|The|of|and|&)\b\.?",
    re.I,
)
# First words too common to stand for one company on their own.
GENERIC: Final = frozenset(
    "american first general united national digital advanced applied global international "
    "royal western southern northern eastern texas universal union pacific atlantic bank "
    "energy health medical capital data american big new old alpha beta delta prime core "
    "smart open live best real total".split()
)
ALIAS_OVERRIDES: Final[dict[str, tuple[str, ...]]] = {
    "GOOGL": ("Google", "Alphabet", "YouTube", "Waymo"),
    "GOOG": ("Google", "Alphabet", "YouTube", "Waymo"),
    "META": ("Facebook", "Meta", "Instagram", "WhatsApp", "FB"),
    "AMZN": ("Amazon", "AWS", "Whole Foods"),
    "MSFT": ("Microsoft", "LinkedIn", "Xbox", "Azure"),
    "AAPL": ("Apple", "iPhone"),
    "TSLA": ("Tesla",),
    "NVDA": ("Nvidia",),
    "AMD": ("AMD", "Advanced Micro"),
}


@dataclass(frozen=True, slots=True, order=True)
class AliasRow:
    symbol: str
    alias: str
    source: Source


# ── the matcher ───────────────────────────────────────────────


def _undotted(values: Iterable[str]) -> tuple[str, ...]:
    """Dots removed, blanks dropped, duplicates merged; longest first."""
    out = {v.replace(".", "").strip() for v in values}
    out.discard("")
    return tuple(sorted(out, key=lambda v: (-len(v), v)))


@dataclass(frozen=True, slots=True)
class AliasMatcher:
    """Does a headline name ``symbol``'s company?

    ``aliases`` match case-insensitively, ``tickers`` case-sensitively with
    an optional ``$``; neither may touch a letter before it or a lowercase
    letter after it (the flag makes that any letter after an alias).
    Dots are removed from both sides, so "J.B. Hunt" and "BRK.B" match.
    A matcher with nothing to match (a one-letter ticker and no name) passes
    every headline: there is nothing to check it against.
    """

    symbol: str
    aliases: tuple[str, ...]
    tickers: tuple[str, ...] = ()
    _rx: re.Pattern[str] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        aliases = _undotted(self.aliases)
        tickers = tuple(t for t in _undotted(self.tickers) if len(t) >= 2)
        parts = [re.escape(a) for a in aliases]
        parts += [r"(?-i:\$?" + re.escape(t) + r")" for t in tickers]
        rx = (
            re.compile(r"(?<![A-Za-z])(?:" + "|".join(parts) + r")(?![a-z])", re.I)
            if parts
            else None
        )
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "tickers", tickers)
        object.__setattr__(self, "_rx", rx)

    @property
    def empty(self) -> bool:
        """Nothing to match: ``matches`` passes every headline."""
        return self._rx is None

    def matches(self, text: str) -> bool:
        if self._rx is None:
            return True
        return self._rx.search(text.replace(".", "")) is not None


def tickers_of(symbol: str) -> tuple[str, ...]:
    """The symbol and its old tickers (source d)."""
    return (symbol, *old_tickers(symbol))


def matcher_for(symbol: str, aliases: Mapping[str, AliasMatcher]) -> AliasMatcher:
    """``symbol``'s matcher, or one that knows only its tickers."""
    known = aliases.get(symbol)
    return known if known is not None else AliasMatcher(symbol, (), tickers_of(symbol))


# ── the sources ───────────────────────────────────────────────


def _distinctive_first_word(word: str) -> bool:
    return (len(word) >= 4 or (word.isupper() and len(word) >= 3)) and word.lower() not in GENERIC


def name_aliases(name: str) -> list[str]:
    """Aliases of one asset name (source a), longest first."""
    clean = re.sub(r"[,.()]", " ", name)
    clean = re.sub(r"\s+", " ", SUFFIX.sub(" ", clean)).strip()
    words = clean.split()
    out: set[str] = set()
    if words:
        out.add(clean)
        if _distinctive_first_word(words[0]):
            out.add(words[0])
        if len(words) >= 2 and len(" ".join(words[:2])) >= 6 and all(len(w) > 1 for w in words[:2]):
            out.add(" ".join(words[:2]))
    return sorted(out, key=lambda a: (-len(a), a))


def slot_of(headline: str) -> str | None:
    """Benzinga's company slot in an analyst, guidance or earnings headline (source b).

    The first template that matches decides; a slot that is too short or
    too long to be a name is no slot. Guidance is tried before earnings:
    the earnings template also matches "Nvidia Sees Q3 Sales ...", with
    "Nvidia Sees" for the company.
    """
    for rx in (ANALYST_SLOT, GUIDE_CO, EARN_CO):
        m = rx.search(headline)
        if m:
            co = next((g for g in m.groups() if g), None)
            if co is None:
                return None
            co = co.strip(" '\"").removesuffix("'s")
            return co if _SLOT_LEN[0] <= len(co) <= _SLOT_LEN[1] else None
    return None


def learned_aliases(counts: Mapping[str, Counter[str]]) -> dict[str, set[str]]:
    """Each symbol's slots seen ``LEARN_MIN_COUNT`` times and ``LEARN_MIN_SHARE``
    of its slots, with their distinctive first words."""
    out: dict[str, set[str]] = {}
    for symbol, seen in counts.items():
        total = sum(seen.values())
        keep = {
            co for co, n in seen.items() if n >= LEARN_MIN_COUNT and n >= LEARN_MIN_SHARE * total
        }
        firsts = {
            words[0] for co in keep if (words := co.split()) and _distinctive_first_word(words[0])
        }
        if keep:
            out[symbol] = keep | firsts
    return out


def alias_rows(
    symbols: Iterable[str],
    names: Mapping[str, Collection[str]],
    learned: Mapping[str, Collection[str]],
) -> list[AliasRow]:
    """Every (symbol, alias, source) row of the four sources, sorted."""
    rows: set[AliasRow] = set()
    for symbol in set(symbols):
        for name in names.get(symbol, ()):
            rows.update(AliasRow(symbol, a, "name") for a in name_aliases(name))
        rows.update(AliasRow(symbol, a, "learned") for a in learned.get(symbol, ()))
        rows.update(AliasRow(symbol, a, "override") for a in ALIAS_OVERRIDES.get(symbol, ()))
        rows.update(AliasRow(symbol, t, "ticker") for t in tickers_of(symbol) if len(t) >= 2)
    return sorted(rows)


async def learn_slots(
    engine: AsyncEngine, *, start: date = LEARN_FROM, end: date = LEARN_TO
) -> dict[str, Counter[str]]:
    """Slot counts per symbol from single-symbol articles published in [start, end] (New York).

    Only rows that are their symbol's own count (``renames.owner``): not the
    rows of a ticker's earlier holder, and not an old ticker's rows, whose
    copies under the current symbol count instead. Streams the rows: only the
    counts are held, never the articles.
    """
    history = ticker_history()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    async with engine.connect() as conn:
        rows = await conn.stream(
            text(
                "SELECT symbol, published_at, payload->'symbols' AS tagged, "
                "payload->>'headline' AS headline FROM events "
                "WHERE kind = 'news' AND published_at >= :lo AND published_at < :hi "
                "AND jsonb_array_length(coalesce(payload->'symbols', '[]'::jsonb)) = 1"
            ),
            {"lo": trading_day_start_utc(start), "hi": trading_day_end_utc(end)},
        )
        async for r in rows:
            day = r.published_at.astimezone(MARKET_TZ).date()
            if history.owner(r.symbol, day, r.tagged) != r.symbol:
                continue
            slot = slot_of(r.headline or "")
            if slot is not None:
                counts[r.symbol][slot] += 1
    return dict(counts)


async def _asset_names(engine: AsyncEngine, market: Any) -> dict[str, set[str]]:
    """Every name each symbol has had in ``market_assets`` or Alpaca's inactive list."""
    names: dict[str, set[str]] = defaultdict(set)
    async with engine.connect() as conn:
        for r in await conn.execute(text("SELECT symbol, name FROM market_assets")):
            if r.name:
                names[r.symbol].add(r.name)
    for asset in await market.inactive_assets():
        if asset.name:
            names[asset.symbol].add(asset.name)
    return names


async def _news_symbols(engine: AsyncEngine) -> set[str]:
    """Every symbol a story can be about: the screen's and the event store's news."""
    from halal_trader.events.history import covered_symbols

    symbols = await covered_symbols(engine)
    async with engine.connect() as conn:
        rows = await conn.execute(text("SELECT DISTINCT symbol FROM events WHERE kind = 'news'"))
        symbols |= {r.symbol for r in rows}
    return symbols | {current for current, _ in TICKER_RENAMES.values()}


# ── persistence ───────────────────────────────────────────────


async def _persist(engine: AsyncEngine, rows: list[AliasRow]) -> None:
    """Replace this builder version's rows, in one transaction."""
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM story_aliases WHERE builder_version = :v"), {"v": BUILDER_VERSION}
        )
        for i in range(0, len(rows), _INSERT_CHUNK):
            await conn.execute(
                text(
                    "INSERT INTO story_aliases (builder_version, symbol, alias, source) "
                    "VALUES (:v, :s, :a, :src)"
                ),
                [
                    {"v": BUILDER_VERSION, "s": r.symbol, "a": r.alias, "src": r.source}
                    for r in rows[i : i + _INSERT_CHUNK]
                ],
            )


async def build_aliases(engine: AsyncEngine, market: Any) -> int:
    """Learn every symbol's aliases and store them (replacing this version's); returns rows."""
    learned = learned_aliases(await learn_slots(engine))
    names = await _asset_names(engine, market)
    symbols = await _news_symbols(engine) | set(learned)
    rows = alias_rows(symbols, names, learned)
    await _persist(engine, rows)
    logger.info(
        "story aliases: %d rows for %d symbols (%d with learned slots)",
        len(rows),
        len(symbols),
        len(learned),
    )
    return len(rows)


async def _rows(engine: AsyncEngine) -> list[AliasRow]:
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT symbol, alias, source FROM story_aliases WHERE builder_version = :v"),
            {"v": BUILDER_VERSION},
        )
        return sorted(AliasRow(r.symbol, r.alias, r.source) for r in result)


async def load_aliases(engine: AsyncEngine) -> dict[str, AliasMatcher]:
    """A matcher for every symbol with a stored alias; its tickers always included."""
    aliases: dict[str, list[str]] = defaultdict(list)
    tickers: dict[str, list[str]] = defaultdict(list)
    for r in await _rows(engine):
        (tickers if r.source == "ticker" else aliases)[r.symbol].append(r.alias)
    return {
        symbol: AliasMatcher(
            symbol, tuple(aliases.get(symbol, ())), (*tickers.get(symbol, ()), *tickers_of(symbol))
        )
        for symbol in sorted(aliases.keys() | tickers.keys())
    }


async def alias_sha(engine: AsyncEngine) -> str:
    """A short hash of this builder version's alias rows, for the pre-registration.

    Rows are sorted here, not by the database, so the hash does not depend on
    the server's collation.
    """
    rows = [[r.symbol, r.alias, r.source] for r in await _rows(engine)]
    blob = json.dumps({"builder_version": BUILDER_VERSION, "rows": rows}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
