"""FOMC / Fed-speak sentiment streaming.

The Fed's policy stance is one of the highest-impact macro signals for
U.S. equities; the formal FOMC statements are the cleanest read but
also the most-already-priced. **Speeches and interviews from FOMC
members between meetings** are where the marginal positioning happens.

The Federal Reserve publishes every speech transcript at
``federalreserve.gov/newsevents/speech/`` and exposes a public
``rss/speeches.xml`` feed. The transcripts are plain English; a
small lexicon-based hawkish/dovish scorer is enough for a
first-pass signal.

Output is a ``FedSpeakSignal`` per cycle: net hawkish-dovish drift
over the last ``window_hours``, plus the count of speeches and the
single most-recent quote that drove the score most. The cycle's
risk policy can use the drift to widen / tighten size pre-FOMC.

Design choices:
* **No FRED key needed.** Fed speech RSS is open.
* **Lexicon scoring**, not FinBERT. Lightweight, deterministic, no
  install. The lexicon below is curated for the central-banker
  register, not generic news. Swappable with FinBERT later via the
  same ``Scorer`` Protocol.
* **15-min cache.** New speech transcripts land at cadence of hours,
  not minutes — over-polling is pure noise.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


_RSS_URL = "https://www.federalreserve.gov/feeds/speeches.xml"
_CACHE_TTL_S = 15 * 60


# ── Lexicon ──────────────────────────────────────────────────────


# Curated for FOMC / central-bank register. Each token's weight is
# a hawkish (+) / dovish (-) score. The scorer counts whole-word
# occurrences and sums the weights.
HAWKISH_TOKENS: dict[str, float] = {
    "tighten": 1.5,
    "tightening": 1.5,
    "tightened": 1.0,
    "restrictive": 1.5,
    "raise": 1.0,
    "raising": 1.0,
    "raised": 0.8,
    "hike": 1.2,
    "hikes": 1.2,
    "hiked": 1.0,
    "inflation": 0.6,  # mention is mildly hawkish; pair with verbs
    "inflationary": 1.0,
    "overheating": 1.5,
    "vigilant": 0.8,
    "patient": -0.3,  # actually mildly dovish
    "elevated": 0.7,
    "above-target": 1.0,
    "persistent": 0.6,
    "stubborn": 0.8,
    "broad-based": 0.5,
    "premature": -0.5,  # "premature to cut"
    "warranted": 0.4,
}

DOVISH_TOKENS: dict[str, float] = {
    "cut": -1.5,
    "cuts": -1.5,
    "cutting": -1.5,
    "ease": -1.5,
    "easing": -1.5,
    "accommodative": -1.5,
    "weaker": -0.8,
    "softening": -1.0,
    "moderating": -0.6,
    "cooling": -0.7,
    "downside": -0.6,
    "downside-risk": -1.0,
    "patience": -0.5,
    "gradual": -0.4,
    "supportive": -0.7,
    "stimulus": -1.2,
    "below-target": -1.0,
    "subdued": -0.6,
    "fragile": -0.8,
    "weakness": -0.6,
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


# ── Speech + signal ──────────────────────────────────────────────


@dataclass(frozen=True)
class FedSpeech:
    """One speech / interview transcript snippet."""

    title: str
    timestamp: datetime
    speaker: str = ""
    url: str = ""
    summary: str = ""  # the body text (or extract) the scorer reads


@dataclass(frozen=True)
class FedSpeakSignal:
    """Aggregated hawkish-dovish drift over the recent window."""

    n_speeches: int
    hawkish_score: float
    dovish_score: float
    net_drift: float  # hawkish - dovish; positive = hawkish lean
    label: str
    most_hawkish_quote: str = ""
    most_dovish_quote: str = ""

    @property
    def stance(self) -> str:
        """Human-readable stance label."""
        return self.label


# ── Scorer ───────────────────────────────────────────────────────


def score_text(text: str) -> tuple[float, float]:
    """Return (hawkish_score, dovish_score) for one block of text.

    Pure stdlib: tokenise on word boundaries, lowercase, look up in
    the two lexicons, sum the weights. Counts each occurrence (not
    just first appearance) so longer speeches with repeated themes
    score proportionately.
    """
    if not text:
        return 0.0, 0.0
    hawkish = 0.0
    dovish = 0.0
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in HAWKISH_TOKENS:
            w = HAWKISH_TOKENS[tok]
            if w > 0:
                hawkish += w
            else:
                dovish += -w
        elif tok in DOVISH_TOKENS:
            w = DOVISH_TOKENS[tok]
            if w < 0:
                dovish += -w
            else:
                hawkish += w
    return hawkish, dovish


def aggregate_signal(speeches: Iterable[FedSpeech]) -> FedSpeakSignal:
    """Roll a list of speeches into a single :class:`FedSpeakSignal`."""
    speeches = list(speeches)
    if not speeches:
        return FedSpeakSignal(
            n_speeches=0,
            hawkish_score=0.0,
            dovish_score=0.0,
            net_drift=0.0,
            label="no_data",
        )
    h_total = 0.0
    d_total = 0.0
    most_hawkish = ("", -1.0)
    most_dovish = ("", -1.0)
    for s in speeches:
        h, d = score_text(s.summary)
        h_total += h
        d_total += d
        if h - d > most_hawkish[1]:
            most_hawkish = (s.title, h - d)
        if d - h > most_dovish[1]:
            most_dovish = (s.title, d - h)
    net = h_total - d_total
    label = _classify(net, n=len(speeches))
    return FedSpeakSignal(
        n_speeches=len(speeches),
        hawkish_score=h_total,
        dovish_score=d_total,
        net_drift=net,
        label=label,
        most_hawkish_quote=most_hawkish[0],
        most_dovish_quote=most_dovish[0],
    )


def _classify(net_drift: float, *, n: int) -> str:
    if n == 0:
        return "no_data"
    # Normalize per-speech so a quiet week with one strong speech
    # isn't drowned by a noisy week with many lukewarm ones.
    per_speech = net_drift / n
    if per_speech >= 2.0:
        return "hawkish_drift"
    if per_speech >= 0.7:
        return "mildly_hawkish"
    if per_speech <= -2.0:
        return "dovish_drift"
    if per_speech <= -0.7:
        return "mildly_dovish"
    return "balanced"


# ── Fetcher ──────────────────────────────────────────────────────


@dataclass
class _CacheEntry:
    fetched_at: float
    speeches: list[FedSpeech]


@dataclass
class FedSpeakFetcher:
    """Pulls recent FOMC speeches from the Fed's public RSS feed.

    No key, no auth. Polite User-Agent. The RSS contains title,
    timestamp, link, and a short summary; for our scorer the summary
    is enough — fetching the full transcript per speech is overkill
    for a directional signal.
    """

    user_agent: str = "halal-trader/0.1 (fed-speak)"
    window_hours: int = 168  # 1 week
    _client: Any | None = None
    _cache: _CacheEntry | None = None

    async def fetch(self) -> FedSpeakSignal:
        if self._cache and (time.monotonic() - self._cache.fetched_at) < _CACHE_TTL_S:
            return aggregate_signal(self._cache.speeches)
        try:
            speeches = await self._fetch_rss()
        except Exception as exc:  # noqa: BLE001
            logger.debug("fed-speak RSS fetch failed: %s", exc)
            return aggregate_signal([])
        self._cache = _CacheEntry(fetched_at=time.monotonic(), speeches=speeches)
        return aggregate_signal(speeches)

    async def _fetch_rss(self) -> list[FedSpeech]:
        client = await self._get_client()
        resp = await client.get(_RSS_URL)
        if resp.status_code != 200:
            return []
        return parse_rss(resp.text, window_hours=self.window_hours)

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=10.0,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/rss+xml, application/xml, text/xml",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


# ── RSS parser ───────────────────────────────────────────────────


_ITEM_RE = re.compile(r"<item>(.*?)</item>", re.DOTALL | re.IGNORECASE)
_TAG_RE = {
    "title": re.compile(r"<title>(.*?)</title>", re.DOTALL | re.IGNORECASE),
    "link": re.compile(r"<link>(.*?)</link>", re.DOTALL | re.IGNORECASE),
    "pubDate": re.compile(r"<pubDate>(.*?)</pubDate>", re.DOTALL | re.IGNORECASE),
    "description": re.compile(r"<description>(.*?)</description>", re.DOTALL | re.IGNORECASE),
}


def parse_rss(xml_text: str, *, window_hours: int = 168) -> list[FedSpeech]:
    """Extract speeches from the Fed's RSS within the last ``window_hours``.

    Stdlib regex parser — defparation against the Fed feed which is
    consistently shaped. Robust to minor XML format drift; bad rows
    are skipped silently.
    """
    out: list[FedSpeech] = []
    cutoff_ts = datetime.now(UTC).timestamp() - window_hours * 3600
    for item in _ITEM_RE.findall(xml_text):
        title = _extract_tag(item, "title")
        link = _extract_tag(item, "link")
        pub = _extract_tag(item, "pubDate")
        desc = _extract_tag(item, "description")
        ts = _parse_pubdate(pub)
        if ts is None or ts.timestamp() < cutoff_ts:
            continue
        out.append(
            FedSpeech(
                title=title,
                timestamp=ts,
                speaker=_extract_speaker(title),
                url=link,
                summary=_clean_html(desc) or _clean_html(title),
            )
        )
    return out


def _extract_tag(blob: str, tag: str) -> str:
    m = _TAG_RE[tag].search(blob)
    if not m:
        return ""
    return _strip_cdata(m.group(1)).strip()


def _strip_cdata(s: str) -> str:
    s = s.strip()
    if s.startswith("<![CDATA[") and s.endswith("]]>"):
        return s[len("<![CDATA[") : -len("]]>")]
    return s


def _clean_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _parse_pubdate(s: str) -> datetime | None:
    if not s:
        return None
    # RFC 822 format: "Tue, 23 Apr 2026 14:30:00 GMT"
    try:
        from email.utils import parsedate_to_datetime

        ts = parsedate_to_datetime(s)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts
    except Exception:  # noqa: BLE001
        return None


def _extract_speaker(title: str) -> str:
    """Best-effort: titles are typically 'Last name, Title — Subject'."""
    if not title:
        return ""
    if "—" in title:
        return title.split("—")[0].strip()
    if "-" in title:
        return title.split("-")[0].strip()
    return title.strip()


# ── Prompt formatting ────────────────────────────────────────────


def format_fed_speak_for_prompt(signal: FedSpeakSignal) -> str:
    """One-block summary for the macro-catalysts section of the prompt.

    Empty result returns "" so the prompt builder can elide it.
    """
    if signal.n_speeches == 0:
        return ""
    sign = "+" if signal.net_drift >= 0 else ""
    lines = [
        "Fed-speak drift (last week):",
        f"  stance={signal.label} | net {sign}{signal.net_drift:.1f} "
        f"(hawkish {signal.hawkish_score:.1f}, dovish {signal.dovish_score:.1f}) "
        f"across {signal.n_speeches} speech(es)",
    ]
    if signal.most_hawkish_quote:
        lines.append(f"  most hawkish: {signal.most_hawkish_quote[:100]}")
    if signal.most_dovish_quote:
        lines.append(f"  most dovish: {signal.most_dovish_quote[:100]}")
    return "\n".join(lines)


# ── Sequence helper for the catalyst feed ────────────────────────


def fed_speak_to_catalysts(signal: FedSpeakSignal, symbols: Sequence[str]) -> list:
    """Convert a ``FedSpeakSignal`` into Catalyst rows the stock feed
    can consume.

    Macro signals aren't tied to a single ticker; we emit one Catalyst
    per requested symbol with kind ``"fed_speak"``. The
    ``CatalystRiskPolicy`` can opt into shrinking sizing on hawkish or
    dovish drifts — for now we set kind to a neutral macro tag and
    leave the policy choice to the operator.
    """
    from halal_trader.trading.catalysts import Catalyst

    if signal.n_speeches == 0:
        return []
    out: list[Catalyst] = []
    now = datetime.now(UTC)
    for sym in symbols:
        out.append(
            Catalyst(
                symbol=sym.upper(),
                kind="fed_speak",
                title=f"Fed-speak {signal.label} (net {signal.net_drift:+.1f})",
                timestamp=now,
                source="fed-rss",
                extra={
                    "n_speeches": signal.n_speeches,
                    "hawkish_score": signal.hawkish_score,
                    "dovish_score": signal.dovish_score,
                },
            )
        )
    return out
