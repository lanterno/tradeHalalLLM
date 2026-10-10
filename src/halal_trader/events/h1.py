"""H1, the overreaction bounce: its pre-registration and the runner that tests it (spec §G).

Pre-registered (written before any result was seen).

**The hypothesis** (spec §G.1). After a large drop on non-structural negative
news (an analyst downgrade or an earnings miss: ``NSN_CORE``), a long
position bought once the selling is exhausted earns a positive mean return
net of costs, abnormal to SPY at the same bars. The rule is
``halabot.playbooks.bounce`` (constants frozen); the cells are intraday
(``ID``, one session) and multi-day (``MD3``, three). One-sided: H0 is
E[r] <= 0.

**Order of work** (spec §G.15), each step refusing until the one before it
is on the ledger (``quant_trials``):

1. :func:`register` writes the pre-registration (``kind="preregistration"``,
   ``config=PREREG``) once the Phase 0 gate rows all pass and the data
   preconditions D1-D6 hold (:func:`preconditions`), with D9's counts, D2/D3's
   residuals, D7/D8's measurements, the code pins and the git state in its
   metrics. It refuses a second registration of the same configuration.
2. :func:`stage_a` counts entries without computing an exit or a return
   (``stop_at="entry"``, ``assume_full_hold=True``) in both windows and
   decides which cells are eligible; data skips above 2% of the eligible NSN
   stories block everything after it until the data is fixed. It freezes
   each window's data (:func:`data_digest`).
3. :func:`run_window` ``"train"`` runs the eligible cells, ``"validation"``
   the train passers only; each cell is one trial
   (``research.record_backtest``, one config hash across both windows).
4. :func:`verdict` records pass, fail or inconclusive (``kind="verdict"``).
5. :func:`implementability` (the delayed feed, a counted trial) and
   :func:`sensitivities` (never trials) come after the verdict, on the
   window rows it cites.

**Code and data are pinned** (spec §G.14: a simulator or loader fix is a
logged amendment, with old and new results both reported). Every step after
the registration compares the checkout (HEAD, a modified tree, and the
sha of each of :data:`PINNED_MODULES`) with the registration's, and every
step after Stage A compares the window's data with Stage A's digest: the
candidates with their ``nsn_at``, ``at_news`` and eligibility reason, the
minute units their paths request, and which of those are done. A
difference, a window that has run, or trial rows an unfinished run left
behind, lets the step run only with an amendment (``--amend <reason>``): a
``kind="amendment"`` row, written before the step runs, with what it found;
recorded code or data is what later steps compare with. Each run row
(``hb_playbook_run.code_sha``) names the HEAD actually running, ``-dirty``
when the tree is modified.

**PREREG** is built at run time (:func:`prereg`): the literal of
:func:`build_prereg`, with every pin ``stories.pins`` reports (the
builder's, the aliases', the renames', the taxonomy's, the parser's, the
shared headline patterns', and any it adds later), the simulator's
constants and the bounce's. A pin that changes after registration changes
the hash, so the runner no longer finds its registration and refuses: a
new registration is a new trial.

**The runner** rebuilds the stories of every symbol with an NSN story in
the window (``stories.load_items`` and ``stories.build``, from the history
start, so parents are the persisted ones) and refuses when they disagree
with ``news_stories`` (the units plan selected paths from those rows). A
story starts its playbook only when it is NSN by S's entry cutoff, inside
the cell's sessions, and eligible at its news time
(``PitContext.eligibility``); every other story of the symbol is passed as a
:class:`Carrier`, so its items still reach a live playbook (they can only
abort it) but it never starts one, never blocks one and never asks for a
path. Paths load behind the loader's window guard with the registration's
id.

**Rows written** (``quant_trials``): ``research.news.h1`` with kind
``preregistration``, ``stage-a``, ``window`` (one per window run: every
cell's statistics, its trial id and DSR at ``n_trials + 6``), ``verdict``,
``amendment`` and ``implementability``; ``research.news.h1.nsn_core.<id|md3>``
backtests (the trials, benchmark "SPY (exposure-matched)"); and
``research.news.h1.sens.<name>`` sensitivities (no Sharpe, so never
counted). None of them but the backtests carries ``active_sr_period``.

**Where this departs from spec §G** (each stated for the record):

* ``PREREG`` is a function of the database's aliases (``alias_sha``), so it
  is :func:`prereg`, not a module constant, and :func:`trial_config` takes
  the registration's hash. It also lists the sensitivities, the pins under
  one ``pins`` key, the context's stated deviations from §C, and the
  simulator's constants (``SimConfig().as_config()``) under ``fills``.
* Gate rows are matched by their exact id (:data:`REQUIRED_GATES`: each of
  G1-G3's sub-gates); every required id must have a row, and every gate
  id's latest row must pass.
* D2's unmapped and pending counts cover the screens of D2's span only, and
  D6 needs the facts of the news published 2016-01-01..2024-12-31 only:
  nothing H1 never reads can block the registration.
* D5's full scan reads every done unit of plan H through
  ``minutes.read_windows``; on a seeded sample of 200 units, ``minutes.read``
  reads the unit and ``minutes.read_sessions`` the unit with its symbol's
  done sessions within a week (a multi-session range scan).
* D7 and D8 are measured at registration through Alpaca (the CLI passes a
  client when its keys are set) and recorded; without a client, or when the
  probe fails, they are recorded as not measured. Neither gates.
* D3 judges a quarter on the names PIT-halal and rank < 1000 at its first
  session; "admitted news" is a news event among a story's items.
* A ``window`` summary row is added (the verdict reads it), and an
  ``amendment`` row precedes any re-run of a window, any step on changed
  code or data, and a window or implementability run after an unfinished
  one.
* With no train passer the verdict is decided on train alone (validation is
  not owed, and an older validation row, from before train was amended, is
  reported as ignored), so an amended train can always be decided. A verdict
  is recorded once per state of the ledger: again only after a newer Stage
  A, window or amendment row. It reports the window rows it does not cite
  (the results amendments replaced), every amendment and every earlier
  verdict.
* News-lag sensitivities keep the 600 s grouping (reaction sessions,
  parents) and move only when each item becomes usable.
* "Unresolved trades at -100%" sets those trades' net and beta-adjusted
  returns to -1.
* A cell whose book is empty or degenerate is recorded with
  ``QuantTrialRepoImpl.record_trial`` (no Sharpe, ``degenerate: true``).
* Eligibility is judged before blocking: an ineligible story is a carrier, so
  it is counted by its eligibility reason, never ``blocked_open`` (it could
  not have blocked anything either way). Stage A's data budget is checked
  per window and cell (the MD3 paths are longer), and its ``stories`` count
  is the persisted stories, noise only left out, with S in the cell's
  sessions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import random
import subprocess
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks import bounce as bounce_rule
from halabot.playbooks.bounce import BounceFactory, BounceParams
from halabot.playbooks.clock import SIP_DELAYED
from halabot.playbooks.interfaces import CardView, StoryView
from halabot.playbooks.loader import (
    SPY,
    WINDOW_LAST,
    WINDOW_START,
    CalendarMismatch,
    Window,
    WindowUnlock,
    check_calendar,
)
from halabot.playbooks.records import (
    DailyBook,
    Leg,
    OutcomeSink,
    PgOutcomeSink,
    RunInfo,
    StoryOutcome,
    TradeRecord,
    daily_book,
)
from halabot.playbooks.sim import RunSummary, start_time
from halabot.playbooks.sim import run as simulate
from halabot.playbooks.types import DATA_SKIPS, Session, SimConfig, path_days
from halal_trader.core.sharpe_stats import deflated_sharpe_ratio
from halal_trader.data import minutes
from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl, config_hash
from halal_trader.events import context as pit
from halal_trader.events import stats
from halal_trader.events import stories as st
from halal_trader.events.aliases import AliasMatcher, load_aliases
from halal_trader.events.context import Eligibility, PitContext, PreEvent, Universe
from halal_trader.events.earnings_parse import EXTRACTOR
from halal_trader.events.study import COST_BPS
from halal_trader.events.taxonomy import DEAD_BAND, FAMILY, FAMILY_TYPES, GUIDE_BAND, RULES_VERSION
from halal_trader.market_hours import (
    EARLY_CLOSE_DATES,
    MARKET_TZ,
    is_trading_day,
    next_trading_day,
)
from halal_trader.research.ledger import PREFIX, criterion_for, record_backtest

logger = logging.getLogger(__name__)

# ── the trial's constants ──────────────────────────────────────

NAME: Final = "research.news.h1"
SENS_PREFIX: Final = NAME + ".sens."
GATE_NAME: Final = "research.news.sim-gate"
TAG: Final = "news-h1-v1"
HYPOTHESIS: Final = "news.h1.overreaction_bounce"
BENCHMARK_LABEL: Final = "SPY (exposure-matched)"

T_MIN: Final = 2.0
ALPHA: Final = 0.05
COST_X: Final = 1.5  # T3: the mean must stay positive at 1.5x costs (one more side)
SKIP_MAX: Final = 0.02  # Stage A: data skips per eligible NSN story, each window
UNRESOLVED_MAX: Final = 0.005  # T6
BOOK_SLOTS: Final = 8
BOOK_LOSS_LIMIT: Final = 0.02
COVID: Final = (date(2020, 2, 20), date(2020, 6, 30))  # T5 leaves these entries out (train)
SCREEN_BREAK: Final = date(2020, 10, 1)  # reported: before and after
TEMPLATE_BREAK: Final = date(2018, 1, 1)  # reported: before and after

WindowName = Literal["train", "validation"]
Status = Literal["pass", "fail", "inconclusive"]
WINDOWS: Final[tuple[WindowName, ...]] = ("train", "validation")
_SIM_WINDOW: Final[dict[WindowName, Window]] = {
    "train": Window.TRAIN,
    "validation": Window.VALIDATION,
}
YEARS: Final[dict[WindowName, float]] = {"train": 5.25, "validation": 3.0}


@dataclass(frozen=True, slots=True)
class Cell:
    """One tested cell: the family, the variant and the sessions it holds."""

    family: str
    variant: str
    hold: int

    @property
    def key(self) -> str:
        """``"NSN_CORE/ID"``: the simulator's ``TradeFacts.cell`` of the cell's trades."""
        return f"{self.family}/{self.variant}"

    @property
    def strategy(self) -> str:
        """The ledger strategy: ``news.h1.nsn_core.id`` (``research.`` is the ledger's)."""
        return f"news.h1.{self.family.lower()}.{self.variant.lower()}"

    def params(self, *, require_family: bool = True) -> BounceParams:
        return BounceParams(hold_sessions=self.hold, require_family=require_family)


CELLS: Final = (Cell("NSN_CORE", "ID", 1), Cell("NSN_CORE", "MD3", 3))
MAX_HOLD: Final = max(c.hold for c in CELLS)


@dataclass(frozen=True, slots=True)
class CountRule:
    """Stage A's eligibility of a cell (spec §G.9)."""

    validation_per_year: float = 200.0
    validation_dates: int = 100
    train_n: int = 500
    train_dates: int = 100


COUNT_RULE: Final = CountRule()

SENSITIVITIES: Final = (
    "tech",
    "broad",
    "news_lag_60s",
    "news_lag_1200s",
    "cost_x2",
    "cost_surcharge",
    "rank_split",
    "unresolved_minus100",
    "participation_le_10pct",
)
NEWS_LAGS: Final = {
    "news_lag_60s": timedelta(seconds=60),
    "news_lag_1200s": timedelta(seconds=1200),
}
PARTICIPATION_MAX: Final = 0.10

# What the context computes differently from spec §C: the deviations section of
# events/context.py's docstring, item for item, whitespace normalised (a test holds
# the two equal, so the PREREG cites what the context says it does).
CONTEXT_DEVIATIONS: Final = (
    "**The news time is required.** ``eligibility(symbol, session, *, at_news, "
    'universe="primary")``; the spec\'s signature has no ``at_news``. σ is judged at that '
    "time, on the window :meth:`PitContext.pre_event` uses, and a time outside (S−2's close, "
    "S's close) is refused by both.",
    "**BROAD admits an unmapped name with no ``ticker_ciks`` row** (see Screen above); the "
    "spec's ``status != 'fund'`` would drop it as NULL.",
    "**History is loaded from 380 days before the first session**, not 100: daily bars over "
    "[start − 380 d, end + 7 d], because the 252-session levels need a year of bars (380 days "
    "hold at least 257 sessions in 2017-2027). With each name's first raw bar day "
    "(Descriptives), no answer depends on where a load starts.",
    "``stories_before`` is not built yet (contracts.md); ``sessions`` and ``daily`` are added "
    "for the simulator. ``adj``, ``daily`` and ``facts_before`` raise outside the loaded "
    "range rather than answer None or nothing.",
)

CRITERION: Final = (
    "H1 passes iff some cell (NSN_CORE, ID|MD3) that Stage A finds eligible (validation >= 200 "
    "entries/yr and >= 100 entry dates; train >= 500 entries and >= 100 dates) has, on train "
    "2016-10-03..2021-12-31: mean net abnormal return > 0 with date-clustered CR1 t >= 2.0 "
    "(MD3 also calendar-time Newey-West(4) t >= 2.0), Holm rejection at one-sided alpha 0.05 "
    "across eligible cells, mean > 0 at 1.5x costs, beta-adjusted mean > 0, and mean > 0 "
    "excluding entries 2020-02-20..2020-06-30; and on validation 2022-01-03..2024-12-31 (train "
    "passers only) the same except the 2020 exclusion, Holm across train passers. Stage-A data "
    "skips <= 2% and unresolved exits <= 0.5% per window, else inconclusive. Nothing is changed "
    "after Stage A counts or any result is seen."
)


class H1Locked(RuntimeError):
    """A step asked for out of order (spec §G.15), or a configuration not registered."""


class RegistrationRefused(RuntimeError):
    """The registration's preconditions fail, or the configuration is registered already."""


class StoriesStale(RuntimeError):
    """The stories rebuilt in memory disagree with ``news_stories`` (rebuild them)."""


# ── windows ────────────────────────────────────────────────────


def window_span(window: WindowName) -> tuple[date, date]:
    """The window's first session and its last day (the loader's guard)."""
    w = _SIM_WINDOW[window]
    return WINDOW_START[w], WINDOW_LAST[w]


def last_session(window: WindowName, cell: Cell) -> date:
    """The last reaction session whose ``cell.hold``-session path ends by the window's end.

    Train: 2021-12-31 (ID), 2021-12-29 (MD3); validation 2024-12-31, 2024-12-27.
    """
    _, end = window_span(window)
    day = end
    while not is_trading_day(day) or path_days(day, cell.hold)[-1] > end:
        day -= timedelta(days=1)
    return day


def sessions_between(first: date, last: date) -> list[date]:
    """Every ``market_hours`` session in [first, last]."""
    out: list[date] = []
    day = first if is_trading_day(first) else next_trading_day(first)
    while day <= last:
        out.append(day)
        day = next_trading_day(day)
    return out


def _windows_text() -> str:
    t0, t1 = window_span("train")
    v0, v1 = window_span("validation")
    return (
        f"train {t0}..{t1} | validation {v0}..{v1} | "
        "holdout 2025-01-02.. (2025-12-01.. non-confirmatory)"
    )


# ── the pre-registration (spec §G.14) ─────────────────────────


def build_prereg(pins: Mapping[str, str]) -> dict[str, Any]:
    """PREREG: the literal with ``pins`` (``stories.pins``) and the code's constants.

    Pure: the same pins give the same dict (and the same ``config_hash``).
    """
    playbook = BounceParams().as_config()
    playbook.pop("hold_sessions", None)  # the cells carry it
    windows: dict[str, Any] = {}
    for w in WINDOWS:
        first, end = window_span(w)
        windows[w] = [first.isoformat(), end.isoformat()]
        windows[f"{w}_last_session"] = {c.variant: last_session(w, c).isoformat() for c in CELLS}
    windows["years"] = dict(YEARS)
    windows["holdout"] = "2025-01-02.. (2025-12-01.. non-confirmatory)"
    return {
        "hypothesis": HYPOTHESIS,
        "version": 1,
        "pins": dict(sorted(pins.items())),
        "builder": {
            "version": st.BUILDER_VERSION,
            "roundup_max_symbols": st.ROUNDUP_MAX_SYMBOLS,
            "news_lag_s": int(st.NEWS_LAG.total_seconds()),
            "react_cutoff_min": int(st.REACT_CUTOFF.total_seconds() // 60),
            "filing": "EDGAR business day [06:00,17:30) else next 06:00",
            "dup_jaccard": st.DUP_JACCARD,
            "repost_jaccard": st.REPOST_JACCARD,
            "follow_sessions": st.FOLLOW_SESSIONS,
        },
        "taxonomy": {
            "version": RULES_VERSION,
            "extractor": EXTRACTOR,
            "dead_band": DEAD_BAND,
            "guide_band": GUIDE_BAND,
        },
        "family": {FAMILY: sorted(FAMILY_TYPES)},
        "universe": {
            "screen": "newest as_of < S, verdict halal, cik not null",
            "rank_lt": pit.MAX_RANK,
            "rank_top_n": pit.TOP_N,
            "min_prev_close": pit.MIN_PREV_CLOSE,
            "one_class_per_cik": "lower rank",
            "sigma_sessions": pit.SIGMA_SESSIONS,
            "sigma_min_obs": pit.SIGMA_MIN_OBS,
            "beta_clip": list(pit.BETA_CLIP),
            "context_deviations": list(CONTEXT_DEVIATIONS),
        },
        "playbook": {
            "name": bounce_rule.NAME,
            "version": bounce_rule.VERSION,
            **playbook,
            "reclaim": "close > AVWAP from anchor",
            "stop": "bar close < L*",
            "abort": "structural item",
            "priority": ["abort", "stop", "target"],
            "compliance_exit": True,
        },
        "cells": [[c.family, c.variant, c.hold] for c in CELLS],
        "fills": {
            **SimConfig().as_config(),
            "orders": ["market/day"],
            "price": "clamped VWAP of first bar ts >= active_at; open after a >=5 min gap",
            "partial": "none",
            "fallback": ["next session open (MD3)", "official close", "next session", "unresolved"],
        },
        "costs": {
            "rule": "study.COST_BPS one-way, two sides",
            "bps": [[rank_lt, bps] for rank_lt, bps in COST_BPS],
        },
        "metric": "net abnormal vs SPY at the same bars, A-ratio adjusted, no drop filter",
        "stats": {
            "se": "CR1 by entry date, df G-1",
            "md3_extra": f"calendar-time Newey-West {stats.NW_LAGS} lags",
            "alpha": ALPHA,
            "multiple": "Holm one-sided over eligible cells",
            "t_min": T_MIN,
        },
        "count_rule": {
            "validation_per_year": COUNT_RULE.validation_per_year,
            "validation_dates": COUNT_RULE.validation_dates,
            "train_n": COUNT_RULE.train_n,
            "train_dates": COUNT_RULE.train_dates,
        },
        "robustness": {
            "cost_x": COST_X,
            "beta_adjusted": True,
            "train_exclude": [COVID[0].isoformat(), COVID[1].isoformat()],
        },
        "data": {"skip_max": SKIP_MAX, "unresolved_max": UNRESOLVED_MAX},
        "windows": windows,
        "book": {
            "slots": BOOK_SLOTS,
            "daily_loss_limit": BOOK_LOSS_LIMIT,
            "benchmark": BENCHMARK_LABEL,
        },
        "sensitivities": list(SENSITIVITIES),
    }


async def prereg(engine: AsyncEngine) -> dict[str, Any]:
    """PREREG with the pins the database and the code hold now."""
    return build_prereg(await st.pins(engine))


def trial_config(
    cell: Cell, prereg_hash: str, *, feed: str = SimConfig().feed.name
) -> dict[str, Any]:
    """One cell's trial: the registration, the cell, the feed (no window: both share it)."""
    return {"prereg": prereg_hash, "cell": [cell.family, cell.variant, cell.hold], "feed": feed}


# ── the ledger ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LedgerRow:
    id: int
    kind: str
    verdict: str | None
    window: str
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Registration:
    id: int
    hash: str
    prereg: dict[str, Any]
    commit: str | None
    code: dict[str, Any] = field(default_factory=dict)  # C0's data: commit, tags, dirty
    files: dict[str, Any] = field(default_factory=dict)  # file_shas() at registration


async def _rows(
    engine: AsyncEngine, *, kind: str, config_h: str, name: str = NAME
) -> list[LedgerRow]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                'SELECT id, kind, verdict, "window", metrics FROM quant_trials '
                "WHERE name = :n AND kind = :k AND config_hash = :h ORDER BY id"
            ),
            {"n": name, "k": kind, "h": config_h},
        )
        return [
            LedgerRow(int(r.id), str(r.kind), r.verdict, str(r.window or ""), dict(r.metrics or {}))
            for r in rows
        ]


async def _latest(engine: AsyncEngine, *, kind: str, config_h: str) -> LedgerRow | None:
    rows = await _rows(engine, kind=kind, config_h=config_h)
    return rows[-1] if rows else None


async def _latest_window(engine: AsyncEngine, reg: Registration, window: str) -> LedgerRow | None:
    rows = [
        r
        for r in await _rows(engine, kind="window", config_h=reg.hash)
        if r.metrics.get("window_role") == window
    ]
    return rows[-1] if rows else None


async def _row(engine: AsyncEngine, reg: Registration, kind: str, row_id: int) -> LedgerRow:
    """The ledger row ``row_id``, which must be the trial's ``kind`` row."""
    rows = [r for r in await _rows(engine, kind=kind, config_h=reg.hash) if r.id == row_id]
    if not rows:
        raise H1Locked(f"quant_trials {row_id} is not a {kind} row of this registration")
    return rows[0]


async def _record(
    engine: AsyncEngine,
    *,
    kind: str,
    config: dict[str, Any],
    window: str,
    metrics: Mapping[str, Any],
    verdict: str | None = None,
    criterion: str | None = None,
    name: str = NAME,
) -> int:
    return await QuantTrialRepoImpl(engine).record_trial(
        name=name,
        kind=kind,
        config=config,
        window=window,
        metrics=clean(metrics),
        criterion=criterion,
        verdict=verdict,
    )


def clean(value: Any) -> Any:
    """``value`` as JSON the ledger stores: no NaN or infinity (None), dates as ISO."""
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float | np.floating):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        items = sorted(value) if isinstance(value, set | frozenset) else list(value)
        return [clean(v) for v in items]
    return str(value)


async def existing_registration(engine: AsyncEngine) -> int | None:
    """The id of the registration of the current configuration, if there is one."""
    rows = await _rows(engine, kind="preregistration", config_h=config_hash(await prereg(engine)))
    return rows[0].id if rows else None


async def registration(engine: AsyncEngine) -> Registration:
    """The registration of the current configuration; :class:`H1Locked` without one.

    The message names the pins that differ from the newest registration on
    record, when one exists under another hash.
    """
    pre = await prereg(engine)
    h = config_hash(pre)
    rows = await _rows(engine, kind="preregistration", config_h=h)
    if rows:
        code = dict(rows[0].metrics.get("code") or {})
        files = dict(rows[0].metrics.get("files") or {})
        return Registration(rows[0].id, h, pre, code.get("commit"), code, files)
    async with engine.connect() as conn:
        other = (
            await conn.execute(
                text(
                    "SELECT config FROM quant_trials WHERE name = :n AND kind = 'preregistration' "
                    "ORDER BY id DESC LIMIT 1"
                ),
                {"n": NAME},
            )
        ).first()
    why = "H1 is not registered: run `halal-trader events h1 register` first"
    if other is not None and other.config:
        old = dict((other.config or {}).get("pins") or {})
        changed = sorted(k for k in set(old) | set(pre["pins"]) if old.get(k) != pre["pins"].get(k))
        why = (
            f"no registration matches the configuration in force (hash {h}); "
            f"pins changed since the last registration: {', '.join(changed) or 'none'} "
            "(a changed pin is a new trial)"
        )
    raise H1Locked(why)


# ── preconditions (spec §E.0, §G.15) ──────────────────────────


@dataclass(frozen=True, slots=True)
class Check:
    """One precondition: whether it holds, why, and what it measured."""

    id: str
    ok: bool
    detail: str
    data: Mapping[str, Any] = field(default_factory=dict)


REQUIRED_CHECKS: Final = ("C0", "G", "D1", "D2", "D3", "D4", "D5", "D6", "D9")
# Measured and recorded in the registration, never gating (spec §E.0).
REPORTED_CHECKS: Final = ("D7", "D8")


@dataclass(frozen=True, slots=True)
class Preconditions:
    checks: tuple[Check, ...]

    def get(self, check_id: str) -> Check | None:
        return next((c for c in self.checks if c.id == check_id), None)

    @property
    def missing(self) -> list[str]:
        have = {c.id for c in self.checks}
        return [i for i in REQUIRED_CHECKS if i not in have]

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.id not in REPORTED_CHECKS]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.failures

    def metrics(self) -> dict[str, Any]:
        return {c.id: {"ok": c.ok, "detail": c.detail, **dict(c.data)} for c in self.checks}


@dataclass(frozen=True, slots=True)
class CodeState:
    """The checkout the registration was made from."""

    commit: str | None
    tags: tuple[str, ...] = ()
    dirty: bool | None = None

    @property
    def frozen(self) -> bool:
        """Tagged :data:`TAG` and nothing tracked modified."""
        return TAG in self.tags and self.dirty is False


def code_state(root: Path | None = None) -> CodeState:
    """HEAD, the tags on it, and whether a tracked file is modified (git; None without it)."""
    where = root or Path(__file__).resolve().parents[3]

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=where, capture_output=True, text=True, timeout=60, check=True
            )
        except OSError, subprocess.SubprocessError:
            return None
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return CodeState(None)
    tags = tuple(sorted((git("tag", "--points-at", "HEAD") or "").split()))
    status = git("status", "--porcelain", "--untracked-files=no")
    return CodeState(commit, tags, None if status is None else bool(status))


PINNED_MODULES: Final = (
    "halal_trader.events.h1",
    "halal_trader.events.stories",
    "halal_trader.events.aliases",
    "halal_trader.events.renames",
    "halal_trader.events.taxonomy",
    "halal_trader.events.earnings_parse",
    "halal_trader.events.headline_patterns",
    "halal_trader.events.context",
    "halal_trader.events.stats",
    "halal_trader.events.study",
    "halal_trader.events.units",
    "halal_trader.events.sim_gate",
    "halal_trader.data.minutes",
    "halal_trader.data.universe",
    "halal_trader.market_hours",
    "halal_trader.research.ledger",
    "halabot.playbooks.bounce",
    "halabot.playbooks.clock",
    "halabot.playbooks.exchange",
    "halabot.playbooks.interfaces",
    "halabot.playbooks.legacy",
    "halabot.playbooks.loader",
    "halabot.playbooks.playbook",
    "halabot.playbooks.records",
    "halabot.playbooks.rules",
    "halabot.playbooks.sim",
    "halabot.playbooks.types",
)


def file_shas() -> dict[str, str | None]:
    """sha256 (12 hex) of each pinned module's source file; None when it is missing."""
    out: dict[str, str | None] = {}
    for name in PINNED_MODULES:
        try:
            spec = importlib.util.find_spec(name)
        except ImportError:
            spec = None
        origin = spec.origin if spec is not None else None
        path = Path(origin) if origin else None
        out[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            if path is not None and path.is_file()
            else None
        )
    return out


def check_code(code: CodeState) -> Check:
    """C0: the code is frozen and tagged (spec §G.15 step 1)."""
    data = {"commit": code.commit, "tags": list(code.tags), "dirty": code.dirty, "tag": TAG}
    if code.commit is None:
        return Check("C0", False, "not a git checkout: the frozen code cannot be named", data)
    if not code.frozen:
        state = "modified" if code.dirty else "clean" if code.dirty is False else "unknown"
        return Check(
            "C0",
            False,
            f"HEAD {code.commit[:12]} is not tagged {TAG} with a clean tree ({state})",
            data,
        )
    return Check("C0", True, f"HEAD {code.commit[:12]} tagged {TAG}, clean", data)


# The Phase 0 gate ids (spec §E; events/sim_gate.py writes one row per id under
# GATE_NAME, config {"gate": <id>, ...}): G1's look-ahead, synthetic and
# determinism checks, G2's R0-R2, G3's S0, S1, the calibration and S2-S3.
REQUIRED_GATES: Final[tuple[str, ...]] = (
    "g1-lookahead",
    "g1-synthetic",
    "g1-determinism",
    "r0",
    "r1",
    "r2",
    "s0",
    "s1",
    "s1-calib",
    "s2",
    "s3",
)


def judge_gates(rows: Sequence[tuple[str, int, str | None]]) -> Check:
    """G: every required gate id has a row, and every gate id's latest row passes.

    ``rows`` are (gate id, row id, verdict), any order. An id outside
    :data:`REQUIRED_GATES` gates nothing by being absent, but its latest row
    must pass too (a failure recorded under the gate name is never ignored).
    """
    latest: dict[str, tuple[int, str | None]] = {}
    for gate, row_id, verdict_ in rows:
        if gate not in latest or row_id > latest[gate][0]:
            latest[gate] = (row_id, verdict_)
    failed = sorted(g for g, (_, v) in latest.items() if v != "pass")
    missing = [g for g in REQUIRED_GATES if g not in latest]
    data = {
        "required": list(REQUIRED_GATES),
        "gates": {g: {"id": i, "verdict": v} for g, (i, v) in sorted(latest.items())},
    }
    if failed or missing:
        parts = []
        if missing:
            parts.append(f"no row for {', '.join(missing)}")
        if failed:
            parts.append(f"latest row not passed: {', '.join(failed)}")
        return Check("G", False, "; ".join(parts), data)
    return Check(
        "G", True, f"{len(REQUIRED_GATES)} required gate(s), every latest row passed", data
    )


async def gate_rows(engine: AsyncEngine) -> list[tuple[str, int, str | None]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT config->>'gate' AS gate, id, verdict FROM quant_trials "
                "WHERE name = :n AND kind = 'gate' AND config->>'gate' IS NOT NULL"
            ),
            {"n": GATE_NAME},
        )
        return [(str(r.gate), int(r.id), r.verdict) for r in rows]


D1_SPAN: Final = (date(2016, 1, 4), date(2026, 10, 9))
D2_SPAN: Final = (date(2016, 9, 30), date(2024, 12, 31))
D2_MAX: Final = 0.03
D3_SPAN: Final = (date(2016, 10, 1), date(2024, 12, 31))
D3_MAX: Final = 0.02
D4_MIN: Final = 0.99
D5_SAMPLE: Final = 200
READ_SESSIONS_SPAN_DAYS: Final = 7  # read_sessions scans sessions this close in one range
D6_SPAN: Final = (date(2016, 1, 1), date(2024, 12, 31))  # news published (NY days)
D7_FROM: Final = date(2026, 10, 8)  # live rows from here on
D7_SAMPLE: Final = 100
D7_WINDOW: Final = timedelta(hours=1)  # the refetch asks for the article's hour around it
D8_UNITS: Final = 20
D8_YEAR: Final = 2024
SCAN_CHUNK: Final = 1000
SEED: Final = 20261010


async def check_calendar_d1(engine: AsyncEngine) -> Check:
    """D1: market_hours' sessions are SPY's raw daily-bar sessions, 2016-01-04..2026-10-09."""
    try:
        await check_calendar(engine, *D1_SPAN)
    except CalendarMismatch as exc:
        return Check("D1", False, str(exc))
    return Check("D1", True, f"sessions {D1_SPAN[0]}..{D1_SPAN[1]} match SPY's daily bars")


async def check_screens_d2(engine: AsyncEngine) -> Check:
    """D2: the unmapped tickers were mapped and re-screened; each screen's residual <= 3%.

    Every count is over the screens H1 reads (``D2_SPAN``): a nightly screen
    after 2024 never reaches the trial, so it cannot block the registration.
    """
    from halal_trader.compliance.runner import UNMAPPED

    span = {"a": D2_SPAN[0], "b": D2_SPAN[1]}
    async with engine.connect() as conn:
        unmatched = await conn.scalar(
            text(
                "SELECT count(DISTINCT r.symbol) FROM halal_screen_current r "
                "LEFT JOIN ticker_ciks t ON t.symbol = r.symbol "
                "WHERE r.sic_description = :u AND t.symbol IS NULL "
                "AND r.as_of BETWEEN :a AND :b"
            ),
            {"u": UNMAPPED, **span},
        )
        pending = await conn.scalar(
            text(
                "SELECT count(*) FROM halal_screen_current r "
                "JOIN ticker_ciks t ON t.symbol = r.symbol AND t.status = 'mapped' "
                "WHERE r.sic_description = :u AND r.as_of BETWEEN :a AND :b"
            ),
            {"u": UNMAPPED, **span},
        )
        rows = (
            await conn.execute(
                text(
                    "SELECT r.as_of, "
                    "count(*) FILTER (WHERE t.status IS DISTINCT FROM 'fund') AS companies, "
                    "count(*) FILTER (WHERE r.cik IS NULL AND t.status IS DISTINCT FROM 'fund') "
                    "AS unmapped "
                    "FROM halal_screen_current r LEFT JOIN ticker_ciks t ON t.symbol = r.symbol "
                    "WHERE r.as_of BETWEEN :a AND :b GROUP BY r.as_of ORDER BY r.as_of"
                ),
                span,
            )
        ).all()
    residual = {
        r.as_of.isoformat(): (int(r.unmapped) / int(r.companies) if r.companies else 0.0)
        for r in rows
    }
    over = sorted(d for d, share in residual.items() if share > D2_MAX)
    data = {
        "unmatched": int(unmatched or 0),
        "pending_rescreen": int(pending or 0),
        "residual": residual,
    }
    problems = []
    if unmatched:
        problems.append(f"{unmatched} unmapped ticker(s) never matched (delisted.map_unmapped)")
    if pending:
        problems.append(f"{pending} mapped row(s) not re-screened (delisted.rescreen_mapped)")
    if not rows:
        problems.append(f"no screen in {D2_SPAN[0]}..{D2_SPAN[1]}")
    if over:
        problems.append(f"{len(over)} screen(s) above {D2_MAX:.0%} unmapped (first {over[0]})")
    if problems:
        return Check("D2", False, "; ".join(problems), data)
    worst = max(residual.values())
    return Check("D2", True, f"{len(rows)} screens, unmapped at most {worst:.2%}", data)


def quarter_starts(first: date, last: date) -> list[date]:
    out: list[date] = []
    d = date(first.year, 3 * ((first.month - 1) // 3) + 1, 1)
    while d <= last:
        out.append(d)
        d = date(d.year + (d.month + 2) // 12, (d.month + 2) % 12 + 1, 1)
    return out


async def check_news_d3(engine: AsyncEngine) -> Check:
    """D3: renamed tickers' news fetched; per quarter, PIT-halal rank<1000 names with no
    admitted news item are at most 2%."""
    from halal_trader.data.universe import universe_at
    from halal_trader.events.renames import missing_units
    from halal_trader.halal.strict import all_screens

    missing = await missing_units(engine)
    screens = await all_screens(engine)
    dates = sorted(screens)
    residual: dict[str, dict[str, Any]] = {}
    for q in quarter_starts(*D3_SPAN):
        q_end = date(q.year + (q.month + 2) // 12, (q.month + 2) % 12 + 1, 1) - timedelta(days=1)
        first = q if is_trading_day(q) else next_trading_day(q)
        as_of = max((d for d in dates if d < first), default=None)
        if as_of is None:
            residual[q.isoformat()] = {"names": 0, "silent": [], "share": 1.0, "screen": None}
            continue
        halal = {r.symbol for r in screens[as_of] if r.verdict == "halal" and r.cik is not None}
        ranked = (await universe_at(engine, first, top_n=pit.TOP_N))[: pit.MAX_RANK]
        names = sorted(halal & set(ranked))
        async with engine.connect() as conn:
            rows = await conn.execute(
                text(
                    "SELECT DISTINCT s.symbol FROM news_stories s "
                    "CROSS JOIN LATERAL jsonb_array_elements(s.items) AS it "
                    "JOIN events e ON e.id = CAST(it->>'event_id' AS bigint) "
                    "WHERE s.builder_version = :v AND s.session BETWEEN :a AND :b "
                    "AND s.symbol = ANY(:names) AND e.kind = 'news'"
                ),
                {"v": st.BUILDER_VERSION, "a": first, "b": q_end, "names": names},
            )
            heard = {str(r.symbol) for r in rows}
        silent = sorted(set(names) - heard)
        residual[q.isoformat()] = {
            "names": len(names),
            "silent": silent,
            "share": len(silent) / len(names) if names else 1.0,
            "screen": as_of.isoformat(),
        }
    over = sorted(q for q, r in residual.items() if r["share"] > D3_MAX)
    data = {"renamed_months_missing": missing, "quarters": residual}
    problems = []
    if missing:
        problems.append(f"{len(missing)} renamed-news month(s) not fetched (first {missing[0]})")
    if over:
        problems.append(f"{len(over)} quarter(s) above {D3_MAX:.0%} without news (first {over[0]})")
    if problems:
        return Check("D3", False, "; ".join(problems), data)
    worst = max((r["share"] for r in residual.values()), default=0.0)
    return Check("D3", True, f"{len(residual)} quarters, silent names at most {worst:.2%}", data)


class UnitPlanLike(Protocol):
    """What D4 and D5 read of ``units.UnitPlan`` (plan H, spec §H)."""

    @property
    def parts(self) -> Mapping[str, Collection[tuple[str, date]]]: ...


def spy_complete(day: date, bars: int) -> bool:
    """D.9's complete SPY session: 300 bars, 150 on an early close."""
    return bars >= (150 if day in EARLY_CLOSE_DATES else 300)


async def bar_counts(
    engine: AsyncEngine, units: Collection[tuple[str, date]]
) -> dict[tuple[str, date], int]:
    """Bars ``minutes.read_windows`` returns for each unit."""
    out: dict[tuple[str, date], int] = {}
    todo = sorted(set(units))
    for i in range(0, len(todo), SCAN_CHUNK):
        got = await minutes.read_windows(engine, todo[i : i + SCAN_CHUNK])
        out.update({u: len(a) for u, a in got.items()})
    return out


async def check_units_d4(engine: AsyncEngine, plan: UnitPlanLike, done: set[str]) -> Check:
    """D4: >= 99% of each part of plan H done, SPY 100%, every SPY session complete."""
    parts: dict[str, dict[str, Any]] = {}
    problems = []
    for name, units in sorted(plan.parts.items()):
        n = len(units)
        got = sum(1 for s, d in units if minutes.unit(s, d) in done)
        share = got / n if n else 1.0
        parts[name] = {"units": n, "done": got, "share": share}
        need = 1.0 if name == "spy" else D4_MIN
        if share < need:
            problems.append(f"{name} {share:.2%} done (needs {need:.0%})")
    spy_units = {(s, d) for units in plan.parts.values() for s, d in units if s == "SPY"}
    counts = await bar_counts(engine, spy_units)
    short = sorted(d.isoformat() for (_, d), n in counts.items() if not spy_complete(d, n))
    if short:
        problems.append(f"{len(short)} SPY session(s) short of bars (first {short[0]})")
    data = {"parts": parts, "spy_sessions": len(spy_units), "spy_short": short[:50]}
    if problems:
        return Check("D4", False, "; ".join(problems), data)
    return Check(
        "D4", True, f"{len(parts)} parts done, {len(spy_units)} SPY sessions complete", data
    )


async def check_readers_d5(
    engine: AsyncEngine, units: Collection[tuple[str, date]], done: set[str]
) -> Check:
    """D5: no reader returns a bar outside [open, effective close).

    Every done unit through ``read_windows``. A seeded sample of
    :data:`D5_SAMPLE` units through ``read`` (one session), and each sampled
    unit with its symbol's other done sessions within a week of it through
    ``read_sessions``, so its multi-session range scan (which reads the
    nights and pre-markets between the sessions) is exercised too.
    """
    todo = sorted(u for u in set(units) if minutes.unit(*u) in done)
    outside = bars = 0
    for i in range(0, len(todo), SCAN_CHUNK):
        got = await minutes.read_windows(engine, todo[i : i + SCAN_CHUNK])
        for (_, day), arrays in got.items():
            lo, hi = (int(t.timestamp()) for t in minutes.session_bounds(day))
            outside += int(np.count_nonzero((arrays.ts < lo) | (arrays.ts >= hi)))
            bars += len(arrays)
    by_symbol: dict[str, list[date]] = defaultdict(list)
    for symbol, day in todo:
        by_symbol[symbol].append(day)
    sample = random.Random(SEED).sample(todo, min(D5_SAMPLE, len(todo)))
    outside_read = outside_sessions = multi = 0
    for symbol, day in sample:
        lo_t, hi_t = minutes.session_bounds(day)
        outside_read += sum(
            1 for b in await minutes.read(engine, symbol, day) if not lo_t <= b.ts < hi_t
        )
        near = [d for d in by_symbol[symbol] if abs((d - day).days) <= READ_SESSIONS_SPAN_DAYS]
        multi += len(near) > 1
        for d, got_bars in (await minutes.read_sessions(engine, symbol, near)).items():
            lo_d, hi_d = minutes.session_bounds(d)
            outside_sessions += sum(1 for b in got_bars if not lo_d <= b.ts < hi_d)
    data = {
        "units": len(todo),
        "bars": bars,
        "outside_read_windows": outside,
        "sampled_read": len(sample),
        "outside_read": outside_read,
        "sampled_read_sessions_multi": multi,
        "outside_read_sessions": outside_sessions,
    }
    total = outside + outside_read + outside_sessions
    if total:
        return Check("D5", False, f"{total} bar(s) outside the session", data)
    return Check(
        "D5",
        True,
        f"{len(todo)} units, {bars} bars, none outside the session "
        f"({multi} multi-session read_sessions scans)",
        data,
    )


def _ny_midnight(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time(), MARKET_TZ)


async def check_facts_d6(engine: AsyncEngine) -> Check:
    """D6: every news event H1 can read has the current extractor's facts; the 2.02 rate.

    "Every" is the news published in ``D6_SPAN`` (NY days): news ingested
    live after the nightly extraction never reaches the trial.
    """
    first, end = window_span("train")
    async with engine.connect() as conn:
        unread = await conn.scalar(
            text(
                "SELECT count(*) FROM events e WHERE e.kind = 'news' "
                "AND e.published_at >= :a AND e.published_at < :b AND NOT EXISTS ("
                "SELECT 1 FROM event_facts f WHERE f.event_id = e.id AND f.extractor = :x)"
            ),
            {
                "x": EXTRACTOR,
                "a": _ny_midnight(D6_SPAN[0]),
                "b": _ny_midnight(D6_SPAN[1] + timedelta(days=1)),
            },
        )
        row = (
            await conn.execute(
                text(
                    "SELECT count(*) AS n, "
                    "count(*) FILTER (WHERE type_close = 'earnings_unparsed') AS unparsed "
                    "FROM news_stories WHERE builder_version = :v AND session BETWEEN :a AND :b "
                    "AND items @> CAST(:probe AS jsonb)"
                ),
                {
                    "v": st.BUILDER_VERSION,
                    "a": first,
                    "b": end,
                    "probe": '[{"itype": "earnings_8k"}]',
                },
            )
        ).one()
    n, unparsed = int(row.n), int(row.unparsed)
    data = {
        "extractor": EXTRACTOR,
        "span": [D6_SPAN[0].isoformat(), D6_SPAN[1].isoformat()],
        "news_unread": int(unread or 0),
        "stories_202": n,
        "stories_202_unparsed": unparsed,
        "unparsed_share": unparsed / n if n else None,
    }
    rate = f"{unparsed}/{n} 2.02 stories left earnings_unparsed (train)"
    span = f"{D6_SPAN[0]}..{D6_SPAN[1]}"
    if unread:
        return Check(
            "D6", False, f"{unread} news event(s) of {span} not read by {EXTRACTOR}; {rate}", data
        )
    return Check("D6", True, f"every news event of {span} read by {EXTRACTOR}; {rate}", data)


class MarketProbe(Protocol):
    """What D7 and D8 ask of ``data.alpaca_market.AlpacaMarketData``."""

    async def news(
        self,
        symbols: Iterable[str] | None,
        *,
        start: datetime,
        end: datetime | None = ...,
        max_pages: int = ...,
    ) -> Sequence[Any]: ...

    async def minute_bars_many(
        self,
        symbols: Iterable[str],
        *,
        start: datetime,
        end: datetime,
        feed: Literal["sip", "iex"] = ...,
    ) -> Mapping[str, Sequence[Any]]: ...


def _not_measured(check_id: str, why: str) -> Check:
    return Check(check_id, True, f"not measured: {why} (reported, not gating)", {"measured": False})


async def check_edits_d7(engine: AsyncEngine, market: MarketProbe | None) -> Check:
    """D7 (reported, never gating): how often a live headline differs from Alpaca's copy now.

    A seeded sample of :data:`D7_SAMPLE` live ``alpaca`` news rows seen from
    :data:`D7_FROM` on is refetched by id (the row's symbol, an hour either
    side of its publication time). ``edited``: the stored headline differs
    from the refetched one (whitespace stripped); ``missing``: the id is not
    returned. A probe that cannot run is recorded as not measured.
    """
    if market is None:
        return _not_measured("D7", "no Alpaca client")
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, symbol, source_id, published_at, payload->>'headline' AS headline "
                    "FROM events WHERE kind = 'news' AND source = 'alpaca' AND seen_at >= :t "
                    "ORDER BY id"
                ),
                {"t": _ny_midnight(D7_FROM)},
            )
        ).all()
    sample = random.Random(SEED).sample(rows, min(D7_SAMPLE, len(rows)))
    found = edited = missing = 0
    examples: list[dict[str, Any]] = []
    try:
        for r in sample:
            article_id = str(r.source_id).removeprefix("alpaca:")
            got = await market.news(
                [r.symbol],
                start=r.published_at - D7_WINDOW,
                end=r.published_at + D7_WINDOW,
                max_pages=5,
            )
            match = next((a for a in got if str(a.id) == article_id), None)
            if match is None:
                missing += 1
                continue
            found += 1
            if str(match.headline).strip() != str(r.headline or "").strip():
                edited += 1
                if len(examples) < 20:
                    examples.append({"event": r.id, "stored": r.headline, "now": match.headline})
    except Exception as exc:  # reported only: a probe that fails is recorded, never fatal
        return _not_measured("D7", f"the refetch failed ({exc!s})")
    data = {
        "measured": True,
        "since": D7_FROM.isoformat(),
        "rows": len(rows),
        "sampled": len(sample),
        "found": found,
        "missing": missing,
        "edited": edited,
        "edit_rate": edited / found if found else None,
        "examples": examples,
    }
    detail = (
        f"{edited}/{found} refetched live headlines edited, {missing} not found "
        "(reported, not gating)"
    )
    return Check("D7", True, detail, data)


async def probe_iex_d8(
    engine: AsyncEngine,
    market: MarketProbe | None,
    units: Collection[tuple[str, date]] | None,
) -> Check:
    """D8 (recorded, never gating): IEX history for :data:`D8_UNITS` plan-H units of 2024.

    A seeded sample of the plan's non-SPY units in :data:`D8_YEAR`, each
    asked of Alpaca with ``feed=iex``; the bar count is recorded beside the
    stored SIP count. IEX bars are never stored or fed to H1 (spec §G.13).
    """
    if market is None:
        return _not_measured("D8", "no Alpaca client")
    if units is None:
        return _not_measured("D8", "plan H is not available")
    pool = sorted(u for u in set(units) if u[0] != SPY and u[1].year == D8_YEAR)
    sample = sorted(random.Random(SEED).sample(pool, min(D8_UNITS, len(pool))))
    stored = await bar_counts(engine, sample)
    probes: list[dict[str, Any]] = []
    try:
        for symbol, day in sample:
            lo, hi = minutes.session_bounds(day)
            got = await market.minute_bars_many([symbol], start=lo, end=hi, feed="iex")
            probes.append(
                {
                    "unit": minutes.unit(symbol, day),
                    "iex_bars": len(got.get(symbol, ())),
                    "sip_bars": stored.get((symbol, day), 0),
                }
            )
    except Exception as exc:  # recorded only: a probe that fails is recorded, never fatal
        return _not_measured("D8", f"the IEX request failed ({exc!s})")
    available = sum(1 for p in probes if p["iex_bars"])
    data = {"measured": True, "year": D8_YEAR, "units": probes, "available": available}
    return Check(
        "D8",
        True,
        f"IEX history: {available}/{len(probes)} sampled {D8_YEAR} units have bars "
        "(recorded, not gating)",
        data,
    )


async def story_counts_d9(engine: AsyncEngine) -> Check:
    """D9: stories by close type and NSN, per year and window, all/PRIMARY/Tech (recorded)."""
    first, _ = window_span("train")
    _, end = window_span("validation")
    counts = await st.count_stories(engine, start=first, end=end)
    data = {
        "types": [[*k, n] for k, n in sorted(counts.types.items())],
        "nsn": [[*k, n] for k, n in sorted(counts.nsn.items())],
        "reasons": [[*k, n] for k, n in sorted(counts.reasons.items())],
        "table": st.counts_table(counts, start=first, end=end),
    }
    total = sum(n for (u, *_), n in counts.nsn.items() if u == "primary")
    return Check("D9", True, f"{total} PRIMARY NSN stories by the entry cutoff (recorded)", data)


async def preconditions(
    engine: AsyncEngine,
    *,
    code: CodeState | None = None,
    plan: UnitPlanLike | None = None,
    scan: bool = True,
    market: MarketProbe | None = None,
) -> Preconditions:
    """Every check the registration needs: C0 (code), G (gates), D1-D6, D9 (counts).

    D7 and D8 are measured through ``market`` (Alpaca) and recorded, never
    gating; without it they are recorded as not measured. ``plan`` defaults
    to ``units.h1_plan(engine)``; ``scan=False`` leaves D5 out (a dry run
    only: the registration then refuses).
    """
    checks: list[Check] = [check_code(code if code is not None else code_state())]
    checks.append(judge_gates(await gate_rows(engine)))
    checks.append(await check_calendar_d1(engine))
    checks.append(await check_screens_d2(engine))
    checks.append(await check_news_d3(engine))
    if plan is None:
        try:
            from halal_trader.events.units import h1_plan
        except ImportError as exc:
            checks.append(Check("D4", False, f"plan H is not available: {exc}"))
            checks.append(Check("D5", False, "plan H is not available"))
            plan = None
        else:
            plan = await h1_plan(engine)
    units: set[tuple[str, date]] | None = None
    if plan is not None:
        done = await minutes.done_units(engine)
        checks.append(await check_units_d4(engine, plan, done))
        units = {u for part in plan.parts.values() for u in part}
        if scan:
            checks.append(await check_readers_d5(engine, units, done))
    checks.append(await check_facts_d6(engine))
    checks.append(await check_edits_d7(engine, market))
    checks.append(await probe_iex_d8(engine, market, units))
    checks.append(await story_counts_d9(engine))
    return Preconditions(tuple(checks))


async def register(
    engine: AsyncEngine, *, report: Preconditions | None = None, code: CodeState | None = None
) -> int:
    """Write the pre-registration; returns its ``quant_trials`` id.

    Refuses (:class:`RegistrationRefused`) when this configuration is
    registered already, or when a required check is missing or fails.
    ``report`` defaults to :func:`preconditions` (computed after the cheap
    refusal); a caller passing one takes responsibility for it.
    """
    pre = await prereg(engine)
    h = config_hash(pre)
    if rows := await _rows(engine, kind="preregistration", config_h=h):
        raise RegistrationRefused(f"H1 is registered already under {h} (quant_trials {rows[0].id})")
    if report is None:
        report = await preconditions(engine, code=code)
    if not report.ok:
        parts = [f"{c.id}: {c.detail}" for c in report.failures]
        parts += [f"{i}: not run" for i in report.missing]
        raise RegistrationRefused("preconditions fail: " + "; ".join(parts))
    c0 = report.get("C0")
    metrics = {
        "pins": pre["pins"],
        "code": dict(c0.data) if c0 is not None else {},
        "files": file_shas(),
        "checks": report.metrics(),
    }
    trial_id = await _record(
        engine,
        kind="preregistration",
        config=pre,
        window=_windows_text(),
        metrics=metrics,
        criterion=CRITERION,
    )
    logger.info("h1: registered as quant_trials %d (config %s)", trial_id, h)
    return trial_id


# ── stories and context for a window ──────────────────────────


@dataclass(frozen=True, slots=True)
class Carrier:
    """A story whose items reach a live playbook of its symbol but that never starts one.

    Everything is the story's, except that it is never NSN (no start, so no
    path, no block and no outcome): a story outside the cell's sessions, not
    NSN by its cutoff, or not eligible.
    """

    story: StoryView

    @property
    def story_id(self) -> str:
        return self.story.story_id

    @property
    def symbol(self) -> str:
        return self.story.symbol

    @property
    def session(self) -> date:
        return self.story.session

    def card_at(self, t: datetime) -> CardView:
        return self.story.card_at(t)

    def nsn_at(self, cutoff: datetime) -> datetime | None:
        return None

    def at_news(self) -> datetime | None:
        return None

    def start_case(self) -> str:
        return self.story.start_case()

    def news_times(self) -> Sequence[datetime]:
        return self.story.news_times()


def relag(story: st.Story, lag: timedelta) -> st.Story:
    """``story`` with each item usable ``lag`` after its public time instead of NEWS_LAG.

    Grouping, parents and the order of items are the builder's (a shift
    keeps the order); only ``available_at`` moves.
    """
    delta = lag - st.NEWS_LAG
    if not delta:
        return story
    items = [replace(i, available_at=i.available_at + delta) for i in story.items]
    return replace(story, items=items)


def nsn_by_cutoff(story: StoryView) -> bool:
    """NSN_CORE by S's entry cutoff: the stories the simulator would start."""
    return story.nsn_at(Session.of(story.session).entry_cutoff) is not None


@dataclass(slots=True)
class WindowData:
    """A window's stories (those that may start and those that may reach them), and context."""

    window: WindowName
    first: date
    end: date
    sessions: list[date]
    stories: list[st.Story]  # session order
    candidates: frozenset[str]  # NSN by the cutoff, S in [first, end]
    ctx: PitContext
    counts: Counter[str] = field(default_factory=Counter)
    _judged: dict[tuple[str, Universe], tuple[PreEvent | None, Eligibility]] = field(
        default_factory=dict
    )

    def judge(self, story: StoryView, universe: Universe) -> tuple[PreEvent | None, Eligibility]:
        """The story's eligibility at its news time, and its pre-event state when eligible."""
        key = (story.story_id, universe)
        hit = self._judged.get(key)
        if hit is None:
            at = story.at_news()
            if at is None:
                raise ValueError(f"{story.story_id} has no news time: it is not NSN")
            elig = self.ctx.eligibility(story.symbol, story.session, at_news=at, universe=universe)
            pre = self.ctx.pre_event(story.symbol, story.session, at) if elig.eligible else None
            hit = self._judged[key] = (pre, elig)
        return hit


async def _nsn_symbols(engine: AsyncEngine, first: date, end: date) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT DISTINCT symbol FROM news_stories WHERE builder_version = :v "
                "AND session BETWEEN :a AND :b AND nsn_at IS NOT NULL ORDER BY symbol"
            ),
            {"v": st.BUILDER_VERSION, "a": first, "b": end},
        )
        return [str(r.symbol) for r in rows]


async def _persisted_nsn(
    engine: AsyncEngine, symbols: Sequence[str], first: date, end: date
) -> dict[str, datetime]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT story_id, nsn_at FROM news_stories WHERE builder_version = :v "
                "AND symbol = ANY(:s) AND session BETWEEN :a AND :b AND nsn_at IS NOT NULL"
            ),
            {"v": st.BUILDER_VERSION, "s": list(symbols), "a": first, "b": end},
        )
        return {str(r.story_id): r.nsn_at for r in rows}


async def _built(
    engine: AsyncEngine, symbols: Sequence[str], end: date, aliases: Mapping[str, AliasMatcher]
) -> AsyncIterator[list[st.Story]]:
    """Each symbol's stories built from the history start to ``end`` (as ``build_range``)."""
    raws = st.load_items(engine, start=st.HISTORY_FROM, end=end, symbols=symbols)
    batch: list[st.RawItem] = []
    async for raw in raws:
        if batch and raw.symbol != batch[0].symbol:
            yield st.build(batch, aliases)
            batch = []
        batch.append(raw)
    if batch:
        yield st.build(batch, aliases)


def _keep(built: Sequence[st.Story], first: date, end: date) -> tuple[list[st.Story], set[str]]:
    """The symbol's candidates and every story with an item while one may be live."""
    candidates = [s for s in built if first <= s.session <= end and nsn_by_cutoff(s)]
    spans: list[tuple[datetime, datetime]] = []
    for s in candidates:
        start = start_time(s)
        assert start is not None
        spans.append((start, Session.of(path_days(s.session, MAX_HOLD)[-1]).close))
    ids = {s.story_id for s in candidates}
    kept = list(candidates)
    for s in built:
        if s.story_id in ids:
            continue
        if any(a < t <= b for t in s.news_times() for a, b in spans):
            kept.append(s)
    return kept, ids


BATCH_SYMBOLS: Final = 100


async def load_window(
    engine: AsyncEngine, window: WindowName, *, lag: timedelta | None = None
) -> WindowData:
    """The stories and context every cell of ``window`` runs on.

    Symbols are those with a persisted NSN story in the window. Their stories
    are rebuilt from the history start; a rebuilt NSN story that disagrees
    with ``news_stories`` (missing, extra, another ``nsn_at``) raises
    :class:`StoriesStale`. ``lag`` moves when items become usable (the news-
    lag sensitivities). Kept: the stories NSN by their entry cutoff with S in
    the window, and the symbol's other stories with an item inside one of
    their possible live spans (from the start to the close of a
    three-session path).
    """
    first, end = window_span(window)
    symbols = await _nsn_symbols(engine, first, end)
    aliases = await load_aliases(engine)
    counts: Counter[str] = Counter(symbols=len(symbols))
    kept: list[st.Story] = []
    candidates: set[str] = set()
    stale: list[str] = []
    for i in range(0, len(symbols), BATCH_SYMBOLS):
        batch = symbols[i : i + BATCH_SYMBOLS]
        persisted = await _persisted_nsn(engine, batch, first, end)
        rebuilt: dict[str, datetime] = {}
        async for built in _built(engine, batch, end, aliases):
            counts["built"] += len(built)
            for s in built:
                if first <= s.session <= end and (nsn := s.nsn_at(s.close)) is not None:
                    rebuilt[s.story_id] = nsn
            if lag is not None:
                built = [relag(s, lag) for s in built]
            keep, ids = _keep(built, first, end)
            kept += keep
            candidates |= ids
        stale += sorted(
            sid for sid in set(rebuilt) | set(persisted) if rebuilt.get(sid) != persisted.get(sid)
        )
        logger.info("h1 %s: %d of %d symbols rebuilt", window, i + len(batch), len(symbols))
    if stale:
        raise StoriesStale(
            f"{len(stale)} NSN stories in {first}..{end} differ from news_stories "
            f"(first: {', '.join(stale[:5])}); "
            "rebuild them with `halal-trader events stories build`"
        )
    kept.sort(key=lambda s: (s.session, s.story_id))
    counts["kept"] = len(kept)
    counts["candidates"] = len(candidates)
    names = sorted({s.symbol for s in kept if s.story_id in candidates})
    ctx = await PitContext.load(engine, symbols=names, start=first, end=end)
    return WindowData(
        window=window,
        first=first,
        end=end,
        sessions=sessions_between(first, end),
        stories=kept,
        candidates=frozenset(candidates),
        ctx=ctx,
        counts=counts,
    )


@dataclass(frozen=True, slots=True)
class RunInputs:
    """One run's stories (those that start, then carriers) and its factory context."""

    stories: list[StoryView]
    context: dict[str, tuple[PreEvent | None, Eligibility]]
    nsn: int  # NSN by the cutoff, S in the cell's sessions
    eligible: int
    reasons: dict[str, int]  # eligibility reasons of the NSN stories
    last: date


def run_inputs(data: WindowData, cell: Cell, *, universe: Universe = "primary") -> RunInputs:
    """Which of the window's stories start a playbook in ``cell``, and the factory's map.

    A story starts iff it is NSN by its cutoff, S is no later than the cell's
    last session and it is eligible in ``universe`` at its news time; every
    other story is a :class:`Carrier`.
    """
    last = last_session(data.window, cell)
    started: list[StoryView] = []
    carriers: list[StoryView] = []
    context: dict[str, tuple[PreEvent | None, Eligibility]] = {}
    reasons: Counter[str] = Counter()
    nsn = 0
    for s in data.stories:
        if s.story_id in data.candidates and s.session <= last:
            nsn += 1
            pre, elig = data.judge(s, universe)
            reasons[elig.reason] += 1
            if elig.eligible:
                started.append(s)
                context[s.story_id] = (pre, elig)
                continue
        carriers.append(Carrier(s))
    return RunInputs([*started, *carriers], context, nsn, len(started), dict(reasons), last)


class _Collect:
    """An :class:`OutcomeSink` that keeps every outcome and passes it on."""

    def __init__(self, inner: OutcomeSink) -> None:
        self.inner = inner
        self.outcomes: list[StoryOutcome] = []

    async def begin(self, info: RunInfo) -> str:
        return await self.inner.begin(info)

    async def write(self, outcomes: Sequence[StoryOutcome]) -> None:
        self.outcomes.extend(outcomes)
        await self.inner.write(outcomes)

    async def finish(self, summary: Mapping[str, object]) -> None:
        await self.inner.finish(summary)


Parallel = bool | Literal["fork", "spawn"] | None


async def _simulate(
    engine: AsyncEngine,
    reg: Registration,
    data: WindowData,
    cell: Cell,
    *,
    role: str,
    cfg: SimConfig,
    stop_at: Literal["entry", "end"] = "end",
    assume_full_hold: bool = False,
    universe: Universe = "primary",
    workers: int = 6,
    parallel: Parallel = None,
    code_sha: str | None = None,
) -> tuple[RunSummary, list[StoryOutcome], RunInputs]:
    """Run ``cell`` on ``data``; ``code_sha`` is the checkout actually running (``CodeNow.sha``)."""
    inputs = run_inputs(data, cell, universe=universe)
    factory = BounceFactory(inputs.context, cell.params())
    sink = _Collect(
        PgOutcomeSink(
            engine,
            mode="sim",
            cell=cell.key,
            config={
                **trial_config(cell, reg.hash, feed=cfg.feed.name),
                "role": role,
                "window": data.window,
                "universe": universe,
                "cost": cfg.cost,
            },
            prereg_trial_id=reg.id,
            code_sha=code_sha,
        )
    )
    summary = await simulate(
        engine,
        inputs.stories,
        factory,
        context=data.ctx,
        window=_SIM_WINDOW[data.window],
        window_end=data.end,
        cfg=cfg,
        unlock=WindowUnlock(prereg_id=reg.id, config_hash=reg.hash),
        sink=sink,
        workers=workers,
        stop_at=stop_at,
        assume_full_hold=assume_full_hold,
        parallel=parallel,
    )
    logger.info(
        "h1 %s %s %s: %d stories started, %d entries, %d trades",
        role,
        data.window,
        cell.key,
        summary.started,
        summary.entries,
        summary.trades,
    )
    return summary, sink.outcomes, inputs


async def _bootstrap(engine: AsyncEngine) -> None:
    from halabot.platform.db import bootstrap_schema

    await bootstrap_schema(engine)


# ── amendments: the code and the data each step runs on ───────
#
# The registration pins the code (its git state and file_shas) and Stage A
# freezes the data (a digest per window). Every later step compares what is in
# force now with the latest pinned state, and runs past a difference, a rerun
# or an unfinished earlier run only with an amendment: a ``kind="amendment"``
# row with its reason and what it found, written before the step runs. An
# amendment that records new code or data becomes the state the next steps
# compare with.


@dataclass(frozen=True, slots=True)
class CodeNow:
    """The checkout a step runs from: git's state and the pinned modules' shas."""

    state: CodeState
    files: dict[str, str | None]

    @property
    def sha(self) -> str | None:
        """What ``hb_playbook_run.code_sha`` records: HEAD, ``-dirty`` when modified."""
        if self.state.commit is None:
            return None
        return self.state.commit + ("" if self.state.dirty is False else "-dirty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "commit": self.state.commit,
            "tags": list(self.state.tags),
            "dirty": self.state.dirty,
        }


def current_code() -> CodeNow:
    return CodeNow(code_state(), file_shas())


def code_drift(
    before_code: Mapping[str, Any], before_files: Mapping[str, Any], now: CodeNow
) -> dict[str, Any]:
    """What differs from the pinned code (empty: nothing), pure.

    ``files``: each pinned module whose sha changed, ``[before, now]``;
    ``commit``: another HEAD; ``dirty``: the tree's modified state changed,
    or cannot be read (``None``), ``[before, now]``. The registration's tree
    is clean, so a modified tree needs one amendment; the same modified tree
    with the same pinned files then runs on.
    """
    out: dict[str, Any] = {}
    files = {
        m: [before_files.get(m), now.files.get(m)]
        for m in sorted(set(before_files) | set(now.files))
        if before_files.get(m) != now.files.get(m)
    }
    if files:
        out["files"] = files
    if now.state.commit != before_code.get("commit"):
        out["commit"] = [before_code.get("commit"), now.state.commit]
    if now.state.dirty is None or now.state.dirty != before_code.get("dirty"):
        out["dirty"] = [before_code.get("dirty"), now.state.dirty]
    return out


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()[:16]


def data_digest(data: WindowData, done: Collection[str]) -> dict[str, Any]:
    """What Stage A freezes of a window's data (spec §G.9: "data is frozen here").

    The candidates (NSN by the cutoff, S in the window) with their ``nsn_at``,
    ``at_news`` and PRIMARY eligibility reason; the minute units their paths
    request in either cell (the eligible stories' path sessions and the
    spare, the symbol's and SPY's); and which of those units are done. A
    rebuilt story, a re-run screen or a unit fetched since changes it.
    """
    rows: list[list[str | None]] = []
    units: set[tuple[str, date]] = set()
    for s in data.stories:
        if s.story_id not in data.candidates:
            continue
        nsn = s.nsn_at(Session.of(s.session).entry_cutoff)
        at = s.at_news()
        _, elig = data.judge(s, "primary")
        rows.append(
            [
                s.story_id,
                nsn.isoformat() if nsn else None,
                at.isoformat() if at else None,
                elig.reason,
            ]
        )
        if not elig.eligible:
            continue
        for cell in CELLS:
            if s.session > last_session(data.window, cell):
                continue
            days = path_days(s.session, cell.hold)
            spare = next_trading_day(days[-1])
            for d in [*days, *([spare] if spare <= data.end else [])]:
                units |= {(s.symbol, d), (SPY, d)}
    lines = sorted(minutes.unit(sym, d) for sym, d in units)
    done_lines = [u for u in lines if u in done]
    rows.sort(key=lambda r: str(r[0]))
    return {
        "candidates": len(rows),
        "candidates_sha": _sha(rows),
        "units": len(lines),
        "units_sha": _sha(lines),
        "done": len(done_lines),
        "done_sha": _sha(done_lines),
    }


def data_drift(before: Mapping[str, Any] | None, now: Mapping[str, Any]) -> dict[str, Any]:
    """Each digest field that differs, ``[before, now]`` (all of them without a baseline)."""
    old = before or {}
    return {k: [old.get(k), v] for k, v in now.items() if old.get(k) != v}


_FOUND: Final[dict[str, str]] = {
    "replaces": "{window} has run (quant_trials {value}); a rerun is an amendment",
    "partial": "{n} backtest row(s) of an unfinished run are on the ledger (quant_trials {value})",
    "code_diff": "the code differs from the code pinned for this trial ({value})",
    "data_diff": "the data differs from the data Stage A froze ({value})",
}


@dataclass(slots=True)
class Amendment:
    """What a step found that only an amendment may run past (spec §G.14)."""

    step: str
    reason: str | None
    window_role: str | None = None
    found: dict[str, Any] = field(default_factory=dict)
    written: list[int] = field(default_factory=list)

    def need(self, key: str, value: Any, **record: Any) -> None:
        """Note ``key`` (a :data:`_FOUND` key) and what the row must also carry."""
        self.found[key] = value
        self.found.update(record)

    def require(self) -> None:
        """:class:`H1Locked` when something was found and no reason was given."""
        why = [
            _FOUND[k].format(window=self.window_role or self.step, value=_brief(v), n=_count(v))
            for k, v in self.found.items()
            if k in _FOUND
        ]
        if why and not self.reason:
            raise H1Locked(
                f"{self.step}: " + "; ".join(why) + ": give the amendment's reason (--amend)"
            )

    async def write(self, engine: AsyncEngine, reg: Registration) -> int | None:
        """Record what was found (refusing without a reason); nothing when nothing was."""
        self.require()
        if not self.found:
            return None
        metrics: dict[str, Any] = {"step": self.step, "reason": self.reason, **self.found}
        if self.window_role is not None:
            metrics["window_role"] = self.window_role
        row = await _record(
            engine,
            kind="amendment",
            config=reg.prereg,
            window=self.window_role or self.step,
            metrics=metrics,
        )
        self.written.append(row)
        self.found = {}
        return row


def _count(value: Any) -> int:
    return len(value) if isinstance(value, list | tuple | dict) else 1


def _brief(value: Any) -> str:
    if isinstance(value, Mapping):
        return ", ".join(str(k) for k in value)
    if isinstance(value, list | tuple):
        shown = ", ".join(str(v) for v in value[:5])
        return shown + (", ..." if len(value) > 5 else "")
    return str(value)


async def _amendments(engine: AsyncEngine, reg: Registration) -> list[LedgerRow]:
    return await _rows(engine, kind="amendment", config_h=reg.hash)


async def check_code_now(
    engine: AsyncEngine, reg: Registration, amendment: Amendment, now: CodeNow
) -> None:
    """Note in ``amendment`` any difference from the pinned code (spec §G.14).

    The pinned code is the registration's, or the latest amendment's that
    recorded new code.
    """
    code, files = reg.code, reg.files
    for row in reversed(await _amendments(engine, reg)):
        if "files" in row.metrics:
            code, files = dict(row.metrics.get("code") or {}), dict(row.metrics["files"])
            break
    if drift := code_drift(code, files, now):
        amendment.need("code_diff", drift, code=now.as_dict(), files=now.files)


async def check_data_now(
    engine: AsyncEngine,
    reg: Registration,
    stage: LedgerRow,
    amendment: Amendment,
    data: WindowData,
) -> dict[str, Any]:
    """Note in ``amendment`` any difference from the data Stage A froze; returns the digest.

    The frozen data is Stage A's digest of the window, or the latest
    amendment's that recorded a new one.
    """
    now = data_digest(data, await minutes.done_units(engine))
    pinned: Mapping[str, Any] | None = (stage.metrics.get("data") or {}).get(data.window)
    for row in reversed(await _amendments(engine, reg)):
        if row.id > stage.id and data.window in (row.metrics.get("data") or {}):
            pinned = row.metrics["data"][data.window]
            break
    if drift := data_drift(pinned, now):
        merged = {**amendment.found.get("data", {}), data.window: now}
        diffs = {**amendment.found.get("data_diff", {}), data.window: drift}
        amendment.need("data_diff", diffs, data=merged)
    return now


async def _unfinished(
    engine: AsyncEngine,
    reg: Registration,
    *,
    feed: str,
    windows: Collection[str],
    after: int,
) -> list[int]:
    """Trial rows of ``feed`` in ``windows`` newer than ``after``: an unfinished run's."""
    hashes = [config_hash(trial_config(c, reg.hash, feed=feed)) for c in CELLS]
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT id FROM quant_trials WHERE kind = 'backtest' AND name = ANY(:n) "
                "AND config_hash = ANY(:h) AND metrics->>'window_role' = ANY(:w) AND id > :a "
                "ORDER BY id"
            ),
            {
                "n": [PREFIX + c.strategy for c in CELLS],
                "h": hashes,
                "w": list(windows),
                "a": after,
            },
        )
        return [int(r.id) for r in rows]


# ── Stage A (spec §G.9) ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CellCounts:
    """Stage A's counts for one cell in one window (no exit, no return)."""

    stories: int  # persisted stories (not noise only) with S in the cell's sessions
    nsn: int
    eligible: int
    reasons: dict[str, int]
    outcomes: int
    blocked_open: int
    triggered: int
    armed: int
    entries: int
    dates: int
    data_skips: int
    skips: dict[str, int]
    dismissed: dict[str, int]
    expired: dict[str, int]
    terminal: dict[str, int]
    years: float

    @property
    def per_year(self) -> float:
        return self.entries / self.years if self.years else 0.0

    @property
    def skip_share(self) -> float:
        return self.data_skips / self.eligible if self.eligible else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "stories": self.stories,
            "nsn": self.nsn,
            "eligible": self.eligible,
            "reasons": dict(sorted(self.reasons.items())),
            "outcomes": self.outcomes,
            "blocked_open": self.blocked_open,
            "triggered": self.triggered,
            "armed": self.armed,
            "entries": self.entries,
            "entry_dates": self.dates,
            "per_year": self.per_year,
            "data_skips": self.data_skips,
            "skip_share": self.skip_share,
            "skips": dict(sorted(self.skips.items())),
            "dismissed": dict(sorted(self.dismissed.items())),
            "expired": dict(sorted(self.expired.items())),
            "terminal": dict(sorted(self.terminal.items())),
        }


def _ny_day(t: datetime) -> date:
    return t.astimezone(MARKET_TZ).date()


def count_outcomes(
    outcomes: Sequence[StoryOutcome],
    *,
    stories: int,
    nsn: int,
    eligible: int,
    reasons: Mapping[str, int],
    years: float,
) -> CellCounts:
    """Stage A's counts from a run that stopped at the entries."""
    skips: Counter[str] = Counter()
    dismissed: Counter[str] = Counter()
    expired: Counter[str] = Counter()
    terminal: Counter[str] = Counter()
    blocked = triggered = armed = entries = 0
    days: set[date] = set()
    for o in outcomes:
        terminal[f"{o.terminal_state}/{o.reason}"] += 1
        if o.skip is not None:
            skips[o.skip] += 1
        if o.terminal_state == "DISMISSED":
            if o.reason == "blocked_open":
                blocked += 1
            else:
                dismissed[o.reason] += 1
        if o.terminal_state == "EXPIRED":
            expired[o.reason] += 1
        triggered += o.triggered_at is not None
        armed += o.armed_at is not None
        if o.entered and o.entry_bar_ts is not None:
            entries += 1
            days.add(_ny_day(o.entry_bar_ts))
    return CellCounts(
        stories=stories,
        nsn=nsn,
        eligible=eligible,
        reasons=dict(reasons),
        outcomes=len(outcomes),
        blocked_open=blocked,
        triggered=triggered,
        armed=armed,
        entries=entries,
        dates=len(days),
        data_skips=sum(n for reason, n in skips.items() if reason in DATA_SKIPS),
        skips=dict(skips),
        dismissed=dict(dismissed),
        expired=dict(expired),
        terminal=dict(terminal),
        years=years,
    )


def cell_eligible(train: CellCounts, validation: CellCounts, rule: CountRule | None = None) -> bool:
    """Validation >= 200 entries a year and >= 100 dates; train >= 500 and >= 100 dates."""
    r = rule if rule is not None else COUNT_RULE
    return (
        validation.per_year >= r.validation_per_year
        and validation.dates >= r.validation_dates
        and train.entries >= r.train_n
        and train.dates >= r.train_dates
    )


@dataclass(frozen=True, slots=True)
class StageA:
    counts: dict[tuple[WindowName, str], CellCounts]
    eligible: tuple[Cell, ...]
    budget_ok: bool
    verdict: str
    trial_id: int


async def _story_total(engine: AsyncEngine, first: date, last: date) -> int:
    async with engine.connect() as conn:
        n = await conn.scalar(
            text(
                "SELECT count(*) FROM news_stories WHERE builder_version = :v "
                "AND session BETWEEN :a AND :b AND type_close <> 'noise_only'"
            ),
            {"v": st.BUILDER_VERSION, "a": first, "b": last},
        )
    return int(n or 0)


def _stage_eligible(row: LedgerRow) -> tuple[Cell, ...]:
    keys = set(row.metrics.get("eligible") or [])
    return tuple(c for c in CELLS if c.key in keys)


async def _stage_row(engine: AsyncEngine, reg: Registration) -> LedgerRow:
    row = await _latest(engine, kind="stage-a", config_h=reg.hash)
    if row is None:
        raise H1Locked("Stage A has not run: `halal-trader events h1 stage-a`")
    if not row.metrics.get("budget_ok"):
        raise H1Locked(
            "Stage A's data skips exceed the budget: fix the data and rerun Stage A first"
        )
    return row


async def stage_a(
    engine: AsyncEngine,
    *,
    workers: int = 6,
    parallel: Parallel = None,
    amend: str | None = None,
) -> StageA:
    """Count both cells in both windows to their entries; record ``kind="stage-a"``.

    Refuses once a Stage A has passed its data budget (the data is frozen
    there) or once any window has results, and on code that differs from
    the pinned code unless ``amend`` gives the amendment's reason. The row
    holds each window's :func:`data_digest`, which every later step checks.
    """
    reg = await registration(engine)
    prior = await _latest(engine, kind="stage-a", config_h=reg.hash)
    if prior is not None and prior.metrics.get("budget_ok"):
        raise H1Locked(f"Stage A passed its data budget (quant_trials {prior.id}): data is frozen")
    if await _rows(engine, kind="window", config_h=reg.hash):
        raise H1Locked("returns exist already: Stage A cannot run again")
    now = current_code()
    amendment = Amendment("stage-a", amend)
    await check_code_now(engine, reg, amendment, now)
    await amendment.write(engine, reg)
    await _bootstrap(engine)
    counts: dict[tuple[WindowName, str], CellCounts] = {}
    digests: dict[str, dict[str, Any]] = {}
    for window in WINDOWS:
        data = await load_window(engine, window)
        for cell in CELLS:
            _, outcomes, inputs = await _simulate(
                engine,
                reg,
                data,
                cell,
                role="stage-a",
                cfg=SimConfig(),
                stop_at="entry",
                assume_full_hold=True,
                workers=workers,
                parallel=parallel,
                code_sha=now.sha,
            )
            counts[(window, cell.key)] = count_outcomes(
                outcomes,
                stories=await _story_total(engine, data.first, inputs.last),
                nsn=inputs.nsn,
                eligible=inputs.eligible,
                reasons=inputs.reasons,
                years=YEARS[window],
            )
        digests[window] = data_digest(data, await minutes.done_units(engine))
        del data
    budget_ok = all(c.skip_share <= SKIP_MAX for c in counts.values())
    eligible = tuple(
        c for c in CELLS if cell_eligible(counts[("train", c.key)], counts[("validation", c.key)])
    )
    if not budget_ok:
        verdict_ = "blocked: data skips"
    elif not eligible:
        verdict_ = "fail: insufficient events"
    else:
        verdict_ = "pass"
    trial_id = await _record(
        engine,
        kind="stage-a",
        config=reg.prereg,
        window=_windows_text(),
        metrics={
            "prereg_id": reg.id,
            "budget_ok": budget_ok,
            "eligible": [c.key for c in eligible],
            "cells": {f"{w}:{k}": c.as_dict() for (w, k), c in counts.items()},
            "data": digests,
            "code": now.as_dict(),
            "code_sha": now.sha,
        },
        verdict=verdict_,
        criterion=(
            f"data skips <= {SKIP_MAX:.0%} of eligible NSN stories per window; a cell is eligible "
            f"with validation >= {COUNT_RULE.validation_per_year:g} entries/yr and "
            f">= {COUNT_RULE.validation_dates} dates, train >= {COUNT_RULE.train_n} entries and "
            f">= {COUNT_RULE.train_dates} dates"
        ),
    )
    return StageA(counts, eligible, budget_ok, verdict_, trial_id)


# ── a window's statistics (spec §G.6, §G.8, §G.10, §G.11) ─────


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else math.nan


def entry_day(t: TradeRecord) -> date:
    return _ny_day(t.entry_bar_ts)


def _group(trades: Sequence[TradeRecord]) -> dict[str, Any]:
    values = [t.r_net_abn for t in trades]
    fit = stats.clustered_mean(values, [entry_day(t) for t in trades]) if trades else None
    return {
        "n": len(trades),
        "mean": _mean(values),
        "se": fit.se if fit is not None else None,
        "t": fit.t if fit is not None else None,
    }


def regimes(trades: Sequence[TradeRecord]) -> dict[str, dict[str, dict[str, Any]]]:
    """Reported, never gated (spec §G.11): years, breaks, family type, start, Tech."""

    def split(key: Any) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[TradeRecord]] = defaultdict(list)
        for t in trades:
            groups[str(key(t))].append(t)
        return {k: _group(v) for k, v in sorted(groups.items())}

    return {
        "year": split(lambda t: entry_day(t).year),
        "screen_break": split(lambda t: "after" if entry_day(t) >= SCREEN_BREAK else "before"),
        "template_break": split(lambda t: "after" if entry_day(t) >= TEMPLATE_BREAK else "before"),
        "family_type": split(lambda t: t.family_type),
        "start_case": split(lambda t: t.start_case),
        "tech": split(lambda t: "tech" if t.tech else "other"),
    }


@dataclass(frozen=True, slots=True)
class WindowStats:
    """One cell's result in one window: the tests of spec §G.10 and what is reported."""

    cell: Cell
    window: WindowName
    n: int
    dates: int
    mean: float
    cr1: stats.ClusteredMean | None
    nw: stats.NWMean | None  # MD3 only
    mean_cost15: float
    mean_beta: float
    mean_ex_covid: float | None  # train only
    unresolved: int
    p: float  # one-sided; MD3: max(p_CR1, p_NW)
    t1: bool
    t3: bool
    t4: bool
    t5: bool  # True in validation (no exclusion there)
    t6: bool
    t2: bool = False  # Holm, set by judge_window
    status: Status = "fail"
    regimes: Mapping[str, Any] = field(default_factory=dict)
    run_id: str = ""
    skips: Mapping[str, int] = field(default_factory=dict)
    trial_id: int | None = None
    dsr: float | None = None
    dsr_plus6: float | None = None
    n_trials: int | None = None
    book: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell.key,
            "window": self.window,
            "trades": self.n,
            "dates": self.dates,
            "mean_trade": self.mean,
            "t_cr1": self.cr1.t if self.cr1 is not None else None,
            "se_cr1": self.cr1.se if self.cr1 is not None else None,
            "deff": self.cr1.deff if self.cr1 is not None else None,
            "t_nw": self.nw.t if self.nw is not None else None,
            "mean_nw": self.nw.mean if self.nw is not None else None,
            "mean_cost15": self.mean_cost15,
            "mean_beta": self.mean_beta,
            "mean_ex_covid": self.mean_ex_covid,
            "unresolved": self.unresolved,
            "p": self.p,
            "tests": {
                "T1": self.t1,
                "T2": self.t2,
                "T3": self.t3,
                "T4": self.t4,
                "T5": self.t5,
                "T6": self.t6,
            },
            "status": self.status,
            "pass": self.passed,
            "regimes": dict(self.regimes),
            "run_id": self.run_id,
            "skips": dict(self.skips),
            "trial_id": self.trial_id,
            "dsr": self.dsr,
            "dsr_plus6": self.dsr_plus6,
            "n_trials": self.n_trials,
            "book": dict(self.book),
        }


def window_stats(
    cell: Cell,
    window: WindowName,
    trades: Sequence[TradeRecord],
    legs: Mapping[str, Sequence[Leg]],
    sessions: Sequence[date],
    *,
    run_id: str = "",
    skips: Mapping[str, int] | None = None,
) -> WindowStats:
    """Every test of spec §G.10 but Holm (:func:`judge_window` adds it), pure."""
    n = len(trades)
    values = [t.r_net_abn for t in trades]
    days = [entry_day(t) for t in trades]
    cr1 = stats.clustered_mean(values, days) if n else None
    nw: stats.NWMean | None = None
    if cell.hold > 1 and n:
        series = stats.calendar_series(
            {t.story_id: legs.get(t.story_id, ()) for t in trades}, sessions
        )
        nw = stats.newey_west_mean([x for _, x in series])
    mean = _mean(values)
    cost15 = _mean([t.r_net_abn - (COST_X - 1.0) * 2.0 * t.cost_bps / 1e4 for t in trades])
    beta = _mean([t.r_beta_adj for t in trades])
    ex_covid: float | None = None
    if window == "train":
        kept = [t.r_net_abn for t in trades if not COVID[0] <= entry_day(t) <= COVID[1]]
        ex_covid = _mean(kept)
    unresolved = sum(1 for t in trades if "unresolved" in t.flags)
    p = cr1.p_one_sided if cr1 is not None else 1.0
    if cell.hold > 1:
        p = max(p, nw.p_one_sided if nw is not None else 1.0)
    t1 = bool(
        n
        and mean > 0
        and cr1 is not None
        and cr1.t >= T_MIN
        and (cell.hold == 1 or (nw is not None and nw.mean > 0 and nw.t >= T_MIN))
    )
    return WindowStats(
        cell=cell,
        window=window,
        n=n,
        dates=len(set(days)),
        mean=mean,
        cr1=cr1,
        nw=nw,
        mean_cost15=cost15,
        mean_beta=beta,
        mean_ex_covid=ex_covid,
        unresolved=unresolved,
        p=p,
        t1=t1,
        t3=bool(n and cost15 > 0),
        t4=bool(n and beta > 0),
        t5=window != "train" or bool(ex_covid is not None and ex_covid > 0),
        t6=unresolved <= UNRESOLVED_MAX * n,
        regimes=regimes(trades),
        run_id=run_id,
        skips=dict(skips or {}),
    )


def judge_window(results: Mapping[Cell, WindowStats]) -> dict[Cell, WindowStats]:
    """Holm across ``results`` (the eligible cells, or the train passers), then each status.

    ``inconclusive`` when unresolved exits exceed 0.5% (T6), else ``pass``
    when T1-T5 hold (T5 only on train), else ``fail``.
    """
    if not results:
        return {}
    rejected = stats.holm({c.key: r.p for c, r in results.items()}, alpha=ALPHA)
    out: dict[Cell, WindowStats] = {}
    for cell, r in results.items():
        t2 = rejected[cell.key]
        status: Status
        if not r.t6:
            status = "inconclusive"
        elif r.t1 and t2 and r.t3 and r.t4 and r.t5:
            status = "pass"
        else:
            status = "fail"
        out[cell] = replace(r, t2=t2, status=status)
    return out


def decide(train: Mapping[str, str], validation: Mapping[str, str]) -> Status:
    """The verdict (spec §G.10) from each cell's status per window.

    ``pass`` iff some cell passes both; else ``inconclusive`` iff some window
    a cell ran was inconclusive (it could still have passed); else ``fail``.
    """
    if any(s == "pass" and validation.get(k) == "pass" for k, s in train.items()):
        return "pass"
    if "inconclusive" in train.values() or "inconclusive" in validation.values():
        return "inconclusive"
    return "fail"


def _book_summary(book: DailyBook) -> dict[str, Any]:
    return {
        "days": len(book.days),
        "first": book.days[0] if book.days else None,
        "last": book.days[-1] if book.days else None,
        "skipped_full": book.skipped_full,
        "skipped_loss_limit": book.skipped_loss_limit,
        "exposure": float(np.mean(book.exposure)) if len(book.exposure) else None,
    }


async def _record_cell(
    engine: AsyncEngine,
    reg: Registration,
    ws: WindowStats,
    trades: Sequence[TradeRecord],
    legs: Mapping[str, Sequence[Leg]],
    sessions: Sequence[date],
    *,
    feed: str,
) -> WindowStats:
    """The cell's trial row (``record_backtest`` on its daily book), DSR at n_trials + 6."""
    book = daily_book(trades, legs, sessions, slots=BOOK_SLOTS, daily_loss_limit=BOOK_LOSS_LIMIT)
    summary = _book_summary(book)
    extra = {
        "window_role": ws.window,
        "feed": feed,
        "trades": ws.n,
        "dates": ws.dates,
        "t_cr1": ws.cr1.t if ws.cr1 is not None else None,
        "t_nw": ws.nw.t if ws.nw is not None else None,
        "mean_trade": ws.mean,
        "mean_cost15": ws.mean_cost15,
        "mean_beta": ws.mean_beta,
        "mean_ex_covid": ws.mean_ex_covid,
        "unresolved": ws.unresolved,
        "p": ws.p,
        "holm": ws.t2,
        "status": ws.status,
        "pass": ws.passed,
        "skipped_full": book.skipped_full,
        "skipped_loss_limit": book.skipped_loss_limit,
        "exposure": summary["exposure"],
        "run_id": ws.run_id,
    }
    config = trial_config(ws.cell, reg.hash, feed=feed)
    assessment = None
    if book.days:
        assessment = await record_backtest(
            engine,
            strategy=ws.cell.strategy,
            config=config,
            days=book.days,
            returns=book.returns,
            benchmark=book.benchmark,
            extra=clean(extra),
            benchmark_label=BENCHMARK_LABEL,
        )
    if assessment is None:
        first = sessions[0] if sessions else None
        last = sessions[-1] if sessions else None
        trial_id = await _record(
            engine,
            name=PREFIX + ws.cell.strategy,
            kind="backtest",
            config=config,
            window=f"{first}..{last} vs {BENCHMARK_LABEL}",
            metrics={**extra, "degenerate": True},
            criterion=criterion_for(BENCHMARK_LABEL),
        )
        return replace(ws, trial_id=trial_id, book=summary)
    async with engine.connect() as conn:
        sr_var = await conn.scalar(
            text("SELECT (metrics->>'sr_variance')::float FROM quant_trials WHERE id = :i"),
            {"i": assessment.trial_id},
        )
    active = np.asarray(book.returns) - np.asarray(book.benchmark)
    plus6 = deflated_sharpe_ratio(
        active, assessment.n_trials + 6, float(sr_var) if sr_var is not None else None
    )
    return replace(
        ws,
        trial_id=assessment.trial_id,
        dsr=assessment.dsr,
        dsr_plus6=plus6,
        n_trials=assessment.n_trials,
        book=summary,
    )


def _trades(
    outcomes: Sequence[StoryOutcome],
) -> tuple[list[TradeRecord], dict[str, tuple[Leg, ...]]]:
    trades = [o.trade for o in outcomes if o.trade is not None]
    legs = {o.story_id: o.legs for o in outcomes if o.trade is not None}
    return trades, legs


async def _window_cells(
    engine: AsyncEngine, reg: Registration, window: WindowName
) -> tuple[Cell, ...]:
    """The cells ``window`` must run: Stage A's eligible ones, or the train passers."""
    stage = await _stage_row(engine, reg)
    if window == "train":
        return _stage_eligible(stage)
    train = await _latest_window(engine, reg, "train")
    if train is None:
        raise H1Locked("train has not run: validation runs on its passers only")
    passed = {k for k, c in (train.metrics.get("cells") or {}).items() if c.get("status") == "pass"}
    return tuple(c for c in CELLS if c.key in passed)


async def _run_cells(
    engine: AsyncEngine,
    reg: Registration,
    data: WindowData,
    cells: Sequence[Cell],
    *,
    role: str,
    cfg: SimConfig,
    workers: int,
    parallel: Parallel,
    code_sha: str | None,
) -> dict[Cell, WindowStats]:
    """Simulate, judge (Holm across ``cells``) and record each cell's trial."""
    raw: dict[Cell, WindowStats] = {}
    kept: dict[Cell, tuple[list[TradeRecord], dict[str, tuple[Leg, ...]]]] = {}
    for cell in cells:
        summary, outcomes, _ = await _simulate(
            engine,
            reg,
            data,
            cell,
            role=role,
            cfg=cfg,
            workers=workers,
            parallel=parallel,
            code_sha=code_sha,
        )
        trades, legs = _trades(outcomes)
        kept[cell] = (trades, legs)
        raw[cell] = window_stats(
            cell,
            data.window,
            trades,
            legs,
            data.sessions,
            run_id=summary.run_id,
            skips=summary.skips,
        )
    judged = judge_window(raw)
    out: dict[Cell, WindowStats] = {}
    for cell, ws in judged.items():
        trades, legs = kept[cell]
        out[cell] = await _record_cell(
            engine, reg, ws, trades, legs, data.sessions, feed=cfg.feed.name
        )
    return out


async def run_window(
    engine: AsyncEngine,
    window: WindowName,
    cells: Sequence[Cell] | None = None,
    *,
    cfg: SimConfig | None = None,
    workers: int = 6,
    parallel: Parallel = None,
    amend: str | None = None,
) -> dict[Cell, WindowStats]:
    """Run ``window`` on the cells it is owed and record each one (spec §G.10, §G.14).

    Train runs Stage A's eligible cells; validation the train passers.
    ``cells``, when given, must be exactly those. The window runs only with
    ``amend`` (the reason, written first as a ``kind="amendment"`` row; the
    earlier results stay on record) when it has run already, when an
    unfinished run left trial rows on the ledger, when the code differs from
    the pinned code, or when the window's data differs from what Stage A
    froze (:func:`data_digest`).
    """
    cfg = cfg if cfg is not None else SimConfig()
    reg = await registration(engine)
    stage = await _stage_row(engine, reg)
    owed = await _window_cells(engine, reg, window)
    if cells is not None and set(cells) != set(owed):
        raise ValueError(
            f"{window} runs {[c.key for c in owed]} (spec §G.10), not {[c.key for c in cells]}"
        )
    if not owed:
        raise H1Locked(f"no cell runs on {window}")
    now = current_code()
    amendment = Amendment(window, amend, window_role=window)
    previous = await _latest_window(engine, reg, window)
    if previous is not None:
        amendment.need("replaces", previous.id)
    covered = [
        r.id for r in await _amendments(engine, reg) if r.metrics.get("window_role") == window
    ]
    after = max([previous.id if previous is not None else 0, *covered])
    if partial := await _unfinished(engine, reg, feed=cfg.feed.name, windows=[window], after=after):
        amendment.need("partial", partial)
    await check_code_now(engine, reg, amendment, now)
    amendment.require()  # before the expensive load
    await _bootstrap(engine)
    data = await load_window(engine, window)
    digest = await check_data_now(engine, reg, stage, amendment, data)
    await amendment.write(engine, reg)
    results = await _run_cells(
        engine,
        reg,
        data,
        owed,
        role=window,
        cfg=cfg,
        workers=workers,
        parallel=parallel,
        code_sha=now.sha,
    )
    await _record(
        engine,
        kind="window",
        config=reg.prereg,
        window=f"{window} {data.first}..{data.end}",
        metrics={
            "window_role": window,
            "feed": cfg.feed.name,
            "prereg_id": reg.id,
            "loaded": dict(data.counts),
            "data": digest,
            "code": now.as_dict(),
            "code_sha": now.sha,
            "amendments": amendment.written,
            "cells": {c.key: ws.as_dict() for c, ws in results.items()},
        },
    )
    return results


# ── the verdict (spec §G.10) ──────────────────────────────────


def _statuses(row: LedgerRow | None) -> dict[str, str]:
    if row is None:
        return {}
    return {k: str(c.get("status")) for k, c in (row.metrics.get("cells") or {}).items()}


def _amendment_summary(row: LedgerRow) -> dict[str, Any]:
    keys = ("step", "reason", "window_role", "replaces", "partial", "code_diff", "data_diff")
    return {"id": row.id, **{k: row.metrics[k] for k in keys if k in row.metrics}}


async def verdict(engine: AsyncEngine, *, amend: str | None = None) -> Status:
    """Record H1's verdict (``kind="verdict"``, ``config=PREREG``) and return it.

    Train's latest row decides with validation's latest, which must be
    newer than train's when train has passers. With no train passer the
    verdict is decided on train alone: validation never runs then, and an
    older validation row (from before train was amended) is reported as
    ignored. The verdict is recorded once per state of the ledger: again
    only after a newer Stage A, window or amendment row (the holdout opens
    on the latest verdict only, ``loader.WindowGuard``). Every window row
    the verdict does not cite (the results an amendment replaced), every
    amendment and every earlier verdict are reported in its metrics.
    """
    reg = await registration(engine)
    stage = await _stage_row(engine, reg)
    now = current_code()
    amendment = Amendment("verdict", amend)
    await check_code_now(engine, reg, amendment, now)
    amendment.require()
    eligible = _stage_eligible(stage)
    train = await _latest_window(engine, reg, "train")
    validation = await _latest_window(engine, reg, "validation")
    cited: LedgerRow | None = None  # the validation row the verdict reads
    metrics: dict[str, Any] = {"prereg_id": reg.id, "stage_a": stage.id, "code": now.as_dict()}
    if not eligible:
        decision: Status = "fail"
        metrics["reason"] = "insufficient events"
    else:
        if train is None:
            raise H1Locked("train has not run")
        passers = [k for k, s in _statuses(train).items() if s == "pass"]
        if passers:
            if validation is None:
                raise H1Locked("validation has not run on the train passers")
            if validation.id < train.id:
                raise H1Locked("train was amended after validation: rerun validation")
            cited = validation
        elif validation is not None:
            metrics["validation_ignored"] = validation.id  # no passer: validation is not owed
        decision = decide(_statuses(train), _statuses(cited))
        metrics.update(
            train=train.metrics.get("cells"),
            validation=cited.metrics.get("cells") if cited is not None else None,
            train_row=train.id,
            validation_row=cited.id if cited is not None else None,
        )
    amendments = await _amendments(engine, reg)
    newest = max(r.id for r in (stage, train, validation, *amendments) if r is not None)
    before = await _latest(engine, kind="verdict", config_h=reg.hash)
    if before is not None and before.id > newest and not amendment.found:
        raise H1Locked(f"the verdict is recorded already (quant_trials {before.id})")
    await amendment.write(engine, reg)
    amendments = await _amendments(engine, reg)
    windows = await _rows(engine, kind="window", config_h=reg.hash)
    used = {train.id if train is not None else None, cited.id if cited is not None else None}
    metrics["replaced"] = [
        {"row": r.id, "window_role": r.metrics.get("window_role"), "cells": r.metrics.get("cells")}
        for r in windows
        if r.id not in used
    ]
    metrics["amendments"] = [_amendment_summary(r) for r in amendments]
    metrics["previous_verdicts"] = [
        {"id": r.id, "verdict": r.verdict}
        for r in await _rows(engine, kind="verdict", config_h=reg.hash)
    ]
    await _record(
        engine,
        kind="verdict",
        config=reg.prereg,
        window=_windows_text(),
        metrics=metrics,
        verdict=decision,
        criterion=CRITERION,
    )
    return decision


async def _after_verdict(engine: AsyncEngine, reg: Registration) -> LedgerRow:
    row = await _latest(engine, kind="verdict", config_h=reg.hash)
    if row is None:
        raise H1Locked("the verdict comes first: `halal-trader events h1 verdict`")
    return row


async def _cited(
    engine: AsyncEngine, reg: Registration, final: LedgerRow
) -> tuple[LedgerRow | None, LedgerRow | None]:
    """The train and validation rows ``final`` (the latest verdict) decided on.

    :class:`H1Locked` when a window ran after the verdict: it no longer
    describes the ledger, so it is recorded again first.
    """
    newer = [r for r in await _rows(engine, kind="window", config_h=reg.hash) if r.id > final.id]
    if newer:
        raise H1Locked(
            f"{newer[-1].metrics.get('window_role')} ran after the verdict "
            f"(quant_trials {newer[-1].id}): record the verdict again first"
        )
    rows: list[LedgerRow | None] = []
    for key in ("train_row", "validation_row"):
        row_id = final.metrics.get(key)
        rows.append(await _row(engine, reg, "window", int(row_id)) if row_id is not None else None)
    return rows[0], rows[1]


# ── sensitivities (spec §G.12) ────────────────────────────────


def summary_stats(
    cell: Cell,
    window: WindowName,
    trades: Sequence[TradeRecord],
    legs: Mapping[str, Sequence[Leg]],
    sessions: Sequence[date],
) -> dict[str, Any]:
    """A sensitivity's numbers: the window statistics without Holm, the book or a status."""
    ws = window_stats(cell, window, trades, legs, sessions)
    return {
        "trades": ws.n,
        "dates": ws.dates,
        "mean_trade": ws.mean,
        "t_cr1": ws.cr1.t if ws.cr1 is not None else None,
        "t_nw": ws.nw.t if ws.nw is not None else None,
        "mean_cost15": ws.mean_cost15,
        "mean_beta": ws.mean_beta,
        "mean_ex_covid": ws.mean_ex_covid,
        "unresolved": ws.unresolved,
    }


def subset_sensitivities(
    cell: Cell,
    window: WindowName,
    trades: Sequence[TradeRecord],
    legs: Mapping[str, Sequence[Leg]],
    sessions: Sequence[date],
) -> dict[str, dict[str, Any]]:
    """The sensitivities read from the base run's trades (no rerun), pure."""

    def of(subset: Sequence[TradeRecord]) -> dict[str, Any]:
        return summary_stats(cell, window, subset, legs, sessions)

    unresolved = [
        replace(t, r_net_abn=-1.0, r_beta_adj=-1.0) if "unresolved" in t.flags else t
        for t in trades
    ]
    return {
        "tech": {
            "tech": of([t for t in trades if t.tech]),
            "other": of([t for t in trades if not t.tech]),
        },
        "rank_split": {
            "lt300": of([t for t in trades if 0 <= t.rank < 300]),
            "300_999": of([t for t in trades if 300 <= t.rank < pit.MAX_RANK]),
        },
        "unresolved_minus100": {"all": of(unresolved)},
        "participation_le_10pct": {
            "kept": of([t for t in trades if not t.participation > PARTICIPATION_MAX]),
            "dropped": sum(1 for t in trades if t.participation > PARTICIPATION_MAX),
        },
    }


async def sensitivities(
    engine: AsyncEngine,
    *,
    workers: int = 6,
    parallel: Parallel = None,
    amend: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Every sensitivity on train (eligible cells) and validation (train passers).

    Each is a ``kind="sensitivity"`` row under ``research.news.h1.sens.<name>``
    with ``config={"prereg", "cell", "sensitivity"}`` and no Sharpe. The
    train passers are those of the train row the latest verdict cites.
    Changed code or data needs ``amend``, as for a window.
    """
    reg = await registration(engine)
    final = await _after_verdict(engine, reg)
    train_row, _ = await _cited(engine, reg, final)
    stage = await _stage_row(engine, reg)
    passed = _statuses(train_row)
    owed: dict[WindowName, tuple[Cell, ...]] = {
        "train": _stage_eligible(stage),
        "validation": tuple(c for c in CELLS if passed.get(c.key) == "pass"),
    }
    if not any(owed.values()):
        return {}
    now = current_code()
    amendment = Amendment("sensitivities", amend)
    await check_code_now(engine, reg, amendment, now)
    amendment.require()
    await _bootstrap(engine)
    out: dict[str, dict[str, Any]] = defaultdict(dict)

    async def keep(name: str, cell: Cell, window: WindowName, numbers: Mapping[str, Any]) -> None:
        out[name][f"{window}:{cell.key}"] = numbers
        await _record(
            engine,
            name=SENS_PREFIX + name,
            kind="sensitivity",
            config={
                "prereg": reg.hash,
                "cell": [cell.family, cell.variant, cell.hold],
                "sensitivity": name,
            },
            window=window,
            metrics={"window_role": window, "code_sha": now.sha, **numbers},
        )

    for window in WINDOWS:
        cells = owed[window]
        if not cells:
            continue
        data = await load_window(engine, window)
        await check_data_now(engine, reg, stage, amendment, data)
        await amendment.write(engine, reg)
        for cell in cells:
            role = f"sens:{window}"
            _, outcomes, _ = await _simulate(
                engine,
                reg,
                data,
                cell,
                role=role,
                cfg=SimConfig(),
                workers=workers,
                parallel=parallel,
                code_sha=now.sha,
            )
            trades, legs = _trades(outcomes)
            for name, numbers in subset_sensitivities(
                cell, window, trades, legs, data.sessions
            ).items():
                await keep(name, cell, window, numbers)
            runs: dict[str, tuple[SimConfig, Universe]] = {
                "broad": (SimConfig(), "broad"),
                "cost_x2": (SimConfig(cost="study_x2"), "primary"),
                "cost_surcharge": (SimConfig(cost="surcharge"), "primary"),
            }
            for name, (cfg, universe) in runs.items():
                _, outcomes, _ = await _simulate(
                    engine,
                    reg,
                    data,
                    cell,
                    role=f"{role}:{name}",
                    cfg=cfg,
                    universe=universe,
                    workers=workers,
                    parallel=parallel,
                    code_sha=now.sha,
                )
                trades, legs = _trades(outcomes)
                await keep(
                    name, cell, window, summary_stats(cell, window, trades, legs, data.sessions)
                )
        del data
        for name, lag in NEWS_LAGS.items():
            lagged = await load_window(engine, window, lag=lag)
            for cell in cells:
                _, outcomes, _ = await _simulate(
                    engine,
                    reg,
                    lagged,
                    cell,
                    role=f"sens:{window}:{name}",
                    cfg=SimConfig(),
                    workers=workers,
                    parallel=parallel,
                    code_sha=now.sha,
                )
                trades, legs = _trades(outcomes)
                await keep(
                    name, cell, window, summary_stats(cell, window, trades, legs, lagged.sessions)
                )
            del lagged
    return dict(out)


# ── the implementability trial (spec §G.13) ───────────────────


async def implementability(
    engine: AsyncEngine,
    *,
    workers: int = 6,
    parallel: Parallel = None,
    amend: str | None = None,
) -> dict[WindowName, dict[Cell, WindowStats]]:
    """The cells that passed H1, on the delayed feed, both windows, by the same rule.

    A counted trial per cell (``trial_config(..., feed="sip-delayed")``);
    validation runs this trial's train passers. A ``kind="implementability"``
    row (config: the registration's hash and the feed) records the outcome.
    The cells are those that pass both window rows the latest verdict cites;
    nothing runs when none did. An unfinished earlier run (delayed-feed
    trial rows and no outcome row), changed code or changed data needs
    ``amend``, as for a window.
    """
    reg = await registration(engine)
    final = await _after_verdict(engine, reg)
    feed = SIP_DELAYED.name
    summary_config = {"prereg": reg.hash, "feed": feed}
    async with engine.connect() as conn:
        done = await conn.scalar(
            text(
                "SELECT count(*) FROM quant_trials WHERE name = :n AND kind = 'implementability' "
                "AND config_hash = :h"
            ),
            {"n": NAME, "h": config_hash(summary_config)},
        )
    if done:
        raise H1Locked("the implementability trial has run already")
    train_row, validation_row = await _cited(engine, reg, final)
    t_status, v_status = _statuses(train_row), _statuses(validation_row)
    cells = tuple(
        c for c in CELLS if t_status.get(c.key) == "pass" and v_status.get(c.key) == "pass"
    )
    if not cells:
        logger.info("h1 implementability: no cell passed H1 (verdict %s)", final.verdict)
        return {}
    stage = await _stage_row(engine, reg)
    now = current_code()
    amendment = Amendment("implementability", amend)
    covered = [
        r.id for r in await _amendments(engine, reg) if r.metrics.get("step") == "implementability"
    ]
    if partial := await _unfinished(
        engine, reg, feed=feed, windows=WINDOWS, after=max([0, *covered])
    ):
        amendment.need("partial", partial)
    await check_code_now(engine, reg, amendment, now)
    amendment.require()
    await _bootstrap(engine)
    cfg = SimConfig(feed=SIP_DELAYED)
    results: dict[WindowName, dict[Cell, WindowStats]] = {}
    owed: tuple[Cell, ...] = cells
    for window in WINDOWS:
        if not owed:
            break
        data = await load_window(engine, window)
        await check_data_now(engine, reg, stage, amendment, data)
        await amendment.write(engine, reg)
        results[window] = await _run_cells(
            engine,
            reg,
            data,
            owed,
            role=f"implementability:{window}",
            cfg=cfg,
            workers=workers,
            parallel=parallel,
            code_sha=now.sha,
        )
        owed = tuple(c for c, ws in results[window].items() if ws.passed)
        del data
    decision = decide(
        {c.key: ws.status for c, ws in results.get("train", {}).items()},
        {c.key: ws.status for c, ws in results.get("validation", {}).items()},
    )
    await _record(
        engine,
        kind="implementability",
        config=summary_config,
        window=_windows_text(),
        metrics={
            "prereg_id": reg.id,
            "verdict_row": final.id,
            "code": now.as_dict(),
            "code_sha": now.sha,
            "amendments": amendment.written,
            "cells": {
                f"{w}:{c.key}": ws.as_dict() for w, by in results.items() for c, ws in by.items()
            },
        },
        verdict=decision,
        criterion="the H1 pass rule on the delayed SIP feed (spec §G.13)",
    )
    return results


__all__ = [
    "BENCHMARK_LABEL",
    "CELLS",
    "CONTEXT_DEVIATIONS",
    "COUNT_RULE",
    "CRITERION",
    "NAME",
    "REPORTED_CHECKS",
    "REQUIRED_CHECKS",
    "REQUIRED_GATES",
    "SENSITIVITIES",
    "TAG",
    "Amendment",
    "Carrier",
    "Cell",
    "CellCounts",
    "Check",
    "CodeNow",
    "CodeState",
    "CountRule",
    "H1Locked",
    "MarketProbe",
    "Preconditions",
    "Registration",
    "RegistrationRefused",
    "RunInputs",
    "StageA",
    "StoriesStale",
    "WindowData",
    "WindowStats",
    "build_prereg",
    "cell_eligible",
    "check_code",
    "check_code_now",
    "check_data_now",
    "check_edits_d7",
    "clean",
    "code_drift",
    "code_state",
    "count_outcomes",
    "current_code",
    "data_digest",
    "data_drift",
    "decide",
    "existing_registration",
    "file_shas",
    "implementability",
    "judge_gates",
    "judge_window",
    "last_session",
    "load_window",
    "preconditions",
    "prereg",
    "probe_iex_d8",
    "register",
    "registration",
    "relag",
    "run_inputs",
    "run_window",
    "sensitivities",
    "stage_a",
    "subset_sensitivities",
    "summary_stats",
    "trial_config",
    "verdict",
    "window_span",
    "window_stats",
]
