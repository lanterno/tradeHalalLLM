"""The story builder's tables: what a story row holds, and what the backup leaves out."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import NewsStory, StoryAlias
from halal_trader.web.operations import NOT_DUMPED


def _story(**overrides: object) -> NewsStory:
    at = datetime(2024, 5, 2, 13, 0, tzinfo=UTC)
    fields: dict[str, object] = {
        "builder_version": "stories-v1",
        "story_id": "AAPL:2024-05-02",
        "symbol": "AAPL",
        "session": date(2024, 5, 2),
        "start_case": "in",
        "detect_at": at,
        "type_detect": "analyst_downgrade",
        "type_close": "analyst_downgrade",
        "follower_close": False,
        "n_items": 1,
        "n_distinct": 1,
        "items": [{"event_id": 7, "at": at.isoformat(), "itype": "analyst_downgrade"}],
        "flags": {"nsym_unknown": False},
    }
    return NewsStory(**(fields | overrides))


async def test_a_story_round_trips_with_its_optional_fields_empty(engine: AsyncEngine) -> None:
    async with AsyncSession(engine) as session:
        session.add(_story())
        session.add(_story(story_id="AAPL:2024-05-03", session=date(2024, 5, 3), flags=["x"]))
        await session.commit()
        rows = (await session.exec(select(NewsStory).order_by(NewsStory.session))).all()
    assert [r.story_id for r in rows] == ["AAPL:2024-05-02", "AAPL:2024-05-03"]
    first = rows[0]
    assert first.nsn_at is None and first.at_news is None and first.parent is None
    assert first.family_ever is None
    assert first.items[0]["event_id"] == 7
    assert first.flags == {"nsym_unknown": False}
    assert rows[1].flags == ["x"]


async def test_a_story_is_one_per_builder_version(engine: AsyncEngine) -> None:
    async with AsyncSession(engine) as session:
        session.add(_story())
        session.add(_story(builder_version="stories-v2"))
        await session.commit()
        session.add(_story())
        with pytest.raises(IntegrityError):
            await session.commit()


async def test_an_alias_is_keyed_by_all_four_columns(engine: AsyncEngine) -> None:
    async with AsyncSession(engine) as session:
        session.add(
            StoryAlias(builder_version="stories-v1", symbol="AAPL", alias="Apple", source="name")
        )
        session.add(
            StoryAlias(builder_version="stories-v1", symbol="AAPL", alias="Apple", source="learned")
        )
        await session.commit()
        assert len((await session.exec(select(StoryAlias))).all()) == 2


def test_stories_are_left_out_of_the_backup_and_aliases_are_kept() -> None:
    assert "news_stories" in NOT_DUMPED
    assert "story_aliases" not in NOT_DUMPED
