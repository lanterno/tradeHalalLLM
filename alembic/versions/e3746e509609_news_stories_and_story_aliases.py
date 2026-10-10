"""news_stories and story_aliases: the news engine's story builder (stories-v1).

``news_stories`` holds one row per (builder version, symbol, reaction
session): the story the builder made of that session's admitted items. It is
derived from the event store and rebuilt by ``events stories build``, so the
nightly backup leaves its rows out.

``story_aliases`` holds the names the entity check accepts for each symbol
(market and Alpaca names, Benzinga's own company slot, overrides, tickers).
It is pinned by the pre-registration (``alias_sha``) and cannot be rebuilt
later -- ``market_assets`` changes -- so it is backed up.

Revision ID: e3746e509609
Revises: d8f0b2c4e6a7
Create Date: 2026-10-10 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e3746e509609"
down_revision: str | Sequence[str] | None = "d8f0b2c4e6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "news_stories",
        sa.Column("builder_version", S(), nullable=False),
        sa.Column("story_id", S(), nullable=False),
        sa.Column("symbol", S(), nullable=False),
        sa.Column("session", sa.Date(), nullable=False),
        sa.Column("start_case", S(), nullable=False),
        sa.Column("detect_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("nsn_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("at_news", sa.DateTime(timezone=True), nullable=True),
        sa.Column("type_detect", S(), nullable=False),
        sa.Column("type_close", S(), nullable=False),
        sa.Column("family_ever", S(), nullable=True),
        sa.Column("follower_close", sa.Boolean(), nullable=False),
        sa.Column("parent", S(), nullable=True),
        sa.Column("n_items", sa.Integer(), nullable=False),
        sa.Column("n_distinct", sa.Integer(), nullable=False),
        sa.Column("items", JSONB, nullable=False),
        sa.Column("flags", JSONB, nullable=False),
        sa.PrimaryKeyConstraint("builder_version", "story_id"),
    )
    op.create_index("ix_news_stories_symbol_session", "news_stories", ["symbol", "session"])
    op.create_index(
        "ix_news_stories_family_ever_session", "news_stories", ["family_ever", "session"]
    )
    op.create_table(
        "story_aliases",
        sa.Column("builder_version", S(), nullable=False),
        sa.Column("symbol", S(), nullable=False),
        sa.Column("alias", S(), nullable=False),
        sa.Column("source", S(), nullable=False),
        sa.PrimaryKeyConstraint("builder_version", "symbol", "alias", "source"),
    )


def downgrade() -> None:
    op.drop_table("story_aliases")
    op.drop_index("ix_news_stories_family_ever_session", table_name="news_stories")
    op.drop_index("ix_news_stories_symbol_session", table_name="news_stories")
    op.drop_table("news_stories")
