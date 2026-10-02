"""events, event_scores, event_labels: the event store (S2 Phase 0)

Revision ID: c3e7a1d5f9b2
Revises: b7d1f3a9c5e2
Create Date: 2026-10-02 07:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3e7a1d5f9b2"
down_revision: Union[str, Sequence[str], None] = "b7d1f3a9c5e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("source", S(), nullable=False),
        sa.Column("source_id", S(), nullable=False),
        sa.Column("kind", S(), nullable=False),
        sa.Column("symbol", S(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=True),
        sa.UniqueConstraint("source", "source_id", "symbol", name="uq_events_source_item"),
    )
    op.create_index("ix_events_symbol_published", "events", ["symbol", "published_at"])
    op.create_index("ix_events_kind_published", "events", ["kind", "published_at"])
    op.create_table(
        "event_scores",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("event_id", sa.BigInteger(), sa.ForeignKey("events.id"), nullable=False),
        sa.Column("scorer", S(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("tag", S(), nullable=True),
        sa.Column("rationale", S(), nullable=True),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("event_id", "scorer", name="uq_event_scores_event_scorer"),
    )
    op.create_table(
        "event_labels",
        sa.Column("event_id", sa.BigInteger(), sa.ForeignKey("events.id"), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("ret", sa.Float(), nullable=False),
        sa.Column("abn_ret", sa.Float(), nullable=False),
        sa.Column("labeled_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("event_id", "horizon"),
    )


def downgrade() -> None:
    op.drop_table("event_labels")
    op.drop_table("event_scores")
    op.drop_index("ix_events_kind_published", table_name="events")
    op.drop_index("ix_events_symbol_published", table_name="events")
    op.drop_table("events")
