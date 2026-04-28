"""timezone-aware datetime columns

Revision ID: f18549dec00a
Revises: 76aa7ebe3b32
Create Date: 2026-04-28 22:15:37.264706

The initial Postgres migration created every datetime column as
``timestamp without time zone`` (Postgres default for SQLAlchemy's
``DateTime()``), but the SQLModel declarations all use
``sa.DateTime(timezone=True)``. Alembic autogenerate detects the
diff on every revision; contributors learn to ignore it and
hand-edit, which is one missed real change away from a deletion-
by-confusion.

One-shot migration: ALTER every drifted column to TIMESTAMPTZ. The
existing naïve values are interpreted as UTC (every writer in the
codebase uses ``datetime.now(UTC)``), so the cast is lossless.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f18549dec00a"
down_revision: Union[str, Sequence[str], None] = "76aa7ebe3b32"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_UTC_COLUMNS: list[tuple[str, str, bool]] = [
    # (table, column, nullable)
    ("crypto_halal_cache", "updated_at", False),
    ("crypto_trades", "timestamp", False),
    ("crypto_trades", "closed_at", True),
    ("crypto_trades", "submitted_at", True),
    ("crypto_trades", "filled_at", True),
    ("halal_cache", "updated_at", False),
    ("halal_screenings", "timestamp", False),
    ("indicator_snapshots", "timestamp", False),
    ("kill_switch", "set_at", True),
    ("llm_decisions", "timestamp", False),
    ("pair_pauses", "set_at", False),
    ("purification_entries", "timestamp", False),
    ("purification_entries", "paid_at", True),
    ("rag_rationales", "timestamp", False),
    ("reconciliation_log", "timestamp", False),
    ("regret_records", "closed_at", False),
    ("research_jobs", "timestamp", False),
    ("research_jobs", "finished_at", True),
    ("runtime_config", "set_at", False),
    ("strategy_adjustments", "timestamp", False),
    ("thesis_tags", "set_at", False),
    ("trades", "timestamp", False),
    ("trades", "submitted_at", True),
    ("trades", "filled_at", True),
    ("trades", "closed_at", True),
    ("web_actions", "timestamp", False),
]


def upgrade() -> None:
    for table, column, nullable in _UTC_COLUMNS:
        op.alter_column(
            table,
            column,
            existing_type=sa.TIMESTAMP(),
            type_=sa.DateTime(timezone=True),
            existing_nullable=nullable,
            postgresql_using=f"\"{column}\" AT TIME ZONE 'UTC'",
        )


def downgrade() -> None:
    for table, column, nullable in reversed(_UTC_COLUMNS):
        op.alter_column(
            table,
            column,
            existing_type=sa.DateTime(timezone=True),
            type_=sa.TIMESTAMP(),
            existing_nullable=nullable,
            postgresql_using=f"\"{column}\" AT TIME ZONE 'UTC'",
        )
