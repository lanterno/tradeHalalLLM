"""entry_type column on trades

Revision ID: 6c4e9bdf2810
Revises: 552a7d6e5862
Create Date: 2026-05-22 11:00:00.000000

Tags each Trade with how it entered the book — currently:

  * "scheduled"        — opened by the 15-min cron cycle (default
                         when unspecified, for backward compat)
  * "reactor_momentum" — opened by the news-momentum reactor on a
                         high-confidence news + price confluence

Used by the slow-out discipline (operator memory:
strategy-fast-in-slow-out): positions tagged "reactor_momentum" are
LLM-untouchable on the SELL side. Only the monitor's rule-based
exit (HWM trailing stop or N-bar trend break) can close them.

Nullable + no default so the column is purely additive — existing
rows stay None and the strategy code treats None as "scheduled".
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "6c4e9bdf2810"
down_revision: str | Sequence[str] | None = "552a7d6e5862"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column("entry_type", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("trades", "entry_type")
