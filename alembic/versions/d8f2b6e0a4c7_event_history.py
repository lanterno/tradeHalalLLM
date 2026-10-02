"""backfill_progress, eps_facts: the event store's history (S2 Phase A)

Revision ID: d8f2b6e0a4c7
Revises: c3e7a1d5f9b2
Create Date: 2026-10-02 09:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d8f2b6e0a4c7"
down_revision: Union[str, Sequence[str], None] = "c3e7a1d5f9b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "backfill_progress",
        sa.Column("task", S(), nullable=False),
        sa.Column("unit", S(), nullable=False),
        sa.Column("items", sa.Integer(), nullable=False),
        sa.Column("done_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("task", "unit"),
    )
    op.create_table(
        "eps_facts",
        sa.Column("cik", sa.Integer(), nullable=False),
        sa.Column("concept", S(), nullable=False),
        sa.Column("start", sa.Date(), nullable=False),
        sa.Column("end", sa.Date(), nullable=False),
        sa.Column("val", sa.Float(), nullable=False),
        sa.Column("form", S(), nullable=False),
        sa.Column("fp", S(), nullable=True),
        sa.Column("fy", sa.Integer(), nullable=True),
        sa.Column("filed", sa.Date(), nullable=False),
        sa.Column("accn", S(), nullable=False),
        sa.PrimaryKeyConstraint("cik", "concept", "end", "accn", "start", name="pk_eps_facts"),
    )
    op.create_index("ix_eps_facts_cik_filed", "eps_facts", ["cik", "filed"])


def downgrade() -> None:
    op.drop_index("ix_eps_facts_cik_filed", table_name="eps_facts")
    op.drop_table("eps_facts")
    op.drop_table("backfill_progress")
