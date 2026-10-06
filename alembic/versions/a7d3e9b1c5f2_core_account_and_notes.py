"""core_orders.account, core_runs.account and core_runs.notes

The core's paper and live accounts are recorded apart (portfolio/core_account.py):
every existing row is the paper account's, so the column defaults to 'core'.
A run's notes (what was scaled, left out, priced from a fallback) are kept with it.

Revision ID: a7d3e9b1c5f2
Revises: f8c2a6d4e0b7
Create Date: 2026-10-06 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7d3e9b1c5f2"
down_revision: str | Sequence[str] | None = "f8c2a6d4e0b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    for table in ("core_orders", "core_runs"):
        op.add_column(table, sa.Column("account", S(), nullable=False, server_default="core"))
    op.add_column(
        "core_runs",
        sa.Column("notes", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    )
    op.create_index("ix_core_runs_account_run_on", "core_runs", ["account", "run_on"])


def downgrade() -> None:
    op.drop_index("ix_core_runs_account_run_on", table_name="core_runs")
    op.drop_column("core_runs", "notes")
    # Live rows have no place in the older schema, which knew one account only.
    op.execute("DELETE FROM core_orders WHERE account <> 'core'")
    op.execute("DELETE FROM core_runs WHERE account <> 'core'")
    for table in ("core_orders", "core_runs"):
        op.drop_column(table, "account")
