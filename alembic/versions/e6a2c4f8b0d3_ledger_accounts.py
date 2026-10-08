"""broker_activities / broker_equity per account: the core's own ledger

Revision ID: e6a2c4f8b0d3
Revises: d4f0b2c8e6a1
Create Date: 2026-10-05 02:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e6a2c4f8b0d3"
down_revision: str | Sequence[str] | None = "d4f0b2c8e6a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    # Existing rows are the day-trader's ("paper", the name the purification
    # ledger already uses for it).
    op.add_column(
        "broker_activities", sa.Column("account", S(), nullable=False, server_default="paper")
    )
    op.create_index(
        "ix_broker_activities_account_time", "broker_activities", ["account", "transaction_time"]
    )
    op.add_column(
        "broker_equity", sa.Column("account", S(), nullable=False, server_default="paper")
    )
    op.drop_constraint("broker_equity_pkey", "broker_equity", type_="primary")
    op.create_primary_key("broker_equity_pkey", "broker_equity", ["account", "day"])


def downgrade() -> None:
    op.drop_constraint("broker_equity_pkey", "broker_equity", type_="primary")
    op.execute("DELETE FROM broker_equity WHERE account <> 'paper'")
    op.create_primary_key("broker_equity_pkey", "broker_equity", ["day"])
    op.drop_column("broker_equity", "account")
    op.drop_index("ix_broker_activities_account_time", table_name="broker_activities")
    op.execute("DELETE FROM broker_activities WHERE account <> 'paper'")
    op.drop_column("broker_activities", "account")
