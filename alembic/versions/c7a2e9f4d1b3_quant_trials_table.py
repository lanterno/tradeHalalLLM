"""quant_trials table — the anti-overfitting trials ledger

Phase 0 of docs/QUANT_PREDICTION_ROADMAP.md: every evaluated quant variant
(including failures) records a row so the Deflated Sharpe Ratio gets an
honest trial count and verdicts have a durable, pre-registered home.

Revision ID: c7a2e9f4d1b3
Revises: b3f1c7d2a9e4
Create Date: 2026-07-13 10:15:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7a2e9f4d1b3"
down_revision: Union[str, Sequence[str], None] = "b3f1c7d2a9e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — create quant_trials."""
    op.create_table(
        "quant_trials",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("kind", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("config_hash", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("config", JSONB, nullable=True),
        sa.Column("window", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("metrics", JSONB, nullable=True),
        sa.Column("criterion", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("verdict", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.create_index("ix_quant_trials_name", "quant_trials", ["name"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_quant_trials_name", table_name="quant_trials")
    op.drop_table("quant_trials")
