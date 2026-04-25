"""add indicator_snapshots, strategy_adjustments, and llm_decisions.thinking

Revision ID: e4f5a6b7c8d9
Revises: d3e4f5a6b7c8
Create Date: 2026-04-25 12:00:00.000000

Catch-up migration for tables and columns that were previously created
implicitly via SQLModel.metadata.create_all + ALTER TABLE in init_db.
Idempotent — safe to apply on databases adopted from the create_all path.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

revision: str = "e4f5a6b7c8d9"
down_revision: Union[str, None] = "d3e4f5a6b7c8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": name},
    )
    return result.first() is not None


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(sa.text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result)


def upgrade() -> None:
    if not _table_exists("indicator_snapshots"):
        op.create_table(
            "indicator_snapshots",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("trade_id", sa.Integer(), nullable=False),
            sa.Column("pair", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.Column("rsi_14", sa.Float(), nullable=True),
            sa.Column("macd_histogram", sa.Float(), nullable=True),
            sa.Column("volume_ratio", sa.Float(), nullable=True),
            sa.Column("atr_14", sa.Float(), nullable=True),
            sa.Column("bb_position", sa.Float(), nullable=True),
            sa.Column("price_change_5m", sa.Float(), nullable=True),
            sa.Column("ema_9", sa.Float(), nullable=True),
            sa.Column("ema_21", sa.Float(), nullable=True),
            sa.Column("vwap", sa.Float(), nullable=True),
            sa.Column("label", sa.Integer(), nullable=True),
            sa.Column("return_pct", sa.Float(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_indicator_snapshots_trade_id",
            "indicator_snapshots",
            ["trade_id"],
        )

    if not _table_exists("strategy_adjustments"):
        op.create_table(
            "strategy_adjustments",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.Column("parameter", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("old_value", sa.Float(), nullable=True),
            sa.Column("new_value", sa.Float(), nullable=False),
            sa.Column("reasoning", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _column_exists("llm_decisions", "thinking"):
        op.add_column(
            "llm_decisions", sa.Column("thinking", sa.String(), nullable=True)
        )


def downgrade() -> None:
    if _column_exists("llm_decisions", "thinking"):
        op.drop_column("llm_decisions", "thinking")
    if _table_exists("strategy_adjustments"):
        op.drop_table("strategy_adjustments")
    if _table_exists("indicator_snapshots"):
        op.drop_index(
            "ix_indicator_snapshots_trade_id", table_name="indicator_snapshots"
        )
        op.drop_table("indicator_snapshots")
