"""Drop the tables nothing reads or writes.

Eight tables left behind by features that were deleted or never wired:
ml_artefacts and prompt_genomes (the dormant ml/ stack and prompt
evolution), regime_snapshots, replay_snapshots and thesis_tags (the replay
and self-review stores), research_jobs (the deleted admin plane),
runtime_config (its repo went with the infra cleanup) and pair_pauses (the
crypto bot's per-pair pause). No code reads or writes any of them.
Downgrade recreates them empty.

Revision ID: 69402fab5aad
Revises: 732583ae4034
Create Date: 2026-10-08 05:55:19.514799

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "69402fab5aad"
down_revision: str | Sequence[str] | None = "732583ae4034"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_table("ml_artefacts")
    op.drop_table("pair_pauses")
    op.drop_table("prompt_genomes")
    op.drop_table("regime_snapshots")
    op.drop_table("replay_snapshots")
    op.drop_table("research_jobs")
    op.drop_table("runtime_config")
    op.drop_table("thesis_tags")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "ml_artefacts",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column("name", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("version", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("payload_format", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("payload_bytes", postgresql.BYTEA(), autoincrement=False, nullable=True),
        sa.Column(
            "payload_json",
            postgresql.JSONB(astext_type=sa.Text()),
            autoincrement=False,
            nullable=True,
        ),
        sa.Column("sklearn_version", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("feature_hash", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "created_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("ml_artefacts_pkey")),
    )
    op.create_index(op.f("ix_ml_artefacts_name"), "ml_artefacts", ["name"], unique=False)
    op.create_table(
        "pair_pauses",
        sa.Column("pair", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("set_by", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column(
            "set_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.Column("reason", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint("pair", name=op.f("pair_pauses_pkey")),
    )
    op.create_table(
        "prompt_genomes",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.Column("name", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "genome", postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=False
        ),
        sa.Column(
            "fitness", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False
        ),
        sa.Column("n_cycles", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column(
            "parent_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "promoted_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True
        ),
        sa.Column("notes", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("prompt_genomes_pkey")),
    )
    op.create_index(op.f("ix_prompt_genomes_name"), "prompt_genomes", ["name"], unique=False)
    op.create_index(
        op.f("ix_prompt_genomes_created_at"), "prompt_genomes", ["created_at"], unique=False
    )
    op.create_table(
        "regime_snapshots",
        sa.Column("date", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "features_json",
            postgresql.JSONB(astext_type=sa.Text()),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "outcome_pnl_pct",
            sa.DOUBLE_PRECISION(precision=53),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "outcome_win_rate",
            sa.DOUBLE_PRECISION(precision=53),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("outcome_n_trades", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("note", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "created_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.Column(
            "embedding",
            Vector(10),
            autoincrement=False,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("date", name=op.f("regime_snapshots_pkey")),
    )
    op.create_table(
        "replay_snapshots",
        sa.Column("cycle_id", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "created_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.Column("market", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("schema_version", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column(
            "payload", postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=False
        ),
        sa.PrimaryKeyConstraint("cycle_id", name=op.f("replay_snapshots_pkey")),
    )
    op.create_index(
        op.f("ix_replay_snapshots_created_at"), "replay_snapshots", ["created_at"], unique=False
    )
    op.create_table(
        "research_jobs",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "timestamp", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.Column("kind", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("name", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column(
            "params", postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=False
        ),
        sa.Column("status", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "result", postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True
        ),
        sa.Column("error", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column(
            "finished_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True
        ),
        sa.Column("pinned", sa.BOOLEAN(), autoincrement=False, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("research_jobs_pkey")),
    )
    op.create_table(
        "runtime_config",
        sa.Column("key", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "value", postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=False
        ),
        sa.Column("set_by", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column(
            "set_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.PrimaryKeyConstraint("key", name=op.f("runtime_config_pkey")),
    )
    op.create_table(
        "thesis_tags",
        sa.Column("trade_id", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("tag", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "confidence", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False
        ),
        sa.Column("reason", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column("method", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "set_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.PrimaryKeyConstraint("trade_id", name=op.f("thesis_tags_pkey")),
    )
