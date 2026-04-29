"""ml_artefacts

Revision ID: efe56a359835
Revises: d48f82af284b
Create Date: 2026-04-29 06:06:08.579450

Wave K — versioned model blob table replaces ``models/*.pkl`` so
the bot's learned state replicates with the DB and rolls back
atomically alongside the schema.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "efe56a359835"
down_revision: Union[str, Sequence[str], None] = "d48f82af284b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ml_artefacts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("payload_format", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("payload_bytes", sa.LargeBinary(), nullable=True),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("sklearn_version", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("feature_hash", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_ml_artefacts_name"), "ml_artefacts", ["name"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_ml_artefacts_name"), table_name="ml_artefacts")
    op.drop_table("ml_artefacts")
