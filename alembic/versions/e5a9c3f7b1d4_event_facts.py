"""event_facts: structured facts extracted from events (earnings vs consensus)

Revision ID: e5a9c3f7b1d4
Revises: d8f2b6e0a4c7
Create Date: 2026-10-02 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5a9c3f7b1d4"
down_revision: str | Sequence[str] | None = "d8f2b6e0a4c7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "event_facts",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("event_id", sa.BigInteger(), sa.ForeignKey("events.id"), nullable=False),
        sa.Column("extractor", S(), nullable=False),
        sa.Column("kind", S(), nullable=False),
        sa.Column("fields", JSONB, nullable=False),
    )
    op.create_index("ix_event_facts_event_extractor", "event_facts", ["event_id", "extractor"])
    op.create_index("ix_event_facts_kind", "event_facts", ["kind"])


def downgrade() -> None:
    op.drop_index("ix_event_facts_kind", table_name="event_facts")
    op.drop_index("ix_event_facts_event_extractor", table_name="event_facts")
    op.drop_table("event_facts")
