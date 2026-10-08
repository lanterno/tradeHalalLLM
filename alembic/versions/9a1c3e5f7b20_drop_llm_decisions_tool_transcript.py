"""Drop llm_decisions.tool_transcript.

The agentic tool-calling mode that wrote it was deleted on 2026-10-01; every
row since stores None, and nothing reads it. Downgrade restores the empty
column.

Revision ID: 9a1c3e5f7b20
Revises: 7d2e4a6c8b10
Create Date: 2026-10-08 12:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9a1c3e5f7b20"
down_revision: str | Sequence[str] | None = "7d2e4a6c8b10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("llm_decisions", "tool_transcript")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "llm_decisions",
        sa.Column("tool_transcript", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
