"""tool_transcript_column

Revision ID: d48f82af284b
Revises: 0c42f379c69b
Create Date: 2026-04-29 05:52:42.066645

Wave H — adds llm_decisions.tool_transcript JSONB so the dashboard
can render the agent's chain-of-thought when the cycle runs in
agentic mode.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "d48f82af284b"
down_revision: Union[str, Sequence[str], None] = "0c42f379c69b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_decisions",
        sa.Column(
            "tool_transcript",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("llm_decisions", "tool_transcript")
