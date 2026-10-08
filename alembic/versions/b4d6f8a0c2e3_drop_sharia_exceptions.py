"""Drop sharia_exceptions.

The exception queue was never fed: nothing wrote a row, and an approval
authorised nothing, since no order path read the queue. A doubtful verdict
has always meant not bought; the strict screen fails closed. Downgrade
recreates the table empty.

Revision ID: b4d6f8a0c2e3
Revises: 9a1c3e5f7b20
Create Date: 2026-10-08 12:20:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b4d6f8a0c2e3"
down_revision: str | Sequence[str] | None = "9a1c3e5f7b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_table("sharia_exceptions")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "sharia_exceptions",
        sa.Column("entry_id", sa.VARCHAR(), nullable=False),
        sa.Column("instrument", sa.VARCHAR(), nullable=False),
        sa.Column("kind", sa.VARCHAR(), nullable=False),
        sa.Column("reasoning", sa.VARCHAR(), nullable=False),
        sa.Column("status", sa.VARCHAR(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sa.VARCHAR(), nullable=False),
        sa.Column("operator_note", sa.VARCHAR(), nullable=False),
        sa.PrimaryKeyConstraint("entry_id"),
    )
