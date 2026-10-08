"""core orders name their screen method

Revision ID: 0afe09d31725
Revises: e27d9fe017e0
Create Date: 2026-10-08 09:00:00.000000

Verdicts are kept per method (b7d3f1a9c5e2), so an order's screen date alone
no longer says which verdict it relied on. The method of that verdict is
recorded beside it; orders placed before this have none.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0afe09d31725"
down_revision: str | Sequence[str] | None = "e27d9fe017e0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("core_orders", sa.Column("screen_method", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("core_orders", "screen_method")
