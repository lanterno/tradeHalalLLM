"""indicator_snapshots.pair is symbol

Revision ID: e27d9fe017e0
Revises: 69402fab5aad
Create Date: 2026-10-08 06:20:00.000000

The column holding the stock symbol was named ``pair``, after the crypto
bot's trading pairs. Renamed in place, so rows keep their values.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e27d9fe017e0"
down_revision: str | Sequence[str] | None = "69402fab5aad"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column("indicator_snapshots", "pair", new_column_name="symbol")


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column("indicator_snapshots", "symbol", new_column_name="pair")
