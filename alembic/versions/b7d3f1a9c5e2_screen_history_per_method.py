"""halal_screen_results keyed per method; halal_screen_current reads the newest

Revision ID: b7d3f1a9c5e2
Revises: a7d3e9b1c5f2
Create Date: 2026-10-06 14:00:00.000000

A re-screen under a new method used to overwrite the stored verdicts for
that date in place (``ON CONFLICT (as_of, symbol) DO UPDATE``), so an order
citing ``screen_as_of`` pointed at a verdict that had since been rewritten.
With ``method`` in the key every method's verdicts are kept side by side.

Readers go through ``halal_screen_current``: per (as_of, symbol), the row
of the newest method (the integer after the last "-v": v11 beats v9, which
a string comparison would not), then the latest screened. A method name
without a version (tests use "test") ranks below every versioned one.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7d3f1a9c5e2"
down_revision: str | Sequence[str] | None = "a7d3e9b1c5f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = "as_of, symbol, cik, sic_description, verdict, reasons, metrics, method, screened_at"


def upgrade() -> None:
    op.drop_constraint("halal_screen_results_pkey", "halal_screen_results", type_="primary")
    op.create_primary_key(
        "halal_screen_results_pkey", "halal_screen_results", ["as_of", "symbol", "method"]
    )
    op.execute(
        """
        CREATE FUNCTION halal_screen_method_rank(method text) RETURNS integer
        LANGUAGE sql IMMUTABLE STRICT
        AS $$ SELECT substring(method FROM '-v([0-9]+)$')::integer $$
        """
    )
    op.execute(
        f"""
        CREATE VIEW halal_screen_current AS
        SELECT DISTINCT ON (as_of, symbol) {_COLUMNS}
        FROM halal_screen_results
        ORDER BY as_of, symbol, halal_screen_method_rank(method) DESC NULLS LAST,
                 screened_at DESC
        """
    )


def downgrade() -> None:
    # Lossy: only the newest method's verdict per (as_of, symbol) fits the old key.
    op.execute(
        "DELETE FROM halal_screen_results WHERE (as_of, symbol, method) NOT IN "
        "(SELECT as_of, symbol, method FROM halal_screen_current)"
    )
    op.execute("DROP VIEW halal_screen_current")
    op.execute("DROP FUNCTION halal_screen_method_rank(text)")
    op.drop_constraint("halal_screen_results_pkey", "halal_screen_results", type_="primary")
    op.create_primary_key("halal_screen_results_pkey", "halal_screen_results", ["as_of", "symbol"])
