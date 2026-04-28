"""JSONB for JSON-shaped string columns

Revision ID: c5b6a92d7311
Revises: f18549dec00a
Create Date: 2026-04-29 00:00:00.000000

Promote the columns we used as TEXT-encoded JSON to native JSONB:
* ``llm_decisions.parsed_action`` (action-counts dict)
* ``llm_decisions.symbols`` (list of strings)
* ``crypto_halal_cache.screening_criteria`` (criteria dict)
* ``research_jobs.params`` (input dict)
* ``research_jobs.result`` (output dict)
* ``runtime_config.value`` (any JSON scalar / list / dict)
* ``halal_screenings.criteria`` (criteria dict)
* ``rag_rationales.vector`` (list[float])
* ``regime_snapshots.features_json`` (dict)
* ``regime_snapshots.vector_json`` (list[float])

Every existing value is already valid JSON (the writers all run
``json.dumps`` on a Python value), so ``USING column::jsonb`` is
lossless.

``web_actions.payload`` stays TEXT — the audit middleware truncates
it to 6KB and appends ``…[truncated]``, which isn't valid JSON.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c5b6a92d7311"
down_revision: Union[str, Sequence[str], None] = "f18549dec00a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_JSONB_COLUMNS: list[tuple[str, str, bool]] = [
    # (table, column, nullable)
    ("llm_decisions", "parsed_action", True),
    ("llm_decisions", "symbols", True),
    ("crypto_halal_cache", "screening_criteria", True),
    ("research_jobs", "params", False),
    ("research_jobs", "result", True),
    ("runtime_config", "value", False),
    ("halal_screenings", "criteria", True),
    ("rag_rationales", "vector", False),
    ("regime_snapshots", "features_json", False),
    ("regime_snapshots", "vector_json", False),
]


def upgrade() -> None:
    for table, column, nullable in _JSONB_COLUMNS:
        op.alter_column(
            table,
            column,
            existing_type=sa.VARCHAR(),
            type_=sa.dialects.postgresql.JSONB(),
            existing_nullable=nullable,
            postgresql_using=f'"{column}"::jsonb',
        )


def downgrade() -> None:
    for table, column, nullable in reversed(_JSONB_COLUMNS):
        op.alter_column(
            table,
            column,
            existing_type=sa.dialects.postgresql.JSONB(),
            type_=sa.VARCHAR(),
            existing_nullable=nullable,
            postgresql_using=f'"{column}"::text',
        )
