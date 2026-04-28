"""pgvector for RAG and regime similarity

Revision ID: 8e2c41a6b54f
Revises: c5b6a92d7311
Create Date: 2026-04-29 01:00:00.000000

Promote the JSON-encoded embedding columns to native pgvector:
* ``rag_rationales.vector`` (JSONB list[float], 512-dim) →
  ``rag_rationales.embedding`` (vector(512)) with HNSW index.
* ``regime_snapshots.vector_json`` (JSONB list[float], 10-dim) →
  ``regime_snapshots.embedding`` (vector(10)).

Cosine similarity queries collapse from a Python loop to one
``ORDER BY embedding <=> :q LIMIT k`` SQL statement; the HNSW
index keeps the RAG query at ~1ms even at 100k rows. The regime
table is bounded to 730 rows so it doesn't get an index.

Backfill walks the existing JSON arrays and casts them with
``column::text::vector(N)``.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8e2c41a6b54f"
down_revision: Union[str, Sequence[str], None] = "c5b6a92d7311"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_RAG_DIM = 512
_REGIME_DIM = 10


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── rag_rationales: vector (JSONB) → embedding (vector(512)) ──
    op.add_column(
        "rag_rationales",
        sa.Column("embedding", Vector(_RAG_DIM), nullable=True),
    )
    op.execute(
        f"UPDATE rag_rationales SET embedding = (vector::text)::vector({_RAG_DIM}) "
        "WHERE vector IS NOT NULL"
    )
    op.alter_column("rag_rationales", "embedding", nullable=False)
    op.drop_column("rag_rationales", "vector")
    # HNSW index for cosine similarity. m=16, ef_construction=64 are
    # pgvector defaults; tune later if recall@k slips.
    op.execute(
        "CREATE INDEX ix_rag_rationales_embedding_hnsw ON rag_rationales "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    # ── regime_snapshots: vector_json (JSONB) → embedding (vector(10)) ──
    op.add_column(
        "regime_snapshots",
        sa.Column("embedding", Vector(_REGIME_DIM), nullable=True),
    )
    op.execute(
        f"UPDATE regime_snapshots SET embedding = (vector_json::text)::vector({_REGIME_DIM}) "
        "WHERE vector_json IS NOT NULL"
    )
    op.alter_column("regime_snapshots", "embedding", nullable=False)
    op.drop_column("regime_snapshots", "vector_json")


def downgrade() -> None:
    # ── regime_snapshots: embedding (vector(10)) → vector_json (JSONB) ──
    op.add_column(
        "regime_snapshots",
        sa.Column("vector_json", sa.dialects.postgresql.JSONB(), nullable=True),
    )
    op.execute(
        "UPDATE regime_snapshots SET vector_json = to_jsonb(embedding::float4[]) "
        "WHERE embedding IS NOT NULL"
    )
    op.alter_column("regime_snapshots", "vector_json", nullable=False)
    op.drop_column("regime_snapshots", "embedding")

    # ── rag_rationales: embedding (vector(512)) → vector (JSONB) ──
    op.drop_index("ix_rag_rationales_embedding_hnsw", table_name="rag_rationales")
    op.add_column(
        "rag_rationales",
        sa.Column("vector", sa.dialects.postgresql.JSONB(), nullable=True),
    )
    op.execute(
        "UPDATE rag_rationales SET vector = to_jsonb(embedding::float4[]) "
        "WHERE embedding IS NOT NULL"
    )
    op.alter_column("rag_rationales", "vector", nullable=False)
    op.drop_column("rag_rationales", "embedding")
