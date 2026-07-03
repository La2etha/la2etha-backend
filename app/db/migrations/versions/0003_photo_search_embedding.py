"""photo.search_embedding for SigLIP-2 natural-language search (F5, FR-015)

Adds a nullable 768-d vector column + HNSW cosine index. Photos are indexed
best-effort during processing; null means "not yet indexed" and is simply
excluded from search results.

Revision ID: 0003_photo_search_embedding
Revises: 0002_gallery_relevance
Create Date: 2026-07-04
"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "0003_photo_search_embedding"
down_revision: Union[str, None] = "0002_gallery_relevance"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SEARCH_EMBED_DIM = 768


def upgrade() -> None:
    op.add_column(
        "photo", sa.Column("search_embedding", Vector(SEARCH_EMBED_DIM), nullable=True)
    )
    op.create_index(
        "ix_photo_search_embedding_hnsw",
        "photo",
        ["search_embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"search_embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_photo_search_embedding_hnsw", table_name="photo")
    op.drop_column("photo", "search_embedding")
