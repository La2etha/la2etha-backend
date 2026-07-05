"""event.cover_key for the host-set cover image (UI spec 002 boarding pass)

Nullable storage key; null means "no cover" and the client falls back to the
people-pass variant.

Revision ID: 0004_event_cover
Revises: 0003_photo_search_embedding
Create Date: 2026-07-04
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0004_event_cover"
down_revision: Union[str, None] = "0003_photo_search_embedding"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("event", sa.Column("cover_key", sa.String(length=512), nullable=True))


def downgrade() -> None:
    op.drop_column("event", "cover_key")
