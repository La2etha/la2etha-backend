"""gallery_entry demotion: relevance + demote_reason (F3/F4, FR-012/013)

Flagged photos are demoted to a secondary gallery section rather than excluded,
so these two columns record whether an entry is a main subject photo or a
low-relevance one and why.

Revision ID: 0002_gallery_relevance
Revises: 0001_initial
Create Date: 2026-07-03
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0002_gallery_relevance"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "gallery_entry",
        sa.Column("relevance", sa.String(length=8), nullable=False, server_default="main"),
    )
    op.add_column(
        "gallery_entry",
        sa.Column("demote_reason", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_gallery_entry_account_event_relevance",
        "gallery_entry",
        ["account_id", "event_id", "relevance"],
    )


def downgrade() -> None:
    op.drop_index("ix_gallery_entry_account_event_relevance", table_name="gallery_entry")
    op.drop_column("gallery_entry", "demote_reason")
    op.drop_column("gallery_entry", "relevance")
