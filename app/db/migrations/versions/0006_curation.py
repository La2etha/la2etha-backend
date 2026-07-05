"""Curation layer (spec 004): event.cover_photo_id/cover_source (host-picked or
auto-picked from event photos, distinct from the spec-002 uploaded cover_key),
and photo.is_highlight/highlight_rank for the event Highlights strip.

Revision ID: 0006_curation
Revises: 0005_identity_host_policy
Create Date: 2026-07-05
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0006_curation"
down_revision: Union[str, None] = "0005_identity_host_policy"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("event", sa.Column("cover_photo_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_event_cover_photo_id",
        "event",
        "photo",
        ["cover_photo_id"],
        ["id"],
        ondelete="set null",
    )
    op.add_column("event", sa.Column("cover_source", sa.String(length=16), nullable=True))
    op.add_column(
        "photo",
        sa.Column("is_highlight", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("photo", sa.Column("highlight_rank", sa.SmallInteger(), nullable=True))
    op.create_index(
        "ix_photo_event_highlight", "photo", ["event_id", "is_highlight"]
    )


def downgrade() -> None:
    op.drop_index("ix_photo_event_highlight", table_name="photo")
    op.drop_column("photo", "highlight_rank")
    op.drop_column("photo", "is_highlight")
    op.drop_column("event", "cover_source")
    op.drop_constraint("fk_event_cover_photo_id", "event", type_="foreignkey")
    op.drop_column("event", "cover_photo_id")
