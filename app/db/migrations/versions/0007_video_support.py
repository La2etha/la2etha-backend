"""Video support (spec 003): photo.media_type/duration_s/poster_key/content_hash,
detected_face.frame_index/frame_ts_s.

Revision ID: 0007_video_support
Revises: 0006_curation
Create Date: 2026-07-05
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0007_video_support"
down_revision: Union[str, None] = "0006_curation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "photo",
        sa.Column("media_type", sa.String(length=8), nullable=False, server_default="photo"),
    )
    op.add_column("photo", sa.Column("duration_s", sa.Float(), nullable=True))
    op.add_column("photo", sa.Column("poster_key", sa.String(length=512), nullable=True))
    op.add_column("photo", sa.Column("content_hash", sa.String(length=64), nullable=True))
    op.create_index("ix_photo_content_hash", "photo", ["content_hash"])
    op.add_column("detected_face", sa.Column("frame_index", sa.Integer(), nullable=True))
    op.add_column("detected_face", sa.Column("frame_ts_s", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("detected_face", "frame_ts_s")
    op.drop_column("detected_face", "frame_index")
    op.drop_index("ix_photo_content_hash", table_name="photo")
    op.drop_column("photo", "content_hash")
    op.drop_column("photo", "poster_key")
    op.drop_column("photo", "duration_s")
    op.drop_column("photo", "media_type")
