"""initial schema: accounts, events, photos, faces, clusters, enrollment, galleries

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

UUID = postgresql.UUID(as_uuid=True)
EMBED_DIM = 512


def upgrade() -> None:
    # pgvector extension must exist before any vector column is created.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- account (FastAPI-Users) ---
    op.create_table(
        "account",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("hashed_password", sa.String(length=1024), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_superuser", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_account_email", "account", ["email"], unique=True)

    # --- oauth_account (FastAPI-Users) ---
    op.create_table(
        "oauth_account",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("oauth_name", sa.String(length=100), nullable=False),
        sa.Column("access_token", sa.String(length=1024), nullable=False),
        sa.Column("expires_at", sa.Integer(), nullable=True),
        sa.Column("refresh_token", sa.String(length=1024), nullable=True),
        sa.Column("account_id", sa.String(length=320), nullable=False),
        sa.Column("account_email", sa.String(length=320), nullable=False),
        sa.Column(
            "user_id",
            UUID,
            sa.ForeignKey("account.id", ondelete="cascade"),
            nullable=False,
        ),
    )
    op.create_index("ix_oauth_account_oauth_name", "oauth_account", ["oauth_name"])
    op.create_index("ix_oauth_account_account_id", "oauth_account", ["account_id"])

    # --- event ---
    op.create_table(
        "event",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "owner_id", UUID, sa.ForeignKey("account.id", ondelete="cascade"), nullable=False
        ),
        sa.Column("join_code", sa.String(length=16), nullable=False),
        sa.Column("join_token", sa.String(length=64), nullable=False),
        sa.Column(
            "privacy_default_remove_strangers",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("retention", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_event_owner_id", "event", ["owner_id"])
    op.create_index("ix_event_join_code", "event", ["join_code"], unique=True)
    op.create_index("ix_event_join_token", "event", ["join_token"], unique=True)

    # --- membership ---
    op.create_table(
        "membership",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "event_id", UUID, sa.ForeignKey("event.id", ondelete="cascade"), nullable=False
        ),
        sa.Column(
            "account_id", UUID, sa.ForeignKey("account.id", ondelete="cascade"), nullable=False
        ),
        sa.Column("role", sa.String(length=16), nullable=False, server_default="member"),
        sa.Column("joined_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("event_id", "account_id", name="uq_membership_event_account"),
    )
    op.create_index("ix_membership_event_id", "membership", ["event_id"])
    op.create_index("ix_membership_account_id", "membership", ["account_id"])

    # --- photo ---
    op.create_table(
        "photo",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "event_id", UUID, sa.ForeignKey("event.id", ondelete="cascade"), nullable=False
        ),
        sa.Column(
            "contributor_id",
            UUID,
            sa.ForeignKey("account.id", ondelete="cascade"),
            nullable=False,
        ),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False, server_default="upload"),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column(
            "orientation_applied", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("phash", sa.String(length=64), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("quality_verdict", sa.String(length=16), nullable=False, server_default="ok"),
        sa.Column("cull_reason", sa.String(length=64), nullable=True),
        sa.Column(
            "processing_status", sa.String(length=16), nullable=False, server_default="pending"
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_photo_event_id", "photo", ["event_id"])
    op.create_index("ix_photo_event_phash", "photo", ["event_id", "phash"])

    # --- face_cluster (before detected_face: detected_face.cluster_id → face_cluster) ---
    op.create_table(
        "face_cluster",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "event_id", UUID, sa.ForeignKey("event.id", ondelete="cascade"), nullable=False
        ),
        sa.Column("centroid", Vector(EMBED_DIM), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("algo", sa.String(length=24), nullable=False, server_default="hdbscan"),
        sa.Column(
            "claimed_by_account_id",
            UUID,
            sa.ForeignKey("account.id", ondelete="set null"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_face_cluster_event_id", "face_cluster", ["event_id"])
    op.create_index(
        "ix_face_cluster_centroid_hnsw",
        "face_cluster",
        ["centroid"],
        postgresql_using="hnsw",
        postgresql_ops={"centroid": "vector_cosine_ops"},
    )

    # --- detected_face ---
    op.create_table(
        "detected_face",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "photo_id", UUID, sa.ForeignKey("photo.id", ondelete="cascade"), nullable=False
        ),
        sa.Column("bbox", postgresql.JSONB(), nullable=False),
        sa.Column("landmarks", postgresql.JSONB(), nullable=True),
        sa.Column("det_score", sa.Float(), nullable=False),
        sa.Column("face_area_ratio", sa.Float(), nullable=True),
        sa.Column("face_sharpness", sa.Float(), nullable=True),
        sa.Column("is_background", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=False),
        sa.Column(
            "cluster_id",
            UUID,
            sa.ForeignKey("face_cluster.id", ondelete="set null"),
            nullable=True,
        ),
    )
    op.create_index("ix_detected_face_photo_id", "detected_face", ["photo_id"])
    op.create_index("ix_detected_face_cluster_id", "detected_face", ["cluster_id"])
    op.create_index(
        "ix_detected_face_embedding_hnsw",
        "detected_face",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # --- identity_enrollment ---
    op.create_table(
        "identity_enrollment",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "account_id", UUID, sa.ForeignKey("account.id", ondelete="cascade"), nullable=False
        ),
        sa.Column(
            "event_id", UUID, sa.ForeignKey("event.id", ondelete="cascade"), nullable=False
        ),
        sa.Column("centroid", Vector(EMBED_DIM), nullable=False),
        sa.Column(
            "per_angle_embeddings",
            postgresql.JSONB(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("quality_ok", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("account_id", "event_id", name="uq_enrollment_account_event"),
    )
    op.create_index("ix_enrollment_account_id", "identity_enrollment", ["account_id"])
    op.create_index("ix_enrollment_event_id", "identity_enrollment", ["event_id"])
    op.create_index(
        "ix_enrollment_centroid_hnsw",
        "identity_enrollment",
        ["centroid"],
        postgresql_using="hnsw",
        postgresql_ops={"centroid": "vector_cosine_ops"},
    )

    # --- gallery_entry ---
    op.create_table(
        "gallery_entry",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "account_id", UUID, sa.ForeignKey("account.id", ondelete="cascade"), nullable=False
        ),
        sa.Column(
            "event_id", UUID, sa.ForeignKey("event.id", ondelete="cascade"), nullable=False
        ),
        sa.Column(
            "photo_id", UUID, sa.ForeignKey("photo.id", ondelete="cascade"), nullable=False
        ),
        sa.Column("origin", sa.String(length=8), nullable=False, server_default="auto"),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("account_id", "photo_id", name="uq_gallery_account_photo"),
    )
    op.create_index("ix_gallery_entry_account_id", "gallery_entry", ["account_id"])
    op.create_index("ix_gallery_entry_event_id", "gallery_entry", ["event_id"])
    op.create_index("ix_gallery_entry_photo_id", "gallery_entry", ["photo_id"])

    # --- gallery_claim ---
    op.create_table(
        "gallery_claim",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "account_id", UUID, sa.ForeignKey("account.id", ondelete="cascade"), nullable=False
        ),
        sa.Column(
            "photo_id", UUID, sa.ForeignKey("photo.id", ondelete="cascade"), nullable=False
        ),
        sa.Column("state", sa.String(length=16), nullable=False, server_default="claimed"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_gallery_claim_account_id", "gallery_claim", ["account_id"])
    op.create_index("ix_gallery_claim_photo_id", "gallery_claim", ["photo_id"])


def downgrade() -> None:
    op.drop_table("gallery_claim")
    op.drop_table("gallery_entry")
    op.drop_table("identity_enrollment")
    op.drop_table("detected_face")
    op.drop_table("face_cluster")
    op.drop_table("photo")
    op.drop_table("membership")
    op.drop_table("event")
    op.drop_table("oauth_account")
    op.drop_table("account")
