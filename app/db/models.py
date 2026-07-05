"""ORM models for every entity in data-model.md.

Metadata and 512-d face embeddings live in one Postgres store (pgvector). Cosine
similarity is served by HNSW indexes on the embedding/centroid columns.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi_users.db import (
    SQLAlchemyBaseOAuthAccountTableUUID,
    SQLAlchemyBaseUserTableUUID,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from app.db.base import Base

EMBED_DIM = 512
# SigLIP-2-Base image/text embedding dimension (F5 semantic search).
SEARCH_EMBED_DIM = 768


# --------------------------------------------------------------------------- #
# Accounts & auth (FastAPI-Users)                                             #
# --------------------------------------------------------------------------- #
class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    __tablename__ = "oauth_account"

    # The base points user_id at "user.id"; our account table is "account".
    @declared_attr
    def user_id(cls) -> Mapped[uuid.UUID]:  # noqa: N805
        return mapped_column(
            ForeignKey("account.id", ondelete="cascade"), nullable=False
        )


class Account(SQLAlchemyBaseUserTableUUID, Base):
    """The unified user (host and/or member). FastAPI-Users supplies
    id, email, hashed_password, is_active, is_superuser, is_verified."""

    __tablename__ = "account"

    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    oauth_accounts: Mapped[list[OAuthAccount]] = relationship(
        "OAuthAccount", lazy="joined", cascade="all, delete-orphan"
    )


# --------------------------------------------------------------------------- #
# Events & membership                                                         #
# --------------------------------------------------------------------------- #
class Event(Base):
    __tablename__ = "event"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255))
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("account.id", ondelete="cascade"), index=True
    )
    join_code: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    join_token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    privacy_default_remove_strangers: Mapped[bool] = mapped_column(Boolean, default=False)
    # active|archived|deleting. "archived" IS the uploads_closed toggle (spec 005
    # US5): no new uploads/enrollments; galleries/search/export keep working.
    status: Mapped[str] = mapped_column(String(16), default="active")
    retention: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Host-set cover image (boarding-pass variant, UI spec 002); storage key, not a URL.
    cover_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # Curated cover picked from the event's own photos (spec 004): auto-picked at
    # the end of each pipeline run, or host-overridden. Never overwritten by auto
    # once cover_source == "host" (FR-010). Distinct from cover_key above, which
    # is an arbitrary uploaded image.
    cover_photo_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("photo.id", ondelete="set null", use_alter=True, name="fk_event_cover_photo_id"),
        nullable=True,
    )
    cover_source: Mapped[str | None] = mapped_column(String(16), nullable=True)  # auto|host

    # --- Identity & host policy (spec 005) ---
    # Who may see enrolled members' names on faces in the trust overlay.
    name_policy: Mapped[str] = mapped_column(String(16), default="nobody")  # nobody|host_only|everyone
    event_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # own_only = today's behavior; everyone_sees_all lets members browse the full pool.
    gallery_visibility: Mapped[str] = mapped_column(String(24), default="own_only")
    # solo_only = today's behavior (AI edit only on a photo of just the caller).
    ai_edit_scope: Mapped[str] = mapped_column(String(16), default="solo_only")
    member_uploads: Mapped[str] = mapped_column(String(16), default="enabled")  # enabled|host_only
    member_delete_own: Mapped[bool] = mapped_column(Boolean, default=True)
    join_approval: Mapped[bool] = mapped_column(Boolean, default=False)
    member_list_visible: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    @property
    def has_cover(self) -> bool:
        return self.cover_key is not None or self.cover_photo_id is not None


class Membership(Base):
    __tablename__ = "membership"
    __table_args__ = (UniqueConstraint("event_id", "account_id", name="uq_membership_event_account"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("event.id", ondelete="cascade"), index=True
    )
    account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("account.id", ondelete="cascade"), index=True
    )
    role: Mapped[str] = mapped_column(String(16), default="member")  # host|member
    # active = full member; pending = awaiting host approval (join_approval on) —
    # every access guard requires "active".
    status: Mapped[str] = mapped_column(String(16), default="active")
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# --------------------------------------------------------------------------- #
# Photos & detected faces                                                     #
# --------------------------------------------------------------------------- #
class Photo(Base):
    __tablename__ = "photo"
    __table_args__ = (
        Index("ix_photo_event_phash", "event_id", "phash"),
        Index(
            "ix_photo_search_embedding_hnsw",
            "search_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"search_embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("event.id", ondelete="cascade"), index=True
    )
    contributor_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("account.id", ondelete="cascade")
    )
    storage_key: Mapped[str] = mapped_column(String(512))
    source: Mapped[str] = mapped_column(String(16), default="upload")  # upload|gdrive
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    orientation_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    phash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_verdict: Mapped[str] = mapped_column(String(16), default="ok")  # ok|culled
    cull_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # SigLIP-2 image embedding for natural-language search (F5); null until indexed.
    search_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(SEARCH_EMBED_DIM), nullable=True
    )
    processing_status: Mapped[str] = mapped_column(
        String(16), default="pending"
    )  # pending|processing|done|failed
    # Curation (spec 004): rewritten wholesale by the post-pipeline curation step.
    is_highlight: Mapped[bool] = mapped_column(Boolean, default=False)
    highlight_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DetectedFace(Base):
    __tablename__ = "detected_face"
    __table_args__ = (
        Index(
            "ix_detected_face_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    photo_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("photo.id", ondelete="cascade"), index=True
    )
    bbox: Mapped[dict] = mapped_column(JSONB)  # {x, y, w, h}
    landmarks: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    det_score: Mapped[float] = mapped_column(Float)
    face_area_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_sharpness: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_background: Mapped[bool] = mapped_column(Boolean, default=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("face_cluster.id", ondelete="set null"), nullable=True, index=True
    )


class FaceCluster(Base):
    __tablename__ = "face_cluster"
    __table_args__ = (
        Index(
            "ix_face_cluster_centroid_hnsw",
            "centroid",
            postgresql_using="hnsw",
            postgresql_ops={"centroid": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("event.id", ondelete="cascade"), index=True
    )
    centroid: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    member_count: Mapped[int] = mapped_column(Integer, default=0)
    algo: Mapped[str] = mapped_column(String(24), default="hdbscan")
    claimed_by_account_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("account.id", ondelete="set null"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# --------------------------------------------------------------------------- #
# Identity enrollment & galleries                                             #
# --------------------------------------------------------------------------- #
class IdentityEnrollment(Base):
    __tablename__ = "identity_enrollment"
    __table_args__ = (
        UniqueConstraint("account_id", "event_id", name="uq_enrollment_account_event"),
        Index(
            "ix_enrollment_centroid_hnsw",
            "centroid",
            postgresql_using="hnsw",
            postgresql_ops={"centroid": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("account.id", ondelete="cascade"), index=True
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("event.id", ondelete="cascade"), index=True
    )
    centroid: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    # List of per-angle L2-normalized embeddings (each a 512-float list). Stored as
    # JSONB: there is one enrollment per (account, event), so the per-angle recall
    # boost is computed in-memory against the small set of cluster centroids.
    per_angle_embeddings: Mapped[list] = mapped_column(JSONB, default=list)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    quality_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class GalleryEntry(Base):
    __tablename__ = "gallery_entry"
    __table_args__ = (
        UniqueConstraint("account_id", "photo_id", name="uq_gallery_account_photo"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("account.id", ondelete="cascade"), index=True
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("event.id", ondelete="cascade"), index=True
    )
    photo_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("photo.id", ondelete="cascade"), index=True
    )
    origin: Mapped[str] = mapped_column(String(8), default="auto")  # auto|claim
    # "main" = a confident, in-focus subject photo; "low" = demoted by quality
    # culling (F3) or background/proximity filtering (F4). Demoted entries are
    # still shown, in the gallery's secondary "probably not interested" section.
    relevance: Mapped[str] = mapped_column(String(8), default="main")  # main|low
    demote_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class GalleryClaim(Base):
    __tablename__ = "gallery_claim"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("account.id", ondelete="cascade"), index=True
    )
    photo_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("photo.id", ondelete="cascade"), index=True
    )
    state: Mapped[str] = mapped_column(String(16), default="claimed")  # claimed|unclaimed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
