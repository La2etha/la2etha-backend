"""Host member management (spec 005 US4): removal cascade + member aggregates.

Removal revokes everything event-scoped for that member — membership,
enrollment, gallery entries/claims, and any cluster they'd claimed reverts to
anonymous — while leaving their account, other events, and photos they
contributed to this event's pool untouched (FR-013).
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Account,
    DetectedFace,
    FaceCluster,
    GalleryClaim,
    GalleryEntry,
    IdentityEnrollment,
    Membership,
    Photo,
)


class CannotRemoveHostError(Exception):
    """Raised when a host attempts to remove themselves."""


async def remove_member(
    session: AsyncSession, event_id: uuid.UUID, account_id: uuid.UUID
) -> None:
    membership = await session.scalar(
        select(Membership).where(
            Membership.event_id == event_id, Membership.account_id == account_id
        )
    )
    if membership is None:
        return  # already gone — removal is idempotent
    if membership.role == "host":
        raise CannotRemoveHostError()

    # Un-claim any cluster they'd claimed in this event — their faces revert to
    # anonymous "Guest" for everyone immediately (mirrors delete-own-identity).
    cluster_ids = select(FaceCluster.id).where(FaceCluster.event_id == event_id)
    await session.execute(
        update(FaceCluster)
        .where(FaceCluster.id.in_(cluster_ids), FaceCluster.claimed_by_account_id == account_id)
        .values(claimed_by_account_id=None)
    )

    await session.execute(
        delete(GalleryEntry).where(
            GalleryEntry.account_id == account_id, GalleryEntry.event_id == event_id
        )
    )
    # GalleryClaim has no event_id column — scope it via the photo's event.
    event_photo_ids = select(Photo.id).where(Photo.event_id == event_id)
    await session.execute(
        delete(GalleryClaim).where(
            GalleryClaim.account_id == account_id, GalleryClaim.photo_id.in_(event_photo_ids)
        )
    )
    await session.execute(
        delete(IdentityEnrollment).where(
            IdentityEnrollment.account_id == account_id, IdentityEnrollment.event_id == event_id
        )
    )
    await session.delete(membership)
    await session.commit()


async def list_members(session: AsyncSession, event_id: uuid.UUID, *, status: str | None = None):
    """Members with display name, enrolled flag, and photo-appearance count
    (host-only aggregate, spec 005 FR-012)."""
    appearance_count = (
        select(func.count(GalleryEntry.id))
        .where(
            GalleryEntry.account_id == Membership.account_id,
            GalleryEntry.event_id == event_id,
        )
        .correlate(Membership)
        .scalar_subquery()
    )
    enrolled = (
        select(func.count(IdentityEnrollment.id))
        .where(
            IdentityEnrollment.account_id == Membership.account_id,
            IdentityEnrollment.event_id == event_id,
        )
        .correlate(Membership)
        .scalar_subquery()
    )
    stmt = (
        select(Membership, Account.name, enrolled, appearance_count)
        .join(Account, Account.id == Membership.account_id)
        .where(Membership.event_id == event_id)
        .order_by(Membership.joined_at)
    )
    if status is not None:
        stmt = stmt.where(Membership.status == status)
    rows = (await session.execute(stmt)).all()
    return [
        {
            "account_id": m.account_id,
            "name": name,
            "role": m.role,
            "status": m.status,
            "enrolled": enrolled_count > 0,
            "appearance_count": appearance_count_val,
            "joined_at": m.joined_at,
        }
        for m, name, enrolled_count, appearance_count_val in rows
    ]
