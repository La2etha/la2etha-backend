"""Export privacy logic (F6, FR-016) — which faces get removed.

A face is removed only if it is BOTH unclaimed (its cluster isn't linked to any
enrolled account — i.e. not a verified member) AND an incidental background
presence. This preserves every claimed member and never erases a prominent
subject, matching "remove unclaimed background people, keep claimed members".

Model-free and query-only, so the privacy rule can be tested without LaMa.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DetectedFace, FaceCluster


async def faces_to_remove(session: AsyncSession, photo_id: uuid.UUID) -> list[dict]:
    """Return the bboxes of unclaimed background faces in a photo (to inpaint out).

    Claimed members (their cluster is linked to an account) are always preserved.
    """
    rows = (
        await session.execute(
            select(DetectedFace.bbox, DetectedFace.is_background, FaceCluster.claimed_by_account_id)
            .outerjoin(FaceCluster, FaceCluster.id == DetectedFace.cluster_id)
            .where(DetectedFace.photo_id == photo_id)
        )
    ).all()

    remove: list[dict] = []
    for bbox, is_background, claimed_by in rows:
        is_claimed_member = claimed_by is not None
        if is_background and not is_claimed_member:
            remove.append(bbox)
    return remove


async def is_solo_editable(
    session: AsyncSession, photo_id: uuid.UUID, account_id: uuid.UUID
) -> bool:
    """True only if the photo is safe to send to the cloud editor (F7): it may
    contain no one but the requesting user.

    Every detected face must belong to the caller's claimed cluster; a photo with
    another member, or any unclaimed face (a stranger, or someone unconfirmed), is
    blocked so no one else's face ever leaves the machine. A person-less photo
    (e.g. scenery) is allowed — there's nobody to protect.
    """
    rows = (
        await session.execute(
            select(FaceCluster.claimed_by_account_id)
            .select_from(DetectedFace)
            .outerjoin(FaceCluster, FaceCluster.id == DetectedFace.cluster_id)
            .where(DetectedFace.photo_id == photo_id)
        )
    ).all()
    return all(claimed_by == account_id for (claimed_by,) in rows)
