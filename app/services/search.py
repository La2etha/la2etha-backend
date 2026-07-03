"""Gallery search query (F5, FR-015) — the access-isolation-critical part.

Ranks a caller's OWN verified photos by cosine similarity to a query vector. The
result set is intersected with the caller's GalleryEntry rows, so search can never
surface a photo the person isn't verified in — the same hard rule as FR-019,
enforced here in the query itself. This function is model-free (it takes a query
vector), which lets the isolation test run without the SigLIP stack.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import GalleryEntry, Photo


@dataclass
class SearchHit:
    photo_id: uuid.UUID
    relevance: str
    demote_reason: str | None
    score: float  # cosine similarity (1 - cosine distance)


async def search_gallery(
    session: AsyncSession,
    account_id: uuid.UUID,
    event_id: uuid.UUID,
    query_vec: np.ndarray,
    limit: int,
    offset: int = 0,
) -> list[SearchHit]:
    """Return the caller's verified photos ranked by similarity to ``query_vec``.

    Only photos with a materialized GalleryEntry for THIS account+event and a
    non-null search embedding are considered.
    """
    query_list = np.asarray(query_vec, dtype=np.float32).tolist()
    distance = Photo.search_embedding.cosine_distance(query_list)

    stmt = (
        select(
            GalleryEntry.photo_id,
            GalleryEntry.relevance,
            GalleryEntry.demote_reason,
            distance.label("distance"),
        )
        .join(Photo, Photo.id == GalleryEntry.photo_id)
        .where(
            GalleryEntry.account_id == account_id,
            GalleryEntry.event_id == event_id,
            Photo.search_embedding.is_not(None),
        )
        .order_by(distance)
        .offset(offset)
        .limit(limit)
    )

    rows = (await session.execute(stmt)).all()
    return [
        SearchHit(
            photo_id=photo_id,
            relevance=relevance,
            demote_reason=demote_reason,
            score=1.0 - float(dist),
        )
        for photo_id, relevance, demote_reason, dist in rows
    ]
