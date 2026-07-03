"""Natural-language gallery search endpoint (F5, FR-015)."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.cv.search_embed import embed_text, search_available
from app.db.base import get_async_session
from app.db.models import Account, Membership
from app.schemas.gallery import SearchPage, SearchResultPhoto

router = APIRouter(tags=["search"])

DEFAULT_LIMIT = 40


async def _require_membership(
    session: AsyncSession, account_id: uuid.UUID, event_id: uuid.UUID
) -> None:
    member = await session.scalar(
        select(Membership.id).where(
            Membership.event_id == event_id, Membership.account_id == account_id
        )
    )
    if member is None:
        raise HTTPException(status_code=404, detail="Event not found")


@router.get("/events/{event_id}/gallery/search", response_model=SearchPage)
async def search_gallery_endpoint(
    event_id: uuid.UUID,
    q: str = Query(..., min_length=1, max_length=200),
    cursor: str | None = Query(None),
    limit: int = Query(DEFAULT_LIMIT, le=100),
    user: Account = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> SearchPage:
    """Free-text search over the caller's verified photos, ranked by relevance."""
    await _require_membership(session, user.id, event_id)

    if not search_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search is unavailable — the SigLIP model isn't installed on the server.",
        )

    # Imported lazily so the module (and tests) don't require the search service.
    from app.services.search import search_gallery

    # ponytail: offset pagination is plenty at event scale (a person's gallery is
    # tens–hundreds of photos); switch to keyset-on-distance if that grows huge.
    offset = int(cursor) if cursor and cursor.isdigit() else 0

    query_vec = embed_text(q)
    hits = await search_gallery(session, user.id, event_id, query_vec, limit, offset)

    next_cursor = str(offset + limit) if len(hits) == limit else None
    items = [
        SearchResultPhoto(
            photo_id=h.photo_id,
            relevance=h.relevance,
            demote_reason=h.demote_reason,
            score=h.score,
        )
        for h in hits
    ]
    return SearchPage(items=items, next_cursor=next_cursor)
