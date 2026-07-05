"""Gallery response schemas."""

import uuid

from pydantic import BaseModel


class GalleryPhoto(BaseModel):
    photo_id: uuid.UUID
    origin: str
    relevance: str = "main"  # main | low
    demote_reason: str | None = None
    confidence: float | None = None
    # Who uploaded it — lets the client offer "delete" when member_delete_own
    # allows it (spec 005 US5/FR-018), without a second round-trip.
    contributor_id: uuid.UUID


class GalleryPage(BaseModel):
    items: list[GalleryPhoto]
    next_cursor: str | None = None


class EmptyState(BaseModel):
    empty: bool
    message: str


class SearchResultPhoto(BaseModel):
    photo_id: uuid.UUID
    relevance: str = "main"
    demote_reason: str | None = None
    score: float  # cosine similarity to the query (higher = better)


class SearchPage(BaseModel):
    items: list[SearchResultPhoto]
    next_cursor: str | None = None
