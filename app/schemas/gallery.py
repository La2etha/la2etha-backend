"""Gallery response schemas."""

import uuid

from pydantic import BaseModel


class GalleryPhoto(BaseModel):
    photo_id: uuid.UUID
    origin: str
    confidence: float | None = None


class GalleryPage(BaseModel):
    items: list[GalleryPhoto]
    next_cursor: str | None = None


class EmptyState(BaseModel):
    empty: bool
    message: str
