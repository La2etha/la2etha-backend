"""Event & membership request/response schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EventCreate(BaseModel):
    name: str


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    owner_id: uuid.UUID
    join_code: str
    status: str
    privacy_default_remove_strangers: bool
    created_at: datetime


class EventCreated(EventRead):
    join_link: str


class EventListItem(EventRead):
    """An event as it appears in the caller's events list (home screen)."""

    role: str
    member_count: int
    photo_count: int


class EventJoin(BaseModel):
    join_code: str


class EventSettingsUpdate(BaseModel):
    privacy_default_remove_strangers: bool | None = None


class MemberRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    account_id: uuid.UUID
    role: str
    joined_at: datetime


class DemotedItem(BaseModel):
    """A gallery photo demoted to a member's secondary section (F3/F4, FR-014)."""

    photo_id: uuid.UUID
    account_id: uuid.UUID
    demote_reason: str | None = None
