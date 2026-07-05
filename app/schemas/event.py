"""Event & membership request/response schemas."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

NamePolicy = Literal["nobody", "host_only", "everyone"]
EventType = Literal["wedding", "graduation", "iftar", "birthday", "trip", "other"]
GalleryVisibility = Literal["own_only", "everyone_sees_all"]
AiEditScope = Literal["solo_only", "any_photo"]
MemberUploads = Literal["enabled", "host_only"]


class EventCreate(BaseModel):
    name: str
    event_type: EventType | None = None


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    owner_id: uuid.UUID
    join_code: str
    status: str
    privacy_default_remove_strangers: bool
    has_cover: bool
    # --- Identity & host policy (spec 005) ---
    name_policy: NamePolicy
    event_type: EventType | None
    gallery_visibility: GalleryVisibility
    ai_edit_scope: AiEditScope
    member_uploads: MemberUploads
    member_delete_own: bool
    join_approval: bool
    member_list_visible: bool
    created_at: datetime

    @property
    def uploads_closed(self) -> bool:
        return self.status == "archived"


class EventCreated(EventRead):
    join_link: str


class EventListItem(EventRead):
    """An event as it appears in the caller's events list (home screen)."""

    role: str
    member_count: int
    photo_count: int


class EventJoin(BaseModel):
    join_code: str


class EventJoined(BaseModel):
    """Join response — distinguishes an instant join from a pending one waiting
    on host approval (join_approval toggle)."""

    status: Literal["active", "pending"]
    event: EventRead | None = None


class EventSettingsUpdate(BaseModel):
    privacy_default_remove_strangers: bool | None = None
    name_policy: NamePolicy | None = None
    event_type: EventType | None = None
    gallery_visibility: GalleryVisibility | None = None
    ai_edit_scope: AiEditScope | None = None
    member_uploads: MemberUploads | None = None
    member_delete_own: bool | None = None
    join_approval: bool | None = None
    member_list_visible: bool | None = None
    # Maps to event.status ("archived" | "active") — reuses the existing column
    # rather than adding a parallel one.
    uploads_closed: bool | None = None


class MemberRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    account_id: uuid.UUID
    name: str
    role: str
    status: str
    enrolled: bool
    appearance_count: int
    joined_at: datetime


class DemotedItem(BaseModel):
    """A gallery photo demoted to a member's secondary section (F3/F4, FR-014)."""

    photo_id: uuid.UUID
    account_id: uuid.UUID
    demote_reason: str | None = None
