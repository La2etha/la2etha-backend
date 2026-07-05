"""Photo upload / processing schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PhotoRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    contributor_id: uuid.UUID
    processing_status: str
    media_type: str = "photo"
    duration_s: float | None = None
    created_at: datetime


class RejectedUpload(BaseModel):
    filename: str | None
    reason: str


class UploadAccepted(BaseModel):
    job_id: str
    photo_ids: list[uuid.UUID]
    accepted: int
    duplicates: int
    rejected: list[RejectedUpload] = []


class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    processed: int | None = None
    total: int | None = None
    progress: float | None = None


class GDriveIngestRequest(BaseModel):
    # Short-lived OAuth access token from the client-side Google Picker
    # (drive.readonly / drive.file scope). Provide a folder_id, explicit
    # file_ids, or both. Drive file IDs are opaque strings, not UUIDs.
    access_token: str
    folder_id: str | None = None
    file_ids: list[str] | None = None


class GDriveIngestAccepted(BaseModel):
    job_id: str


class PhotoFace(BaseModel):
    # Bounding box normalized to 0..1 of the photo's width/height, so the client
    # can scale it to whatever size the image is rendered at (FR-024).
    x: float
    y: float
    w: float
    h: float
    is_me: bool  # this face's cluster is claimed by the requesting account
    # Populated only when the event's name_policy permits it for this viewer
    # (spec 005 FR-002); null is indistinguishable from "unclaimed guest" — the
    # policy gate happens server-side, never by a client-side hide.
    name: str | None = None
