"""Photo upload / processing schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PhotoRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    processing_status: str
    created_at: datetime


class UploadAccepted(BaseModel):
    job_id: str
    photo_ids: list[uuid.UUID]
    accepted: int
    duplicates: int


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
