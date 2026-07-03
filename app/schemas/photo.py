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
