"""Enrollment schemas (F2)."""

from pydantic import BaseModel


class EnrollmentAccepted(BaseModel):
    job_id: str
    sample_count: int


class EnrollmentStatus(BaseModel):
    job_id: str
    status: str
    enrolled: bool | None = None
    quality_ok: bool | None = None
    sample_count: int | None = None
    gallery_entries: int | None = None
    reason: str | None = None
