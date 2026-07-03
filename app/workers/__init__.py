"""RQ + Redis wiring and a job-status helper for batch-processing progress."""

from typing import Any

from redis import Redis
from rq import Queue
from rq.job import Job

from app.config import get_settings

QUEUE_NAME = "la2etha"

_settings = get_settings()
_redis: Redis | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(_settings.redis_url)
    return _redis


def get_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=get_redis())


def job_status(job_id: str) -> dict[str, Any]:
    """Return a small status dict for a queued/running job (SC-006 progress UX)."""
    try:
        job = Job.fetch(job_id, connection=get_redis())
    except Exception:
        return {"job_id": job_id, "status": "unknown"}
    return {
        "job_id": job_id,
        "status": job.get_status(refresh=True),
        "progress": job.meta.get("progress"),
        "processed": job.meta.get("processed"),
        "total": job.meta.get("total"),
    }
