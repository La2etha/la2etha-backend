"""Storage/ingestion interface.

Photo bytes live behind an opaque ``storage_key`` so the pipeline never cares
which backend holds them. Local filesystem is the MVP adapter; R2 and Google
Drive adapters (Phase 8) implement the same contract.
"""

from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import get_settings


class StorageAdapter(ABC):
    """Put/get/delete photo bytes by opaque key."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> str:
        """Store ``data`` under ``key``; return the (possibly normalized) key."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Return the bytes stored under ``key``."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the object at ``key`` (no error if already absent)."""


@lru_cache
def get_storage() -> StorageAdapter:
    """Return the storage adapter selected by ``STORAGE_BACKEND``."""
    settings = get_settings()
    backend = settings.storage_backend
    if backend == "local_fs":
        from app.storage.local_fs import LocalFilesystemStorage

        return LocalFilesystemStorage(settings.media_root)
    # r2 / gdrive adapters arrive in Phase 8.
    raise ValueError(f"Unsupported STORAGE_BACKEND: {backend!r}")
