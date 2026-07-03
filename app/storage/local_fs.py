"""Local filesystem storage adapter (MVP / demo)."""

from pathlib import Path

from app.storage.base import StorageAdapter


class LocalFilesystemStorage(StorageAdapter):
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Guard against path traversal: the resolved path must stay under root.
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Illegal storage key: {key!r}")
        return path

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
