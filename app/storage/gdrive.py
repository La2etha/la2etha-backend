"""Google Drive link ingestion (F-infra, Phase 8).

Pulls images the user already has in Drive into an event's pool. This is an
INGESTION source, not a storage backend: files are downloaded here, then stored
via the configured storage adapter (local_fs / r2) like any upload.

Free by design: the Drive API has a generous free quota. Auth is a short-lived
OAuth access token the frontend obtains via the Google Picker (narrow
``drive.readonly``/``drive.file`` scope) and passes per request — the server
stores no Google credentials. Uses stdlib HTTP only (no google client library,
no new dependency).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

_API = "https://www.googleapis.com/drive/v3"
_UPLOAD_GET = "https://www.googleapis.com/drive/v3/files"


@dataclass
class DriveFile:
    id: str
    name: str
    mime_type: str


def _get(url: str, token: str) -> bytes:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted Google host)
        return resp.read()


def parse_file_list(payload: dict) -> tuple[list[DriveFile], str | None]:
    """Extract image files + the next page token from a Drive files.list response.

    Non-image files (and Google-native docs) are skipped — we only ingest photos.
    Pure function so it can be unit-tested without hitting the network.
    """
    files: list[DriveFile] = []
    for f in payload.get("files", []):
        mime = f.get("mimeType", "")
        if mime.startswith("image/"):
            files.append(DriveFile(id=f["id"], name=f.get("name", f["id"]), mime_type=mime))
    return files, payload.get("nextPageToken")


def list_folder_images(folder_id: str, token: str, max_files: int = 2000) -> list[DriveFile]:
    """List image files directly inside a Drive folder, following pagination."""
    out: list[DriveFile] = []
    page_token: str | None = None
    while len(out) < max_files:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id, name, mimeType)",
            "pageSize": "100",
        }
        if page_token:
            params["pageToken"] = page_token
        url = f"{_API}/files?{urllib.parse.urlencode(params)}"
        payload = json.loads(_get(url, token))
        files, page_token = parse_file_list(payload)
        out.extend(files)
        if not page_token:
            break
    return out[:max_files]


def download_file(file_id: str, token: str) -> bytes:
    """Download a Drive file's bytes (alt=media)."""
    url = f"{_UPLOAD_GET}/{urllib.parse.quote(file_id)}?alt=media"
    return _get(url, token)


if __name__ == "__main__":  # runnable self-check (Constitution V)
    sample = {
        "files": [
            {"id": "1", "name": "a.jpg", "mimeType": "image/jpeg"},
            {"id": "2", "name": "notes", "mimeType": "application/vnd.google-apps.document"},
            {"id": "3", "name": "b.heic", "mimeType": "image/heic"},
        ],
        "nextPageToken": "TOKEN",
    }
    files, token = parse_file_list(sample)
    assert [f.id for f in files] == ["1", "3"], files  # only images
    assert token == "TOKEN"
    empty, none = parse_file_list({"files": []})
    assert empty == [] and none is None
    print("gdrive parse self-check OK")
