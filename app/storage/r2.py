"""Cloudflare R2 storage adapter (S3-compatible).

Same put/get/delete contract as the local-FS adapter, so the pipeline is
unchanged — only ``STORAGE_BACKEND=r2`` differs. R2's free tier (10 GB, no egress
fees) keeps this within the no-paid-services rule.

boto3 is an optional dependency (the `cloud` extra); it's imported lazily so the
app runs without it whenever R2 isn't the selected backend.
"""

from __future__ import annotations

from app.storage.base import StorageAdapter


class R2Storage(StorageAdapter):
    def __init__(
        self, account_id: str, access_key_id: str, secret_access_key: str, bucket: str
    ) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "STORAGE_BACKEND=r2 needs boto3 — install with `pip install -e .[cloud]`."
            ) from exc

        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )

    def put(self, key: str, data: bytes) -> str:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return key

    def get(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)
