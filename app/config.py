"""Application settings, loaded from environment (.env) with sane local defaults."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the backend root's .env (…/la2etha-backend/.env). Anchoring to
# this file's location — rather than a bare relative ".env" resolved against the
# current working directory — means the API and the RQ worker load the SAME config
# no matter which directory they're launched from. A worker started from the repo
# root would otherwise silently miss .env and fall back to the localhost:5432
# defaults (wrong Postgres → auth failures, no photos processed).
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    # --- Database ---
    # Async driver used by the app; sync driver used by Alembic.
    database_url: str = "postgresql+asyncpg://la2etha:la2etha@localhost:5432/la2etha"
    alembic_database_url: str = "postgresql+psycopg://la2etha:la2etha@localhost:5432/la2etha"

    # --- Redis / RQ ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Auth ---
    jwt_secret: str = "change-me-in-production"
    jwt_lifetime_seconds: int = 3600
    # Brute-force guard on login: max attempts per client IP per window (Redis).
    # 0 disables it (e.g. tests). Fail-open if Redis is down — availability over
    # lockout; the Cloudflare Tunnel edge also rate-limits in production.
    login_rate_limit: int = 10
    login_rate_window_seconds: int = 60

    # --- Uploads ---
    # Per-file ceiling for uploaded photos (memory/DoS guard). 25 MB covers large
    # phone shots and HEIC; oversize files are rejected 413.
    max_upload_bytes: int = 25 * 1024 * 1024

    # --- CORS (comma-separated allowed origins for the web/mobile clients) ---
    cors_origins: str = "http://localhost:5173"

    # --- Storage ---
    storage_backend: str = "local_fs"  # local_fs | r2 | gdrive
    media_root: str = "./media"

    # --- CV models ---
    insightface_model: str = "buffalo_l"
    onnx_providers: str = "CUDAExecutionProvider,CPUExecutionProvider"

    # --- Semantic search (F5, SigLIP-2) — optional; requires the `search` extra ---
    # If transformers/torch aren't installed the app still runs; search indexing is
    # skipped and the search endpoint returns 503 until the model is available.
    search_model: str = "google/siglip2-base-patch16-224"

    # --- Matching ---
    # Cosine similarity floor for linking an enrollment centroid to a face cluster.
    enroll_match_threshold: float = 0.35

    # --- Quality culling (F3) & background/proximity filtering (F4) ---
    # These are calibration knobs: the "right" values depend on the camera mix at
    # a given event, so they are env-tunable rather than hard-coded. All are used
    # only to DEMOTE photos to the secondary gallery section — never to hide them.
    # Variance-of-Laplacian below this = motion-blur / blank floor-or-pocket shot.
    quality_blur_min: float = 55.0
    # A face smaller than this fraction of the frame is an incidental bystander.
    proximity_area_min: float = 0.012
    # Face-crop sharpness (VoL) below this corroborates an out-of-focus background face.
    proximity_sharpness_min: float = 30.0

    # --- Curation (spec 004): best-shot / highlight scoring weights ---
    # Signals-only composition of already-stored per-face/photo columns (research
    # R1/R2) — tunable per event's camera mix, like the quality thresholds above.
    curation_w_quality: float = 0.3
    curation_w_det: float = 0.2
    curation_w_area: float = 0.3
    curation_w_sharpness: float = 0.2
    curation_area_ref: float = 0.05
    curation_confident_det_min: float = 0.6
    curation_confident_area_min: float = 0.01
    curation_group_bonus: float = 1.25
    curation_group_min_faces: int = 3

    # --- Optional / later phases ---
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: str | None = None

    # Cloudflare R2 (S3-compatible) — only used when STORAGE_BACKEND=r2.
    # Free tier: 10 GB storage + no egress fees. All optional; blank = disabled.
    r2_account_id: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket: str | None = None

    # Gemini "Nano Banana" AI photo edit (F7 stretch). Opt-in, consented,
    # solo-photo-only. Free tier via Google AI Studio. Blank = feature disabled
    # (the edit endpoint returns 503). This is the ONLY path where an image may
    # leave the machine, and only for a photo of just the requesting user.
    gemini_api_key: str | None = None
    gemini_image_model: str = "gemini-2.5-flash-image"
    # Solo-photo guard for AI edit (FR-017). Secure by default: only a photo of
    # just the caller may be sent to the cloud editor. Set EDIT_SOLO_ONLY=false
    # in a dev .env to relax it while trying the system out — never in a deploy.
    edit_solo_only: bool = True

    @property
    def onnx_provider_list(self) -> list[str]:
        return [p.strip() for p in self.onnx_providers.split(",") if p.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
