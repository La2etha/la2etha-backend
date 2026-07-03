"""Application settings, loaded from environment (.env) with sane local defaults."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database ---
    # Async driver used by the app; sync driver used by Alembic.
    database_url: str = "postgresql+asyncpg://la2etha:la2etha@localhost:5432/la2etha"
    alembic_database_url: str = "postgresql+psycopg://la2etha:la2etha@localhost:5432/la2etha"

    # --- Redis / RQ ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Auth ---
    jwt_secret: str = "change-me-in-production"
    jwt_lifetime_seconds: int = 3600

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

    @property
    def onnx_provider_list(self) -> list[str]:
        return [p.strip() for p in self.onnx_providers.split(",") if p.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
