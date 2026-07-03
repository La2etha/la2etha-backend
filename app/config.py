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

    # --- Matching ---
    # Cosine similarity floor for linking an enrollment centroid to a face cluster.
    enroll_match_threshold: float = 0.35

    # --- Optional / later phases ---
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: str | None = None

    @property
    def onnx_provider_list(self) -> list[str]:
        return [p.strip() for p in self.onnx_providers.split(",") if p.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
