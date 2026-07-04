"""FastAPI application entrypoint: OpenAPI docs, logging, error handlers, CORS."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.auth.ratelimit import LOGIN_PATH_SUFFIX, register_attempt
from app.config import get_settings
from app.workers import get_redis

settings = get_settings()


def setup_logging() -> None:
    """Minimal structured logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='{"level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )


setup_logging()

app = FastAPI(
    title="Lahza API",
    version="0.1.0",
    description="Personalized event photo galleries — pooled photos, private per-person galleries.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def login_rate_limit(request: Request, call_next):
    """Throttle repeated login attempts per client IP (brute-force guard)."""
    if request.method == "POST" and request.url.path.endswith(LOGIN_PATH_SUFFIX):
        ip = request.client.host if request.client else "unknown"
        allowed = register_attempt(
            get_redis(),
            ip,
            limit=settings.login_rate_limit,
            window=settings.login_rate_window_seconds,
        )
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many login attempts. Please wait and try again."},
            )
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logging.getLogger("la2etha").exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(api_router)
