"""Shared test fixtures. Integration tests need Postgres + pgvector (the vector
columns can't run on SQLite); they skip cleanly when the DB is unreachable.

Tests run against a SEPARATE database (``<app-db>_test``), auto-created if
missing, so the create_all/drop_all lifecycle never touches the app's dev data.
"""

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.base import Base

# Import models so metadata is populated for create_all.
from app.db import models  # noqa: F401


def _test_url(app_url: str) -> str:
    """Derive the test database URL: same server, ``<name>_test`` database."""
    url = make_url(app_url)
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


async def _ensure_test_database(test_url: str) -> bool:
    """Create the test database if it doesn't exist. Returns False if the server
    is unreachable (so the fixture can skip cleanly)."""
    url = make_url(test_url)
    # CREATE DATABASE can't run in a transaction — connect to the maintenance
    # "postgres" db with autocommit.
    admin_url = url.set(database="postgres").render_as_string(hide_password=False)
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
            )
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{url.database}"'))
        return True
    except Exception:
        return False
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session():
    test_url = _test_url(get_settings().database_url)
    if not await _ensure_test_database(test_url):
        pytest.skip("Postgres not reachable — start docker-compose to run integration tests")

    engine = create_async_engine(test_url)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Drop first so the schema always matches the current models even if a
        # stale table lingers from a prior run (create_all won't alter columns).
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
