"""Shared test fixtures. Integration tests need Postgres + pgvector (the vector
columns can't run on SQLite); they skip cleanly when the DB is unreachable."""

import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.base import Base

# Import models so metadata is populated for create_all.
from app.db import models  # noqa: F401


async def _db_reachable(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session():
    settings = get_settings()
    url = settings.database_url
    if not await _db_reachable(url):
        pytest.skip("Postgres not reachable — start docker-compose to run integration tests")

    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
