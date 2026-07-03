"""Synchronous DB session for RQ workers.

CV inference is blocking GPU work, so the worker runs sync code. It uses the
psycopg (sync) URL; the API keeps using the async engine in ``base.py``.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

settings = get_settings()

sync_engine = create_engine(settings.alembic_database_url, pool_pre_ping=True)
SyncSessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False, class_=Session)
