import os
import tempfile
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend import db_models


@pytest.fixture()
def temp_db_session():
    """
    Postgres-only test database session factory with table truncation.
    Uses TEST_DATABASE_URL or DATABASE_URL, otherwise defaults to docker-compose DSN.
    Truncates tables before each test for isolation.
    """
    pg_url = (
        os.getenv("TEST_DATABASE_URL")
        or "postgresql+psycopg2://workperf:%s@postgres:5432/performance" % os.getenv("DB_PASSWORD", "devpassword")
    )

    engine = create_engine(pg_url)
    # connectivity check
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise RuntimeError(
            f"Postgres is required for tests. Could not connect to {pg_url}. Error: {e}"
        )

    Base.metadata.create_all(bind=engine)
    
    # Truncate tables for test isolation
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE utterances, analyses RESTART IDENTITY CASCADE"))
        conn.commit()
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    try:
        yield SessionLocal
    finally:
        # Clean up after test
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE utterances, analyses RESTART IDENTITY CASCADE"))
            conn.commit()


class FakeRedis:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


@pytest.fixture()
def fake_redis_client():
    return FakeRedis()
