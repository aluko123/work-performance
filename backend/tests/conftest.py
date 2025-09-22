import os
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base


@pytest.fixture()
def temp_db_session():
    """
    Creates a temporary SQLite database and returns a Session factory bound to it.
    Tables are created from backend.database.Base metadata.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    url = f"sqlite:///{path}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    try:
        yield SessionLocal
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


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

