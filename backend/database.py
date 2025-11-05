from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os


# Database URL (Postgres only)
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://workperf:%s@postgres:5432/performance" % os.getenv("DB_PASSWORD", "devpassword")
)

# Postgres engine with pooling
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base declarative
Base = declarative_base()


def init_db():
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")
