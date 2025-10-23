from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os


#path for SQLite db file
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/analysis.db")

#engine
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})


# Enable SQLite performance optimizations (WAL mode + other pragmas)
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """
    Configure SQLite for better write performance.
    WAL mode allows concurrent reads during writes and significantly improves throughput.
    """
    if "sqlite" in SQLALCHEMY_DATABASE_URL:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.execute("PRAGMA busy_timeout=5000")   # 5 second timeout
        cursor.close()
        print("✅ SQLite optimizations enabled: WAL mode, NORMAL sync, 64MB cache")


#session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#inherit from base class to create each table
Base = declarative_base()

def init_db():
    #create all tables
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")

