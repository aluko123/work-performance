from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os


#path for SQLite db file
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/analysis.db")

#engine
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

#session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#inherit from base class to create each table
Base = declarative_base()

def init_db():
    #create all tables
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")

