# db/connection.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from .models import Base

# Database URL from environment with fallback for development
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://sentiment_user:sentiment_pass@localhost:5432/sentiment_db"
)

# Hide password in logs for security
safe_url = DATABASE_URL.replace(DATABASE_URL.split('@')[0].split('://')[-1], '***')
print(f"Connecting to database: {safe_url}")

# Create the database engine
engine = create_engine(
    DATABASE_URL,
    echo=True,  # Shows SQL queries in logs 
    pool_size=10,  # Number of connections to keep open
    max_overflow=20  # Additional connections if needed
)

# Create the database engine
engine = create_engine(
    DATABASE_URL,
    echo=True,  # Shows SQL queries in logs 
    pool_size=10,  # Number of connections to keep open
    max_overflow=20  # Additional connections if needed
)

# Create session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,  # Don't auto-commit changes 
    autoflush=False,   # Don't auto-flush changes to DB
    expire_on_commit=False  # Keep objects usable after commit
)

def get_db():
    """
    Creates a database session and ensures it gets closed properly
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """
    Creates all database tables defined in our models
    Only creates tables that don't already exist
    """
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise
