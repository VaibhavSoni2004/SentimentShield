import logging
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import dotenv_values, load_dotenv
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

# Safely load environment variables from local .env if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def get_database_url() -> str:
    """Retrieves, validates, and normalizes the database connection URL based on the environment.

    - In development/test: If DATABASE_URL is unset, defaults to local SQLite (feedback.db).
    - In production ('production' or 'prod'): Strictly requires DATABASE_URL to be set to a
      valid PostgreSQL connection string. Silently falling back to SQLite is rejected.
    - Normalizes legacy 'postgres://' URLs (provided by some cloud hosts) to 'postgresql://'.
    - Automatically appends 'sslmode=require' for remote PostgreSQL hosts (e.g. Supabase).
    """
    env = os.getenv("ENVIRONMENT", "development").strip().lower()
    raw_url = os.getenv("DATABASE_URL")
    url = raw_url.strip().strip("'\"") if raw_url else ""

    if env in ("production", "prod"):
        if not url:
            raise RuntimeError(
                "CRITICAL CONFIGURATION ERROR: In production, DATABASE_URL must be explicitly "
                "configured with a PostgreSQL connection string. SQLite fallback is not permitted."
            )
        if url.startswith("sqlite"):
            raise RuntimeError(
                "CRITICAL CONFIGURATION ERROR: In production, DATABASE_URL must point to a PostgreSQL "
                "database (e.g. postgresql://...), not SQLite."
            )

    # If unset in development/test, default to local SQLite database in project directory
    if not url:
        return f"sqlite:///{BASE_DIR / 'feedback.db'}"

    # Normalize 'postgres://' to 'postgresql://' for SQLAlchemy 1.4+ compatibility
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Ensure SSL is enforced for remote PostgreSQL hosts (such as Supabase or Render)
    if url.startswith("postgresql://") or url.startswith("postgresql+"):
        parsed = urlparse(url)
        if parsed.hostname and parsed.hostname not in ("localhost", "127.0.0.1"):
            qs = parse_qs(parsed.query)
            if "sslmode" not in qs:
                delimiter = "&" if parsed.query else "?"
                url = f"{url}{delimiter}sslmode=require"

    return url


def create_db_engine(url: str | None = None):
    """Creates a SQLAlchemy engine configured appropriately for the target database dialect."""
    if url is None:
        url = get_database_url()

    if url.startswith("sqlite"):
        return create_engine(
            url,
            connect_args={"check_same_thread": False}
        )

    # PostgreSQL configuration (optimized for Supabase / Render connection pooling)
    connect_args = {}
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname not in ("localhost", "127.0.0.1"):
        connect_args["sslmode"] = "require"

    return create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,  # Transparently recycles dropped/idle connections
        pool_size=10,
        max_overflow=20,
    )


DATABASE_URL = get_database_url()
engine = create_db_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class FeedbackModel(Base):
    __tablename__ = "feedback_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    text = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    polarity_score = Column(Float, nullable=False)
    urgency = Column(String, nullable=False)
    urgency_score = Column(Integer, default=50)
    status = Column(String, default="Pending")


class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False)


def init_db(target_engine=None):
    """Initializes tables in the configured database (idempotent, does not drop data)."""
    Base.metadata.create_all(bind=target_engine or engine)


def get_db():
    """Yields a database session from the session maker."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
