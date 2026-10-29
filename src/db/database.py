"""Database connection."""
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings

_settings = get_settings()
engine = create_engine(_settings.active_database_url, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


@contextmanager
def transaction():
    """Context manager pentru tranzacții."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    return SessionLocal()
