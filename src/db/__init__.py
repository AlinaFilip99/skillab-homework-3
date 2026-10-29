"""
Database Module - SQLAlchemy models and repositories.

Usage:
    from src.db import transaction, DocumentChunkRepository

    with transaction() as session:
        repo = DocumentChunkRepository(session)
        chunks = repo.search_similar(embedding, top_k=5)
"""
from .database import engine, get_session, SessionLocal, transaction
from .models import Base, DocumentChunk, AchizitieDirecta, AnuntInitiere, EMBEDDING_DIM
from .repositories import DocumentChunkRepository, AchizitieRepository, AnuntRepository

__all__ = [
    "engine",
    "get_session",
    "SessionLocal",
    "transaction",
    "Base",
    "DocumentChunk",
    "AchizitieDirecta",
    "AnuntInitiere",
    "EMBEDDING_DIM",
    "DocumentChunkRepository",
    "AchizitieRepository",
    "AnuntRepository",
]
