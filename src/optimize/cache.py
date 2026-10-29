"""
Semantic Cache - Cache pentru query-uri RAG bazat pe similaritate.

Usage:
    from src.optimize.cache import SemanticCache
    from src.db.database import transaction

    with transaction() as session:
        cache = SemanticCache(session)

        # Verifică cache
        hit = cache.get(query_embedding)
        if hit:
            return hit  # Cache hit

        # Cache miss → compute & store
        result = expensive_rag_search(query)
        cache.set(query_text, query_embedding, result)
"""
import json
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from src.config import get_settings
from src.db.repositories import QueryCacheRepository

logger = logging.getLogger(__name__)
_settings = get_settings()


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Rezultat din cache."""

    query_text: str
    response: Any


class SemanticCache:
    """
    Cache semantic pentru query-uri RAG.

    Folosește pgvector pentru a găsi query-uri similare (cosine similarity).
    Cache hit dacă similaritatea >= threshold (default 0.95).
    Entries expiră după TTL (default 1 oră).
    """

    __slots__ = ("_repo", "_threshold", "_ttl_hours")

    def __init__(
        self,
        db_session: Session,
        threshold: float | None = None,
        ttl_hours: int | None = None,
    ):
        """
        Args:
            db_session: SQLAlchemy session activ
            threshold: Similaritate minimă pentru cache hit (default: din config)
            ttl_hours: Expirare în ore (default: din config)
        """
        self._repo = QueryCacheRepository(db_session)
        self._threshold = threshold or _settings.cache_similarity_threshold
        self._ttl_hours = ttl_hours or _settings.cache_ttl_hours

    def get(self, query_embedding: list[float]) -> CacheEntry | None:
        """
        Caută un query similar în cache.

        Args:
            query_embedding: Embedding-ul query-ului curent

        Returns:
            CacheEntry dacă există hit, None altfel
        """
        entry = self._repo.find_similar(
            embedding=query_embedding,
            threshold=self._threshold,
            max_age_hours=self._ttl_hours,
        )

        if entry:
            logger.info(f"[CACHE] Hit for query: {entry.query_text[:50]}...")
            return CacheEntry(
                query_text=entry.query_text,
                response=json.loads(entry.response_json),
            )

        logger.debug("[CACHE] Miss")
        return None

    def set(
        self,
        query_text: str,
        query_embedding: list[float],
        response: Any,
    ) -> None:
        """
        Adaugă un rezultat în cache.

        Args:
            query_text: Textul original al query-ului
            query_embedding: Embedding-ul query-ului
            response: Rezultatul de cache-uit (serializabil JSON)
        """
        response_json = json.dumps(response, ensure_ascii=False, default=str)
        self._repo.add(query_text, query_embedding, response_json)
        logger.debug(f"[CACHE] Stored: {query_text[:50]}...")

    def cleanup(self) -> int:
        """Șterge entries expirate."""
        deleted = self._repo.delete_expired(self._ttl_hours)
        if deleted:
            logger.info(f"[CACHE] Cleanup: {deleted} expired entries")
        return deleted

    def clear(self) -> int:
        """Șterge tot cache-ul."""
        deleted = self._repo.clear()
        logger.info(f"[CACHE] Cleared {deleted} entries")
        return deleted

    def stats(self) -> dict[str, int]:
        """Statistici cache."""
        return {"entries": self._repo.count()}
