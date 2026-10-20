# ─────────────────────────────────────────────────────────
# Slide 3 · Cache Invalidation — TTL + topic-based + manual
# Extinde semantic cache-ul (Secțiunea 5) cu logică de expirare.
# "There are only two hard things in CS: cache invalidation
#  and naming things." — Phil Karlton
# ─────────────────────────────────────────────────────────
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, delete
from sqlalchemy.orm import sessionmaker

from models import SemanticCache
from database import transaction

from typing import Union

def embed(texts: Union[str, list[str]]) -> list[list[float]]:
    """
    Tiny embedding helper for the caching demos (semantic cache, embedding
    cache). Anthropic has no first-party embeddings endpoint, so we use
    sentence-transformers locally — zero extra API keys, runs offline.

        pip install sentence-transformers
    """
    from sentence_transformers import SentenceTransformer

    global _embed_model
    try:
        _embed_model
    except NameError:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if isinstance(texts, str):
        texts = [texts]
    return _embed_model.encode(texts, normalize_embeddings=True).tolist()

class SmartCache:
    def __init__(self, threshold: float = 0.92, ttl: int = 3600):
        self.threshold = threshold
        self.ttl = ttl                          # time-to-live în secunde (1h)

    def get(self, query: str) -> str | None:
        emb = embed(query)[0]
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.ttl)
        with transaction() as db:
            distance = SemanticCache.query_embedding.cosine_distance(emb)
            row = db.execute(
                select(SemanticCache, distance.label("dist"))
                .where(SemanticCache.created_at >= cutoff)   # ⏱️ doar entries ne-expirate
                .order_by(distance)
                .limit(1)
            ).first()

            if row is None:
                return None
            cache, dist = row
            if (1 - dist) < self.threshold:
                return None
            cache.hit_count += 1
            db.commit()
            return cache.response

    def set(self, query: str, response: str) -> None:
        """Salvează un query + răspuns nou în cache."""
        emb = embed(query)[0]
        with transaction() as db:
            db.add(SemanticCache(query_text=query, query_embedding=emb, response=response))
            db.commit()

    # 🔄 Topic-based: șterge tot ce ține de un subiect (ex: doc sursă schimbat)
    def invalidate_by_topic(self, topic: str) -> int:
        with transaction() as db:
            result = db.execute(
                delete(SemanticCache).where(SemanticCache.query_text.ilike(f"%{topic}%"))
            )
            db.commit()
            return result.rowcount

    # 👆 Manual: user cere explicit "refresh" → golește tot
    def clear(self) -> None:
        with transaction() as db:
            db.query(SemanticCache).delete()
            db.commit()

    # 🧹 Curățenie: șterge proactiv entries expirate (rulat periodic / cron)
    def purge_expired(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.ttl)
        with transaction() as db:
            result = db.execute(delete(SemanticCache).where(SemanticCache.created_at < cutoff))
            db.commit()
            return result.rowcount
