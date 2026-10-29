"""RAG Service."""
import hashlib
import logging
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from src.config import get_settings
from src.db.models import DocumentChunk
from src.db.repositories import DocumentChunkRepository

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
_settings = get_settings()

EMBEDDING_VERSION = "1.0"

_embedding_model: "SentenceTransformer | None" = None


def get_embedding_model() -> "SentenceTransformer":
    """Încarcă modelul de embedding (lazy singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {_settings.embedding_model}")
        _embedding_model = SentenceTransformer(_settings.embedding_model)
    return _embedding_model


class RAGService:
    """Embedding + Search."""

    def __init__(self, session: Session):
        self.session = session
        self.repo = DocumentChunkRepository(session)

    def embed_text(self, text: str) -> list[float]:
        embedding = get_embedding_model().encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = get_embedding_model().encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def add_chunk(
        self,
        file_name: str,
        chunk_index: int,
        content: str,
        chunk_type: str = "paragraph",
        summary: str = "",
        metadata_json: str = "{}",
    ) -> DocumentChunk:
        chunk = DocumentChunk(
            file_name=file_name,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            content=content,
            summary=summary,
            metadata_json=metadata_json,
            embedding=self.embed_text(content),
        )
        return self.repo.add(chunk)

    def add_chunks_batch(self, file_name: str, chunks: list[dict]) -> int:
        """Adaugă chunks cu batch encoding."""
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        embeddings = self.embed_batch(texts)

        db_chunks = [
            DocumentChunk(
                file_name=file_name,
                chunk_index=i,
                chunk_type=c.get("chunk_type", "paragraph"),
                content=c["content"],
                summary=c.get("summary", ""),
                metadata_json=c.get("metadata_json", "{}"),
                embedding=emb,
                content_hash=c.get("content_hash") or hashlib.md5(c["content"].encode()).hexdigest(),
                embedding_model=_settings.embedding_model,
                embedding_version=EMBEDDING_VERSION,
            )
            for i, (c, emb) in enumerate(zip(chunks, embeddings))
        ]
        return self.repo.add_batch(db_chunks)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list[tuple[DocumentChunk, float]]:
        """Caută chunks similare semantic."""
        return self.repo.search_similar(self.embed_text(query), top_k, threshold)

    def count(self) -> int:
        return self.repo.count()

    def delete_file(self, file_name: str) -> int:
        return self.repo.delete_by_file(file_name)
