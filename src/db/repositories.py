"""Repository Pattern."""
from datetime import datetime, timedelta

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from .models import (
    AchizitieDirecta,
    AnuntInitiere,
    ConversationMessage,
    DocumentChunk,
    ExtractedContract,
    ExtractedInvoice,
    QueryCache,
)


class DocumentChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, chunk: DocumentChunk) -> DocumentChunk:
        self.session.add(chunk)
        self.session.flush()
        return chunk

    def add_batch(self, chunks: list[DocumentChunk]) -> int:
        self.session.add_all(chunks)
        self.session.flush()
        return len(chunks)

    def get_by_id(self, chunk_id: int) -> DocumentChunk | None:
        return self.session.query(DocumentChunk).filter_by(id=chunk_id).first()

    def get_by_file(self, file_name: str) -> list[DocumentChunk]:
        return (
            self.session.query(DocumentChunk)
            .filter_by(file_name=file_name)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

    def search_similar(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """Caută chunks similare folosind cosine similarity."""
        query = text("""
            SELECT id, 1 - (embedding <=> CAST(:embedding AS vector)) as score
            FROM document_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        embedding_str = f"[{','.join(map(str, embedding))}]"
        results = self.session.execute(
            query, {"embedding": embedding_str, "top_k": top_k}
        ).fetchall()

        chunk_ids = [row.id for row in results if row.score >= threshold]
        if not chunk_ids:
            return []

        chunks = (
            self.session.query(DocumentChunk)
            .filter(DocumentChunk.id.in_(chunk_ids))
            .all()
        )
        chunk_map = {c.id: c for c in chunks}

        return [
            (chunk_map[row.id], row.score)
            for row in results
            if row.score >= threshold and row.id in chunk_map
        ]

    def count(self) -> int:
        return self.session.query(func.count(DocumentChunk.id)).scalar() or 0

    def delete_by_file(self, file_name: str) -> int:
        return self.session.query(DocumentChunk).filter_by(file_name=file_name).delete()

    def delete_all(self) -> int:
        return self.session.query(DocumentChunk).delete()

    def get_by_content_hash(self, content_hash: str) -> list[DocumentChunk]:
        """Find chunks by content hash for deduplication."""
        return (
            self.session.query(DocumentChunk)
            .filter_by(content_hash=content_hash)
            .all()
        )


class AchizitieRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_batch(self, records: list[dict]) -> int:
        self.session.add_all([AchizitieDirecta(**r) for r in records])
        self.session.flush()
        return len(records)

    def count(self) -> int:
        return self.session.query(func.count(AchizitieDirecta.id)).scalar() or 0

    def delete_all(self) -> int:
        return self.session.query(AchizitieDirecta).delete()


class AnuntRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_batch(self, records: list[dict]) -> int:
        self.session.add_all([AnuntInitiere(**r) for r in records])
        self.session.flush()
        return len(records)

    def count(self) -> int:
        return self.session.query(func.count(AnuntInitiere.id)).scalar() or 0

    def delete_all(self) -> int:
        return self.session.query(AnuntInitiere).delete()


class ConversationRepository:
    """Repository pentru mesaje de conversație."""

    def __init__(self, session: Session):
        self.session = session

    def add(self, session_id: str, role: str, content: str) -> ConversationMessage:
        msg = ConversationMessage(session_id=session_id, role=role, content=content)
        self.session.add(msg)
        self.session.flush()
        return msg

    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ConversationMessage]:
        """Returnează mesajele unei conversații, ordonate cronologic."""
        if limit:
            # Subquery pentru ultimele N, apoi ordonare ASC
            subq = (
                self.session.query(ConversationMessage.id)
                .filter_by(session_id=session_id)
                .order_by(ConversationMessage.created_at.desc())
                .limit(limit)
                .subquery()
            )
            return (
                self.session.query(ConversationMessage)
                .filter(ConversationMessage.id.in_(subq))
                .order_by(ConversationMessage.created_at.asc())
                .all()
            )

        return (
            self.session.query(ConversationMessage)
            .filter_by(session_id=session_id)
            .order_by(ConversationMessage.created_at.asc())
            .all()
        )

    def count(self, session_id: str) -> int:
        return (
            self.session.query(func.count(ConversationMessage.id))
            .filter_by(session_id=session_id)
            .scalar() or 0
        )

    def clear(self, session_id: str) -> int:
        return (
            self.session.query(ConversationMessage)
            .filter_by(session_id=session_id)
            .delete()
        )

    def delete_older_than(self, days: int) -> int:
        """Șterge mesajele mai vechi de N zile."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        return (
            self.session.query(ConversationMessage)
            .filter(ConversationMessage.created_at < cutoff)
            .delete()
        )

    def get_all_session_ids(self) -> list[str]:
        results = self.session.query(ConversationMessage.session_id).distinct().all()
        return [r[0] for r in results]


class QueryCacheRepository:
    """Repository pentru cache semantic RAG."""

    def __init__(self, session: Session):
        self.session = session

    def add(
        self,
        query_text: str,
        query_embedding: list[float],
        response_json: str,
    ) -> QueryCache:
        entry = QueryCache(
            query_text=query_text,
            query_embedding=query_embedding,
            response_json=response_json,
        )
        self.session.add(entry)
        self.session.flush()
        return entry

    def find_similar(
        self,
        embedding: list[float],
        threshold: float = 0.95,
        max_age_hours: int = 1,
    ) -> QueryCache | None:
        """Găsește un query similar în cache (dacă există și nu e expirat)."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        embedding_str = f"[{','.join(map(str, embedding))}]"

        query = text("""
            SELECT id, 1 - (query_embedding <=> CAST(:embedding AS vector)) as score
            FROM query_cache
            WHERE created_at > :cutoff
            ORDER BY query_embedding <=> CAST(:embedding AS vector)
            LIMIT 1
        """)

        result = self.session.execute(
            query, {"embedding": embedding_str, "cutoff": cutoff}
        ).first()

        if result and result.score >= threshold:
            return self.session.query(QueryCache).filter_by(id=result.id).first()

        return None

    def delete_expired(self, max_age_hours: int) -> int:
        """Șterge entries expirate."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        return self.session.query(QueryCache).filter(QueryCache.created_at < cutoff).delete()

    def count(self) -> int:
        return self.session.query(func.count(QueryCache.id)).scalar() or 0

    def clear(self) -> int:
        return self.session.query(QueryCache).delete()


class ExtractedInvoiceRepository:
    """Repository pentru facturi extrase."""

    def __init__(self, session: Session):
        self.session = session

    def add(self, invoice: ExtractedInvoice) -> ExtractedInvoice:
        self.session.add(invoice)
        self.session.flush()
        return invoice

    def get_by_id(self, invoice_id: int) -> ExtractedInvoice | None:
        return self.session.query(ExtractedInvoice).filter_by(id=invoice_id).first()

    def get_by_file(self, file_name: str) -> ExtractedInvoice | None:
        return self.session.query(ExtractedInvoice).filter_by(file_name=file_name).first()

    def list_all(self, limit: int = 100) -> list[ExtractedInvoice]:
        return self.session.query(ExtractedInvoice).order_by(ExtractedInvoice.extracted_at.desc()).limit(limit).all()

    def delete_by_file(self, file_name: str) -> int:
        return self.session.query(ExtractedInvoice).filter_by(file_name=file_name).delete()

    def count(self) -> int:
        return self.session.query(func.count(ExtractedInvoice.id)).scalar() or 0


class ExtractedContractRepository:
    """Repository pentru contracte extrase."""

    def __init__(self, session: Session):
        self.session = session

    def add(self, contract: ExtractedContract) -> ExtractedContract:
        self.session.add(contract)
        self.session.flush()
        return contract

    def get_by_id(self, contract_id: int) -> ExtractedContract | None:
        return self.session.query(ExtractedContract).filter_by(id=contract_id).first()

    def get_by_file(self, file_name: str) -> ExtractedContract | None:
        return self.session.query(ExtractedContract).filter_by(file_name=file_name).first()

    def list_all(self, limit: int = 100) -> list[ExtractedContract]:
        return self.session.query(ExtractedContract).order_by(ExtractedContract.extracted_at.desc()).limit(limit).all()

    def delete_by_file(self, file_name: str) -> int:
        return self.session.query(ExtractedContract).filter_by(file_name=file_name).delete()

    def count(self) -> int:
        return self.session.query(func.count(ExtractedContract.id)).scalar() or 0
