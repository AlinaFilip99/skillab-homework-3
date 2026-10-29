"""
SQLAlchemy Models
"""
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, DateTime, Numeric, Index, func
from sqlalchemy.orm import declarative_base, Mapped, mapped_column

Base = declarative_base()

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class DocumentChunk(Base):
    """Chunk pentru RAG cu pgvector."""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(50))
    content = Column(Text, nullable=False)
    summary = Column(Text)
    metadata_json = Column(Text)
    embedding = Column(Vector(EMBEDDING_DIM))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_document_chunks_embedding', 'embedding', postgresql_using='ivfflat'),
    )


class AchizitieDirecta(Base):
    """Achiziție directă din SEAP."""
    __tablename__ = "achizitii_directe"

    id = Column(Integer, primary_key=True)
    castigator = Column(String(500))
    castigator_cui = Column(String(50))
    castigator_tara = Column(String(100))
    castigator_localitate = Column(String(200))
    castigator_adresa = Column(Text)
    tip_procedura = Column(String(200))
    autoritate_contractanta = Column(String(500))
    autoritate_contractanta_cui = Column(String(50))
    numar_anunt = Column(String(100))
    data_anunt = Column(DateTime)
    descriere = Column(Text)
    tip_incheiere_contract = Column(String(200))
    numar_contract = Column(String(100))
    data_contract = Column(DateTime)
    titlu_contract = Column(Text)
    valoare = Column(Numeric(15, 2))
    moneda = Column(String(10))
    valoare_ron = Column(Numeric(15, 2))
    valoare_eur = Column(Numeric(15, 2))
    cpv_code_id = Column(String(50))
    cpv_code = Column(String(200))


class AnuntInitiere(Base):
    """Anunț inițiere licitație."""
    __tablename__ = "anunturi_initiere"

    id = Column(Integer, primary_key=True)
    tip_anunt = Column(String(200))
    numar_anunt_invitatie = Column(String(100))
    data_publicare = Column(DateTime)
    denumire_ac = Column(String(500))
    cui = Column(String(50))
    judet = Column(String(100))
    tip_contract = Column(String(200))
    utilitati = Column(String(100))
    tip_procedura = Column(String(200))
    criteriu_atribuire = Column(String(200))
    valoare_estimata = Column(Numeric(15, 2))
    moneda = Column(String(10))
    modalitate_desfasurare = Column(String(200))
    trimis_ojeu = Column(String(50))
    fonduri_comunitare = Column(String(50))
    main_cpv_code = Column(String(50))
    main_cpv_name = Column(String(500))

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)                # "user" / "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("ix_conversation_session_created", "session_id", "timestamp"),
    )


class SemanticCache(Base):
    __tablename__ = "semantic_cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    query_text: Mapped[str] = mapped_column(Text)              # query original (debugging)
    query_embedding: Mapped[list[float]] = mapped_column(Vector(384))  # vectorul
    response: Mapped[str] = mapped_column(Text)               # răspunsul cache-uit
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    hit_count: Mapped[int] = mapped_column(Integer, default=0)  # de câte ori a fost servit

    # Index HNSW pentru cosine distance — operatorul <=>.
    # vector_cosine_ops TREBUIE să se potrivească cu distanța din query,
    # altfel Postgres ignoră indexul și face full scan.
    __table_args__ = (
        Index(
            "ix_semantic_cache_embedding_hnsw",
            "query_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"query_embedding": "vector_cosine_ops"},
        ),
    )