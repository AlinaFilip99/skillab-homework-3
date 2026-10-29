"""
SQLAlchemy Models - DocumentChunk (RAG) + SEAP tables + Memory + Cache + Extraction.
"""
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import declarative_base

from src.config import get_settings

Base = declarative_base()

_settings = get_settings()
EMBEDDING_DIM = _settings.embedding_dim


class DocumentChunk(Base):
    """Chunk de document pentru RAG cu pgvector."""

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
    content_hash = Column(String(32), index=True)  # MD5 for dedup
    embedding_model = Column(String(100))  # e.g., "all-MiniLM-L6-v2"
    embedding_version = Column(String(20))  # e.g., "1.0"

    __table_args__ = (
        Index(
            "ix_document_chunks_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class AchizitieDirecta(Base):
    """Achiziție directă SEAP."""

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
    """Anunț inițiere licitație SEAP."""

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


class ConversationMessage(Base):
    """Mesaj dintr-o conversație persistată."""

    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_conversation_session_created", "session_id", "created_at"),
    )


class QueryCache(Base):
    """Cache pentru query-uri RAG cu semantic matching."""

    __tablename__ = "query_cache"

    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    query_embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    response_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index(
            "ix_query_cache_embedding",
            "query_embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"query_embedding": "vector_cosine_ops"},
        ),
    )


class ExtractedInvoice(Base):
    """Extracted invoice data from documents."""

    __tablename__ = "extracted_invoices"

    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), nullable=False, index=True)
    numar = Column(String(100))
    data = Column(String(50))
    furnizor = Column(String(500))
    client = Column(String(500))
    produse_json = Column(Text)  # JSON array of products
    subtotal = Column(Numeric(15, 2))
    tva = Column(Numeric(15, 2))
    total = Column(Numeric(15, 2))
    extracted_at = Column(DateTime, default=datetime.utcnow)


class ExtractedContract(Base):
    """Extracted contract data from documents."""

    __tablename__ = "extracted_contracts"

    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), nullable=False, index=True)
    numar = Column(String(100))
    data_incheiere = Column(String(50))
    prestator = Column(String(500))
    beneficiar = Column(String(500))
    valoare = Column(Numeric(15, 2))
    durata_luni = Column(Integer)
    obligatii_json = Column(Text)  # JSON array of obligations
    extracted_at = Column(DateTime, default=datetime.utcnow)
