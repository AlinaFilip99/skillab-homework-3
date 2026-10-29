"""
RAG Module - Embedding and semantic search.

Usage:
    from src.db import transaction
    from src.rag import RAGService

    with transaction() as session:
        rag = RAGService(session)
        results = rag.search("query", top_k=5)
"""
from .service import RAGService, get_embedding_model

__all__ = ["RAGService", "get_embedding_model"]
