"""
Application Settings - Centralized configuration via environment variables.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+psycopg2://demo:demo123@localhost:5433/rag_demo"
    database_url_test: str = "postgresql+psycopg2://demo:demo123@localhost:5433/rag_demo_test"
    test_mode: bool = False

    @property
    def active_database_url(self) -> str:
        """Return test or production database URL based on test_mode."""
        return self.database_url_test if self.test_mode else self.database_url

    # LLM
    llm_provider: str = "ollama"
    llm_model: str | None = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.0

    # LLM API Keys
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # RAG
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50
    rag_top_k: int = 5
    rag_threshold: float = 0.3

    # Ingest
    ingest_batch_size: int = 100
    ingest_extensions: list[str] = [".pdf", ".txt", ".md"]

    # Extraction
    extraction_model: str | None = None  # Falls back to llm_model

    # Ingest validation
    max_file_size_mb: int = 50
    min_chunk_size: int = 50

    # Checkpoint cleanup
    checkpoint_ttl_hours: int = 24

    # Agents
    max_iterations: int = 3
    max_retries: int = 2

    # Memory
    memory_ttl_days: int = 30

    # Cache
    cache_ttl_hours: int = 1
    cache_similarity_threshold: float = 0.95

    # Guardrails
    guardrail_max_chars: int = 2000
    guardrail_max_words: int = 500
    guardrail_block_urls: bool = True
    guardrail_block_emails: bool = True


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
