import logging
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

SEPARATORS_RO = [
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
]


@dataclass
class ChunkConfig:
    size: int = field(default_factory=lambda: _settings.rag_chunk_size)
    overlap: int = field(default_factory=lambda: _settings.rag_chunk_overlap)
    min_size_to_chunk: int = 4000
    separators: list[str] = field(default_factory=lambda: SEPARATORS_RO.copy())
    preprocessors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.overlap >= self.size:
            raise ValueError(f"overlap ({self.overlap}) must be < size ({self.size})")


CONFIGS: dict[str, ChunkConfig] = {
    "contract": ChunkConfig(size=1500, overlap=200),
    "specificatie": ChunkConfig(size=1000, overlap=150),
    "licitatie": ChunkConfig(size=1200, overlap=150),
    "raport": ChunkConfig(size=1000, overlap=100),
    "default": ChunkConfig(size=1000, overlap=100),
}


def get_config(doc_type: str) -> ChunkConfig:
    return CONFIGS.get(doc_type, CONFIGS["default"])


def validate_chunk(content: str, min_size: int | None = None) -> bool:
    """Validate chunk has meaningful content."""
    min_size = min_size if min_size is not None else _settings.min_chunk_size
    stripped = content.strip()
    return len(stripped) >= min_size and not stripped.isspace()


def register_config(doc_type: str, config: ChunkConfig):
    CONFIGS[doc_type] = config


def should_chunk(docs: list[Document], config: ChunkConfig) -> bool:
    total_chars = sum(len(d.page_content) for d in docs)
    return total_chars > config.min_size_to_chunk


def chunk(docs: list[Document], config: ChunkConfig) -> list[Document]:
    if not should_chunk(docs, config):
        total = sum(len(d.page_content) for d in docs)
        logger.debug(f"Skipping chunking: {total} chars < {config.min_size_to_chunk} threshold")
        return docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.size,
        chunk_overlap=config.overlap,
        separators=config.separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(docs)
    logger.debug(f"Split {len(docs)} docs into {len(chunks)} chunks")
    return chunks
