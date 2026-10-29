import logging
import sys
from pathlib import Path

from .chunking import CONFIGS, ChunkConfig, get_config, register_config
from .extraction import ExtractionService
from .loaders import LOADERS, load, register_loader
from .pipeline import (
    BatchResult,
    DocumentPipeline,
    IngestResult,
    ingest,
    ingest_batch,
    ingest_directory,
)
from .preprocessors import PREPROCESSORS, preprocess, register_preprocessor
from .schemas import SCHEMA_REGISTRY, Contract, Invoice, Product, get_schema

__all__ = [
    "DocumentPipeline",
    "IngestResult",
    "BatchResult",
    "ChunkConfig",
    "ingest",
    "ingest_batch",
    "ingest_directory",
    "load",
    "preprocess",
    "get_config",
    "register_config",
    "register_loader",
    "register_preprocessor",
    "LOADERS",
    "PREPROCESSORS",
    "CONFIGS",
    # Extraction
    "ExtractionService",
    "Invoice",
    "Contract",
    "Product",
    "SCHEMA_REGISTRY",
    "get_schema",
]

logger = logging.getLogger(__name__)


def ingest_file(path: str | Path, doc_type: str = "default") -> int:
    result = ingest(path, doc_type)
    if not result.success:
        raise RuntimeError(result.error)
    return result.chunks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.documents <path> [doc_type]")
        print()
        print("Document types:", ", ".join(CONFIGS.keys()))
        print("Supported formats:", ", ".join(LOADERS.keys()))
        sys.exit(1)

    target = Path(sys.argv[1])
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "default"

    if doc_type not in CONFIGS:
        print(f"Unknown doc_type: {doc_type}")
        print("Available:", ", ".join(CONFIGS.keys()))
        sys.exit(1)

    config = get_config(doc_type)
    print(f"Config: chunk_size={config.size}, overlap={config.overlap}")
    print()

    if target.is_file():
        result = ingest(target, doc_type)
        status = "✓" if result.success else "✗"
        if result.success:
            print(f"{status} {result.file_name}: {result.chunks} chunks, {result.chars} chars")
        else:
            print(f"{status} {result.file_name}: {result.error}")

    elif target.is_dir():
        batch = ingest_directory(target, doc_type)
        for name, result in batch.results.items():
            status = "✓" if result.success else "✗"
            info = f"{result.chunks} chunks" if result.success else result.error
            print(f"{status} {name}: {info}")
        print()
        print(f"Total: {batch.success_count} files, {batch.total_chunks} chunks, {batch.total_chars} chars")
        if batch.failed_count:
            print(f"Failed: {batch.failed_count}")

    else:
        print(f"Not found: {target}")
        sys.exit(1)
