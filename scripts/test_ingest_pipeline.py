#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# CONFIGURATION - Set these before running
# =============================================================================
FILE_PATH = "scripts/sample_docs/contract_test.txt"
DOC_TYPE = "default"  # "default", "invoice", "contract"
DRY_RUN = True
EXTRACT = False  # Set to True to extract structured data (requires DOC_TYPE = "invoice" or "contract")
GENERATE_EMBEDDINGS = True  # Generate embeddings in dry-run mode (no DB storage)
# =============================================================================

from src.ingest.pipeline import DocumentPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    path = Path(FILE_PATH)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return 1

    extract_schema = None
    if EXTRACT and DOC_TYPE in ("invoice", "contract"):
        extract_schema = DOC_TYPE

    pipeline = DocumentPipeline(
        doc_type=DOC_TYPE,
        use_checkpoints=not DRY_RUN,
        extract_schema=extract_schema,
        dry_run=DRY_RUN,
        generate_embeddings=GENERATE_EMBEDDINGS if DRY_RUN else False,
    )

    mode = "[DRY RUN] " if DRY_RUN else ""
    logger.info(f"{mode}Processing: {path}")

    result = pipeline.ingest(path)

    if not result.success:
        logger.error(f"Failed: {result.error}")
        return 1

    print(f"\n{mode}Result:")
    print(f"  File: {result.file_name}")
    print(f"  Chunks: {result.chunks}")
    print(f"  Characters: {result.chars}")
    print(f"  Pages: {result.pages}")

    if result.extracted_data:
        print(f"  Extracted data: {json.dumps(result.extracted_data, indent=2, ensure_ascii=False)}")

    if result.chunks_data:
        print(f"\n  Chunks preview (first 3):")
        for i, chunk in enumerate(result.chunks_data[:3]):
            preview = chunk["content"][:80].replace("\n", " ")
            print(f"    [{i}] {preview}...")
            print(f"        hash: {chunk['content_hash']}")
            if "embedding" in chunk:
                print(f"        embedding: dim={len(chunk['embedding'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
