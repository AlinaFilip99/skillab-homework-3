#!/usr/bin/env python3
"""
Test script for RAG search functionality.

Usage:
    TEST_MODE=1 python scripts/test_rag_search.py "search query"
    TEST_MODE=1 python scripts/test_rag_search.py --stats
    TEST_MODE=1 python scripts/test_rag_search.py "query" --top-k 10 --threshold 0.3

Examples:
    TEST_MODE=1 python scripts/test_rag_search.py "contract servicii"
    TEST_MODE=1 python scripts/test_rag_search.py --stats
    TEST_MODE=1 python scripts/test_rag_search.py "delivery terms" --top-k 3
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.db.database import transaction
from src.rag.service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_search(args: argparse.Namespace) -> int:
    """Search for similar chunks."""
    settings = get_settings()
    logger.info(f"Test mode: {settings.test_mode}")
    logger.info(f"Database: {settings.active_database_url}")

    query = args.query
    top_k = args.top_k
    threshold = args.threshold

    print(f"\nSearching for: '{query}'")
    print(f"Parameters: top_k={top_k}, threshold={threshold}")
    print("-" * 60)

    with transaction() as session:
        rag = RAGService(session)
        results = rag.search(query, top_k=top_k, threshold=threshold)

        if not results:
            print("No results found.")
            return 0

        print(f"Found {len(results)} results:\n")

        for i, (chunk, score) in enumerate(results, 1):
            print(f"--- Result {i} (score: {score:.4f}) ---")
            print(f"File: {chunk.file_name}")
            print(f"Chunk index: {chunk.chunk_index}")

            # Parse and show metadata
            try:
                metadata = json.loads(chunk.metadata_json)
                if metadata:
                    print(f"Metadata: {metadata}")
            except json.JSONDecodeError:
                pass

            # Show content (truncated)
            content = chunk.content
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"Content:\n{content}")
            print()

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show database statistics."""
    settings = get_settings()
    print(f"\nTest mode: {settings.test_mode}")
    print(f"Database: {settings.active_database_url}")
    print("-" * 40)

    with transaction() as session:
        rag = RAGService(session)

        # Total count
        total = rag.count()
        print(f"Total chunks: {total}")

        # Get file breakdown
        from sqlalchemy import func

        from src.db.models import DocumentChunk

        file_stats = (
            session.query(
                DocumentChunk.file_name,
                func.count(DocumentChunk.id).label("count"),
            )
            .group_by(DocumentChunk.file_name)
            .all()
        )

        if file_stats:
            print("\nChunks per file:")
            for file_name, count in file_stats:
                print(f"  {file_name}: {count}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test RAG search functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Search query",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum similarity threshold (default: 0.0)",
    )

    args = parser.parse_args()

    if args.stats:
        return cmd_stats(args)
    elif args.query:
        return cmd_search(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
