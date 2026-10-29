#!/usr/bin/env python3
"""
Cleanup script for test database.

Usage:
    TEST_MODE=1 python scripts/cleanup_test_db.py
    TEST_MODE=1 python scripts/cleanup_test_db.py --confirm

Warning: This will truncate all tables in the test database!
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def truncate_all_tables(confirm: bool = False) -> int:
    """Truncate all tables in the test database."""
    settings = get_settings()

    if not settings.test_mode:
        logger.error("This script can only run in TEST_MODE!")
        logger.error("Set TEST_MODE=1 environment variable to enable.")
        return 1

    print(f"\nDatabase: {settings.active_database_url}")
    print("-" * 40)

    if not confirm:
        print("\nThis will TRUNCATE ALL TABLES in the test database!")
        print("Use --confirm to proceed.")
        return 0

    from sqlalchemy import text

    from src.db.database import engine

    # Tables to truncate (in order to respect foreign keys)
    tables = [
        "document_chunks",
        # Add other tables as needed
    ]

    with engine.begin() as conn:
        for table in tables:
            try:
                conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                logger.info(f"Truncated: {table}")
            except Exception as e:
                logger.warning(f"Could not truncate {table}: {e}")

    print("\nCleanup complete!")

    # Show remaining counts
    from src.db.database import transaction
    from src.rag.service import RAGService

    with transaction() as session:
        rag = RAGService(session)
        print(f"Remaining chunks: {rag.count()}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cleanup test database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm cleanup (required to actually truncate)",
    )

    args = parser.parse_args()
    return truncate_all_tables(confirm=args.confirm)


if __name__ == "__main__":
    sys.exit(main())
