"""Tabular Ingest - Load CSV into SEAP tables."""
import logging
from pathlib import Path

import pandas as pd

from src.db.database import transaction
from src.db.repositories import AchizitieRepository, AnuntRepository

logger = logging.getLogger(__name__)

REPOS = {
    "achizitii": AchizitieRepository,
    "anunturi": AnuntRepository,
}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.where(pd.notnull(df), None)
    for col in df.columns:
        if "data" in col.lower() or "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ingest_csv(table: str, path: str | Path) -> int:
    """Ingestează CSV în tabel."""
    if table not in REPOS:
        raise ValueError(f"Unknown table: {table}. Use: {list(REPOS.keys())}")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Ingesting {table} from: {path.name}")

    df = clean_dataframe(pd.read_csv(path))
    records = df.to_dict("records")

    with transaction() as session:
        repo = REPOS[table](session)
        deleted = repo.delete_all()
        if deleted:
            logger.info(f"Deleted {deleted} old records")
        count = repo.add_batch(records)

    logger.info(f"Ingested {count} {table} records")
    return count


def ingest_achizitii(path: str | Path) -> int:
    return ingest_csv("achizitii", path)


def ingest_anunturi(path: str | Path) -> int:
    return ingest_csv("anunturi", path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python -m src.ingest.tabular <achizitii|anunturi> <csv_file>")
        sys.exit(1)

    table_type = sys.argv[1].lower()
    try:
        print(f"Ingested {ingest_csv(table_type, sys.argv[2])} records")
    except ValueError as e:
        print(e)
        sys.exit(1)
