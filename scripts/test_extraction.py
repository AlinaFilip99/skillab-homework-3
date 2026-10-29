#!/usr/bin/env python3
"""
Test script for querying extracted data (invoices and contracts).

Usage:
    TEST_MODE=1 python scripts/test_extraction.py list invoices
    TEST_MODE=1 python scripts/test_extraction.py list contracts
    TEST_MODE=1 python scripts/test_extraction.py get invoice <id>
    TEST_MODE=1 python scripts/test_extraction.py get contract <id>

Examples:
    TEST_MODE=1 python scripts/test_extraction.py list invoices
    TEST_MODE=1 python scripts/test_extraction.py get invoice 1
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
from src.db.repositories import ExtractedContractRepository, ExtractedInvoiceRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_list(args: argparse.Namespace) -> int:
    """List extracted records."""
    settings = get_settings()
    logger.info(f"Test mode: {settings.test_mode}")

    with transaction() as session:
        if args.type == "invoices":
            repo = ExtractedInvoiceRepository(session)
            records = repo.list_all(limit=args.limit)

            if not records:
                print("No invoices found.")
                return 0

            print(f"\nFound {len(records)} invoice(s):\n")
            print("-" * 80)
            for inv in records:
                produse = json.loads(inv.produse_json) if inv.produse_json else []
                print(f"ID: {inv.id}")
                print(f"  File: {inv.file_name}")
                print(f"  Number: {inv.numar}")
                print(f"  Date: {inv.data}")
                print(f"  Supplier: {inv.furnizor}")
                print(f"  Client: {inv.client}")
                print(f"  Products: {len(produse)} item(s)")
                print(f"  Subtotal: {inv.subtotal}")
                print(f"  TVA: {inv.tva}")
                print(f"  Total: {inv.total}")
                print(f"  Extracted: {inv.extracted_at}")
                print("-" * 80)

        elif args.type == "contracts":
            repo = ExtractedContractRepository(session)
            records = repo.list_all(limit=args.limit)

            if not records:
                print("No contracts found.")
                return 0

            print(f"\nFound {len(records)} contract(s):\n")
            print("-" * 80)
            for ctr in records:
                obligatii = json.loads(ctr.obligatii_json) if ctr.obligatii_json else []
                print(f"ID: {ctr.id}")
                print(f"  File: {ctr.file_name}")
                print(f"  Number: {ctr.numar}")
                print(f"  Date: {ctr.data_incheiere}")
                print(f"  Provider: {ctr.prestator}")
                print(f"  Beneficiary: {ctr.beneficiar}")
                print(f"  Value: {ctr.valoare}")
                print(f"  Duration: {ctr.durata_luni} months")
                print(f"  Obligations: {len(obligatii)} item(s)")
                print(f"  Extracted: {ctr.extracted_at}")
                print("-" * 80)

    return 0


def cmd_get(args: argparse.Namespace) -> int:
    """Get a specific extracted record."""
    settings = get_settings()
    logger.info(f"Test mode: {settings.test_mode}")

    with transaction() as session:
        if args.type == "invoice":
            repo = ExtractedInvoiceRepository(session)
            record = repo.get_by_id(args.id)

            if not record:
                print(f"Invoice with ID {args.id} not found.")
                return 1

            produse = json.loads(record.produse_json) if record.produse_json else []
            data = {
                "id": record.id,
                "file_name": record.file_name,
                "numar": record.numar,
                "data": record.data,
                "furnizor": record.furnizor,
                "client": record.client,
                "produse": produse,
                "subtotal": float(record.subtotal) if record.subtotal else None,
                "tva": float(record.tva) if record.tva else None,
                "total": float(record.total) if record.total else None,
                "extracted_at": str(record.extracted_at),
            }
            print(json.dumps(data, indent=2, ensure_ascii=False))

        elif args.type == "contract":
            repo = ExtractedContractRepository(session)
            record = repo.get_by_id(args.id)

            if not record:
                print(f"Contract with ID {args.id} not found.")
                return 1

            obligatii = json.loads(record.obligatii_json) if record.obligatii_json else []
            data = {
                "id": record.id,
                "file_name": record.file_name,
                "numar": record.numar,
                "data_incheiere": record.data_incheiere,
                "prestator": record.prestator,
                "beneficiar": record.beneficiar,
                "valoare": float(record.valoare) if record.valoare else None,
                "durata_luni": record.durata_luni,
                "obligatii_prestator": obligatii,
                "extracted_at": str(record.extracted_at),
            }
            print(json.dumps(data, indent=2, ensure_ascii=False))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query extracted data (invoices and contracts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    p_list = subparsers.add_parser("list", help="List extracted records")
    p_list.add_argument("type", choices=["invoices", "contracts"], help="Record type")
    p_list.add_argument("--limit", type=int, default=100, help="Max records to return")
    p_list.set_defaults(func=cmd_list)

    # get command
    p_get = subparsers.add_parser("get", help="Get a specific record")
    p_get.add_argument("type", choices=["invoice", "contract"], help="Record type")
    p_get.add_argument("id", type=int, help="Record ID")
    p_get.set_defaults(func=cmd_get)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
