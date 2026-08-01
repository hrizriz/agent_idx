"""After Stockbit fundamentals farm finishes: transform markdown → Postgres parquet.

Writes into ``exports/postgres/``:
  - fundamental_metric.parquet
  - financial_statement_line.parquet
  - company_profile.parquet
  - schema.sql

Usage:
    py -3 scripts/export_pg_fundamentals.py
    py -3 scripts/export_pg_fundamentals.py --skip-transform
    py -3 scripts/export_pg_fundamentals.py --symbols BBCA,BBRI

Same thing via main:
    py -3 main.py export-pg --fundamentals-only --transform
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.pg_export import export_fundamentals  # noqa: E402
from agent_idx.transforms import run_data_transform  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Default: exports/postgres",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated subset; default = all in transforms DB",
    )
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        help="Skip re-ingest of exports/stockbit markdown (export only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional transform limit (0 = all scraped symbols)",
    )
    args = parser.parse_args()

    if not args.skip_transform:
        print(run_data_transform(scope="fundamentals", limit=args.limit))
        print()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    print(export_fundamentals(out_dir=args.out_dir, symbols=symbols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
