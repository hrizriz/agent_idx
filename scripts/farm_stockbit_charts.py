"""Farm IDX OHLCV via tvkit into parquet + indicators.

Usage:
    py -3 scripts/farm_stockbit_charts.py --limit 3
    py -3 scripts/farm_stockbit_charts.py --symbols BBCA,DSSA,BBRI
    py -3 scripts/farm_stockbit_charts.py --tfs 1H,1D,1W

Output:
    data/charts/{SYMBOL}/{TF}.parquet   — OHLCV + technical indicators
    data/charts/_meta/{SYMBOL}.json     — fetch status
    data/parquet/daily_stock_summary_stockbit_*.parquet  — optional daily sync
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.chart_farm import (  # noqa: E402
    DEFAULT_TIMEFRAMES,
    add_indicators,
    bars_to_df,
    load_symbols,
    resolve_symbols,
    run_farm,
    save_tf_parquet,
)

__all__ = ["add_indicators", "bars_to_df", "load_symbols", "save_tf_parquet"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-file", default=str(ROOT / "data" / "symbols_list.txt"))
    parser.add_argument("--symbols", default="", help="Comma-separated override, e.g. BBCA,DSSA")
    parser.add_argument("--tfs", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--limit", type=int, default=0, help="Max symbols (0=all)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sync-daily", action="store_true", help="Rebuild daily panel from 1D/1H")
    parser.add_argument("--sleep", type=float, default=0.25, help="Pause between symbols")
    # Kept so old cron/scripts don't break; ignored (no browser).
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-login-wait", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--login-wait", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--no-profile", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.symbols.strip():
        symbols = resolve_symbols(args.symbols, limit=args.limit)
    else:
        symbols = load_symbols(Path(args.symbols_file))
        symbols = symbols[max(0, args.offset) :]
        if args.limit > 0:
            symbols = symbols[: args.limit]

    timeframes = [t.strip().upper() for t in args.tfs.split(",") if t.strip()]
    print(f"Farming {len(symbols)} symbols × {timeframes} via tvkit")
    summary = run_farm(
        symbols,
        timeframes,
        sync_daily=bool(args.sync_daily),
        sleep_sec=float(args.sleep),
        on_progress=print,
    )
    print(summary)
    return 0 if not summary.startswith("ERROR:") else 1


if __name__ == "__main__":
    raise SystemExit(main())
