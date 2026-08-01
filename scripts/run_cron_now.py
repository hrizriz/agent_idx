"""Run configured cron jobs once (manual trigger)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.config import load_settings
from agent_idx.data import StockDataStore
from agent_idx.jobs import (
    job_daily_chart_farm,
    job_daily_market_digest,
    job_daily_news,
    job_daily_stockbit_reports,
)
from agent_idx.knowledge import KnowledgeBase
from agent_idx.stockbit_store import StockbitReportsStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["all", "market", "reports", "farm", "news"],
        default="all",
    )
    parser.add_argument("--farm-limit", type=int, default=0, help="Limit symbols for chart farm")
    args = parser.parse_args()
    s = load_settings()
    store = StockDataStore(s.parquet_dir)
    knowledge = KnowledgeBase(s.knowledge_dir)
    sb = StockbitReportsStore(s.chroma_dir, export_dir=s.export_dir, auto_sync=False)

    jobs = []
    if args.only in ("all", "market"):
        jobs.append(("market", lambda: job_daily_market_digest(store, knowledge)))
    if args.only in ("all", "reports"):
        jobs.append(
            (
                "reports",
                lambda: job_daily_stockbit_reports(
                    s.export_dir,
                    days=s.cron_reports_days,
                    stockbit_store=sb,
                ),
            )
        )
    if args.only in ("all", "farm"):
        def _farm():
            from agent_idx import chart_farm

            if args.farm_limit and args.farm_limit > 0:
                symbols = chart_farm.load_symbols()[: args.farm_limit]
                summary = chart_farm.run_farm(
                    symbols,
                    [t.strip().upper() for t in s.cron_farm_timeframes.split(",") if t.strip()],
                    sync_daily=s.cron_farm_sync_daily,
                )
                out = knowledge.root / "daily" / "farm_manual.md"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(summary, encoding="utf-8")
                return out
            return job_daily_chart_farm(
                knowledge,
                timeframes=s.cron_farm_timeframes,
                sync_daily=s.cron_farm_sync_daily,
            )

        jobs.append(("farm", _farm))
    if args.only in ("all", "news"):
        jobs.append(("news", lambda: job_daily_news(knowledge)))

    for name, fn in jobs:
        print(f"=== RUN {name} ===", flush=True)
        try:
            path = fn()
            print(f"OK {name}: {path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {exc}", flush=True)
            return 1
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
