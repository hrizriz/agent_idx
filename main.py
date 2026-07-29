from __future__ import annotations

import argparse
import logging
import os
import sys

from agent_idx.agent import AnalystAgent
from agent_idx.config import load_settings
from agent_idx.data import StockDataStore
from agent_idx.format import to_plain_text
from agent_idx.knowledge import KnowledgeBase
from agent_idx.multi_agent import MultiAgentCouncil
from agent_idx.stockbit_store import StockbitReportsStore


def _make_stockbit_store(settings) -> StockbitReportsStore:
    return StockbitReportsStore(
        settings.chroma_dir,
        export_dir=settings.export_dir,
        auto_sync=False,
    )


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_cli(store: StockDataStore, settings) -> int:
    knowledge = KnowledgeBase(settings.knowledge_dir)
    stockbit_store = _make_stockbit_store(settings)
    agent = AnalystAgent(settings, store, knowledge=knowledge, stockbit_store=stockbit_store)
    print("IDX Analyst CLI. Ketik pertanyaan, atau 'exit' untuk keluar.")
    print("Perintah: /reset | /multi (pindah ke council 3-agent)")
    print(f"Data: {store.list_date_range()}")
    print(f"LLM:  {settings.llm_base_url} model={settings.llm_model}")
    history: list[dict] = []
    while True:
        try:
            text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            return 0
        if text.lower() == "/reset":
            history.clear()
            print("history cleared")
            continue
        if text.lower() in {"/multi", "/council"}:
            return cmd_multi(store, settings)
        try:
            result = agent.ask(text, history)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")
            continue
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": result.text})
        history = history[-20:]
        print(f"\nAgent>\n{to_plain_text(result.text)}")
        for path in result.files:
            print(f"[file] {path}")


def cmd_multi(store: StockDataStore, settings) -> int:
    knowledge = KnowledgeBase(settings.knowledge_dir)
    stockbit_store = _make_stockbit_store(settings)
    council = MultiAgentCouncil(
        settings, store, knowledge=knowledge, stockbit_store=stockbit_store
    )
    print("IDX Multi-Agent Council (Researcher → Analyst → Critic).")
    print("Ketik pertanyaan, atau 'exit'. Perintah: /reset | /single")
    print(f"Data: {store.list_date_range()}")
    print(f"LLM:  {settings.llm_base_url} model={settings.llm_model}")
    print(f"Max rounds: {council.max_rounds}")
    while True:
        try:
            text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            return 0
        if text.lower() == "/reset":
            print("council is stateless per question (ok)")
            continue
        if text.lower() in {"/single", "/cli"}:
            return cmd_cli(store, settings)
        try:
            result = council.ask(text, on_progress=print)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")
            continue
        print(f"\nCouncil>\n{to_plain_text(result.text)}")
        for path in result.files:
            print(f"[file] {path}")


def cmd_ask_multi(
    store: StockDataStore, settings, question: str, max_rounds: int = 3
) -> int:
    knowledge = KnowledgeBase(settings.knowledge_dir)
    stockbit_store = _make_stockbit_store(settings)
    council = MultiAgentCouncil(
        settings,
        store,
        knowledge=knowledge,
        stockbit_store=stockbit_store,
        max_rounds=max_rounds,
    )
    result = council.ask(question, on_progress=print)
    print(to_plain_text(result.text))
    for path in result.files:
        print(f"[file] {path}")
    return 0


def cmd_range(store: StockDataStore) -> int:
    print(store.list_date_range())
    print()
    print(store.describe_schema())
    return 0


def cmd_ask(store: StockDataStore, settings, question: str) -> int:
    knowledge = KnowledgeBase(settings.knowledge_dir)
    stockbit_store = _make_stockbit_store(settings)
    agent = AnalystAgent(settings, store, knowledge=knowledge, stockbit_store=stockbit_store)
    result = agent.ask(question)
    print(to_plain_text(result.text))
    for path in result.files:
        print(f"[file] {path}")
    return 0


def cmd_learn(store: StockDataStore, settings, which: str) -> int:
    from agent_idx.jobs import (
        job_daily_market_digest,
        job_daily_news,
        job_weekly_lesson,
        run_all_learning_jobs,
    )

    knowledge = KnowledgeBase(settings.knowledge_dir)
    if which == "market":
        paths = [job_daily_market_digest(store, knowledge)]
    elif which == "news":
        paths = [job_daily_news(knowledge)]
    elif which == "weekly":
        paths = [job_weekly_lesson(knowledge)]
    else:
        paths = run_all_learning_jobs(store, knowledge)
    for path in paths:
        print(f"OK {path}")
    print()
    print(knowledge.list_topics())
    return 0


def cmd_backtest(store: StockDataStore, settings, args) -> int:
    from agent_idx.backtest import run_backtest_scan

    print(
        run_backtest_scan(
            store,
            settings.export_dir,
            start_date=args.start_date,
            min_win_rate=args.min_win_rate,
            max_win_rate=args.max_win_rate,
            min_trades=args.min_trades,
            supplement_external=not args.no_supplement,
            cap_filter=args.cap_filter,
        )
    )
    return 0


def cmd_ma_squeeze(store: StockDataStore, settings, args) -> int:
    from agent_idx.ma_squeeze_study import run_ma_squeeze_study

    print(
        run_ma_squeeze_study(
            store,
            settings.export_dir,
            start_date=int(str(args.start_date).replace("-", "")[:8]),
            cap_filter=args.cap_filter,
            min_forward_gain=args.min_gain,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IDX thinking-only stock analyst agent")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("bot", help="Run Telegram bot (+ cron learning jobs)")
    sub.add_parser("cli", help="Interactive local chat")
    sub.add_parser("multi", help="Interactive 3-agent council (Researcher/Analyst/Critic)")
    sub.add_parser("range", help="Show available date range and schema")

    ask_p = sub.add_parser("ask", help="One-shot question")
    ask_p.add_argument("question", nargs="+", help="Question text")

    ask_multi_p = sub.add_parser(
        "ask-multi",
        help="One-shot question via 3-agent council",
    )
    ask_multi_p.add_argument("question", nargs="+", help="Question text")
    ask_multi_p.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Max Researcher→Analyst→Critic rounds (1-5)",
    )

    learn_p = sub.add_parser("learn", help="Run learning jobs now (market/news/weekly/all)")
    learn_p.add_argument(
        "job",
        nargs="?",
        default="all",
        choices=["all", "market", "news", "weekly"],
    )

    tier_p = sub.add_parser(
        "tiers",
        help="Show the fundamentals farm tiers (top N vs rest)",
    )
    tier_p.add_argument("--top-n", type=int, default=200)
    tier_p.add_argument("--refresh", action="store_true", help="Rebuild the ranking")
    tier_p.add_argument(
        "--list",
        default="",
        choices=["", "top", "rest"],
        help="Print the full symbol list for a tier",
    )

    pg_p = sub.add_parser(
        "export-pg",
        help="Export fundamentals as Postgres-ready parquet + schema.sql",
    )
    pg_p.add_argument("--out-dir", default=None, help="Default exports/postgres")
    pg_p.add_argument("--symbols", default="", help="Comma-separated; default all")

    bt_p = sub.add_parser("backtest", help="Scan trading scenarios on parquet (large cap)")
    bt_p.add_argument("--min-win-rate", type=float, default=0.75)
    bt_p.add_argument("--max-win-rate", type=float, default=0.85)
    bt_p.add_argument("--min-trades", type=int, default=15)
    bt_p.add_argument("--start-date", default="20220101")
    bt_p.add_argument(
        "--cap-filter",
        "--universe",
        dest="cap_filter",
        default="large_cap",
        choices=["large_cap", "all", "none"],
        help="Entry market-cap filter (legacy alias: --universe)",
    )
    bt_p.add_argument("--no-supplement", action="store_true", help="Skip yfinance data fill")

    ma_p = sub.add_parser(
        "ma-squeeze",
        help="Investigate if MA5/20/50/200 narrow before price rides MAs up",
    )
    ma_p.add_argument("--start-date", default="20220101")
    ma_p.add_argument(
        "--cap-filter",
        "--universe",
        dest="cap_filter",
        default="large_cap",
        choices=["large_cap", "all", "none"],
    )
    ma_p.add_argument("--min-gain", type=float, default=10.0, help="Min forward 20d gain pct")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    settings = load_settings()
    store = StockDataStore(settings.parquet_dir, max_rows=settings.max_sql_rows)

    try:
        command = args.command or "cli"
        if command == "bot":
            from agent_idx.bot import run_bot

            run_bot(settings, store)
            return 0
        if command == "range":
            return cmd_range(store)
        if command == "tiers":
            from agent_idx import universe

            print(universe.describe_tiers(args.top_n, refresh=args.refresh))
            if args.list:
                _, syms = universe.resolve_tier(args.list, top_n=args.top_n)
                print(f"\n{args.list} ({len(syms)}):")
                print("\n".join(syms))
            return 0
        if command == "export-pg":
            from agent_idx.pg_export import export_fundamentals

            print(
                export_fundamentals(
                    out_dir=args.out_dir,
                    symbols=[s for s in args.symbols.split(",") if s.strip()] or None,
                )
            )
            return 0
        if command == "ask":
            return cmd_ask(store, settings, " ".join(args.question))
        if command == "multi":
            return cmd_multi(store, settings)
        if command == "ask-multi":
            return cmd_ask_multi(
                store,
                settings,
                " ".join(args.question),
                max_rounds=args.rounds,
            )
        if command == "learn":
            return cmd_learn(store, settings, args.job)
        if command == "backtest":
            return cmd_backtest(store, settings, args)
        if command == "ma-squeeze":
            return cmd_ma_squeeze(store, settings, args)
        return cmd_cli(store, settings)
    except KeyboardInterrupt:
        # Agent may be mid-flight in a worker thread; force-exit so Ctrl+C
        # does not hang on thread join (common on Windows).
        print("\nStopped.", flush=True)
        os._exit(130)
    finally:
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        os._exit(130)
