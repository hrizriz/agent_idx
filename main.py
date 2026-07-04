from __future__ import annotations

import argparse
import logging
import os
import sys

from agent_idx.agent import AnalystAgent
from agent_idx.config import load_settings
from agent_idx.data import StockDataStore
from agent_idx.knowledge import KnowledgeBase


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_cli(store: StockDataStore, settings) -> int:
    knowledge = KnowledgeBase(settings.knowledge_dir)
    agent = AnalystAgent(settings, store, knowledge=knowledge)
    print("IDX Analyst CLI. Ketik pertanyaan, atau 'exit' untuk keluar.")
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
        try:
            result = agent.ask(text, history)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")
            continue
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": result.text})
        history = history[-20:]
        print(f"\nAgent>\n{result.text}")
        for path in result.files:
            print(f"[file] {path}")


def cmd_range(store: StockDataStore) -> int:
    print(store.list_date_range())
    print()
    print(store.describe_schema())
    return 0


def cmd_ask(store: StockDataStore, settings, question: str) -> int:
    knowledge = KnowledgeBase(settings.knowledge_dir)
    agent = AnalystAgent(settings, store, knowledge=knowledge)
    result = agent.ask(question)
    print(result.text)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IDX thinking-only stock analyst agent")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("bot", help="Run Telegram bot (+ cron learning jobs)")
    sub.add_parser("cli", help="Interactive local chat")
    sub.add_parser("range", help="Show available date range and schema")

    ask_p = sub.add_parser("ask", help="One-shot question")
    ask_p.add_argument("question", nargs="+", help="Question text")

    learn_p = sub.add_parser("learn", help="Run learning jobs now (market/news/weekly/all)")
    learn_p.add_argument(
        "job",
        nargs="?",
        default="all",
        choices=["all", "market", "news", "weekly"],
    )

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
        if command == "ask":
            return cmd_ask(store, settings, " ".join(args.question))
        if command == "learn":
            return cmd_learn(store, settings, args.job)
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
