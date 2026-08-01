"""Backtest 200 agent QA scenarios (routing / context / optional LLM).

Usage:
    py -3 scripts/backtest_agent_qa.py
    py -3 scripts/backtest_agent_qa.py --llm --limit 30
    py -3 scripts/backtest_agent_qa.py --category meta_bot
    py -3 scripts/backtest_agent_qa.py --fail-only

Layer A (default): deterministic — intent, drop_history, needs_tools.
Layer B (--llm): call AnalystAgent.ask and regex-check the answer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.agent import AnalystAgent  # noqa: E402
from agent_idx.config import load_settings  # noqa: E402
from agent_idx.context import select_history, should_drop_history  # noqa: E402
from agent_idx.data import StockDataStore  # noqa: E402
from agent_idx.decision import classify_intent  # noqa: E402
from agent_idx.eval_scenarios import Scenario, build_scenarios, scenarios_by_category  # noqa: E402
from agent_idx.knowledge import KnowledgeBase  # noqa: E402


def _check_route(s: Scenario) -> list[str]:
    fails: list[str] = []
    plan = classify_intent(s.question)
    intent = plan.intent.value

    if s.intent_in and intent not in s.intent_in:
        fails.append(f"intent={intent} not in {s.intent_in}")
    if s.intent_not and intent in s.intent_not:
        fails.append(f"intent={intent} forbidden")

    drop, reason = should_drop_history(s.question, s.history)
    if s.drop_history is True and not drop:
        fails.append(f"expected drop_history, got keep ({reason or 'no-reason'})")
    if s.drop_history is False and drop:
        fails.append(f"expected keep history, got drop ({reason})")

    hist = select_history(s.question, s.history)
    needs = AnalystAgent._needs_tools(s.question, hist)
    if s.needs_tools is True and not needs:
        fails.append("expected needs_tools=True")
    if s.needs_tools is False and needs:
        fails.append("expected needs_tools=False")

    return fails


def _check_answer(s: Scenario, text: str) -> list[str]:
    fails: list[str] = []
    body = text or ""
    for pat in s.answer_must:
        if not re.search(pat, body, re.I | re.S):
            fails.append(f"answer missing /{pat}/")
    for pat in s.answer_must_not:
        if re.search(pat, body, re.I | re.S):
            fails.append(f"answer hit forbidden /{pat}/")
    return fails


def run_layer_a(scenarios: list[Scenario]) -> list[dict]:
    rows: list[dict] = []
    for s in scenarios:
        fails = _check_route(s)
        rows.append(
            {
                "id": s.id,
                "category": s.category,
                "question": s.question,
                "layer": "route",
                "ok": not fails,
                "fails": fails,
                "intent": classify_intent(s.question).intent.value,
                "drop": should_drop_history(s.question, s.history)[0],
            }
        )
    return rows


def run_layer_b(
    scenarios: list[Scenario],
    agent: AnalystAgent,
    *,
    limit: int = 0,
) -> list[dict]:
    rows: list[dict] = []
    # Prefer scenarios that have answer expectations, then meta/trap.
    prioritized = sorted(
        scenarios,
        key=lambda s: (
            0 if (s.answer_must or s.answer_must_not) else 1,
            0 if s.category in {"meta_bot", "trap", "topic_switch", "pdf_pref"} else 1,
            s.id,
        ),
    )
    if limit > 0:
        prioritized = prioritized[:limit]

    for i, s in enumerate(prioritized, start=1):
        hist = select_history(s.question, s.history)
        t0 = time.time()
        try:
            result = agent.ask(s.question, hist)
            text = result.text or ""
            err = ""
        except Exception as exc:  # noqa: BLE001
            text = ""
            err = str(exc)
        elapsed = round(time.time() - t0, 2)
        fails = _check_answer(s, text) if not err else [f"exception: {err}"]
        # Flag only clear history-bleed (Elliott thread), not legitimate news mentions.
        drop, _ = should_drop_history(s.question, s.history)
        if drop and s.history and "DSSA" not in s.question.upper():
            if re.search(r"Elliott|Wave\s*\(C\)|Rp432|Rp870", text, re.I):
                if not any("Elliott" in f or "forbidden" in f for f in fails):
                    fails.append("answer leaked prior DSSA Elliott context")
        rows.append(
            {
                "id": s.id,
                "category": s.category,
                "question": s.question,
                "layer": "llm",
                "ok": not fails,
                "fails": fails,
                "elapsed_sec": elapsed,
                "answer_preview": (text[:240] + "…") if len(text) > 240 else text,
            }
        )
        print(f"[{i}/{len(prioritized)}] {s.id} ok={not fails} {elapsed}s")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="Also run LLM answer checks")
    parser.add_argument("--limit", type=int, default=0, help="Limit LLM scenarios (0=all prioritized)")
    parser.add_argument("--category", default="", help="Filter category")
    parser.add_argument("--fail-only", action="store_true")
    parser.add_argument(
        "--out",
        default="",
        help="Write JSON report path (default exports/eval/...)",
    )
    args = parser.parse_args()

    scenarios = build_scenarios()
    if args.category:
        scenarios = [s for s in scenarios if s.category == args.category]
    print(f"Scenarios: {len(scenarios)}")
    print("By category:", scenarios_by_category(scenarios))

    rows = run_layer_a(scenarios)
    a_fail = [r for r in rows if not r["ok"]]
    print(f"\nLayer A (route): {len(rows) - len(a_fail)}/{len(rows)} OK, fail={len(a_fail)}")

    b_rows: list[dict] = []
    if args.llm:
        settings = load_settings()
        store = StockDataStore(settings.parquet_dir, max_rows=settings.max_sql_rows)
        try:
            agent = AnalystAgent(
                settings,
                store,
                knowledge=KnowledgeBase(settings.knowledge_dir),
                chat_id=settings.telegram_chat_id,
            )
            b_rows = run_layer_b(scenarios, agent, limit=args.limit)
        finally:
            store.close()
        b_fail = [r for r in b_rows if not r["ok"]]
        print(f"Layer B (llm): {len(b_rows) - len(b_fail)}/{len(b_rows)} OK, fail={len(b_fail)}")

    all_rows = rows + b_rows
    show = [r for r in all_rows if (not r["ok"] if args.fail_only else True)]
    if args.fail_only:
        print("\n--- FAILURES ---")
        for r in show:
            print(f"{r['id']} [{r['layer']}] {r['question'][:80]}")
            for f in r["fails"]:
                print(f"  - {f}")
            if r.get("answer_preview"):
                print(f"  preview: {r['answer_preview'][:160]}")

    out = Path(args.out) if args.out else (
        ROOT / "exports" / "eval" / f"qa_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_scenarios": len(scenarios),
        "categories": scenarios_by_category(scenarios),
        "layer_a_ok": sum(1 for r in rows if r["ok"]),
        "layer_a_fail": sum(1 for r in rows if not r["ok"]),
        "layer_b_ok": sum(1 for r in b_rows if r["ok"]),
        "layer_b_fail": sum(1 for r in b_rows if not r["ok"]),
        "results": all_rows,
        "scenario_catalog": [asdict(s) for s in scenarios],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {out}")

    return 1 if any(not r["ok"] for r in all_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
