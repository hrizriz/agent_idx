"""Kirim 200 pertanyaan eval ke group Telegram + jawaban agent (untuk pantauan live).

Bot Telegram tidak memproses pesan dari dirinya sendiri, jadi skrip ini:
1) post pertanyaan ke group (terlihat di chat)
2) jalankan AnalystAgent.ask (stack yang sama dengan bot)
3) post jawaban ke group

Usage:
    py -3 scripts/telegram_probe_200.py
    py -3 scripts/telegram_probe_200.py --limit 20 --sleep 3
    py -3 scripts/telegram_probe_200.py --offset 50 --limit 50
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.agent import AnalystAgent  # noqa: E402
from agent_idx.config import load_settings  # noqa: E402
from agent_idx.data import StockDataStore  # noqa: E402
from agent_idx.eval_scenarios import build_scenarios  # noqa: E402
from agent_idx.format import to_plain_text  # noqa: E402
from agent_idx.knowledge import KnowledgeBase  # noqa: E402
from agent_idx.stockbit_store import StockbitReportsStore  # noqa: E402

logger = logging.getLogger("telegram_probe")


def _chunk(text: str, n: int = 3500) -> list[str]:
    text = (text or "").strip() or "(kosong)"
    if len(text) <= n:
        return [text]
    parts: list[str] = []
    while text:
        parts.append(text[:n])
        text = text[n:]
    return parts


def tg_send(token: str, chat_id: int, text: str, *, reply_to: int | None = None) -> int | None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    mid = None
    for i, part in enumerate(_chunk(text)):
        payload: dict = {
            "chat_id": chat_id,
            "text": part,
            "disable_web_page_preview": True,
        }
        if reply_to is not None and i == 0:
            payload["reply_to_message_id"] = reply_to
        for attempt in range(4):
            try:
                r = httpx.post(url, json=payload, timeout=60.0)
                data = r.json()
                if data.get("ok"):
                    mid = data["result"]["message_id"]
                    break
                desc = str(data.get("description") or "")
                if "retry after" in desc.lower() or r.status_code == 429:
                    wait = 5 + attempt * 5
                    time.sleep(wait)
                    continue
                logger.warning("sendMessage fail: %s", data)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("sendMessage error: %s", exc)
                time.sleep(2 + attempt)
        time.sleep(0.35)
    return mid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=2.0, help="pause antar soal (detik)")
    parser.add_argument("--timeout", type=float, default=120.0, help="timeout ask per soal")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    settings = load_settings()
    if not settings.telegram_bot_token or settings.telegram_chat_id is None:
        print("ERROR: TELEGRAM_BOT_TOKEN / CHAT_ID belum di .env")
        return 1

    scenarios = build_scenarios()
    start = max(0, args.offset)
    end = min(len(scenarios), start + max(1, args.limit))
    batch = scenarios[start:end]
    total = len(batch)

    token = settings.telegram_bot_token
    chat_id = int(settings.telegram_chat_id)

    store = StockDataStore(settings.parquet_dir, max_rows=settings.max_sql_rows)
    knowledge = KnowledgeBase(settings.knowledge_dir)
    stockbit_store = StockbitReportsStore(
        settings.chroma_dir, export_dir=settings.export_dir, auto_sync=False
    )
    agent = AnalystAgent(
        settings,
        store,
        knowledge=knowledge,
        stockbit_store=stockbit_store,
        chat_id=chat_id,
    )

    tg_send(
        token,
        chat_id,
        (
            f"🧪 PROBE QA dimulai — {total} pertanyaan "
            f"(offset={start}, total katalog=200).\n"
            "Format: pertanyaan lalu jawaban agent.\n"
            "Ini tes routing/kualitas — pantau saja di group ini.\n"
            "Ketik /cancel di Cursor terminal untuk stop skrip (bukan di Telegram)."
        ),
    )

    ok = 0
    fail = 0
    try:
        for i, sc in enumerate(batch, start=1):
            n = start + i
            q_header = (
                f"🧪 Q{n}/200 [{sc.id}|{sc.category}]\n"
                f"{sc.question}"
            )
            logger.info("=== %s/%s %s ===", i, total, sc.id)
            q_mid = tg_send(token, chat_id, q_header)

            def _ask():
                return agent.ask(sc.question, history=sc.history or None)

            t0 = time.time()
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(_ask)
                    result = fut.result(timeout=args.timeout)
                text = to_plain_text(result.text or "")
                elapsed = time.time() - t0
                preview = text if len(text) <= 3200 else text[:3200] + "\n…(dipotong)"
                ans = (
                    f"✅ A{n}/200 [{sc.id}] ({elapsed:.0f}s)\n"
                    f"{preview}"
                )
                tg_send(token, chat_id, ans, reply_to=q_mid)
                ok += 1
            except FuturesTimeout:
                fail += 1
                tg_send(
                    token,
                    chat_id,
                    f"⏱️ A{n}/200 [{sc.id}] TIMEOUT >{args.timeout:.0f}s",
                    reply_to=q_mid,
                )
            except Exception as exc:  # noqa: BLE001
                fail += 1
                logger.exception("ask failed %s", sc.id)
                tg_send(
                    token,
                    chat_id,
                    f"❌ A{n}/200 [{sc.id}] ERROR: {exc}",
                    reply_to=q_mid,
                )

            time.sleep(max(0.5, args.sleep))
    finally:
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass
        tg_send(
            token,
            chat_id,
            f"🧪 PROBE selesai — OK={ok} gagal={fail} dari {total} (offset={start}).",
        )

    print(f"done ok={ok} fail={fail} total={total}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
