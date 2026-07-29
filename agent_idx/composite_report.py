from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agent_idx.data import StockDataStore
from agent_idx.knowledge import KnowledgeBase
from agent_idx.news import search_news
from agent_idx.stockbit_reports import parse_user_date, search_posts
from agent_idx.stockbit_store import StockbitReportsStore

if TYPE_CHECKING:
    from agent_idx.chat_log import ChatLog
    from agent_idx.decision import QueryPlan

_REPORT_QUERY = re.compile(
    r"\b("
    r"report|laporan|ringkas|rangkum|summary|summarize|digest|ytd|"
    r"rekap|ikhtisar|briefing"
    r")\b",
    re.I,
)
_SCRAPE_VERB = re.compile(
    r"\b(scrape|tarik|ambil|unduh|download|export|riwayat|histori)\b",
    re.I,
)
_PDF_OUTPUT = re.compile(
    r"\b(buat|generate|kirim|export)\b.*\bpdf\b|\bpdf\b.*\b(buat|kirim|export)\b",
    re.I,
)


def is_composite_report_query(text: str) -> bool:
    """General report intent — not PDF attachment, not scrape, not single-date lookup."""
    raw = text or ""
    if not _REPORT_QUERY.search(raw):
        return False
    if re.search(r"\b(pdf|dokumen|attachment|lampiran|file)\b", raw, re.I):
        return False
    if re.search(r"\b(chat|group|grup|diskusi|telegram|siapa\s+bilang)\b", raw, re.I):
        return False
    if _SCRAPE_VERB.search(raw):
        return False
    if _PDF_OUTPUT.search(raw):
        return False
    if parse_user_date(raw) and re.search(r"\b(tanggal|tgl)\b", raw, re.I):
        return False
    return True


def gather_report_context(
    *,
    user_message: str,
    store: StockDataStore,
    knowledge: KnowledgeBase | None = None,
    stockbit_store: StockbitReportsStore | None = None,
    chat_log: ChatLog | None = None,
    plan: QueryPlan | None = None,
) -> dict[str, str]:
    from agent_idx.decision import extract_report_focus

    focus = plan.focus if plan else extract_report_focus(user_message)
    symbols = plan.symbols if plan else []
    wanted = set(plan.sources if plan else ["pasar", "berita", "stockbit_reports", "knowledge"])
    sections: dict[str, str] = {}

    if "pasar" in wanted:
        try:
            coverage = store.list_date_range()
            lines = [f"coverage={coverage}"]
            if symbols:
                for sym in symbols[:5]:
                    try:
                        rows = store._con.execute(
                            """
                            SELECT date, close, volume, value, change,
                                   foreign_buy, foreign_sell
                            FROM daily_stock
                            WHERE symbol = ?
                            ORDER BY date DESC
                            LIMIT 5
                            """,
                            [sym],
                        ).fetchall()
                        if rows:
                            lines.append(f"\n### {sym} (5 sesi terakhir)")
                            for r in rows:
                                nf = (
                                    f"{r[5] - r[6]:,.0f}"
                                    if r[5] is not None and r[6] is not None
                                    else "GAP_DATA"
                                )
                                lines.append(
                                    f"- {r[0]} close={r[1]} vol={r[2]:,.0f} "
                                    f"value={r[3]:,.0f} chg={r[4]:.2f}% net_foreign={nf}"
                                )
                        else:
                            lines.append(f"\n### {sym}: (tidak ada di parquet)")
                    except Exception as exc:  # noqa: BLE001
                        lines.append(f"\n### {sym}: (tidak tersedia: {exc})")
            top_flow = store._con.execute(
                """
                WITH latest AS (SELECT MAX(date) AS d FROM daily_stock)
                SELECT symbol, name,
                       foreign_buy - foreign_sell AS net_foreign,
                       value, close, change
                FROM daily_stock, latest
                WHERE date = latest.d
                  AND foreign_buy IS NOT NULL
                  AND foreign_sell IS NOT NULL
                ORDER BY ABS(net_foreign) DESC
                LIMIT 8
                """
            ).fetchall()
            lines.append("\ntop_net_foreign (sesi terakhir):")
            if not top_flow:
                lines.append("- GAP_DATA: sesi terakhir tidak memiliki foreign buy/sell")
            else:
                for row in top_flow:
                    lines.append(
                        f"- {row[0]} {row[1] or ''}: net_foreign={row[2]:,.0f} "
                        f"value={row[3]:,.0f} close={row[4]} chg={row[5]:.2f}%"
                    )
            sections["pasar"] = "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            sections["pasar"] = f"(tidak tersedia: {exc})"

    if "berita" in wanted:
        try:
            sections["berita"] = search_news(focus, limit=8)
        except Exception as exc:  # noqa: BLE001
            sections["berita"] = f"(tidak tersedia: {exc})"

    if "stockbit_reports" in wanted:
        stockbit_q = "YTD" if re.search(r"\bytd\b", user_message, re.I) else focus
        sb = search_posts(stockbit_store, stockbit_q, limit=20)
        sections["stockbit_reports"] = (
            sb if not sb.startswith("ERROR:") else "(kosong / belum siap)"
        )

    if "knowledge" in wanted and knowledge is not None:
        try:
            sections["knowledge"] = knowledge.search(focus, limit=3)
        except Exception as exc:  # noqa: BLE001
            sections["knowledge"] = f"(tidak tersedia: {exc})"

    if "chat" in wanted and chat_log is not None:
        try:
            q = symbols[0] if symbols else focus.split()[0]
            sections["chat"] = chat_log.search(q, limit=10)
        except Exception as exc:  # noqa: BLE001
            sections["chat"] = f"(tidak tersedia: {exc})"

    return sections


def format_report_context(sections: dict[str, str]) -> str:
    blocks: list[str] = []
    for key, body in sections.items():
        blocks.append(f"=== {key.upper()} ===\n{body.strip()}")
    return "\n\n".join(blocks)


def composite_report_system_prompt() -> str:
    return (
        "Kamu analis IDX. Susun LAPORAN GABUNGAN dari semua sumber di bawah "
        "(pasar, berita, Stockbit Reports, knowledge, chat jika ada). "
        "Bahasa Indonesia, profesional.\n"
        "Struktur:\n"
        "1) Ringkasan eksekutif (3-5 bullet) + rekomendasi watchlist jika relevan\n"
        "2) Pasar & foreign flow (angka dari pasar)\n"
        "3) Berita penting\n"
        "4) Pengumuman / Stockbit Reports relevan\n"
        "5) Insight knowledge / chat group (jika ada)\n"
        "6) Risiko & invalidasi\n"
        "WAJIB sebut sumber per poin (pasar/berita/stockbit/knowledge/chat). "
        "Jangan minta user kirim PDF. Jangan mengarang di luar data."
    )
