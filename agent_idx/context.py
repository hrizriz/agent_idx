"""Conversation context helpers: meta queries & topic-switch detection.

Prevents the agent from continuing an old stock analysis when the user
asks something unrelated (e.g. "bisa buat cronjob?" after a DSSA thread).
"""
from __future__ import annotations

import re
from typing import Any

# Explicit follow-ups / corrections that SHOULD keep history.
_CONTINUE = re.compile(
    r"(?:"
    r"^\s*itu\b|"
    r"\bbukan\b.{0,60}\btapi\b|"
    r"\bbukan\b.{0,40}\b(itu|yang|soal|tentang)\b|"
    r"\b("
    r"tadi|lanjut|lanjutkan|yang\s+barusan|yang\s+kemarin|"
    r"sebelumnya|tersebut|di\s+atas|masih\s+tentang|soal\s+itu|"
    r"analisis\s+itu|proyeksi\s+tadi|hitung\s+lagi|revisi|perjelas|"
    r"jelasin\s+lagi|dari\s+perhitungan\s+tadi|yang\s+kamu\s+bilang|"
    r"close-nya|angka\s+(?:close|tadi)|maksudku|maksud\s+saya|"
    r"salah|bukan\s+itu|yang\s+aku\s+maksud"
    r")\b"
    r")",
    re.I,
)

# Questions about the bot itself / ops — never inherit stock context.
_META_BOT = re.compile(
    r"(?:"
    r"\b(cron|cronjob|cron_[a-z]+|jadwal|scheduler|apscheduler)\b|"
    r"\b(reset|clear\s+history|/reset)\b|"
    r"\b(/ask|/council|/learn|/stockbit|/help|/start)\b|"
    r"\b(kamu\s+bisa|bisa\s+gak|bisa\s+nggak|bisa\s+tidak|sanggup\s+kah|"
    r"mampu\s+kah|fitur\s+apa|kemampuan|apa\s+yang\s+bisa|"
    r"cara\s+pakai|gimana\s+cara|how\s+do\s+you|bedanya\s+/ask|"
    r"apa\s+bedanya)\b|"
    r"\b(siapa\s+kamu|who\s+are\s+you|kamu\s+siapa|nama\s+kamu)\b|"
    r"\b(versi|status\s+bot|health\s*check|sudah\s+restart|bot\s+sudah)\b|"
    r"\b(jangan\s+buat\s+pdf|preferensi|tanpa\s+(pdf|dokumen)|stop\s+bikin\s+pdf|"
    r"jangan\s+export\s+pdf|pdf\s+otomatis|jangan.{0,25}\bpdf\b|"
    r"bikin\s+pdf\s+otomatis|jawab(?:an)?\s+teks\s+saja)\b|"
    r"\b(buat|tambah|atur|set|ubah|ganti)\b.{0,40}\b(cron|jadwal|scheduler)\b|"
    r"\bcron\b.{0,40}\b(buat|tambah|atur|set)\b|"
    r"\b(status\s+login|sesi\s+stockbit|login\s+stockbit|stockbit\s+masih\s+aktif|"
    r"jangan\s+scrape|cuma\s+status)\b|"
    r"\b(digest\s+harian|list\s+(folder|project|dir)|baca\s+file|project\s+dir)\b|"
    r"\b(jangan\s+nyambung|pertanyaan\s+baru|ganti\s+topik)\b"
    r")",
    re.I,
)

_SYMBOL = re.compile(r"\b([A-Za-z]{4})\b")
_SKIP_SYMBOLS = frozenset(
    {
        "YTD",
        "IDX",
        "PDF",
        "BEI",
        "NEWS",
        "CHAT",
        "SQL",
        "DATA",
        "HTML",
        "JSON",
        "HTTP",
        "HTTPS",
        "WIB",
        "OTC",
        "IPO",
        "ETF",
        "RHS",
        "LHS",
        "SMA",
        "EMA",
        "MACD",
        "RSI",
        "OHLC",
        "PPA",
        "TKDN",
        "BUMN",
        "BUMD",
        "NRE",
        "WTE",
        "PLN",
        "NPL",
        "ROE",
        "ROA",
        "DER",
        "APA",
        "ITU",
        "YANG",
        "DARI",
        "UNTUK",
        "DENGAN",
        "ATAU",
        "PADA",
        "SAJA",
        "SUDAH",
        "MASIH",
        "BISA",
        "TIDAK",
        "KAMU",
        "SAYA",
        "FILE",
        "LIST",
        "TEXT",
        "BOTS",
        "CARI",
        "SOAL",
        "KALA",
        "KALO",
        "BUAT",
        "STOP",
        "BARU",
        "HARI",
        "BULAN",
        "TAHU",
        "INFO",
        "OPEN",
        "READ",
        "SHOW",
        "HELP",
        "TEST",
        "TRUE",
        "JANU",
        "FEBR",
        "MARE",
        "APRI",
        "JUNI",
        "JULI",
        "AGUS",
        "SEPT",
        "OKTO",
        "NOVE",
        "DESE",
        "WAVE",
        "DONG",
        "TIME",
        "CLOSE",
        "CYCLE",
        "LALU",
        "MAJOR",
        "BOTTOM",
        "TARGET",
        "ENTRY",
        "RISK",
        "FLOW",
        # Indonesian / English fillers mistaken as 4-letter symbols
        "JADI",
        "TAPI",
        "BAIK",
        "TERM",
        "SHORT",
        "LONG",
        "HOLD",
        "SELL",
        "EMANG",
        "MAKA",
        "KARE",
        "SAMA",
        "LEBI",
        "KURA",
        "SANG",
        "MUNG",
        "NANTI",
        "LAMA",
        "HANY",
        "JUGA",
        "SERT",
        "UNTU",
        "DENG",
        "GINI",
        "GITU",
        "KALA",
        "TENT",
        "MENU",
        "SENT",
        "BERI",
        "REPO",
        "RUPS",
        "VIEW",
        "PLAN",
        "IDEA",
        "CALL",
        "SIZE",
        "LAST",
        "NEXT",
        "WEEK",
        "YEAR",
        "DATE",
        "DAYS",
        "HOUR",
        "MINS",
        "PLUS",
        "ONLY",
        "ALSO",
        "MORE",
        "LESS",
        "OVER",
        "INTO",
        "LIKE",
        "JUST",
        "VERY",
        "MUCH",
        "MANY",
        "SOME",
        "EACH",
        "BOTH",
        "MOST",
        "SUCH",
        "THAN",
        "THEN",
        "ONCE",
        "AWAY",
        "BACK",
        "EVEN",
        "STILL",
        "SAME",
        "FROM",
        "WITH",
        "THAT",
        "THIS",
        "WHEN",
        "WHAT",
        "FREE",
        "CASH",
        "DEBT",
        "NOTE",
        "FALS",
        "NULL",
        "NONE",
        "HIGH",
        "LOW",
        "TTM",
        "WACC",
        "BVPS",
        # Extra ID fillers seen in Fed/BI chats → false symbol_switch
        "ARAH",
        "AREA",
        "ATAS",
        "BELI",
        "DANA",
        "FOMC",
        "GUNA",
        "HERE",
        "LABA",
        "MASA",
        "OLEH",
        "PARA",
        "POIN",
        "SATU",
        "SUKU",
        "TAKE",
        "TREN",
        "UMUM",
        "SAAT",
        "RATE",
        "CUT",
        "HIKE",
        "FED",
        "THE",
        "AND",
        "FOR",
        "NOT",
        "ARE",
        "WAS",
        "HAS",
        "HAD",
        "WILL",
        "CAN",
        "MAY",
        "ALSO",
    }
)

# Soft topic keywords (non-symbol) that signal a domain switch.
_TOPIC_KEYS = re.compile(
    r"\b("
    r"cron|cronjob|jadwal|farming|scrape|chart\s*1h|elliott|wave|"
    r"foreign(?:\s*flow)?|net\s*foreign|backtest|pdf|gambar|foto|knowledge|belajar|"
    r"login\s*stockbit|stockbit\s*reports|folder|project|symbols_list|"
    r"pe\s*ratio|npl|macd|digest"
    r")\b",
    re.I,
)


def is_meta_bot_query(text: str) -> bool:
    """True if the user is asking about the bot / ops, not market analysis."""
    raw = text or ""
    if not _META_BOT.search(raw):
        return False
    # Preference + analysis in one sentence → treat as analysis (PDF preference secondary).
    symbols = extract_symbols(raw)
    if symbols and re.search(
        r"\b(ringkas|analisis|laporan|export\s+csv|foreign|harga|volume|ytd)\b",
        raw,
        re.I,
    ):
        return False
    return True


def extract_symbols(text: str) -> set[str]:
    found: set[str] = set()
    for m in _SYMBOL.finditer(text or ""):
        sym = m.group(1).upper()
        if sym not in _SKIP_SYMBOLS:
            found.add(sym)
    return found


def extract_tickers(text: str) -> set[str]:
    """Deprecated alias for extract_symbols."""
    return extract_symbols(text)


def _history_blob(history: list[dict[str, Any]], *, last_n: int = 6) -> str:
    parts: list[str] = []
    for msg in history[-last_n:]:
        content = str(msg.get("content") or "")
        parts.append(content[:1200])
    return "\n".join(parts)


def should_drop_history(
    question: str, history: list[dict[str, Any]] | None
) -> tuple[bool, str]:
    """Decide whether to ignore prior turns for this question."""
    if not history:
        return False, ""

    q = (question or "").strip()
    if not q:
        return False, ""

    if "--- ISI PDF" in q or "--- ISI GAMBAR" in q:
        return True, "attachment_context"

    if is_meta_bot_query(q):
        return True, "meta_bot_query"

    # User explicitly continues prior thread.
    if _CONTINUE.search(q):
        return False, "explicit_continue"

    q_symbols = extract_symbols(q)
    hist_text = _history_blob(history)
    hist_symbols = extract_symbols(hist_text)

    if q_symbols and hist_symbols and q_symbols.isdisjoint(hist_symbols):
        return True, f"symbol_switch:{','.join(sorted(hist_symbols))}->{','.join(sorted(q_symbols))}"

    q_topics = {m.group(0).lower() for m in _TOPIC_KEYS.finditer(q)}
    h_topics = {m.group(0).lower() for m in _TOPIC_KEYS.finditer(hist_text)}
    # Meta/ops topic in question while history is stock-analysis heavy.
    if q_topics and h_topics and q_topics.isdisjoint(h_topics):
        # Only drop when question looks like a fresh ask (not a tiny pronoun reply).
        if len(q) >= 12:
            return True, f"topic_switch:{','.join(sorted(h_topics)[:3])}->{','.join(sorted(q_topics)[:3])}"

    return False, ""


def select_history(
    question: str, history: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Return history to send to the LLM (possibly empty)."""
    if not history:
        return []
    drop, reason = should_drop_history(question, history)
    if drop:
        return []
    # Keep recent turns only; truncate oversized assistant blobs.
    trimmed: list[dict[str, Any]] = []
    for msg in history[-8:]:
        role = msg.get("role") or "user"
        content = str(msg.get("content") or "")
        if role == "assistant" and len(content) > 900:
            content = content[:900] + "…"
        trimmed.append({"role": role, "content": content})
    return trimmed


def summarize_history_for_log(history: list[dict[str, Any]]) -> str:
    if not history:
        return "history=0"
    symbols = extract_symbols(_history_blob(history))
    return f"history={len(history)} symbols={','.join(sorted(symbols)[:5]) or '-'}"
