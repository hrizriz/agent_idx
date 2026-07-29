from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_idx.stockbit_store import StockbitReportsStore, iter_stream_markdown

_MONTH_ID = {
    "januari": 1,
    "jan": 1,
    "februari": 2,
    "feb": 2,
    "maret": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "mei": 5,
    "juni": 6,
    "jun": 6,
    "juli": 7,
    "jul": 7,
    "agustus": 8,
    "agu": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "oktober": 10,
    "okt": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "desember": 12,
    "des": 12,
    "dec": 12,
}

_STOCKBIT_SCRAPE_QUERY = re.compile(
    r"\b("
    r"scrape|stockbit\s*reports?|hasil\s+scrape|"
    r"(report|laporan).*(tanggal|tgl)|"
    r"(tanggal|tgl).*(report|laporan|stockbit)"
    r")\b",
    re.I,
)

_PENDING_STREAM_SCRAPES: dict[int, dict[str, Any]] = {}


def is_stockbit_scrape_query(text: str) -> bool:
    """Stockbit scrape stream or read-by-date — not general composite report."""
    return bool(_STOCKBIT_SCRAPE_QUERY.search(text or ""))


def parse_user_date(text: str) -> str | None:
    """Parse user text to YYYY-MM-DD if possible."""
    raw = text or ""
    iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", raw)
    if iso:
        return iso.group(1)

    month_names = (
        r"januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|"
        r"november|desember|jan|feb|mar|apr|jun|jul|agu|aug|sep|okt|oct|nov|des|dec"
    )
    for pattern in (
        rf"\b(?:tanggal|tgl\.?)\s*(\d{{1,2}})\s+({month_names})\s+(20\d{{2}})\b",
        rf"\b(\d{{1,2}})\s+({month_names})\s+(20\d{{2}})\b",
        # "22 juli" / "yang 22 juli" without year → assume current calendar year
        rf"\b(?:tanggal|tgl\.?|yang|per|pada)?\s*(\d{{1,2}})\s+({month_names})\b",
    ):
        match = re.search(pattern, raw, re.I)
        if not match:
            continue
        day = int(match.group(1))
        month = _MONTH_ID[match.group(2).lower()]
        if match.lastindex and match.lastindex >= 3 and match.group(3):
            year = int(match.group(3))
        else:
            year = datetime.now().year
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def read_posts_for_date(
    store: StockbitReportsStore | None,
    export_dir: Path,
    target_date: str,
    *,
    limit: int = 50,
) -> str:
    if store is not None:
        return store.posts_by_date(target_date, limit=limit)

    folder = export_dir / "stockbit"
    if not folder.is_dir():
        return (
            "ERROR: belum ada data Stockbit Reports.\n"
            "Jalankan scrape dulu: /ask scrape Stockbit Reports ..."
        )
    temp = StockbitReportsStore(Path("_unused_chroma"), export_dir=export_dir)
    return temp._posts_by_date_markdown(target_date, limit=limit)


def scrape_stats(store: StockbitReportsStore | None, export_dir: Path) -> str:
    if store is not None:
        return store.stats()
    folder = export_dir / "stockbit"
    if not folder.is_dir():
        return "belum ada file scrape Stockbit Reports"
    files = sorted(
        iter_stream_markdown(folder),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return f"markdown_files={len(files)} (Chroma belum aktif, path: {folder})"


def search_posts(
    store: StockbitReportsStore | None,
    query: str,
    *,
    limit: int = 20,
) -> str:
    if store is None:
        return "ERROR: stockbit vector store belum dikonfigurasi"
    return store.search(query, limit=limit)


def stage_stream_scrape(chat_key: int, arguments: dict[str, Any]) -> str:
    """Stage a user-triggered Stream scrape without writing anything."""
    source = (arguments.get("source") or "").strip().lower() or None
    if source not in {None, "reports", "official", "both", "all", "semua"}:
        return "ERROR: source harus reports, official, atau both"
    days_raw = arguments.get("days")
    days = max(1, min(int(days_raw), 90)) if days_raw is not None else None
    action = {
        "days": days,
        "date_from": (arguments.get("date_from") or "").strip() or None,
        "date_to": (arguments.get("date_to") or "").strip() or None,
        "url": (arguments.get("url") or "").strip() or None,
        "source": source,
    }
    _PENDING_STREAM_SCRAPES[int(chat_key)] = action
    return "STAGED (menunggu konfirmasi user)\n" + describe_pending_stream_scrape(
        int(chat_key)
    )


def has_pending_stream_scrape(chat_key: int) -> bool:
    return int(chat_key) in _PENDING_STREAM_SCRAPES


def peek_pending_stream_scrape(chat_key: int) -> dict[str, Any] | None:
    action = _PENDING_STREAM_SCRAPES.get(int(chat_key))
    return dict(action) if action else None


def clear_pending_stream_scrape(chat_key: int) -> bool:
    return _PENDING_STREAM_SCRAPES.pop(int(chat_key), None) is not None


def describe_pending_stream_scrape(chat_key: int) -> str:
    action = _PENDING_STREAM_SCRAPES.get(int(chat_key))
    if not action:
        return "(tidak ada Stream scrape tertunda)"
    source = action.get("source") or ("custom-url" if action.get("url") else "reports")
    window = (
        f"days={action['days']}"
        if action.get("days") is not None
        else f"{action.get('date_from') or 'awal'} s/d {action.get('date_to') or 'sekarang'}"
    )
    return (
        "Stream scrape menunggu konfirmasi:\n"
        f"• source: {source}\n"
        f"• window: {window}\n"
        "• output: markdown exports/stockbit + ingest Vector store"
    )


def run_pending_stream_scrape(
    chat_key: int,
    *,
    stockbit_store: StockbitReportsStore | None = None,
) -> str:
    """Execute a previously confirmed Stream scrape."""
    action = _PENDING_STREAM_SCRAPES.pop(int(chat_key), None)
    if not action:
        return "Tidak ada Stream scrape untuk dijalankan."

    from agent_idx.browser_stockbit import OFFICIAL_URL, REPORTS_URL
    from agent_idx.tools import (
        _run_stockbit,
        _stockbit_browser,
        _stockbit_headless,
        extract_files,
    )

    def _scrape() -> str:
        browser = _stockbit_browser(_stockbit_headless())
        browser.start()
        kwargs = {
            "days": action.get("days"),
            "date_from": action.get("date_from"),
            "date_to": action.get("date_to"),
        }
        source = action.get("source")
        if source in {"both", "all", "semua"} and not action.get("url"):
            parts: list[str] = []
            for label, target in (
                ("reports", REPORTS_URL),
                ("official", OFFICIAL_URL),
            ):
                result = browser.scrape_reports_stream(url=target, **kwargs)
                parts.append(f"### source={label}\n{result}")
            return "\n\n".join(parts)
        return browser.scrape_reports_stream(
            url=action.get("url"),
            source=source,
            **kwargs,
        )

    result = _run_stockbit(_scrape)
    if stockbit_store is not None and (
        result.startswith("OK:") or "\nOK:" in result
    ):
        inserted = sum(
            stockbit_store.ingest_markdown(path) for path in extract_files(result)
        )
        if inserted:
            result += (
                f"\nVector: {inserted} post disimpan ke Chroma "
                f"({stockbit_store.store_path})"
            )
    return result
