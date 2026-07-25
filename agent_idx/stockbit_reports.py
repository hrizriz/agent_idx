from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from agent_idx.stockbit_store import StockbitReportsStore

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


def is_stockbit_scrape_query(text: str) -> bool:
    """Stockbit scrape stream or read-by-date — not general composite report."""
    return bool(_STOCKBIT_SCRAPE_QUERY.search(text or ""))


def parse_user_date(text: str) -> str | None:
    """Parse user text to YYYY-MM-DD if possible."""
    raw = text or ""
    iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", raw)
    if iso:
        return iso.group(1)

    for pattern in (
        r"\b(?:tanggal|tgl\.?)\s*(\d{1,2})\s+"
        r"(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|"
        r"november|desember|jan|feb|mar|apr|jun|jul|agu|aug|sep|okt|oct|nov|des|dec)"
        r"\s+(20\d{2})\b",
        r"\b(\d{1,2})\s+"
        r"(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|"
        r"november|desember|jan|feb|mar|apr|jun|jul|agu|aug|sep|okt|oct|nov|des|dec)"
        r"\s+(20\d{2})\b",
    ):
        match = re.search(pattern, raw, re.I)
        if not match:
            continue
        day = int(match.group(1))
        month = _MONTH_ID[match.group(2).lower()]
        year = int(match.group(3))
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
    files = sorted(folder.glob("StockbitReports_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
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
