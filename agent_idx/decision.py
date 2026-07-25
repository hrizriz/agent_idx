from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from agent_idx.composite_report import is_composite_report_query
from agent_idx.knowledge import is_meta_learning_query
from agent_idx.stockbit_reports import is_stockbit_scrape_query, parse_user_date

_TICKER = re.compile(r"\b([A-Z]{4})\b")
_SKIP_TICKERS = frozenset({"YTD", "IDX", "PDF", "BEI", "NEWS", "CHAT", "SQL", "DATA"})
_SCRAPE_VERB = re.compile(
    r"\b(scrape|tarik|ambil|unduh|download|export|riwayat|histori)\b",
    re.I,
)
_OPEN_VERB = re.compile(r"\b(buka|open|lihat|tampilkan|cek|pindah)\b", re.I)
_CHART_QUERY = re.compile(r"\b(chart|chartbit|grafik)\b", re.I)
_STOCKBIT_PAGE_URL = re.compile(
    r"https?://(?:www\.)?stockbit\.com/[^\s)>\"']+",
    re.I,
)
_REPORTS_URL = re.compile(
    r"stockbitreports|stockbit\s+reports|/StockbitReports",
    re.I,
)
_BACKTEST_HINT = re.compile(
    r"\b("
    r"backtest|win\s*rate|skenario|scenario|setup\s+trading|"
    r"trading\s+setup|scan\s+trading|uji\s+strategi|strategi\s+trading"
    r")\b",
    re.I,
)
_MARKET_HINT = re.compile(
    r"\b("
    r"foreign|net\s*(buy|sell)|volume|return|ohlc|harga|transaksi|"
    r"ranking|top|banding|analisis|rekomendasi|sql|ticker|emiten|ihsg"
    r")\b",
    re.I,
)
_NEWS_HINT = re.compile(r"\b(berita|news|sentimen|headline)\b", re.I)
_CHAT_HINT = re.compile(
    r"\b(chat|group|grup|siapa\s+bilang|diskusi|telegram)\b",
    re.I,
)
_KNOWLEDGE_HINT = re.compile(
    r"\b(fundamental|valuasi|rasio|teori|kurikulum|pe\s+ratio|roe|car\s+analisis)\b",
    re.I,
)
_PDF_INLINE = re.compile(r"--- ISI PDF", re.I)


class Intent(str, Enum):
    PDF_ANALYSIS = "pdf_analysis"
    STOCKBIT_READ_DATE = "stockbit_read_date"
    LEARNING_META = "learning_meta"
    STOCKBIT_SCRAPE = "stockbit_scrape"
    STOCKBIT_CHART = "stockbit_chart"
    COMPOSITE_REPORT = "composite_report"
    BACKTEST_SCAN = "backtest_scan"
    MARKET_ANALYSIS = "market_analysis"
    NEWS = "news"
    CHAT_SEARCH = "chat_search"
    KNOWLEDGE = "knowledge"
    GENERAL = "general"


@dataclass
class QueryPlan:
    intent: Intent
    focus: str
    tickers: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    retrieval: str = "mixed"
    reason: str = ""

    def to_prompt(self) -> str:
        lines = [
            f"intent={self.intent.value}",
            f"focus={self.focus}",
            f"sources={','.join(self.sources) or 'llm_only'}",
            f"retrieval={self.retrieval}",
            f"reason={self.reason}",
        ]
        if self.tickers:
            lines.append(f"tickers={','.join(self.tickers)}")
        return "\n".join(lines)


def extract_tickers(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _TICKER.finditer(text or ""):
        sym = m.group(1).upper()
        if sym in _SKIP_TICKERS or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def extract_stockbit_url(text: str) -> str | None:
    match = _STOCKBIT_PAGE_URL.search(text or "")
    if not match:
        return None
    return match.group(0).rstrip(".,;:)")


def extract_report_focus(text: str) -> str:
    if re.search(r"\bytd\b", text, re.I):
        return "YTD IDX saham"
    tickers = extract_tickers(text)
    if tickers:
        return " ".join(tickers)
    if re.search(r"\b(stockbit|pengumuman|corporate\s+action)\b", text, re.I):
        return "pengumuman BEI Stockbit"
    return "IDX saham Indonesia"


def _default_sources(
    *,
    pasar: bool = False,
    berita: bool = False,
    stockbit: bool = False,
    knowledge: bool = False,
    chat: bool = False,
) -> list[str]:
    out: list[str] = []
    if pasar:
        out.append("pasar")
    if berita:
        out.append("berita")
    if stockbit:
        out.append("stockbit_reports")
    if knowledge:
        out.append("knowledge")
    if chat:
        out.append("chat")
    return out


def classify_intent(text: str) -> QueryPlan:
    raw = (text or "").strip()
    focus = extract_report_focus(raw)
    tickers = extract_tickers(raw)

    if _PDF_INLINE.search(raw):
        return QueryPlan(
            intent=Intent.PDF_ANALYSIS,
            focus="pdf user",
            sources=[],
            retrieval="inline_text",
            reason="PDF sudah di-inject ke prompt",
        )

    if is_stockbit_scrape_query(raw) and parse_user_date(raw):
        if not (_SCRAPE_VERB.search(raw) and len(re.findall(
            r"\b(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s+(20\d{2})\b",
            raw,
            re.I,
        )) >= 2):
            return QueryPlan(
                intent=Intent.STOCKBIT_READ_DATE,
                focus=parse_user_date(raw) or focus,
                tickers=tickers,
                sources=["stockbit_reports"],
                retrieval="metadata_filter",
                reason="baca post Stockbit Reports per tanggal",
            )

    if is_meta_learning_query(raw):
        return QueryPlan(
            intent=Intent.LEARNING_META,
            focus="bot knowledge inventory",
            sources=["knowledge"],
            retrieval="file_list",
            reason="meta pertanyaan progres belajar bot",
        )

    page_url = extract_stockbit_url(raw)
    if _SCRAPE_VERB.search(raw) and (page_url or _REPORTS_URL.search(raw)):
        return QueryPlan(
            intent=Intent.STOCKBIT_SCRAPE,
            focus=page_url or "Stockbit Reports",
            tickers=tickers,
            sources=["stockbit_reports"],
            retrieval="browser_scrape",
            reason="scrape stream Stockbit",
        )

    if page_url and "chartbit" in page_url.lower():
        return QueryPlan(
            intent=Intent.STOCKBIT_CHART,
            focus=page_url,
            tickers=tickers,
            sources=["stockbit_browser"],
            retrieval="browser_open",
            reason="buka chart Stockbit dari URL",
        )
    if _CHART_QUERY.search(raw) and (_OPEN_VERB.search(raw) or tickers):
        return QueryPlan(
            intent=Intent.STOCKBIT_CHART,
            focus=tickers[0] if tickers else focus,
            tickers=tickers,
            sources=["stockbit_browser"],
            retrieval="browser_open",
            reason="buka chart ticker",
        )

    if is_composite_report_query(raw):
        return QueryPlan(
            intent=Intent.COMPOSITE_REPORT,
            focus=focus,
            tickers=tickers,
            sources=_default_sources(
                pasar=True,
                berita=True,
                stockbit=True,
                knowledge=True,
                chat=_CHAT_HINT.search(raw) is not None,
            ),
            retrieval="multi_source_fusion",
            reason="laporan gabungan multi-sumber",
        )

    if _BACKTEST_HINT.search(raw):
        return QueryPlan(
            intent=Intent.BACKTEST_SCAN,
            focus=focus,
            tickers=tickers,
            sources=["pasar"],
            retrieval="backtest_engine",
            reason="scan skenario trading pada parquet",
        )

    if _CHAT_HINT.search(raw):
        return QueryPlan(
            intent=Intent.CHAT_SEARCH,
            focus=focus,
            tickers=tickers,
            sources=["chat"],
            retrieval="sql_like",
            reason="cari di chat Telegram",
        )

    if _NEWS_HINT.search(raw):
        return QueryPlan(
            intent=Intent.NEWS,
            focus=focus,
            tickers=tickers,
            sources=["berita"],
            retrieval="live_rss",
            reason="permintaan berita",
        )

    if _KNOWLEDGE_HINT.search(raw):
        return QueryPlan(
            intent=Intent.KNOWLEDGE,
            focus=focus,
            tickers=tickers,
            sources=["knowledge"],
            retrieval="keyword",
            reason="teori/fundamental",
        )

    if tickers or _MARKET_HINT.search(raw) or re.search(r"\b20\d{2}\b", raw):
        return QueryPlan(
            intent=Intent.MARKET_ANALYSIS,
            focus=focus,
            tickers=tickers,
            sources=_default_sources(
                pasar=True,
                berita=_NEWS_HINT.search(raw) is not None,
                stockbit=re.search(r"\bstockbit\b", raw, re.I) is not None,
                knowledge=_KNOWLEDGE_HINT.search(raw) is not None,
            ),
            retrieval="structured_sql",
            reason="analisis data transaksi parquet",
        )

    return QueryPlan(
        intent=Intent.GENERAL,
        focus=focus,
        tickers=tickers,
        sources=_default_sources(pasar=True, berita=True, stockbit=True),
        retrieval="llm_tools",
        reason="fallback ke tool loop LLM",
    )
