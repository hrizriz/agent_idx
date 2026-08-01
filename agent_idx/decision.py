from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from agent_idx.composite_report import is_composite_report_query
from agent_idx.context import is_meta_bot_query
from agent_idx.knowledge import is_meta_learning_query
from agent_idx.stockbit_reports import is_stockbit_scrape_query, parse_user_date

_SYMBOL = re.compile(r"\b([A-Z]{4})\b")
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
        "BUKA",
        "OPEN",
        "CEK",
        "HARI",
    }
)
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
    r"stockbitreports|stockbit\s+reports|/StockbitReports|"
    r"@Stockbit\b|stockbit\.com/Stockbit\b|"
    r"akun\s+(?:resmi\s+)?stockbit|official\s+stockbit",
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
    r"\b("
    r"chat|telegram|"
    r"siapa\s+bilang|diskusi\s+(?:di\s+)?(?:group|grup|chat)|"
    r"di\s+(?:group|grup)\b|group\s+ini|grup\s+(?:ini|chat|telegram|kita)\b"
    r")\b",
    re.I,
)
_KNOWLEDGE_HINT = re.compile(
    r"\b("
    r"fundamental|valuasi|rasio|teori|kurikulum|pe\s+ratio|roe|car\s+analisis|"
    r"macd|sma\b|ema\b|npl|"
    r"apa\s+itu\s+(?:macd|sma|ema|roe|npl|rsi|pe|pbv|der|hsc|free\s*float)|"
    r"apa\s+beda(?:nya)?\s+(?:macd|sma|ema)|"
    r"jelaskan\s+(?:macd|sma|ema|roe|npl|rsi|hsc)|"
    r"kategori\s+hsc|notasi\s+khusus|special\s+notation"
    r")\b",
    re.I,
)
_KEYSTATS_HINT = re.compile(
    r"\b("
    r"key\s*stats?|keystats?|key-stat|"
    r"profil(?:e)?(?:\s+perusahaan|\s+emiten)?|"
    r"shareholder|pemegang\s+saham|bod\b|komisaris|direksi|"
    r"company\s+background|financials?\s+stockbit"
    r")\b",
    re.I,
)
_FUNDAMENTAL_FACT = re.compile(
    r"\b("
    r"free\s*float|freefloat|float\s+saham|"
    r"market\s*cap|kapitalisasi|"
    r"jumlah\s+saham|saham\s+beredar|listed\s+shares|outstanding|"
    r"treasury|saham\s+tresuri|saham\s+treasury|"
    r"\bpbv\b|\bper\b|\broe\b|harga\s+wajar|valuasi|"
    r"\bhsc\b|notasi\s+khusus|special\s+notation|"
    r"kepemilikan|%\s*(?:saham|kepemilikan|dari\s+saham)|"
    r"(?:setara|value|nilai)\s+(?:dengan\s+)?(?:berapa\s+)?(?:rupiah|rp)|"
    r"berapa\s+(?:value|nilai|rupiah)"
    r")\b",
    re.I,
)
_TECH_AND_FUND = re.compile(
    r"\b("
    r"teknikal\s*(?:&|dan|/)\s*fundamental|"
    r"fundamental\s*(?:&|dan|/)\s*teknikal|"
    r"analyze_stock|analisa(?:kan)?\s+saham|"
    r"cek\s+(?:\$?[A-Za-z]{3,5})\s+fundamental"
    r")\b",
    re.I,
)
_MACRO_EXTERNAL = re.compile(
    r"\b("
    r"sbn|sun\b|yield\s+(?:sbn|sun|obligasi)|"
    r"usd/?idr|kurs\s+(?:dolar|dollar|usd)|"
    r"pengangguran|phk\b|kemnaker|"
    r"gubernur\s+(?:bi|bank\s+indonesia)|bank\s+indonesia|"
    r"the\s+fed|fomc|powell|chairman\s+(?:the\s+)?fed|"
    r"calon\s+gubernur|kredibilitas\s+data"
    r")\b",
    re.I,
)
_CORP_OWNERSHIP = re.compile(
    r"\b("
    r"(?:grup|group)\s+(?:sinarmas|astra|salim|djarum|lippo|bakrie)|"
    r"punya\s+emiten|emiten\s+apa\s+saja|"
    r"osint|afiliasi|anak\s+usaha|holding|"
    r"akuisisi|tender\s+offer|takeover|pengambilalihan"
    r")\b",
    re.I,
)
_STOCKBIT_CONTENT = re.compile(
    r"(?:"
    r"stockbitreports|stockbit\s+reports?|/StockbitReports|"
    r"@Stockbit\b|stockbit\.com/Stockbit\b|"
    r"akun\s+(?:resmi\s+)?stockbit|official\s+stockbit|"
    r"berita\s+report|report(?:s)?\s+(?:stockbit|per\s+\d)|"
    r"\brups(?:lb)?\b|\brpuslb\b|\brupslb\b"
    r")",
    re.I,
)
_FARM_CHART = re.compile(
    r"\b("
    r"farming|farm\s+(?:stockbit\s+)?charts?|"
    r"scrape\s+charts?|"
    r"(?:chart|ohlcv).{0,40}\b(?:semua|all)\s+emiten\b|"
    r"\bsemua\s+emiten\b.{0,40}\b(?:chart|1[hdw]|ohlcv)\b"
    r")\b",
    re.I,
)
_STOCKBIT_POST = re.compile(
    r"\b("
    r"posting|buat\s+post|kirim\s+post|publish\s+(?:post|ide)|"
    r"create\s+post|post\s+(?:di\s+|ke\s+)?stockbit|stockbit\s+post|"
    r"post\s+['\"]|bikin\s+post"
    r")\b",
    re.I,
)
_STOCKBIT_STATUS = re.compile(
    r"\b("
    r"status\s+(?:login\s+)?stockbit|sesi\s+stockbit|stockbit\s+masih\s+aktif|"
    r"apakah\s+(?:sudah\s+)?login\s+stockbit|jangan\s+scrape.{0,20}status"
    r")\b",
    re.I,
)
_PDF_INLINE = re.compile(r"--- ISI PDF", re.I)
_CHART_LOCAL = re.compile(
    r"\b("
    r"duckdb|dari\s+database|pakai\s+data\s+lokal|tanpa\s+farm|"
    r"query\s+chart|data\s+tersimpan|sudah\s+di-?farm"
    r")\b",
    re.I,
)


def _attach_pipeline(plan: QueryPlan) -> QueryPlan:
    """Bind system-defined tool pipeline + allowed_tools to the plan."""
    from agent_idx.pipelines import PIPELINES, pipeline_for_intent

    if plan.pipeline and plan.pipeline in PIPELINES:
        pdef = PIPELINES[plan.pipeline]
        plan.allowed_tools = list(pdef.allowed_tools)
        return plan
    pdef = pipeline_for_intent(plan.intent)
    if pdef:
        plan.pipeline = pdef.id
        plan.allowed_tools = list(pdef.allowed_tools)
    return plan


class Intent(str, Enum):
    PDF_ANALYSIS = "pdf_analysis"
    DOCUMENT_INGEST = "document_ingest"
    STOCKBIT_READ_DATE = "stockbit_read_date"
    LEARNING_META = "learning_meta"
    BOT_META = "bot_meta"
    STOCKBIT_SCRAPE = "stockbit_scrape"
    STOCKBIT_CHART = "stockbit_chart"
    STOCKBIT_POST = "stockbit_post"
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
    symbols: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    retrieval: str = "mixed"
    reason: str = ""
    # Structured backend pipeline (system-defined tool graph).
    pipeline: str = ""
    allowed_tools: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines = [
            f"intent={self.intent.value}",
            f"focus={self.focus}",
            f"sources={','.join(self.sources) or 'llm_only'}",
            f"retrieval={self.retrieval}",
            f"reason={self.reason}",
        ]
        if self.symbols:
            lines.append(f"symbols={','.join(self.symbols)}")
        if self.pipeline:
            lines.append(f"pipeline={self.pipeline}")
            from agent_idx.pipelines import get_pipeline

            pdef = get_pipeline(self.pipeline)
            if pdef:
                lines.append(
                    "pipeline_tools="
                    + " -> ".join(s.tool for s in pdef.steps)
                )
        if self.allowed_tools:
            lines.append(f"allowed_tools={','.join(self.allowed_tools)}")
        return "\n".join(lines)


def extract_symbols(text: str) -> list[str]:
    from agent_idx.context import extract_symbols as _ctx_symbols

    return sorted(_ctx_symbols(text))


def extract_tickers(text: str) -> list[str]:
    """Deprecated alias for extract_symbols."""
    return extract_symbols(text)


def extract_stockbit_url(text: str) -> str | None:
    match = _STOCKBIT_PAGE_URL.search(text or "")
    if not match:
        return None
    return match.group(0).rstrip(".,;:)")


def extract_report_focus(text: str) -> str:
    if re.search(r"\bytd\b", text, re.I):
        return "YTD IDX saham"
    symbols = extract_symbols(text)
    if symbols:
        return " ".join(symbols)
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
    symbols = extract_symbols(raw)

    if _PDF_INLINE.search(raw):
        from agent_idx.doc_ingest import is_document_save_request

        if is_document_save_request(raw):
            return _attach_pipeline(
                QueryPlan(
                    intent=Intent.DOCUMENT_INGEST,
                    focus="user document",
                    sources=["document_db"],
                    retrieval="pipeline",
                    reason="user minta simpan dokumen ke DB (staged)",
                    pipeline="document_ingest",
                )
            )
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
                symbols=symbols,
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

    if is_meta_bot_query(raw) or _STOCKBIT_STATUS.search(raw):
        return QueryPlan(
            intent=Intent.BOT_META,
            focus="bot capabilities / ops",
            sources=[],
            retrieval="direct_answer",
            reason="pertanyaan tentang kemampuan/cron/ops bot",
        )

    if _FARM_CHART.search(raw):
        return QueryPlan(
            intent=Intent.GENERAL,
            focus=symbols[0] if symbols else "farm stockbit charts",
            symbols=symbols,
            sources=["stockbit_browser"],
            retrieval="farm_charts",
            reason="farming OHLCV chart Stockbit",
        )

    if _STOCKBIT_POST.search(raw):
        return QueryPlan(
            intent=Intent.STOCKBIT_POST,
            focus="stockbit stream post",
            symbols=symbols,
            sources=["stockbit_browser"],
            retrieval="create_post",
            reason="buat postingan di stream Stockbit",
        )

    page_url = extract_stockbit_url(raw)
    if _SCRAPE_VERB.search(raw) and (page_url or _REPORTS_URL.search(raw)):
        return _attach_pipeline(
            QueryPlan(
                intent=Intent.STOCKBIT_SCRAPE,
                focus=page_url or "Stockbit Reports",
                symbols=symbols,
                sources=["stockbit_reports"],
                retrieval="browser_scrape",
                reason="scrape stream Stockbit",
            )
        )

    if page_url and "chartbit" in page_url.lower():
        pipe = "chart_query" if _CHART_LOCAL.search(raw) else "chart_analyze"
        return _attach_pipeline(
            QueryPlan(
                intent=Intent.STOCKBIT_CHART,
                focus=page_url,
                symbols=symbols,
                sources=["stockbit_browser", "chart_duckdb"],
                retrieval="pipeline",
                reason="buka/analisa chart Stockbit dari URL",
                pipeline=pipe,
            )
        )
    if _CHART_QUERY.search(raw) and (_OPEN_VERB.search(raw) or symbols):
        pipe = "chart_query" if _CHART_LOCAL.search(raw) else "chart_analyze"
        return _attach_pipeline(
            QueryPlan(
                intent=Intent.STOCKBIT_CHART,
                focus=symbols[0] if symbols else focus,
                symbols=symbols,
                sources=["stockbit_browser", "chart_duckdb"],
                retrieval="pipeline",
                reason="buka/analisa chart symbol",
                pipeline=pipe,
            )
        )
    if re.search(
        r"\btimeframe\b|\b\d+\s*menit\b|\b(1m|5m|15m|30m|1h|4h|1d|1w)\b",
        raw,
        re.I,
    ) and symbols:
        pipe = "chart_query" if _CHART_LOCAL.search(raw) else "chart_analyze"
        return _attach_pipeline(
            QueryPlan(
                intent=Intent.STOCKBIT_CHART,
                focus=symbols[0],
                symbols=symbols,
                sources=["stockbit_browser", "chart_duckdb"],
                retrieval="pipeline",
                reason="baca OHLCV Chartbit per timeframe",
                pipeline=pipe,
            )
        )
    if _CHART_LOCAL.search(raw) and symbols:
        return _attach_pipeline(
            QueryPlan(
                intent=Intent.STOCKBIT_CHART,
                focus=symbols[0],
                symbols=symbols,
                sources=["chart_duckdb"],
                retrieval="pipeline",
                reason="query chart dari DuckDB",
                pipeline="chart_query",
            )
        )

    # Keystats / company profile → Stockbit scrape (bukan parquet transaksi)
    if _KEYSTATS_HINT.search(raw) and symbols:
        return QueryPlan(
            intent=Intent.GENERAL,
            focus=symbols[0],
            symbols=symbols,
            sources=["stockbit_browser"],
            retrieval="stockbit_scrape",
            reason="keystats/profile Stockbit (bukan daily_stock)",
            allowed_tools=[
                "stockbit_scrape",
                "farm_stockbit_fundamentals",
                "stockbit_status",
                "stockbit_open",
                "stockbit_read",
                "get_fundamentals",
                "analyze_stock",
                "get_stockbit_ohlcv",
                "search_stockbit_reports",
            ],
        )

    # Free float / PBV / treasury / HSC / nilai % kepemilikan → fundamentals + harga
    if _FUNDAMENTAL_FACT.search(raw):
        if symbols:
            return QueryPlan(
                intent=Intent.GENERAL,
                focus=symbols[0],
                symbols=symbols,
                sources=["stockbit_browser", "pasar", "knowledge"],
                retrieval="fundamentals_tools",
                reason=(
                    "fakta fundamental emiten (float/shares/valuasi/HSC) — "
                    "WAJIB get_fundamentals/analyze_stock/scrape; GAP_DATA jika kosong"
                ),
                allowed_tools=[
                    "analyze_stock",
                    "get_fundamentals",
                    "get_stockbit_ohlcv",
                    "get_technicals",
                    "stockbit_scrape",
                    "farm_stockbit_fundamentals",
                    "search_stockbit_reports",
                    "search_knowledge",
                    "search_news",
                    "list_date_range",
                ],
            )
        return QueryPlan(
            intent=Intent.KNOWLEDGE,
            focus=focus,
            symbols=symbols,
            sources=["knowledge", "berita"],
            retrieval="keyword",
            reason="definisi/aturan fundamental (HSC/MSCI/float) tanpa ticker — jangan pakai OHLCV",
            allowed_tools=[
                "search_knowledge",
                "search_news",
                "search_stockbit_reports",
            ],
        )

    # Teknikal + fundamental bersama → analyze_stock, bukan knowledge teori saja
    if _TECH_AND_FUND.search(raw) and symbols:
        return QueryPlan(
            intent=Intent.GENERAL,
            focus=symbols[0],
            symbols=symbols,
            sources=["stockbit_browser", "pasar"],
            retrieval="analyze_stock",
            reason="analisis teknikal+fundamental emiten via analyze_stock",
            allowed_tools=[
                "analyze_stock",
                "get_technicals",
                "get_fundamentals",
                "get_stockbit_ohlcv",
                "search_stockbit_reports",
                "search_news",
                "list_date_range",
            ],
        )

    # Makro/eksternal (SBN, USDIDR, PHK, Gubernur BI, Fed) — bukan parquet saham
    if _MACRO_EXTERNAL.search(raw) and not (
        symbols and re.search(r"\b(harga|chart|ohlc|volume|teknikal)\b", raw, re.I)
    ):
        return QueryPlan(
            intent=Intent.GENERAL,
            focus=focus,
            symbols=symbols,
            sources=["berita", "knowledge"],
            retrieval="news_gap",
            reason=(
                "makro/eksternal di luar store OHLCV — search_news/knowledge; "
                "angka yang tidak ada di tool = GAP_DATA (jangan mengarang)"
            ),
            allowed_tools=[
                "search_news",
                "search_knowledge",
                "search_stockbit_reports",
                "list_date_range",
                "get_stock_history",
            ],
        )

    # OSINT grup usaha / daftar emiten afiliasi / akuisisi
    # (jangan override permintaan "berita …")
    if _CORP_OWNERSHIP.search(raw) and not _NEWS_HINT.search(raw):
        return QueryPlan(
            intent=Intent.GENERAL,
            focus=focus,
            symbols=symbols,
            sources=["stockbit_browser", "knowledge", "berita", "stockbit_reports"],
            retrieval="ownership_osint",
            reason="afiliasi/grup usaha — profile/reports/knowledge; GAP_DATA jika tidak tercatat",
            allowed_tools=[
                "stockbit_scrape",
                "get_fundamentals",
                "search_stockbit_reports",
                "search_knowledge",
                "search_news",
                "search_chat_history",
            ],
        )

    # "tentang apa ... 22 juli" / proyeksi earnings → Stockbit reports / financials
    about_date = parse_user_date(raw)
    if about_date and re.search(
        r"\b(tentang\s+apa|apa\s+(?:itu|isinya)|isi\s+(?:post|berita|report)|"
        r"maksud(?:nya)?|bahas(?:an)?)\b",
        raw,
        re.I,
    ):
        return QueryPlan(
            intent=Intent.STOCKBIT_READ_DATE,
            focus=about_date,
            symbols=symbols,
            sources=["stockbit_reports"],
            retrieval="metadata_filter",
            reason="tanya isi post Stockbit Reports di tanggal tersebut",
        )

    if re.search(
        r"\b(proyeksi|estimasi|guidance|outlook)\b.{0,40}\b("
        r"pendapatan|revenue|laba|earnings|eps|q[1-4]|kuartal)\b|"
        r"\b(pendapatan|revenue|laba|earnings)\b.{0,40}\b("
        r"proyeksi|estimasi|q[1-4]|kuartal)\b",
        raw,
        re.I,
    ) and symbols:
        return QueryPlan(
            intent=Intent.GENERAL,
            focus=symbols[0],
            symbols=symbols,
            sources=["stockbit_reports", "stockbit_browser", "knowledge"],
            retrieval="search_posts",
            reason="proyeksi/earnings — Stockbit reports + financials, bukan OHLCV saja",
            allowed_tools=[
                "search_stockbit_reports",
                "read_stockbit_scrape_date",
                "stockbit_scrape",
                "search_knowledge",
                "search_news",
                "get_stock_history",
            ],
        )

    # Tanya isi Stockbit Reports / RUPS / berita report → bukan RSS umum
    if _STOCKBIT_CONTENT.search(raw) or (
        _REPORTS_URL.search(raw) and not _SCRAPE_VERB.search(raw)
    ):
        date = parse_user_date(raw)
        if date and not _SCRAPE_VERB.search(raw):
            return QueryPlan(
                intent=Intent.STOCKBIT_READ_DATE,
                focus=date,
                symbols=symbols,
                sources=["stockbit_reports"],
                retrieval="metadata_filter",
                reason="baca post Stockbit Reports per tanggal",
            )
        return QueryPlan(
            intent=Intent.GENERAL,
            focus=focus,
            symbols=symbols,
            sources=["stockbit_reports"],
            retrieval="search_posts",
            reason="cari/analisis isi Stockbit Reports",
            allowed_tools=[
                "search_stockbit_reports",
                "read_stockbit_scrape_date",
                "list_stockbit_scrape_files",
                "search_news",
                "get_stock_history",
                "list_date_range",
            ],
        )

    if is_composite_report_query(raw):
        # Koreksi singkat "bukan X, tapi berita report..." bukan laporan gabungan.
        if re.search(r"\bbukan\b", raw, re.I) and len(raw) < 100:
            pass
        else:
            return QueryPlan(
                intent=Intent.COMPOSITE_REPORT,
                focus=focus,
                symbols=symbols,
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
            symbols=symbols,
            sources=["pasar"],
            retrieval="backtest_engine",
            reason="scan skenario trading pada parquet",
        )

    if _CHAT_HINT.search(raw):
        return QueryPlan(
            intent=Intent.CHAT_SEARCH,
            focus=focus,
            symbols=symbols,
            sources=["chat"],
            retrieval="sql_like",
            reason="cari di chat Telegram",
        )

    if _NEWS_HINT.search(raw):
        # "berita" + symbol sering maksudnya Stockbit Reports / katalis, bukan RSS.
        if symbols and re.search(
            r"\b(stockbit|reports?|rups|corporate\s+action|katalis)\b",
            raw,
            re.I,
        ):
            return QueryPlan(
                intent=Intent.GENERAL,
                focus=focus,
                symbols=symbols,
                sources=["stockbit_reports", "berita"],
                retrieval="search_posts",
                reason="berita/katalis emiten via Stockbit Reports (+RSS cadangan)",
                allowed_tools=[
                    "search_stockbit_reports",
                    "read_stockbit_scrape_date",
                    "list_stockbit_scrape_files",
                    "search_news",
                ],
            )
        return QueryPlan(
            intent=Intent.NEWS,
            focus=focus,
            symbols=symbols,
            sources=["berita"],
            retrieval="live_rss",
            reason="permintaan berita",
        )

    if _KNOWLEDGE_HINT.search(raw):
        return QueryPlan(
            intent=Intent.KNOWLEDGE,
            focus=focus,
            symbols=symbols,
            sources=["knowledge"],
            retrieval="keyword",
            reason="teori/fundamental",
        )

    if symbols or _MARKET_HINT.search(raw) or re.search(r"\b20\d{2}\b", raw):
        return QueryPlan(
            intent=Intent.MARKET_ANALYSIS,
            focus=focus,
            symbols=symbols,
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
        symbols=symbols,
        sources=_default_sources(pasar=True, berita=True, stockbit=True),
        retrieval="llm_tools",
        reason="fallback ke tool loop LLM",
    )
