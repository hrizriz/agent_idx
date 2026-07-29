from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent_idx.chat_log import ChatLog
from agent_idx.config import Settings
from agent_idx.data import StockDataStore
from agent_idx.knowledge import KnowledgeBase, is_meta_learning_query
from agent_idx.composite_report import (
    composite_report_system_prompt,
    format_report_context,
    gather_report_context,
)
from agent_idx.decision import Intent, QueryPlan, classify_intent, extract_stockbit_url
from agent_idx.stockbit_reports import is_stockbit_scrape_query, parse_user_date
from agent_idx.stockbit_store import StockbitReportsStore
from agent_idx.tools import TOOL_DEFINITIONS, dispatch_tool, extract_files

logger = logging.getLogger(__name__)

_REPORTS_URL = re.compile(
    r"stockbitreports|stockbit\s+reports|/StockbitReports|"
    r"@Stockbit\b|stockbit\.com/Stockbit\b|"
    r"akun\s+(?:resmi\s+)?stockbit|official\s+stockbit",
    re.I,
)
_SCRAPE_VERB = re.compile(
    r"\b(scrape|tarik|ambil|unduh|download|export|riwayat|histori)\b",
    re.I,
)
_OPEN_VERB = re.compile(r"\b(buka|open|lihat|tampilkan|cek|pindah)\b", re.I)
_CHART_QUERY = re.compile(r"\b(chart|chartbit|grafik)\b", re.I)
_SYMBOL_CHART_URL = re.compile(r"/symbol/([A-Za-z]{3,5})/chartbit", re.I)
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

_TOOL_HINT = re.compile(
    r"\b("
    r"analisis|rekomendasi|pantau|watchlist|foreign|volume|return|"
    r"berita|news|laporan|report|export|csv|data|sql|ticker|emiten|"
    r"banding|bandingkan|ranking|top|net\s*sell|net\s*buy|ohlc|harga|"
    r"periode|tahun|ytd|bbca|bmri|bbri|bbni|goto|idx|saham|"
    r"buat\s+pdf|kirim\s+pdf|generate\s+pdf|"
    r"fundamental|valuasi|rasio|charting|teknikal|nim|npl|roe|roa|der|pe\b|pb\b|"
    r"belajar|kurikulum|knowledge|digest|ajarin|ingat\s+bahwa|catat|"
    r"chat|group|grup|siapa\s+bilang|diskusi|stockbit|scrape|reports|stockbitreports|"
    r"posting|buat\s+post|kirim\s+post|create\s+post|"
    r"posisi|advice|entry|stop\s*loss|cut\s*loss|hold|average"
    r")\b|"
    r"\b20\d{2}\b|"
    r"\b[A-Z]{3,5}\b",
    re.I,
)

_POST_QUOTED = re.compile(
    r"""['"“”‘’](.+?)['"“”‘’]""",
    re.S,
)
_POST_AFTER = re.compile(
    r"(?:posting|post|ide)\s+(?:di\s+|ke\s+)?(?:stockbit\s+)?[:\-]?\s*(.+)$",
    re.I | re.S,
)


def _extract_post_text(user_message: str) -> str:
    """Pull post body from quotes or trailing text after 'posting...'."""
    raw = (user_message or "").strip()
    if not raw:
        return ""
    m = _POST_QUOTED.search(raw)
    if m:
        return m.group(1).strip()
    m = _POST_AFTER.search(raw)
    if m:
        body = (m.group(1) or "").strip().strip("'\"").strip()
        # Drop leftover filler like "di stockbit"
        body = re.sub(r"^(?:di\s+|ke\s+)?stockbit\s*", "", body, flags=re.I).strip()
        if body and body.lower() not in {"stockbit", "di stockbit", "ke stockbit"}:
            return body
    return ""


SYSTEM_PROMPT = """\
Kamu adalah analis & asisten pasar saham IDX.
User yang memerintah; kamu yang mengerjakan. JANGAN pernah menyuruh user mencari data.

PERAN
- Kamu eksekutor: tools, analisis, catat knowledge. PDF/CSV hanya jika user minta.
- DILARANG membalas dengan tugas untuk user (contoh: "tolong carikan data ADRO...").
- DILARANG meniru gaya user yang sedang memberi instruksi kepadamu.

KEMAMPUAN TOOLS
- Data pasar: list_date_range, describe_schema, get_stock_history, summarize_period,
  rank_foreign_flow, run_sql (SELECT only), run_backtest_scan
- Knowledge: search_knowledge, list_knowledge_topics, save_learning_note, run_learning_job
- Chat group: list_collected_chats, search_chat_history, recent_chat_history, chat_log_stats
- Berita: search_news — jawab RINGKASAN di chat sebagai teks.
  DILARANG mengirim file digest (.md) atau memanggil run_learning_job
  hanya karena user tanya berita. File/PDF hanya jika user minta eksplisit.
- Laporan gabungan: jika user minta report/laporan/ringkas/ytd (tanpa PDF lampiran),
  gabungkan data pasar + berita + Stockbit Reports + knowledge — jangan minta PDF user.
- File: create_pdf_report (HANYA jika user EXPLISIT minta PDF/dokumen),
  export_query_csv (jika user minta CSV/export tabel).
  DEFAULT: jawab di chat Telegram sebagai teks. DILARANG memanggil
  create_pdf_report kalau user tidak bilang "buat PDF" / "kirim PDF" /
  "generate PDF". Instruksi "jangan buat PDF" = preferensi permanen: patuhi.
- Project folder agent_idx: list_project_dir, read_project_file (baca),
  write_project_file (USULKAN perubahan file teks — di-stage, WAJIB dikonfirmasi
  user ya/tidak sebelum ditulis; sertakan konten lengkap file).
  (.env / cookies / browser profile / binary diblokir)
- Stockbit (browser + profile persistent, allowlist stockbit.com saja):
  stockbit_status, stockbit_open, stockbit_read, stockbit_scrape, stockbit_scrape_reports,
  stockbit_create_post (publish ide ke stream — di-stage, konfirmasi ya/tidak).
  Tanpa order/trading. Cek stockbit_status dulu jika ragu sesi login.
  Chart OHLCV: tool UTAMA get_stockbit_ohlcv / get_technicals
  (MA5, MA20, MA50, MA200 + RSI/MACD — data pasar = teknikal).
  Fundamental: get_fundamentals (hasil transform keystats/profile).
  Berita + sentimen: search_news_sentiment (positif/negatif/neutral).
  Setelah farm/scrape: run_data_transform(scope='technicals'|'fundamentals'|'news'|'all').
  Foreign flow: Chartbit TIDAK punya F Buy/F Sell (NULL di daily_stock).
  Pakai scrape overview → transform foreign_flow_daily, atau akui GAP_DATA.
  Cadangan: query_chart_ohlcv / chart_features / farm_stockbit_charts.
  Chart tersimpan di parquet + DuckDB (data/charts.duckdb).
  Tools lain: sync_chart_db, chart_db_stats, rebuild_market_from_stockbit.
  Pipeline sistem chart_analyze =
    farm_stockbit_charts → sync_chart_db → chart_features → query_chart_ohlcv
    (atau langsung get_stockbit_ohlcv) lalu ringkas dari output tools.
  Pipeline chart_query = chart_features → query_chart_ohlcv (tanpa farm).
  Jika QUERY PLAN menyebut pipeline/allowed_tools: ikuti itu; jangan panggil tool di luar daftar.
  DILARANG menjawab hanya dengan URL Chartbit / link preview.
  JANGAN tutup browser — sesi hilang jika ditutup/restart.
  Farming chart = farm_stockbit_charts (simpan parquet + DuckDB + indikator).
  Job >3 emiten di-stage (konfirmasi ya/tidak).
- Fundamentals/profile Stockbit: farm_stockbit_fundamentals
  (overview,keystats,financials,profile; company hanya alias input) — pace PELAN + jitter; batch kecil
  (limit/offset). Output exports/stockbit/ + data/fundamentals/.
  Universe dipecah dua: tier='top' (200 emiten terbesar) dan tier='rest' (sisanya).
  Jika user minta "semua emiten", tawarkan tier='top' dulu — 'all' makan ~26 jam.
- Jika user minta scrape Stockbit Reports / @Stockbit / Stream official: WAJIB
  stockbit_scrape_reports (pipeline reports_scrape). source=both untuk keduanya.
  Tool hanya men-stage; scrape dan write berjalan setelah konfirmasi ya.
- Jika tool mengembalikan NEED_STOCKBIT_CREDENTIALS: berhenti; bot minta login interaktif.
- Blok "--- ISI PDF ... ---" = dokumen user (boleh dianalisis).
- Simpan dokumen ke DB: HANYA jika user EXPLISIT minta simpan/catat/arsip.
  WAJIB stage_document_ingest (pipeline document_ingest) — TIDAK boleh tulis DB langsung.
  Setelah STAGED, user konfirmasi ya/tidak di Telegram. Baru commit ke DuckDB + knowledge/docs.
  Cari dokumen tersimpan: search_documents / doc_store_stats.
  DILARANG menyimpan PDF otomatis hanya karena dilampirkan.

PEMBELAJARAN / CATAT
- Jika user tanya sejauh mana / apa yang sudah kamu pelajari (meta, tentang BOT):
  WAJIB list_knowledge_topics — jangan search_knowledge teori pasar / berita MSCI.
- Jika user bilang "ingat bahwa" / "catat": WAJIB save_learning_note.
- Jika user bilang "catat", "selalu catat", "ingat bahwa", "ajarin":
  WAJIB panggil save_learning_note(title, body) dengan isi instruksi user,
  lalu balas singkat: sudah dicatat + ringkasan poinnya.
  JANGAN ganti topik. JANGAN minta user melakukan analisis.
- Jika user bilang "belajar sekarang" / "update digest": panggil run_learning_job.
- Pertanyaan teori: search_knowledge dulu.

ATURAN KEPUTUSAN
- Setiap pertanyaan punya QUERY PLAN (intent + sumber data). Ikuti plan itu.
- Angka pasar → SQL/tools parquet. Teori → search_knowledge. Berita → search_news.
- Stockbit Reports → search/read tools. Laporan gabungan sudah di-handle otomatis.
- Jangan minta PDF kecuali user benar-benar membahas lampiran PDF.

ATURAN DATA
- WAJIB tools untuk fakta angka/berita/knowledge. Jangan mengarang.
- date = INTEGER YYYYMMDD. net_foreign = foreign_buy - foreign_sell.
- Data transaksi BUKAN laporan keuangan.
- HARGA SEKARANG = close bar TERAKHIR + tanggalnya saja (dari tools / blok FAKTA HARGA).
  JANGAN pakai rentang multi-hari (mis. "855–875") sebagai harga terkini.
  Kalau ada blok FAKTA HARGA TERKINI di pesan: WAJIB pakai angka itu.

GAYA
- Bahasa Indonesia baku, profesional, netral, ringkas.
- Markdown sederhana saja: ## heading, - bullet, **tebal**, *miring*.
- DILARANG output HTML mentah (<b>, <i>, &amp;, dll).
- DILARANG tabel markdown (| col |). Untuk daftar emiten/proxy pakai bullet:
  - **BNBR** — PT Bakrie & Brothers | Direct Proxy | catalyst singkat
- DILARANG self-puji, upsell, dan menyuruh user.
- Setelah mencatat: 1–3 kalimat konfirmasi saja, tanpa ajakan lain.

KONTEKS PERCAKAPAN
- Jawab HANYA pertanyaan saat ini. Jangan lanjutkan analisis saham/symbol lama
  kecuali user bilang "tadi", "lanjut", "itu", atau menyebut symbol yang sama.
- Jika user tanya kemampuan bot / cron / jadwal / reset / fitur: jawab langsung
  soal bot, JANGAN tarik data pasar atau lanjut topik saham sebelumnya.
- Cron yang sudah ada (bukan dibuat dari chat): market digest,
  **Stockbit stream farm** (CRON_REPORTS → @StockbitReports + https://stockbit.com/Stockbit),
  **Chartbit farm** (CRON_FARM), news digest, weekly lesson — lihat /learn / .env.

STRUKTUR ANALISIS LENGKAP (jika diminta rekomendasi/fundamental)
1) Ringkasan
2) Kerangka fundamental (dari knowledge / PDF) — lewati jika tidak relevan
3) Data pasar & flow (tools)
4) Charting ringkas (jika relevan)
5) Risiko & invalidasi
6) Catatan / sumber
"""


@dataclass
class AgentResult:
    text: str
    files: list[Path] = field(default_factory=list)
    need_stockbit_login: bool = False
    resume_question: str = ""


def _parse_reports_date_range(text: str) -> dict[str, Any]:
    from calendar import monthrange

    low = text.lower()
    if re.search(
        r"(?:stockbit\.com/stockbit\b|@stockbit\b|"
        r"akun\s+(?:resmi\s+)?stockbit|official\s+stockbit|"
        r"foreign\s*net|net\s*foreign)",
        low,
    ) and not re.search(r"stockbitreports|/stockbitreports", low):
        if re.search(r"\bboth\b|\bkeduanya\b|\bsemua\b", low):
            args: dict[str, Any] = {"source": "both"}
        else:
            args = {"source": "official", "url": "https://stockbit.com/Stockbit"}
    elif re.search(r"\bboth\b|\bkeduanya\b", low) and re.search(
        r"stockbit|report", low
    ):
        args = {"source": "both"}
    else:
        args = {"url": "https://stockbit.com/StockbitReports?source=0"}
    iso_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if len(iso_dates) >= 2:
        args["date_from"] = iso_dates[0]
        args["date_to"] = iso_dates[1]
        return args
    if len(iso_dates) == 1:
        args["date_from"] = iso_dates[0]
        return args

    months: list[tuple[int, int]] = []
    for match in re.finditer(
        r"\b("
        r"januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|"
        r"november|desember|jan|feb|mar|apr|jun|jul|agu|aug|sep|okt|oct|nov|des|dec"
        r")\s+(20\d{2})\b",
        text,
        re.I,
    ):
        month = _MONTH_ID[match.group(1).lower()]
        year = int(match.group(2))
        months.append((year, month))

    if len(months) >= 2:
        months.sort()
        y1, m1 = months[0]
        y2, m2 = months[-1]
        args["date_from"] = f"{y1:04d}-{m1:02d}-01"
        args["date_to"] = f"{y2:04d}-{m2:02d}-{monthrange(y2, m2)[1]:02d}"
        return args
    if len(months) == 1:
        y, m = months[0]
        args["date_from"] = f"{y:04d}-{m:02d}-01"
        args["date_to"] = f"{y:04d}-{m:02d}-{monthrange(y, m)[1]:02d}"
        return args

    days_match = re.search(r"(\d+)\s*hari", text, re.I)
    args["days"] = int(days_match.group(1)) if days_match else 10
    return args


class AnalystAgent:
    def __init__(
        self,
        settings: Settings,
        store: StockDataStore,
        knowledge: KnowledgeBase | None = None,
        chat_log: ChatLog | None = None,
        stockbit_store: StockbitReportsStore | None = None,
        chat_id: int | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.knowledge = knowledge or KnowledgeBase(settings.knowledge_dir)
        self.chat_log = chat_log
        self.stockbit_store = stockbit_store
        self.chat_id = chat_id if chat_id is not None else settings.telegram_chat_id
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self._coverage = store.list_date_range()
        self.export_dir = settings.export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _system_prompt(self) -> str:
        chat_stats = ""
        if self.chat_log is not None:
            chat_stats = f"\nCHAT LOG: {self.chat_log.stats()}"
            if self.chat_id is not None:
                chat_stats += f"\nINTERACTION CHAT_ID: {self.chat_id}"
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"CAKUPAN DATA TERSEDIA:\n{self._coverage}\n"
            f"MARKET_SOURCE: Stockbit Chartbit (bukan Yahoo) — "
            f"rebuild_market_from_stockbit / farm_stockbit_charts\n"
            f"FOLDER EXPORT: {self.export_dir}\n"
            f"PROJECT ROOT (list/read tools): {Path(__file__).resolve().parent.parent}\n"
            f"STOCKBIT VECTOR: {self.settings.chroma_dir} (ChromaDB)\n"
            f"KNOWLEDGE DIR: {self.knowledge.root}"
            f"{chat_stats}"
        )

    def _try_force_reports_scrape(self, user_message: str) -> AgentResult | None:
        if not _SCRAPE_VERB.search(user_message):
            return None

        page_url = extract_stockbit_url(user_message)
        has_reports = bool(_REPORTS_URL.search(user_message))
        if not page_url and not has_reports:
            return None

        from agent_idx.pipelines import PipelineContext, run_pipeline

        args = _parse_reports_date_range(user_message)
        if page_url:
            args["url"] = page_url
        ctx = PipelineContext(
            user_message=user_message,
            intent=Intent.STOCKBIT_SCRAPE.value,
            url=args.get("url"),
            days=args.get("days"),
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
        )

        def _dispatch(name: str, tool_args: dict[str, Any]) -> str:
            return dispatch_tool(
                self.store,
                name,
                tool_args,
                export_dir=self.export_dir,
                knowledge=self.knowledge,
                chat_log=self.chat_log,
                chat_id=self.chat_id,
                stockbit_store=self.stockbit_store,
            )

        logger.info("Pipeline reports_scrape args=%s", args)
        pipe = run_pipeline(
            "reports_scrape",
            ctx,
            dispatch=_dispatch,
            extract_files=extract_files,
        )
        files = pipe.all_files()
        text = pipe.combined_text()
        if pipe.need_stockbit_login:
            return AgentResult(
                text=text.split("\n", 1)[0] if text else "NEED_STOCKBIT_CREDENTIALS",
                files=files,
                need_stockbit_login=True,
                resume_question=user_message,
            )
        return AgentResult(text=text, files=files)

    def _try_read_scrape_by_date(self, user_message: str) -> AgentResult | None:
        if not is_stockbit_scrape_query(user_message):
            return None
        target = parse_user_date(user_message)
        if not target:
            return None
        # Rentang multi-bulan = scrape baru, bukan baca file lokal.
        month_hits = re.findall(
            r"\b("
            r"januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|"
            r"november|desember"
            r")\s+(20\d{2})\b",
            user_message,
            re.I,
        )
        if len(month_hits) >= 2 and _SCRAPE_VERB.search(user_message):
            return None

        logger.info("Force read_stockbit_scrape_date date=%s", target)
        result = dispatch_tool(
            self.store,
            "read_stockbit_scrape_date",
            {"date": target},
            export_dir=self.export_dir,
            knowledge=self.knowledge,
            chat_log=self.chat_log,
            chat_id=self.chat_id,
            stockbit_store=self.stockbit_store,
        )
        return AgentResult(text=result, files=extract_files(result))

    def _try_force_open_chart(self, user_message: str) -> AgentResult | None:
        if _SCRAPE_VERB.search(user_message) and not _CHART_QUERY.search(user_message):
            return None

        from agent_idx import chart_farm

        args: dict[str, Any] | None = None
        symbol: str | None = None
        url = extract_stockbit_url(user_message)
        if url and "chartbit" in url.lower():
            args = {"url": url}
            m = re.search(r"/symbol/([A-Za-z]{3,5})/", url, re.I)
            if m:
                symbol = m.group(1).upper()
        elif _CHART_QUERY.search(user_message) or re.search(
            r"\btimeframe\b|\b\d+\s*menit\b|\b\d+m\b|\b1[hdw]\b",
            user_message,
            re.I,
        ):
            skip = {
                "CHART",
                "GRAFIK",
                "SAHAM",
                "STOCK",
                "BUKA",
                "OPEN",
                "CEK",
                "HARI",
                "INI",
                "DATA",
                "YANG",
                "MENIT",
                "ANALISA",
                "ANALISIS",
                "STOCKBIT",
                "TIMEFRAME",
                "UNTUK",
                "TERAKHIR",
                "TERBARU",
            }
            sym_match = _SYMBOL_CHART_URL.search(user_message)
            if not sym_match:
                sym_match = re.search(
                    r"(?:chart|chartbit|grafik)\s+(?:saham\s+)?([A-Za-z]{3,5})\b",
                    user_message,
                    re.I,
                )
            if not sym_match:
                # "chart bbca" / "bbca timeframe 5 menit"
                sym_match = re.search(
                    r"\b([A-Za-z]{3,5})\b(?:\s+\S+){0,8}\b(?:chart|chartbit|grafik|timeframe|\d+\s*menit|\d+m)\b",
                    user_message,
                    re.I,
                )
            if not sym_match:
                sym_match = re.search(
                    r"\b(?:chart|chartbit|grafik|timeframe)\b(?:\s+\S+){0,6}\b([A-Za-z]{3,5})\b",
                    user_message,
                    re.I,
                )
            if sym_match:
                sym = sym_match.group(1).upper()
                if sym not in skip:
                    symbol = sym
                    args = {"symbol": sym, "page": "chart"}
                else:
                    # Retry: "chart bbca" style if first token was OPEN/BUKA/CEK
                    sym_match2 = re.search(
                        r"(?:chart|chartbit|grafik)\s+(?:saham\s+)?([A-Za-z]{3,5})\b",
                        user_message,
                        re.I,
                    )
                    if sym_match2:
                        sym2 = sym_match2.group(1).upper()
                        if sym2 not in skip:
                            symbol = sym2
                            args = {"symbol": sym2, "page": "chart"}

            # Fallback: "Cek DSSA timeframe 1H" — CEK skipped, ambil symbol dari extractor
            if symbol is None:
                from agent_idx.decision import extract_symbols as _plan_symbols

                for cand in _plan_symbols(user_message):
                    if cand not in skip:
                        symbol = cand
                        args = {"symbol": cand, "page": "chart"}
                        break

        if args is None and symbol is None:
            return None

        has_tf = bool(
            re.search(
                r"\b(timeframe|\d+\s*menit|\d+\s*min|\d+m\b|1[hdw]|5m|15m|30m|4h)\b",
                user_message,
                re.I,
            )
        )
        want_analysis = bool(
            re.search(r"\b(analisa|analisis|analyze|bacakan|jelaskan)\b", user_message, re.I)
        )
        if (
            url is None
            and not _OPEN_VERB.search(user_message)
            and not has_tf
            and not want_analysis
            and not re.search(r"\b(hari\s+ini|terbaru|terakhir)\b", user_message, re.I)
        ):
            return None

        timeframe = chart_farm.parse_timeframe(user_message, default="1H")
        # Always pull OHLCV when TF / analisa / "open chart X" — jangan cuma dump URL.
        want_ohlcv = bool(symbol) and (
            has_tf
            or want_analysis
            or _OPEN_VERB.search(user_message) is not None
            or re.search(
                r"\b(hari\s+ini|intraday|terbaru|terakhir|ohlc|candle|duckdb|database)\b",
                user_message,
                re.I,
            )
            is not None
        )

        files: list[Path] = []

        if want_ohlcv and symbol:
            from agent_idx.pipelines import PipelineContext, run_pipeline

            plan = classify_intent(user_message)
            pipeline_id = plan.pipeline or "chart_analyze"
            if re.search(
                r"\b(duckdb|dari\s+database|pakai\s+data\s+lokal|tanpa\s+farm)\b",
                user_message,
                re.I,
            ):
                pipeline_id = "chart_query"

            tfs = [timeframe]
            for extra in ("1H", "1D"):
                if extra != timeframe:
                    tfs.append(extra)

            ctx = PipelineContext(
                user_message=user_message,
                intent=Intent.STOCKBIT_CHART.value,
                symbol=symbol,
                symbols=[symbol],
                timeframe=timeframe,
                timeframes=tfs,
            )

            def _dispatch(name: str, tool_args: dict[str, Any]) -> str:
                return dispatch_tool(
                    self.store,
                    name,
                    tool_args,
                    export_dir=self.export_dir,
                    knowledge=self.knowledge,
                    chat_log=self.chat_log,
                    chat_id=self.chat_id,
                    stockbit_store=self.stockbit_store,
                )

            logger.info(
                "Pipeline chart id=%s symbol=%s timeframe=%s",
                pipeline_id,
                symbol,
                timeframe,
            )
            pipe = run_pipeline(
                pipeline_id,
                ctx,
                dispatch=_dispatch,
                extract_files=extract_files,
            )
            files.extend(pipe.all_files())
            evidence = pipe.combined_text()
            url_line = f"https://stockbit.com/symbol/{symbol}/chartbit"

            if pipe.need_stockbit_login:
                return AgentResult(
                    text=(
                        f"Pipeline `{pipeline_id}` butuh login Stockbit.\n"
                        "Kirim /stockbit, OTP jika diminta, lalu ulangi.\n\n"
                        f"{evidence}"
                    ),
                    files=files,
                    need_stockbit_login=True,
                    resume_question=user_message,
                )
            if pipe.status in {"failed", "stopped"} and "CHARTBIT_BLANK" in evidence:
                return AgentResult(
                    text=(
                        f"Chartbit {symbol} kosong meski login.\n"
                        "Coba STOCKBIT_HEADLESS=false + /stockbit ulang.\n\n"
                        f"{evidence}"
                    ),
                    files=files,
                )
            if pipe.status == "failed":
                return AgentResult(
                    text=(
                        f"Pipeline chart gagal untuk {symbol} TF {timeframe}.\n"
                        f"{evidence}\nURL: {url_line}"
                    ),
                    files=files,
                )

            local = chart_farm.quick_chart_analysis(symbol, timeframe)
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Kamu analis teknikal IDX. Jawab bahasa Indonesia, "
                                "comprehensive tapi terstruktur. "
                                "Hanya pakai evidence dari pipeline tools (DuckDB/Chartbit). "
                                "Sebut pipeline tools yang dipakai, timeframe, last price, "
                                "arah, S/R kasar, RSI/MACD jika ada, multi-TF bila ada, "
                                "dan risiko. Jangan mengarang angka di luar evidence."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Permintaan: {user_message}\n"
                                f"Symbol: {symbol}\nTimeframe utama: {timeframe}\n"
                                f"pipeline={pipeline_id} run={pipe.run_id}\n"
                                f"Chartbit: {url_line}\n\n"
                                f"EVIDENCE TOOLS:\n{evidence}\n\n"
                                f"Ringkasan lokal:\n{local}"
                            ),
                        },
                    ],
                    tool_choice="none",
                    timeout=45.0,
                )
                analysis = (response.choices[0].message.content or "").strip()
                if not analysis:
                    analysis = local
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Chart analysis LLM failed/timeout: %s — using local summary", exc
                )
                analysis = f"{local}\n\n(Catatan: analisa LLM timeout/gagal: {exc})"

            text = (
                f"{analysis}\n\n"
                f"---\npipeline={pipeline_id} run={pipe.run_id} status={pipe.status}\n"
                f"URL: {url_line}"
            )
            return AgentResult(text=text, files=files)

        if args is None:
            return None
        logger.info("Force stockbit_open chart args=%s", args)
        result = dispatch_tool(
            self.store,
            "stockbit_open",
            args,
            export_dir=self.export_dir,
            knowledge=self.knowledge,
            chat_log=self.chat_log,
            chat_id=self.chat_id,
            stockbit_store=self.stockbit_store,
        )
        files.extend(extract_files(result))
        if "NEED_STOCKBIT_CREDENTIALS" in result or "NEED_STOCKBIT_OTP" in result:
            return AgentResult(
                text=result.split("\n", 1)[0],
                files=files,
                need_stockbit_login=True,
                resume_question=user_message,
            )
        return AgentResult(text=result, files=files)

    def _try_force_composite_report(
        self, user_message: str, plan: QueryPlan | None = None
    ) -> AgentResult | None:
        if plan is None or plan.intent != Intent.COMPOSITE_REPORT:
            return None

        logger.info("Force composite report: %s", user_message[:120])
        sections = gather_report_context(
            user_message=user_message,
            store=self.store,
            knowledge=self.knowledge,
            stockbit_store=self.stockbit_store,
            chat_log=self.chat_log,
            plan=plan,
        )
        context = format_report_context(sections)

        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": composite_report_system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"Permintaan user: {user_message}\n\n"
                        f"Data agregat:\n{context}"
                    ),
                },
            ],
            tool_choice="none",
        )
        text = (response.choices[0].message.content or "").strip() or context
        return AgentResult(text=text)

    def _try_force_learning_status(self, user_message: str) -> AgentResult | None:
        if not is_meta_learning_query(user_message):
            return None

        logger.info("Force learning status query")
        topics = dispatch_tool(
            self.store,
            "list_knowledge_topics",
            {},
            export_dir=self.export_dir,
            knowledge=self.knowledge,
            chat_log=self.chat_log,
            chat_id=self.chat_id,
            stockbit_store=self.stockbit_store,
        )
        stats = self.knowledge.stats()
        coverage = self._coverage

        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Kamu bot analis IDX. User menanyakan PROGRES BELAJAR / knowledge base "
                        "BOT (bukan analisis pasar). Jawab singkat, bahasa Indonesia (boleh juga "
                        "jika user tanya English). Sebut jumlah materi per kategori, cron belajar "
                        "(/learn), dan cakupan data pasar. Jangan mengarang analisis MSCI/regulasi."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Pertanyaan: {user_message}\n\n"
                        f"Stats: {stats}\n"
                        f"Data pasar: {coverage}\n\n"
                        f"Inventaris knowledge:\n{topics}"
                    ),
                },
            ],
            tool_choice="none",
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            text = f"{stats}\n\n{topics}"
        return AgentResult(text=text)

    def _try_force_document_ingest(self, user_message: str) -> AgentResult | None:
        """Stage document save — never writes DB until Telegram ya/tidak."""
        from agent_idx import doc_ingest
        from agent_idx.pipelines import PipelineContext, run_pipeline

        if not doc_ingest.is_document_save_request(user_message):
            return None
        parsed = doc_ingest.extract_pdf_block(user_message)
        if not parsed:
            # Allow plain-text body after "simpan dokumen:" style
            m = re.search(
                r"(?:simpan|catat|arsipkan)\s+(?:dokumen|pdf|file)?\s*[:\-]\s*(.+)$",
                user_message,
                re.I | re.S,
            )
            if not m or len(m.group(1).strip()) < 40:
                return AgentResult(
                    text=(
                        "Siap simpan dokumen ke DB, tetapi isi belum ada.\n"
                        "Kirim PDF dengan caption mis. "
                        "\"simpan dokumen ini ke knowledge\" "
                        "atau tempel teks cukup panjang."
                    )
                )
            source_name = "pasted_text"
            body = m.group(1).strip()
            title = "dokumen_user"
        else:
            source_name, body = parsed
            title = re.sub(r"\.[Pp][Dd][Ff]$", "", source_name) or "dokumen_user"

        cat = "user_docs"
        low = user_message.lower()
        if "curriculum" in low or "kurikulum" in low:
            cat = "curriculum"
        elif "research" in low or "riset" in low:
            cat = "research"
        elif "report" in low or "laporan" in low:
            cat = "reports"
        elif "note" in low or "catatan" in low:
            cat = "notes"

        ctx = PipelineContext(
            user_message=user_message,
            intent=Intent.DOCUMENT_INGEST.value,
            extra={
                "title": title[:80],
                "body": body,
                "category": cat,
                "source_name": source_name,
            },
        )

        def _dispatch(name: str, tool_args: dict[str, Any]) -> str:
            return dispatch_tool(
                self.store,
                name,
                tool_args,
                export_dir=self.export_dir,
                knowledge=self.knowledge,
                chat_log=self.chat_log,
                chat_id=self.chat_id,
                stockbit_store=self.stockbit_store,
            )

        logger.info(
            "Pipeline document_ingest title=%r chars=%s cat=%s",
            title,
            len(body),
            cat,
        )
        pipe = run_pipeline(
            "document_ingest",
            ctx,
            dispatch=_dispatch,
            extract_files=extract_files,
        )
        return AgentResult(text=pipe.combined_text(), files=pipe.all_files())

    def _execute_plan(self, plan: QueryPlan, user_message: str) -> AgentResult | None:
        if plan.intent == Intent.DOCUMENT_INGEST:
            return self._try_force_document_ingest(user_message)
        if plan.intent == Intent.STOCKBIT_READ_DATE:
            return self._try_read_scrape_by_date(user_message)
        if plan.intent == Intent.LEARNING_META:
            return self._try_force_learning_status(user_message)
        if plan.intent == Intent.BOT_META:
            return self._try_force_bot_meta(user_message)
        if plan.intent == Intent.STOCKBIT_SCRAPE:
            return self._try_force_reports_scrape(user_message)
        if plan.intent == Intent.STOCKBIT_CHART:
            return self._try_force_open_chart(user_message)
        if plan.intent == Intent.STOCKBIT_POST:
            return self._try_force_create_post(user_message)
        if plan.intent == Intent.COMPOSITE_REPORT:
            return self._try_force_composite_report(user_message, plan)
        if plan.intent == Intent.BACKTEST_SCAN:
            return self._try_force_backtest(user_message)
        return None

    def _try_force_create_post(self, user_message: str) -> AgentResult:
        """Stage a Stockbit stream post (Telegram ya/tidak before publish)."""
        text = _extract_post_text(user_message)
        if not text:
            return AgentResult(
                text=(
                    "Siap post ke Stockbit. Kirim ulang dengan isi postingan, "
                    "contoh: posting di stockbit 'ide singkat di sini'"
                )
            )
        logger.info("Force stockbit_create_post text=%r", text[:80])
        result = dispatch_tool(
            self.store,
            "stockbit_create_post",
            {"text": text},
            export_dir=self.export_dir,
            knowledge=self.knowledge,
            chat_log=self.chat_log,
            chat_id=self.chat_id,
            stockbit_store=self.stockbit_store,
        )
        return AgentResult(text=result)

    def _try_force_bot_meta(self, user_message: str) -> AgentResult:
        """Direct answer for cron / capability / ops — never inherit stock context."""
        from agent_idx.context import is_meta_bot_query

        assert is_meta_bot_query(user_message) or True
        s = self.settings
        q = (user_message or "").lower()
        lines = [
            "Ini jawaban tentang **bot / operasional**, bukan analisis saham.",
            "",
        ]
        if re.search(r"cron|jadwal|scheduler", q):
            lines += [
                "**Cron yang sudah ada** (APScheduler di dalam bot):",
                f"- enabled: `{s.cron_enabled}` (timezone `{s.cron_timezone}`)",
                f"- market digest: `{s.cron_market}` (Sen–Jum)",
                f"- Stockbit stream farm: `{s.cron_reports}` days=`{s.cron_reports_days}` "
                "(@StockbitReports + https://stockbit.com/Stockbit)",
                f"- Chartbit farm: `{s.cron_farm}` tf=`{s.cron_farm_timeframes}` (Sen–Jum) "
                "→ `https://stockbit.com/symbol/{TICKER}/chartbit`",
                f"- Fundamentals farm: `{s.cron_fundamentals}` (Jumat 07:00 WIB; "
                "semua IDX; financials, keystats, profile)",
                f"- news digest: `{s.cron_news}` (Sen–Jum)",
                f"- weekly lesson: `{s.cron_weekly}` (Minggu)",
                "",
                "Jadwal diubah lewat `.env` (`CRON_*`) lalu restart bot.",
                "Farming Stockbit Reports + Chartbit weekday sudah aktif.",
                "Cek status cepat: `/learn`",
            ]
        elif re.search(r"reset|clear\s+history", q):
            lines += [
                "Untuk menghapus riwayat percakapan di group ini: ketik `/reset`.",
                "Bot juga otomatis membuang history kalau topik/symbol berganti atau pertanyaan meta.",
            ]
        elif re.search(r"pdf", q):
            lines += [
                "Default: jawaban **teks di chat**.",
                "PDF hanya dibuat jika kamu minta eksplisit (\"buat PDF\" / \"kirim PDF\").",
            ]
        else:
            lines += [
                "**Yang bisa saya lakukan:**",
                "- Analisis data pasar parquet (harga, volume, foreign flow, SQL, backtest)",
                "- Berita, knowledge base, chat group",
                "- Stockbit: login interaktif, scrape reports, buka/baca Chartbit "
                "(`.../symbol/{T}/chartbit`), farming OHLCV, buat post (konfirmasi)",
                "- Baca PDF/foto, list/baca file project (tulis file perlu konfirmasi ya/tidak)",
                "- Council 3-agent: `/council`",
                "",
                "**Cron otomatis:** market / Stockbit Reports / Chartbit farm / news / weekly "
                "(lihat `/learn`).",
                "**Batasan:** PDF tidak otomatis; tidak ada order/trading Stockbit.",
            ]
        return AgentResult(text="\n".join(lines))

    def _try_force_backtest(self, user_message: str) -> AgentResult | None:
        min_win = 0.75
        max_win = 0.85
        band = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})\s*%?", user_message)
        if band:
            min_win = min(int(band.group(1)), 100) / 100
            max_win = min(int(band.group(2)), 100) / 100
        else:
            m = re.search(r"win\s*rate\s*(?:>|>=)?\s*(\d{2,3})", user_message, re.I)
            if m:
                min_win = min(int(m.group(1)), 100) / 100
                max_win = min(min_win + 0.10, 1.0)
        min_trades = 15
        tm = re.search(r"min(?:imum)?\s*(\d+)\s*trade", user_message, re.I)
        if tm:
            min_trades = int(tm.group(1))

        logger.info(
            "Force backtest scan min_win=%s max_win=%s min_trades=%s",
            min_win, max_win, min_trades,
        )
        result = dispatch_tool(
            self.store,
            "run_backtest_scan",
            {
                "start_date": 20220101,
                "min_win_rate": min_win,
                "max_win_rate": max_win,
                "min_trades": min_trades,
                "min_market_cap": 1_000_000_000_000,
            },
            export_dir=self.export_dir,
            knowledge=self.knowledge,
            chat_log=self.chat_log,
            chat_id=self.chat_id,
            stockbit_store=self.stockbit_store,
        )
        return AgentResult(text=result, files=extract_files(result))

    def ask(
        self, user_message: str, history: list[dict[str, Any]] | None = None
    ) -> AgentResult:
        plan = classify_intent(user_message)
        logger.info("Decision plan:\n%s", plan.to_prompt())

        routed = self._execute_plan(plan, user_message)
        if routed is not None:
            return routed

        price_facts = self._prefetch_price_facts(user_message, history, plan)
        system_body = (
            f"{self._system_prompt()}\n\n"
            f"QUERY PLAN (ikuti sumber & intent ini):\n{plan.to_prompt()}"
        )
        if price_facts:
            system_body += (
                "\n\nFAKTA HARGA TERKINI (wajib dipakai; jangan mengarang "
                "rentang multi-hari sebagai harga sekarang):\n"
                f"{price_facts}"
            )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_body,
            }
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        files: list[Path] = []
        used_tool = False
        needs_tools = self._needs_tools(user_message, history)
        # No user-facing step limit: run until the model answers.
        # Silent safety cap only (prevents rare infinite tool loops).
        safety_cap = 100
        step = 0

        # System may restrict which tools the LLM can call for this intent.
        active_tools = TOOL_DEFINITIONS
        if plan.allowed_tools:
            allowed = set(plan.allowed_tools)
            active_tools = [
                t
                for t in TOOL_DEFINITIONS
                if (t.get("function") or {}).get("name") in allowed
            ]
            if not active_tools:
                active_tools = TOOL_DEFINITIONS

        while True:
            step += 1
            logger.info("Agent step %s", step)
            force_answer = step >= safety_cap
            if force_answer:
                tool_choice: Any = "none"
            elif step == 1 and needs_tools:
                tool_choice = "required"
            else:
                tool_choice = "auto"

            create_kwargs: dict[str, Any] = {
                "model": self.settings.llm_model,
                "messages": messages,
                "tool_choice": tool_choice,
            }
            if not force_answer:
                create_kwargs["tools"] = active_tools

            response = self.client.chat.completions.create(**create_kwargs)
            message = response.choices[0].message
            tool_calls = [] if force_answer else (message.tool_calls or [])

            assistant_payload: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if tool_calls:
                assistant_payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_payload)

            if not tool_calls:
                content = (message.content or "").strip()
                if needs_tools and not used_tool and not force_answer:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Panggil tools yang relevan dulu "
                                "(data / search_news / create_pdf_report / export_query_csv)."
                            ),
                        }
                    )
                    needs_tools = True
                    continue
                return AgentResult(
                    text=content or "(model tidak mengembalikan jawaban)",
                    files=files,
                )

            used_tool = True
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    result = f"ERROR: invalid JSON arguments: {tc.function.arguments!r}"
                else:
                    logger.info("Tool %s args=%s", name, args)
                    result = dispatch_tool(
                        self.store,
                        name,
                        args,
                        export_dir=self.export_dir,
                        knowledge=self.knowledge,
                        chat_log=self.chat_log,
                        chat_id=self.chat_id,
                        stockbit_store=self.stockbit_store,
                    )
                    files.extend(extract_files(result))
                    if "NEED_STOCKBIT_CREDENTIALS" in result or "NEED_STOCKBIT_OTP" in result:
                        hint = (
                            "Stockbit meminta kode OTP."
                            if "NEED_STOCKBIT_OTP" in result
                            else "Stockbit meminta login."
                        )
                        return AgentResult(
                            text=(
                                f"{hint} "
                                "User perlu /stockbit untuk login interaktif, "
                                "lalu ulangi pertanyaan."
                            ),
                            files=files,
                            need_stockbit_login=True,
                            resume_question=user_message,
                        )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

    def _prefetch_price_facts(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None,
        plan: QueryPlan,
    ) -> str:
        """Inject last Stockbit close so advice can't invent mid-range prices."""
        from agent_idx.context import extract_symbols

        syms: list[str] = []
        for t in plan.symbols or []:
            u = str(t).strip().upper()
            if u and u not in syms:
                syms.append(u)
        for t in sorted(extract_symbols(user_message)):
            if t not in syms:
                syms.append(t)
        if history and not syms:
            blob = "\n".join(
                str(m.get("content") or "")
                for m in history[-8:]
                if isinstance(m, dict)
            )
            for t in sorted(extract_symbols(blob)):
                if t not in syms:
                    syms.append(t)
        if not syms:
            return ""
        # Price advice / position questions — or any market plan with symbols.
        q = (user_message or "").lower()
        wants_price = bool(
            re.search(
                r"posisi|advice|entry|harga|close|pantau|analisis|rekomendasi|"
                r"stop\s*loss|cut\s*loss|hold|average",
                q,
                re.I,
            )
            or plan.intent
            in {
                Intent.MARKET_ANALYSIS,
                Intent.STOCKBIT_CHART,
                Intent.COMPOSITE_REPORT,
            }
        )
        if not wants_price:
            return ""

        lines: list[str] = []
        for sym in syms[:5]:
            try:
                sql = (
                    "SELECT date, close, high, low, open_price, volume "
                    f"FROM daily_stock WHERE symbol = '{sym}' "
                    "ORDER BY date DESC LIMIT 1"
                )
                raw = self.store.run_sql(sql)
            except Exception as exc:  # noqa: BLE001
                lines.append(f"- {sym}: ERROR query ({exc})")
                continue
            # Parse simple table row if present
            m = re.search(r"\b(20\d{6})\s+([\d.]+)", raw)
            if m:
                lines.append(
                    f"- {sym}: last_close={m.group(2)} date={m.group(1)} "
                    "source=stockbit_daily"
                )
            else:
                # Fallback: keep a short snippet
                snippet = " ".join(raw.split())[:180]
                lines.append(f"- {sym}: {snippet}")
        if not lines:
            return ""
        return "\n".join(lines)

    @staticmethod
    def _needs_tools(
        user_message: str, history: list[dict[str, Any]] | None
    ) -> bool:
        text = user_message.strip()
        if not text:
            return False
        # PDF content already injected — answer from context, no forced tools.
        if "--- ISI PDF" in text:
            return False
        if is_meta_learning_query(text):
            return False
        from agent_idx.context import is_meta_bot_query

        if is_meta_bot_query(text):
            return False
        # Greetings / short chat.
        if re.fullmatch(r"(halo|hai|hi|hello|pagi|siang|sore|malam)[.! ]*", text, re.I):
            return False
        if history and len(text) < 12 and not _TOOL_HINT.search(text):
            return False
        if _TOOL_HINT.search(text):
            return True
        if history and re.search(
            r"\b(itu|tadi|lanjut|buatkan|kirim|export|csv|berita|buat\s+pdf)\b",
            text,
            re.I,
        ):
            return True
        return len(text) > 40
