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
    r"stockbitreports|stockbit\s+reports|/StockbitReports",
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
    r"chat|group|grup|siapa\s+bilang|diskusi|stockbit|scrape|reports|stockbitreports"
    r")\b|"
    r"\b20\d{2}\b|"
    r"\b[A-Z]{3,5}\b",
    re.I,
)

SYSTEM_PROMPT = """\
Kamu adalah analis & asisten pasar saham IDX.
User yang memerintah; kamu yang mengerjakan. JANGAN pernah menyuruh user mencari data.

PERAN
- Kamu eksekutor: tools, analisis, catat knowledge, buat PDF/CSV.
- DILARANG membalas dengan tugas untuk user (contoh: "tolong carikan data ADRO...").
- DILARANG meniru gaya user yang sedang memberi instruksi kepadamu.

KEMAMPUAN TOOLS
- Data pasar: list_date_range, describe_schema, get_stock_history, summarize_period,
  rank_foreign_flow, run_sql (SELECT only), run_backtest_scan
- Knowledge: search_knowledge, list_knowledge_topics, save_learning_note, run_learning_job
- Chat group: list_collected_chats, search_chat_history, recent_chat_history, chat_log_stats
- Berita: search_news
- Laporan gabungan: jika user minta report/laporan/ringkas/ytd (tanpa PDF lampiran),
  gabungkan data pasar + berita + Stockbit Reports + knowledge — jangan minta PDF user.
- File: create_pdf_report, export_query_csv
- Stockbit (browser ephemeral/guest-like, allowlist stockbit.com saja):
  stockbit_status, stockbit_open, stockbit_read, stockbit_scrape, stockbit_scrape_reports
  (read-only, tanpa order). Cek stockbit_status dulu jika ragu sesi login.
  Chart: stockbit_open(symbol=TICKER, page=chart) → .../symbol/TICKER/chartbit
  JANGAN tutup browser — sesi guest hilang jika ditutup/restart.
- Jika user minta scrape/ambil data Stockbit untuk ticker: WAJIB stockbit_scrape(symbol).
- Jika user minta scrape Stockbit Reports / stream @StockbitReports N hari:
  WAJIB stockbit_scrape_reports(days=N).
- Jika user minta rentang tanggal (mis. Jan–Jul 2026): WAJIB stockbit_scrape_reports
  dengan date_from/date_to (YYYY-MM-DD). JANGAN hanya stockbit_open.
- Jika user tanya report/laporan scrape tanggal tertentu: WAJIB read_stockbit_scrape_date(date).
  Data tersimpan di ChromaDB (CHROMA_DIR), bukan PDF Telegram.
- Pipeline: stockbit_scrape_reports -> markdown backup + ingest Chroma -> baca via read/search tools.
- DILARANG mengarang error teknis (thread switching, dll). Jika tool gagal, sampaikan
  output tool apa adanya. JANGAN redirect ke data parquet sebagai pengganti scrape.
- Jika pertanyaan tentang halaman Stockbit / "setelah login":
  WAJIB stockbit_open / stockbit_read / stockbit_scrape.
  JANGAN pakai rank_foreign_flow / run_sql / get_stock_history sebagai pengganti Stockbit.
- Jika tool mengembalikan NEED_STOCKBIT_CREDENTIALS: berhenti dan bilang perlu login interaktif
  (bot akan minta user/pass di chat). Jangan mengarang isi Stockbit.
- Blok "--- ISI PDF ... ---" = dokumen user.

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

GAYA
- Bahasa Indonesia baku, profesional, netral, ringkas.
- Markdown: ## ### - **teks**
- DILARANG self-puji, upsell, dan menyuruh user.
- Setelah mencatat: 1–3 kalimat konfirmasi saja, tanpa ajakan lain.

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

    args: dict[str, Any] = {"url": "https://stockbit.com/StockbitReports?source=0"}
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
            f"FOLDER EXPORT: {self.export_dir}\n"
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

        args = _parse_reports_date_range(user_message)
        if page_url:
            args["url"] = page_url
        logger.info("Force stockbit_scrape_reports args=%s", args)
        result = dispatch_tool(
            self.store,
            "stockbit_scrape_reports",
            args,
            export_dir=self.export_dir,
            knowledge=self.knowledge,
            chat_log=self.chat_log,
            chat_id=self.chat_id,
            stockbit_store=self.stockbit_store,
        )
        files = extract_files(result)
        if "NEED_STOCKBIT_CREDENTIALS" in result or "NEED_STOCKBIT_OTP" in result:
            return AgentResult(
                text=result.split("\n", 1)[0],
                files=files,
                need_stockbit_login=True,
                resume_question=user_message,
            )
        return AgentResult(text=result, files=files)

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
        if _SCRAPE_VERB.search(user_message):
            return None

        args: dict[str, Any] | None = None
        url = extract_stockbit_url(user_message)
        if url and "chartbit" in url.lower():
            args = {"url": url}
        elif _CHART_QUERY.search(user_message):
            sym_match = _SYMBOL_CHART_URL.search(user_message)
            if not sym_match:
                sym_match = re.search(
                    r"(?:chart|chartbit|grafik)\s+(?:saham\s+)?([A-Za-z]{3,5})\b",
                    user_message,
                    re.I,
                )
            if not sym_match:
                sym_match = re.search(
                    r"\bsaham\s+([A-Za-z]{3,5})\b.*\b(?:chart|chartbit|grafik)\b",
                    user_message,
                    re.I,
                )
            if sym_match:
                sym = sym_match.group(1).upper()
                if sym not in {"CHART", "GRAFIK", "SAHAM", "STOCK", "BUKA", "OPEN"}:
                    args = {"symbol": sym, "page": "chart"}

        if args is None:
            return None
        if url is None and not _OPEN_VERB.search(user_message):
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
        files = extract_files(result)
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

    def _execute_plan(self, plan: QueryPlan, user_message: str) -> AgentResult | None:
        if plan.intent == Intent.STOCKBIT_READ_DATE:
            return self._try_read_scrape_by_date(user_message)
        if plan.intent == Intent.LEARNING_META:
            return self._try_force_learning_status(user_message)
        if plan.intent == Intent.STOCKBIT_SCRAPE:
            return self._try_force_reports_scrape(user_message)
        if plan.intent == Intent.STOCKBIT_CHART:
            return self._try_force_open_chart(user_message)
        if plan.intent == Intent.COMPOSITE_REPORT:
            return self._try_force_composite_report(user_message, plan)
        if plan.intent == Intent.BACKTEST_SCAN:
            return self._try_force_backtest(user_message)
        return None

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

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"{self._system_prompt()}\n\n"
                    f"QUERY PLAN (ikuti sumber & intent ini):\n{plan.to_prompt()}"
                ),
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
                create_kwargs["tools"] = TOOL_DEFINITIONS

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
