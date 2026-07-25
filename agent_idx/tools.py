from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent_idx.chat_log import ChatLog
from agent_idx.data import TOOL_DEFINITIONS as DATA_TOOLS
from agent_idx.data import StockDataStore, dispatch_tool as dispatch_data_tool
from agent_idx.knowledge import KnowledgeBase
from agent_idx.news import search_news
from agent_idx.report import create_pdf_report, export_query_csv
from agent_idx.stockbit_store import StockbitReportsStore

FILE_LINE_RE = re.compile(r"^FILE:\s*(.+)\s*$", re.M)


def _stockbit_headless() -> bool:
    import os

    return os.getenv("STOCKBIT_HEADLESS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _run_stockbit(fn):
    from agent_idx.browser_stockbit import run_sync

    return run_sync(fn)


EXTRA_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Cari materi belajar IDX di knowledge base: fundamental, rasio keuangan, "
                "charting, foreign flow, playbook analisis, daily digest, dan berita harian. "
                "WAJIB dipakai untuk pertanyaan teori/fundamental/cara analisis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_knowledge_topics",
            "description": "List file knowledge base (curriculum, daily, news, lessons).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_collected_chats",
            "description": (
                "List semua group Telegram yang sudah di-collect (chat_id, judul, jumlah pesan). "
                "Pakai dulu jika user menyinggung group tertentu."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_chat_history",
            "description": (
                "Cari pesan yang sudah dikumpulkan dari group Telegram (bisa semua group "
                "atau satu chat_id). Pakai untuk 'siapa bilang', 'ringkas chat group X'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "chat_id": {
                        "type": "integer",
                        "description": "Opsional. Kosongkan = cari di semua group.",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recent_chat_history",
            "description": (
                "Ambil N pesan terbaru. chat_id opsional; kosongkan = semua group."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "integer"},
                    "limit": {"type": "integer", "description": "Default 30, max 100"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chat_log_stats",
            "description": "Statistik chat terkumpul (semua group, atau satu chat_id).",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Cari berita terkini terkait saham/IDX beserta judul, sumber, waktu, dan link. "
                "Pakai untuk permintaan berita, sentimen berita, atau konteks berita ticker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Kata kunci, mis. BBCA, net foreign sell bank, GOTO",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Jumlah berita (default 5, max 15)",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest_scan",
            "description": (
                "Scan skenario trading pada parquet IDX (2022–sekarang). "
                "Universe default: market cap >= 1 Triliun IDR. "
                "Uji win rate band 75–85%, grid parameter, per-emiten, supplement yfinance. "
                "Pakai jika user minta backtest, setup trading, win rate X–Y%."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Tanggal awal YYYYMMDD, default 20220101",
                    },
                    "min_win_rate": {
                        "type": "number",
                        "description": "Win rate minimum 0–1, default 0.75",
                    },
                    "max_win_rate": {
                        "type": "number",
                        "description": "Win rate maksimum 0–1, default 0.85",
                    },
                    "min_trades": {
                        "type": "integer",
                        "description": "Min jumlah trade agregat, default 15",
                    },
                    "min_market_cap": {
                        "type": "number",
                        "description": "Market cap min IDR, default 1e12 (1T)",
                    },
                    "supplement_external": {
                        "type": "boolean",
                        "description": "Supplement gaps via yfinance (default true)",
                    },
                    "universe": {
                        "type": "string",
                        "description": "large_cap (default) atau all",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_pdf_report",
            "description": (
                "Buat laporan PDF profesional dari teks analisis (markdown sederhana). "
                "Pakai jika user minta PDF, laporan, report, atau kirim dokumen. "
                "Isi body dengan ringkasan lengkap berbasis hasil tools sebelumnya."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {
                        "type": "string",
                        "description": "Isi laporan (## judul, - bullet, paragraf)",
                    },
                    "subtitle": {
                        "type": "string",
                        "description": "Opsional, mis. periode 2026 YTD",
                    },
                    "filename_prefix": {
                        "type": "string",
                        "description": "Prefix nama file, default laporan_idx",
                    },
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_query_csv",
            "description": (
                "Jalankan SQL SELECT dan simpan hasil penuh ke CSV. "
                "Pakai jika user minta export data, CSV, atau 'get data' ke file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "filename_prefix": {"type": "string"},
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_learning_job",
            "description": (
                "Jalankan job belajar sekarang: market digest, news digest, weekly lesson, "
                "atau all. Pakai jika user bilang 'belajar sekarang', 'update knowledge', "
                "'generate digest'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job": {
                        "type": "string",
                        "enum": ["all", "market", "news", "weekly"],
                    }
                },
                "required": ["job"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stockbit_status",
            "description": (
                "Cek status sesi Stockbit: browser hidup/tidak, sudah login, butuh OTP, URL aktif. "
                "Pakai sebelum scrape atau jika user tanya 'sudah login stockbit belum'."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stockbit_open",
            "description": (
                "Buka halaman Stockbit di browser ephemeral (guest-like, allowlist stockbit.com). "
                "Untuk chart: symbol=DSSA + page=chart, atau url langsung .../symbol/DSSA/chartbit. "
                "Jika perlu login, tool mengembalikan NEED_STOCKBIT_CREDENTIALS — "
                "jangan mengarang data; minta bot alur login interaktif."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker, mis. ADRO / BBCA / DSSA",
                    },
                    "page": {
                        "type": "string",
                        "description": (
                            "Sub-halaman symbol: chart/chartbit, keystats, financials, "
                            "company, bit. Default overview."
                        ),
                    },
                    "url": {
                        "type": "string",
                        "description": "URL Stockbit opsional (mis. .../symbol/DSSA/chartbit)",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stockbit_read",
            "description": (
                "Baca teks halaman Stockbit yang sedang terbuka (read-only snapshot)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stockbit_scrape",
            "description": (
                "Scrape data ticker di Stockbit (overview, keystats, financials, company). "
                "Butuh sudah login. Simpan laporan markdown ke exports/stockbit/. "
                "Pakai jika user bilang scrape/ambil data Stockbit untuk suatu emiten."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker, mis. BBCA, ADRO",
                    },
                    "sections": {
                        "type": "string",
                        "description": (
                            "Opsional, comma-separated: overview,keystats,financials,company,bit. "
                            "Default: overview,keystats,financials,company"
                        ),
                    },
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stockbit_scrape_reports",
            "description": (
                "Scrape stream/post Stockbit Reports (@StockbitReports) ke markdown. "
                "Butuh sudah login. Simpan ke exports/stockbit/. "
                "Pakai days ATAU date_from/date_to (YYYY-MM-DD)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Hari ke belakang (default 10, max 90) jika tanpa date_from/to",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Tanggal awal YYYY-MM-DD, mis. 2026-01-01",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Tanggal akhir YYYY-MM-DD, mis. 2026-07-04",
                    },
                    "url": {
                        "type": "string",
                        "description": (
                            "Opsional. Default https://stockbit.com/StockbitReports?source=0"
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_stockbit_scrape_date",
            "description": (
                "Baca post Stockbit Reports dari file scrape lokal (exports/stockbit/) "
                "untuk tanggal tertentu YYYY-MM-DD. "
                "Pakai jika user tanya report/laporan scrape tanggal X."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Tanggal YYYY-MM-DD, mis. 2026-03-25",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maks post ditampilkan (default 50)",
                    },
                },
                "required": ["date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_stockbit_scrape_files",
            "description": (
                "Statistik Stockbit Reports di ChromaDB (posts, rentang tanggal, runs)."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_stockbit_reports",
            "description": (
                "Cari post Stockbit Reports di ChromaDB (semantic/vector search)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_learning_note",
            "description": (
                "Simpan catatan ajar dari user ke knowledge/notes agar bisa dicari lagi. "
                "WAJIB dipakai jika user bilang 'ingat bahwa', 'catat', 'selalu catat', "
                "'ajarin'. Jangan balas dengan menyuruh user; catat lalu konfirmasi."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_DEFINITIONS: list[dict[str, Any]] = DATA_TOOLS + EXTRA_TOOLS


def dispatch_tool(
    store: StockDataStore,
    name: str,
    arguments: dict[str, Any],
    *,
    export_dir: Path,
    knowledge: KnowledgeBase | None = None,
    chat_log: ChatLog | None = None,
    chat_id: int | None = None,
    stockbit_store: StockbitReportsStore | None = None,
) -> str:
    try:
        if name == "search_knowledge":
            if knowledge is None:
                return "ERROR: knowledge base not configured"
            return knowledge.search(
                query=arguments["query"],
                limit=int(arguments.get("limit", 5)),
            )
        if name == "list_knowledge_topics":
            if knowledge is None:
                return "ERROR: knowledge base not configured"
            return knowledge.list_topics()
        if name == "list_collected_chats":
            if chat_log is None:
                return "ERROR: chat log not configured"
            return chat_log.list_chats()
        if name == "search_chat_history":
            if chat_log is None:
                return "ERROR: chat log not configured"
            target = arguments.get("chat_id")
            if target is not None:
                target = int(target)
            return chat_log.search(
                query=arguments["query"],
                chat_id=target,
                limit=int(arguments.get("limit", 20)),
            )
        if name == "recent_chat_history":
            if chat_log is None:
                return "ERROR: chat log not configured"
            target = arguments.get("chat_id")
            if target is not None:
                target = int(target)
            return chat_log.recent(chat_id=target, limit=int(arguments.get("limit", 30)))
        if name == "chat_log_stats":
            if chat_log is None:
                return "ERROR: chat log not configured"
            target = arguments.get("chat_id")
            if target is not None:
                target = int(target)
            return chat_log.stats(target)
        if name == "search_news":
            return search_news(
                query=arguments["query"],
                limit=int(arguments.get("limit", 5)),
            )
        if name == "run_backtest_scan":
            from agent_idx.backtest import run_backtest_scan

            return run_backtest_scan(
                store,
                export_dir,
                start_date=arguments.get("start_date", 20220101),
                min_win_rate=float(arguments.get("min_win_rate", 0.75)),
                max_win_rate=float(arguments.get("max_win_rate", 0.85)),
                min_trades=int(arguments.get("min_trades", 15)),
                min_market_cap=float(arguments.get("min_market_cap", 1_000_000_000_000)),
                supplement_external=bool(arguments.get("supplement_external", True)),
                universe=str(arguments.get("universe", "large_cap")),
            )
        if name == "create_pdf_report":
            return create_pdf_report(
                export_dir=export_dir,
                title=arguments["title"],
                body=arguments["body"],
                subtitle=arguments.get("subtitle"),
                filename_prefix=arguments.get("filename_prefix", "laporan_idx"),
            )
        if name == "export_query_csv":
            return export_query_csv(
                export_dir=export_dir,
                store=store,
                sql=arguments["sql"],
                filename_prefix=arguments.get("filename_prefix", "query_idx"),
            )
        if name == "run_learning_job":
            if knowledge is None:
                return "ERROR: knowledge base not configured"
            from agent_idx.jobs import (
                job_daily_market_digest,
                job_daily_news,
                job_weekly_lesson,
                run_all_learning_jobs,
            )

            job = (arguments.get("job") or "all").strip().lower()
            if job == "market":
                paths = [job_daily_market_digest(store, knowledge)]
            elif job == "news":
                paths = [job_daily_news(knowledge)]
            elif job == "weekly":
                paths = [job_weekly_lesson(knowledge)]
            else:
                paths = run_all_learning_jobs(store, knowledge)
            lines = [f"OK: learning job={job}"]
            for path in paths:
                lines.append(f"FILE: {path}")
            return "\n".join(lines)
        if name == "save_learning_note":
            if knowledge is None:
                return "ERROR: knowledge base not configured"
            return knowledge.save_note(
                title=arguments["title"],
                body=arguments["body"],
            )
        if name == "stockbit_status":
            from agent_idx import browser_stockbit as _sb

            if _sb._BROWSER is None:
                return (
                    "browser_ready=False\n"
                    "login=not_started\n"
                    "mode=ephemeral/guest-like\n"
                    "hint: /stockbit untuk login interaktif"
                )

            headless = _stockbit_headless()

            def _status() -> str:
                from agent_idx.browser_stockbit import get_browser

                return get_browser(headless=headless).status()

            return _run_stockbit(_status)
        if name == "stockbit_open":
            headless = _stockbit_headless()
            url = arguments.get("url")
            symbol = arguments.get("symbol")
            page = arguments.get("page")

            def _open() -> str:
                from agent_idx.browser_stockbit import get_browser

                browser = get_browser(headless=headless)
                browser.start()
                return browser.open(url=url, symbol=symbol, page=page)

            return _run_stockbit(_open)
        if name == "stockbit_read":
            max_chars = int(arguments.get("max_chars", 8000))

            def _read() -> str:
                from agent_idx.browser_stockbit import get_browser

                browser = get_browser()
                if not browser.ready:
                    return "ERROR: browser belum dibuka. Panggil stockbit_open dulu."
                return browser.snapshot(max_chars=max_chars)

            return _run_stockbit(_read)
        if name == "stockbit_scrape":
            headless = _stockbit_headless()
            symbol = arguments["symbol"]
            sections = arguments.get("sections")

            def _scrape() -> str:
                from agent_idx.browser_stockbit import get_browser

                browser = get_browser(headless=headless)
                browser.start()
                return browser.scrape_symbol(symbol=symbol, sections=sections)

            return _run_stockbit(_scrape)
        if name == "stockbit_scrape_reports":
            headless = _stockbit_headless()
            days = arguments.get("days")
            date_from = arguments.get("date_from")
            date_to = arguments.get("date_to")
            url = arguments.get("url")

            def _scrape_reports() -> str:
                from agent_idx.browser_stockbit import get_browser

                browser = get_browser(headless=headless)
                browser.start()
                return browser.scrape_reports_stream(
                    days=int(days) if days is not None else None,
                    url=url,
                    date_from=date_from,
                    date_to=date_to,
                )

            result = _run_stockbit(_scrape_reports)
            if stockbit_store is not None and result.startswith("OK:"):
                inserted = 0
                for path in extract_files(result):
                    inserted += stockbit_store.ingest_markdown(path)
                if inserted:
                    result += (
                        f"\nVector: {inserted} post disimpan ke Chroma "
                        f"({stockbit_store.store_path})"
                    )
            return result
        if name == "read_stockbit_scrape_date":
            from agent_idx.stockbit_reports import read_posts_for_date

            return read_posts_for_date(
                stockbit_store,
                export_dir,
                arguments["date"],
                limit=int(arguments.get("limit", 50)),
            )
        if name == "list_stockbit_scrape_files":
            from agent_idx.stockbit_reports import scrape_stats

            return scrape_stats(stockbit_store, export_dir)
        if name == "search_stockbit_reports":
            from agent_idx.stockbit_reports import search_posts

            if stockbit_store is None:
                return "ERROR: stockbit vector store belum dikonfigurasi"
            return search_posts(
                stockbit_store,
                arguments["query"],
                limit=int(arguments.get("limit", 20)),
            )
        if name == "stockbit_close":
            return (
                "ERROR: agent tidak boleh menutup browser Stockbit. "
                "Biarkan sesi terbuka; user menutup via /stockbit close."
            )
        return dispatch_data_tool(store, name, arguments)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def extract_files(tool_output: str) -> list[Path]:
    files: list[Path] = []
    for match in FILE_LINE_RE.finditer(tool_output or ""):
        path = Path(match.group(1).strip().strip('"'))
        if path.is_file():
            files.append(path)
    return files
