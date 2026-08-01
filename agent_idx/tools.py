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


def _stockbit_profile_dir() -> str | None:
    """Persistent Chrome profile so agent reuses Stockbit login cookies."""
    import os
    from pathlib import Path

    raw = (os.getenv("STOCKBIT_PROFILE_DIR") or "").strip()
    if raw.lower() in {"", "0", "false", "none", "off"}:
        # Default: share profile with scripts/farm_stockbit_charts.py
        root = Path(__file__).resolve().parent.parent
        path = root / "data" / "stockbit_profile"
    else:
        path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _stockbit_browser(headless: bool | None = None):
    from agent_idx.browser_stockbit import get_browser

    if headless is None:
        headless = _stockbit_headless()
    return get_browser(headless=headless, user_data_dir=_stockbit_profile_dir())


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
                "Pakai untuk permintaan berita, sentimen berita, atau konteks berita symbol. "
                "Hasil untuk dijawab sebagai teks di chat — jangan buat/kirim file."
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
                "Scan Trading scenario pada parquet IDX (2022–sekarang). "
                "Cap filter default: market cap >= 1 Triliun IDR. "
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
                    "cap_filter": {
                        "type": "string",
                        "description": (
                            "large_cap (default) atau all; none diterima sebagai alias legacy"
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
            "name": "create_pdf_report",
            "description": (
                "Buat file PDF dari teks analisis. "
                "HANYA panggil jika user EXPLISIT bilang buat/kirim/generate PDF. "
                "JANGAN dipakai sebagai default output — jawaban biasa cukup teks chat. "
                "Jika user bilang 'jangan buat PDF', JANGAN panggil tool ini."
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
                "Buka halaman Stockbit di Browser session dengan Persistent profile "
                "(allowlist stockbit.com). "
                "Untuk chart: symbol=DSSA + page=chart, atau url langsung .../symbol/DSSA/chartbit. "
                "Jika perlu login, tool mengembalikan NEED_STOCKBIT_CREDENTIALS — "
                "jangan mengarang data; minta bot alur login interaktif."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol, mis. ADRO / BBCA / DSSA",
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
                "Scrape data symbol di Stockbit (overview, keystats, financials, profile). "
                "Butuh sudah login. Simpan laporan markdown ke exports/stockbit/. "
                "Pakai jika user bilang scrape/ambil data Stockbit untuk suatu emiten."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol, mis. BBCA, ADRO",
                    },
                    "sections": {
                        "type": "string",
                        "description": (
                            "Opsional, comma-separated: overview,keystats,financials,profile,bit. "
                            "Default: overview,keystats,financials,profile. "
                            "Alias: company=profile. Keystats deep-scrape Net Income/EPS/Revenue tabs."
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
                "Scrape stream/post akun Stockbit ke markdown (exports/stockbit/). "
                "Default @StockbitReports. Pakai source=official atau "
                "url=https://stockbit.com/Stockbit untuk akun resmi @Stockbit "
                "(sering ada foreign net siang hari). source=both = kedua akun. "
                "Aksi di-stage dan baru menulis setelah konfirmasi ya/tidak. "
                "Butuh sudah login. Pakai days ATAU date_from/date_to (YYYY-MM-DD)."
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
                    "source": {
                        "type": "string",
                        "description": (
                            "reports | official | both. "
                            "reports=@StockbitReports, official=@Stockbit, both=keduanya. "
                            "Diabaikan jika url diisi (kecuali both)."
                        ),
                    },
                    "url": {
                        "type": "string",
                        "description": (
                            "Opsional URL profil absolut. "
                            "Contoh: https://stockbit.com/Stockbit "
                            "atau https://stockbit.com/StockbitReports?source=0"
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
    {
        "type": "function",
        "function": {
            "name": "list_project_dir",
            "description": (
                "List file/folder di dalam project agent_idx (read-only). "
                "Path relatif dari root repo, contoh: 'data', 'data/charts', 'scripts'. "
                "Pakai untuk melihat isi data/, exports/, knowledge/, symbols_list, dll."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relatif, default '.' (root agent_idx)",
                    },
                    "recursive": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_project_file",
            "description": (
                "Baca isi file teks di project agent_idx (read-only). "
                "Contoh: 'symbols_list.txt', 'data/symbols_list.txt', "
                "'knowledge/curriculum/01_idx_market_structure.md', "
                "'data/charts/_meta/BBCA.json'. "
                ".env dan profile/cookie diblokir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stockbit_create_post",
            "description": (
                "Buat/publish postingan di stream Stockbit akun yang sedang login "
                "(kotak 'Tulis ide kamu disini...'). "
                "INI DIDUKUNG — jangan menolak dengan alasan read-only. "
                "Post di-stage dulu dan WAJIB dikonfirmasi user (ya/tidak) di Telegram "
                "sebelum dikirim. Butuh sesi login Stockbit. "
                "Pakai jika user minta posting/buat post/kirim ide ke Stockbit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Isi postingan (teks biasa, max ~2000 karakter)",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "farm_stockbit_fundamentals",
            "description": (
                "Scrape halaman Stockbit per emiten: overview, keystats (deep: valuation/"
                "solvency/series Net Income·EPS·Revenue), financials deep (Income "
                "Statement·Balance Sheet·Cash Flow x Quarterly/Annual/TTM + Key Ratio "
                "Financial Health/Efficiency), profile deep "
                "(background, shareholder >1%, komposisi KSEI, holding, BOD/BOC, UBO, "
                "history, number of shareholders, subsidiaries, address). "
                "Pace PELAN + jitter supaya tidak "
                "seperti bot agresif. Simpan markdown ke exports/stockbit/ dan "
                "sidecar JSON ke data/fundamentals/. "
                "Tanpa symbols = universe data/symbols_list.txt. "
                "Job >3 emiten di-stage (konfirmasi ya/tidak). "
                "Default skip emiten yang sections-nya lengkap dan di-scrape <7 hari; "
                "pakai force=true untuk scrape ulang. "
                "Pakai limit/offset untuk batch kecil (disarankan 10–30/hari). "
                "Pakai tier='top' untuk 200 emiten terbesar, tier='rest' untuk sisanya."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "string",
                        "description": (
                            "Opsional comma-separated. Kosong = seluruh symbols_list.txt"
                        ),
                    },
                    "tier": {
                        "type": "string",
                        "description": (
                            "top = 200 emiten terbesar, rest = sisanya, all = semua. "
                            "Diabaikan jika symbols diisi."
                        ),
                    },
                    "sections": {
                        "type": "string",
                        "description": (
                            "Comma-separated: overview,keystats,financials,profile,bit. "
                            "Default overview,keystats,financials,profile (company→profile)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Batasi jumlah emiten (0=semua). Disarankan batch kecil.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Lewati N emiten pertama (resume batch)",
                    },
                    "skip_fresh_days": {
                        "type": "integer",
                        "description": (
                            "Skip emiten dengan Sidecar lengkap yang lebih muda dari N "
                            "hari kalender WIB (default 7)."
                        ),
                    },
                    "force": {
                        "type": "boolean",
                        "description": "true = abaikan freshness dan scrape ulang",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "farm_stockbit_charts",
            "description": (
                "Farm data chart OHLCV via tvkit (TradingView) per timeframe "
                "(1M/5M/15M/30M/1H/4H/1D/1W) lalu simpan ke parquet + DuckDB "
                "(data/charts.duckdb) + indikator (SMA/EMA/MACD/RSI). "
                "INI BISA DILAKUKAN — jangan menolak permintaan farming chart. "
                "Butuh sesi login Stockbit (profile persisten). "
                "Tanpa 'symbols' = pakai seluruh universe di data/symbols_list.txt "
                "(~980 emiten, pakai limit untuk batch). "
                "Job dengan >3 emiten di-stage dan minta konfirmasi user dulu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "string",
                        "description": (
                            "Opsional, comma-separated: 'BBCA,BBRI,DSSA'. "
                            "Kosongkan untuk seluruh emiten dari symbols_list.txt"
                        ),
                    },
                    "timeframes": {
                        "type": "string",
                        "description": (
                            "Comma-separated: 1M,5M,15M,30M,1H,4H,1D,1W. "
                            "Default 1H,1D,1W"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Batasi jumlah emiten (0 = semua). Pakai untuk batch.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Lewati N emiten pertama (untuk resume batch)",
                    },
                    "sync_daily": {
                        "type": "boolean",
                        "description": "Gabungkan hasil 1D ke parquet daily_stock_summary",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stockbit_ohlcv",
            "description": (
                "Ambil OHLCV (tvkit) untuk symbol + timeframe. "
                "Timeframe didukung: 5M, 30M, 1H, 1D, 1W (juga 1M/15M/4H). "
                "Bisa multi-TF sekaligus (mis. '5M,1H,1D'). "
                "Baca dari store lokal; set refresh=true untuk tarik ulang via tvkit. "
                "INI tool utama untuk data chart Stockbit — prefer over Yahoo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol IDX, mis. BBCA / DSSA",
                    },
                    "timeframes": {
                        "type": "string",
                        "description": (
                            "Satu atau beberapa TF comma-separated. "
                            "Contoh: '1D' | '5M' | '30M,1H,1D' | '1W'. Default 1D."
                        ),
                    },
                    "n": {
                        "type": "integer",
                        "description": "Jumlah bar per TF (default 40, max 500)",
                    },
                    "refresh": {
                        "type": "boolean",
                        "description": (
                            "true = farm ulang dari Chartbit sebelum baca. "
                            "Default false; otomatis farm jika TF belum ada di disk."
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
            "name": "query_chart_ohlcv",
            "description": (
                "Query OHLCV Chartbit dari DuckDB saja (tanpa farm). "
                "Lebih baik pakai get_stockbit_ohlcv untuk 5M/30M/1H/1D/1W."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol IDX, mis. BBCA"},
                    "timeframe": {
                        "type": "string",
                        "description": "5M/30M/1H/1D/1W (atau 1M/15M/4H)",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Jumlah bar (default 50, max 500)",
                    },
                },
                "required": ["symbol", "timeframe"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chart_features",
            "description": (
                "Ringkasan fitur teknikal multi-timeframe dari DuckDB "
                "(last, range, RSI, MACD, SMA, volume vs avg). "
                "WAJIB dipakai sebelum analisa chart comprehensive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "timeframes": {
                        "type": "string",
                        "description": "Comma-separated TFs, default 5M,1H,1D",
                    },
                    "lookback": {
                        "type": "integer",
                        "description": "Bar per TF untuk hitung fitur (default 40)",
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
            "name": "sync_chart_db",
            "description": (
                "Sinkronkan parquet data/charts/{SYM}/{TF}.parquet ke DuckDB. "
                "Opsional filter symbols/timeframes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "string",
                        "description": "Comma-separated; kosong = semua di folder charts",
                    },
                    "timeframes": {
                        "type": "string",
                        "description": "Comma-separated; kosong = semua TF di folder",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chart_db_stats",
            "description": "Statistik isi DuckDB chart store (symbols, bars, last update).",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technicals",
            "description": (
                "Data PASAR → TEKNIKAL: OHLCV + garis MA5, MA20, MA50, MA200, "
                "RSI, MACD untuk symbol/timeframe. "
                "Pakai untuk analisa teknikal / posisi vs moving average."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "timeframe": {
                        "type": "string",
                        "description": "5M/30M/1H/1D/1W — default 1D",
                    },
                    "n": {"type": "integer", "description": "Jumlah bar (default 5)"},
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": (
                "Data KEYSTATS/PROFILE → FUNDAMENTAL metrics + snapshot F Buy/F Sell "
                "dari transform store (hasil scrape Stockbit yang sudah diolah)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_stock",
            "description": (
                "Stock Analysis gabungan technical + fundamental untuk SATU Symbol. "
                "Read-only (tidak scrape). Output JSON: freshness, sinyal per dimensi "
                "(trend/momentum/volume/volatility/structure + valuation/profitability/"
                "growth/solvency/cash_flow/dividend; bank_quality untuk bank), "
                "conflicts[], gaps (GAP_DATA/STALE_DATA). "
                "Tanpa buy/sell. horizon=intraday|swing|position. "
                "Utama untuk analisis emiten; get_technicals/get_fundamentals untuk drill-down."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "horizon": {
                        "type": "string",
                        "description": "intraday | swing (default) | position",
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
            "name": "search_news_sentiment",
            "description": (
                "Ambil berita lalu TRANSFORM sentiment positif/negatif/neutral "
                "(lexicon ID/EN). Simpan ke data/transforms.duckdb. "
                "Pakai untuk pertanyaan sentimen berita / katalis."
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
            "name": "run_data_transform",
            "description": (
                "Jalankan pipeline TRANSFORM setelah scrape/farm: "
                "scope=technicals|fundamentals|news|foreign|all. "
                "Technicals = hitung ulang MA5/20/50/200 di parquet+DuckDB. "
                "Fundamentals = parse keystats markdown → metrics. "
                "News = sentiment. Foreign = ranking dari snapshot scrape."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "all|technicals|fundamentals|news|foreign",
                    },
                    "symbols": {
                        "type": "string",
                        "description": "Opsional filter symbol comma-separated",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Batasi jumlah file/symbol (0=semua)",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rebuild_market_from_stockbit",
            "description": (
                "Rebuild panel harian pasar dari Chartbit Stockbit "
                "(data/charts/*/1D atau 1H) → daily_stock_summary_stockbit_*.parquet. "
                "INI sumber pasar utama (bukan Yahoo). "
                "Pakai setelah farm chart atau jika list_date_range masih menunjuk Yahoo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "string",
                        "description": "Opsional comma-separated; kosong = semua chart di disk",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stage_document_ingest",
            "description": (
                "STAGE simpan dokumen user (PDF/teks) ke DuckDB documents + "
                "knowledge/docs/*.md. TIDAK langsung menulis — menunggu konfirmasi "
                "ya/tidak di Telegram. WAJIB dipakai HANYA jika user EXPLISIT minta "
                "simpan/catat/arsipkan dokumen ke DB/knowledge. "
                "Body bisa dari blok --- ISI PDF --- di pesan, atau teks yang user tempel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Judul dokumen (dari nama file atau ringkas)",
                    },
                    "body": {
                        "type": "string",
                        "description": "Isi teks dokumen lengkap yang akan disimpan",
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "notes",
                            "curriculum",
                            "reports",
                            "research",
                            "user_docs",
                        ],
                        "description": "Default user_docs",
                    },
                    "source_name": {
                        "type": "string",
                        "description": "Nama file sumber, mis. laporan.pdf",
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
            "name": "search_documents",
            "description": (
                "Cari dokumen yang sudah di-commit ke DuckDB document store "
                "(setelah konfirmasi user)."
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
            "name": "doc_store_stats",
            "description": "Statistik dokumen tersimpan di DuckDB document store.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_project_file",
            "description": (
                "USULKAN perubahan file teks di project agent_idx. "
                "Perubahan TIDAK langsung ditulis — di-stage dan WAJIB dikonfirmasi "
                "user (ya/tidak) di Telegram sebelum benar-benar disimpan. "
                "Gunakan untuk membuat/mengubah file seperti catatan, symbols_list.txt, "
                "config teks, atau kode. Sertakan konten LENGKAP file (bukan patch) "
                "untuk mode overwrite/create. Hanya file teks (.py .md .txt .json .csv dll). "
                ".env, cookie, profile, dan file binary diblokir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relatif dari root agent_idx, mis. 'data/notes.md'",
                    },
                    "content": {
                        "type": "string",
                        "description": "Konten lengkap file (untuk append: teks yang ditambahkan)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append", "create"],
                        "description": "overwrite=ganti isi, append=tambah di akhir, create=file baru",
                    },
                },
                "required": ["path", "content"],
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
                cap_filter=str(
                    arguments.get("cap_filter")
                    or arguments.get("universe")
                    or "large_cap"
                ),
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
                # Digests stay on disk for knowledge; don't expose FILE: (bot would attach).
                lines.append(f"SAVED: {path}")
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

            headless = _stockbit_headless()
            if _sb._BROWSER is None:
                # Don't start browser just for status — report profile path.
                return (
                    "browser_ready=False\n"
                    "login=not_started_in_this_process\n"
                    f"profile={_stockbit_profile_dir()}\n"
                    "mode=persistent_profile (shared with farm script)\n"
                    "hint: stockbit_open / /stockbit — reuse cookies di profile"
                )

            def _status() -> str:
                return _stockbit_browser(headless).status()

            return _run_stockbit(_status)
        if name == "stockbit_open":
            headless = _stockbit_headless()
            url = arguments.get("url")
            symbol = arguments.get("symbol")
            page = arguments.get("page")

            def _open() -> str:
                browser = _stockbit_browser(headless)
                browser.start()
                return browser.open(url=url, symbol=symbol, page=page)

            return _run_stockbit(_open)
        if name == "stockbit_read":
            max_chars = int(arguments.get("max_chars", 8000))

            def _read() -> str:
                browser = _stockbit_browser()
                if not browser.ready:
                    return "ERROR: browser belum dibuka. Panggil stockbit_open dulu."
                return browser.snapshot(max_chars=max_chars)

            return _run_stockbit(_read)
        if name == "stockbit_scrape":
            headless = _stockbit_headless()
            symbol = arguments["symbol"]
            sections = arguments.get("sections")

            def _scrape() -> str:
                browser = _stockbit_browser(headless)
                browser.start()
                return browser.scrape_symbol(symbol=symbol, sections=sections)

            return _run_stockbit(_scrape)
        if name == "stockbit_scrape_reports":
            from agent_idx.stockbit_reports import stage_stream_scrape

            key = chat_id if chat_id is not None else 0
            return stage_stream_scrape(int(key), arguments)
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
        if name == "list_project_dir":
            from agent_idx.project_fs import list_project_dir

            return list_project_dir(
                arguments.get("path") or ".",
                recursive=bool(arguments.get("recursive", False)),
                limit=int(arguments.get("limit", 100)),
            )
        if name == "read_project_file":
            from agent_idx.project_fs import read_project_file

            return read_project_file(
                arguments["path"],
                max_chars=int(arguments.get("max_chars", 12_000)),
            )
        if name == "stockbit_create_post":
            from agent_idx import stockbit_post

            key = chat_id if chat_id is not None else 0
            return stockbit_post.stage_post(int(key), arguments.get("text") or "")
        if name == "farm_stockbit_fundamentals":
            from agent_idx import fund_farm, universe

            tier_raw = (arguments.get("tier") or "").strip()
            limit = int(arguments.get("limit") or 0)
            offset = int(arguments.get("offset") or 0)
            if tier_raw and not (arguments.get("symbols") or "").strip():
                _, picked = universe.resolve_tier(tier_raw)
                picked = picked[offset:]
                if limit > 0:
                    picked = picked[:limit]
            else:
                picked = fund_farm.resolve_symbols(
                    arguments.get("symbols"),
                    limit=limit,
                    offset=offset,
                )
            if not picked:
                return "ERROR: tidak ada simbol (cek data/symbols_list.txt)"
            secs = fund_farm.normalize_sections(arguments.get("sections"))
            force = bool(arguments.get("force", False))
            skip_fresh_days = (
                0 if force else max(0, int(arguments.get("skip_fresh_days") or 7))
            )
            if len(picked) > fund_farm.BULK_THRESHOLD:
                key = chat_id if chat_id is not None else 0
                return fund_farm.stage_fund_job(
                    int(key),
                    picked,
                    secs,
                    skip_fresh_days=skip_fresh_days,
                )
            return fund_farm.run_fund_farm(
                picked,
                secs,
                skip_fresh_days=skip_fresh_days,
            )
        if name == "farm_stockbit_charts":
            from agent_idx import chart_farm

            picked = chart_farm.resolve_symbols(
                arguments.get("symbols"),
                limit=int(arguments.get("limit") or 0),
                offset=int(arguments.get("offset") or 0),
            )
            if not picked:
                return "ERROR: tidak ada simbol (cek data/symbols_list.txt)"
            raw_tfs = (arguments.get("timeframes") or "").strip()
            tfs = (
                chart_farm.normalize_timeframes(raw_tfs)
                if raw_tfs
                else list(chart_farm.DEFAULT_TIMEFRAMES)
            )
            sync_daily = bool(arguments.get("sync_daily", False))

            if len(picked) > chart_farm.BULK_THRESHOLD:
                key = chat_id if chat_id is not None else 0
                return chart_farm.stage_farm_job(
                    int(key), picked, tfs, sync_daily=sync_daily
                )
            return chart_farm.run_farm(picked, tfs, sync_daily=sync_daily)
        if name == "get_stockbit_ohlcv":
            from agent_idx import chart_db

            return chart_db.get_stockbit_ohlcv(
                arguments.get("symbol") or "",
                arguments.get("timeframes") or "1D",
                n=int(arguments.get("n") or 40),
                refresh=bool(arguments.get("refresh", False)),
            )
        if name == "get_technicals":
            from agent_idx.transforms import get_technicals

            return get_technicals(
                arguments.get("symbol") or "",
                arguments.get("timeframe") or "1D",
                n=int(arguments.get("n") or 5),
            )
        if name == "get_fundamentals":
            from agent_idx.transforms import get_fundamentals

            return get_fundamentals(
                arguments.get("symbol") or "",
                limit=int(arguments.get("limit") or 40),
            )
        if name == "analyze_stock":
            from agent_idx.stock_analysis import analyze_stock

            return analyze_stock(
                arguments.get("symbol") or "",
                arguments.get("horizon") or "swing",
            )
        if name == "search_news_sentiment":
            from agent_idx.transforms import search_news_sentiment

            return search_news_sentiment(
                arguments.get("query") or "",
                limit=int(arguments.get("limit") or 8),
            )
        if name == "run_data_transform":
            from agent_idx.transforms import run_data_transform

            return run_data_transform(
                arguments.get("scope") or "all",
                symbols=arguments.get("symbols"),
                limit=int(arguments.get("limit") or 0),
            )
        if name == "query_chart_ohlcv":
            from agent_idx import chart_db

            return chart_db.query_ohlcv(
                arguments.get("symbol") or "",
                arguments.get("timeframe") or "1H",
                n=int(arguments.get("n") or 50),
            )
        if name == "chart_features":
            from agent_idx import chart_db

            return chart_db.chart_features(
                arguments.get("symbol") or "",
                arguments.get("timeframes"),
                lookback=int(arguments.get("lookback") or 40),
            )
        if name == "sync_chart_db":
            from agent_idx import chart_db

            syms = [
                s.strip().upper()
                for s in (arguments.get("symbols") or "").split(",")
                if s.strip()
            ]
            tfs = [
                t.strip().upper()
                for t in (arguments.get("timeframes") or "").split(",")
                if t.strip()
            ]
            return chart_db.sync_from_parquet(syms or None, tfs or None)
        if name == "chart_db_stats":
            from agent_idx import chart_db

            return chart_db.stats()
        if name == "rebuild_market_from_stockbit":
            from agent_idx import chart_farm

            raw = (arguments.get("symbols") or "").strip()
            syms = [s.strip().upper() for s in raw.split(",") if s.strip()] or None
            path = chart_farm.rebuild_stockbit_daily_summary(symbols=syms)
            if path is None:
                return (
                    "ERROR: tidak ada Chartbit 1D/1H di data/charts. "
                    "Jalankan farm_stockbit_charts(timeframes=1H,1D) dulu."
                )
            # Refresh in-process store view if possible
            try:
                store._register_view()  # noqa: SLF001
            except Exception:  # noqa: BLE001
                pass
            return (
                f"OK: market panel dari Stockbit Chartbit\n"
                f"FILE: {path}\n"
                f"{store.list_date_range()}\n"
                f"{store.describe_schema().split(chr(10))[0]}"
            )
        if name == "stage_document_ingest":
            from agent_idx import doc_ingest

            key = chat_id if chat_id is not None else 0
            body = (arguments.get("body") or "").strip()
            source = (arguments.get("source_name") or "").strip()
            title = (arguments.get("title") or "").strip()
            # Allow tool to omit body if PDF block is passed via title misuse —
            # prefer explicit body. If empty, error clearly.
            if not body:
                return (
                    "ERROR: body kosong. Ambil teks dari blok --- ISI PDF --- "
                    "di pesan user, lalu panggil stage_document_ingest lagi."
                )
            if not title:
                title = source or "dokumen_user"
            return doc_ingest.stage_ingest(
                int(key),
                title=title,
                body=body,
                category=arguments.get("category") or "user_docs",
                source_name=source,
            )
        if name == "search_documents":
            from agent_idx import doc_store

            return doc_store.search_documents(
                arguments.get("query") or "",
                limit=int(arguments.get("limit") or 5),
            )
        if name == "doc_store_stats":
            from agent_idx import doc_store

            return doc_store.stats()
        if name == "write_project_file":
            from agent_idx.project_fs import stage_project_write

            key = chat_id if chat_id is not None else 0
            return stage_project_write(
                int(key),
                arguments["path"],
                arguments.get("content") or "",
                mode=arguments.get("mode") or "overwrite",
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
