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

FILE_LINE_RE = re.compile(r"^FILE:\s*(.+)\s*$", re.M)

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
            "name": "save_learning_note",
            "description": (
                "Simpan catatan ajar dari user ke knowledge/notes agar bisa dicari lagi. "
                "Pakai jika user bilang 'ingat bahwa', 'catat', 'ajarin agent'."
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
