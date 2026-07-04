from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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
]

TOOL_DEFINITIONS: list[dict[str, Any]] = DATA_TOOLS + EXTRA_TOOLS


def dispatch_tool(
    store: StockDataStore,
    name: str,
    arguments: dict[str, Any],
    *,
    export_dir: Path,
    knowledge: KnowledgeBase | None = None,
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
