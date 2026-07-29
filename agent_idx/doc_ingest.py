"""Stage user-document ingest — NEVER writes DB until Telegram ya/tidak confirm."""
from __future__ import annotations

import re
from typing import Any

from agent_idx import doc_store

_PENDING: dict[int, dict[str, Any]] = {}

_PDF_BLOCK = re.compile(
    r"---\s*ISI PDF\s*\((?P<name>[^)]*)\)\s*---\s*(?P<body>.*?)\s*---\s*END PDF\s*---",
    re.I | re.S,
)
_SAVE_HINT = re.compile(
    r"\b("
    r"simpan|catat|arsip|ingest|masukkan|tambahkan|"
    r"simpan\s+ke\s+(?:db|database|duckdb|knowledge|docs?)|"
    r"save\s+(?:to\s+)?(?:db|database|knowledge)|"
    r"ingat\s+(?:dokumen|pdf|file\s+ini)"
    r")\b",
    re.I,
)


def is_document_save_request(text: str) -> bool:
    return bool(_SAVE_HINT.search(text or ""))


def extract_pdf_block(text: str) -> tuple[str, str] | None:
    """Return (source_name, body) from injected PDF block, if any."""
    m = _PDF_BLOCK.search(text or "")
    if not m:
        return None
    name = (m.group("name") or "document.pdf").strip()
    body = (m.group("body") or "").strip()
    if not body:
        return None
    return name, body


def stage_ingest(
    chat_key: int,
    *,
    title: str,
    body: str,
    category: str = "user_docs",
    source_name: str = "",
) -> str:
    title = (title or "").strip() or "dokumen_user"
    body = (body or "").strip()
    category = (category or "user_docs").strip().lower()
    if category not in doc_store.ALLOWED_CATEGORIES:
        category = "user_docs"
    if not body:
        return "ERROR: body dokumen kosong — lampirkan PDF atau tempel teks"
    if len(body) < 40:
        return "ERROR: teks terlalu pendek untuk di-ingest"

    h = doc_store.content_hash(body)
    existing = doc_store.find_by_hash(h)
    if existing:
        return (
            f"SKIP: dokumen ini sudah ada di DB (hash={h})\n"
            f"doc_id={existing['doc_id']} title={existing['title']}\n"
            "Tidak di-stage ulang."
        )

    preview = body if len(body) <= 500 else body[:497] + "..."
    _PENDING[int(chat_key)] = {
        "title": title,
        "body": body,
        "category": category,
        "source_name": source_name or "",
        "content_hash": h,
        "char_count": len(body),
    }
    return (
        "STAGED (menunggu konfirmasi user)\n"
        "job=document_ingest\n"
        f"title={title}\n"
        f"category={category}\n"
        f"source={source_name or '-'}\n"
        f"chars={len(body)} hash={h}\n"
        f"preview:\n{preview}\n"
        "CATATAN: dokumen BELUM ditulis ke DuckDB/knowledge. "
        "User harus menyetujui (ya/tidak) di Telegram."
    )


def has_pending(chat_key: int) -> bool:
    return int(chat_key) in _PENDING


def peek_pending(chat_key: int) -> dict[str, Any] | None:
    return _PENDING.get(int(chat_key))


def describe_pending(chat_key: int) -> str:
    job = peek_pending(chat_key)
    if not job:
        return "(tidak ada dokumen tertunda)"
    preview = job["body"]
    if len(preview) > 600:
        preview = preview[:597] + "..."
    return (
        "Dokumen menunggu konfirmasi simpan ke DB:\n"
        f"• title: {job['title']}\n"
        f"• category: {job['category']}\n"
        f"• source: {job.get('source_name') or '-'}\n"
        f"• chars: {job['char_count']} hash={job['content_hash']}\n"
        f"• target DB: {doc_store.db_path()}\n"
        f"• target md: knowledge/docs/\n\n"
        f"Preview:\n{preview}"
    )


def clear_pending(chat_key: int) -> bool:
    return _PENDING.pop(int(chat_key), None) is not None


def apply_pending(chat_key: int) -> str:
    """Commit staged document after explicit user confirmation."""
    job = _PENDING.pop(int(chat_key), None)
    if not job:
        return "ERROR: tidak ada dokumen tertunda"
    return doc_store.commit_document(
        title=job["title"],
        body=job["body"],
        category=job["category"],
        source_name=job.get("source_name") or "",
        chat_id=int(chat_key),
    )
