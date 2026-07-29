"""DuckDB + markdown store for user documents (confirmation-gated ingest only)."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "documents.duckdb"
_LOCK = threading.RLock()

ALLOWED_CATEGORIES = frozenset(
    {"notes", "curriculum", "reports", "research", "user_docs"}
)


def db_path() -> Path:
    raw = (os.getenv("DOC_DB_PATH") or "").strip()
    path = Path(raw) if raw else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def knowledge_docs_dir() -> Path:
    raw = (os.getenv("KNOWLEDGE_DIR") or "").strip()
    root = Path(raw) if raw else (ROOT / "knowledge")
    out = root / "docs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path()))


def init_schema(con: duckdb.DuckDBPyConnection | None = None) -> None:
    own = con is None
    if own:
        con = connect()
    assert con is not None
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id VARCHAR PRIMARY KEY,
            title VARCHAR,
            category VARCHAR,
            source_name VARCHAR,
            content TEXT,
            content_hash VARCHAR,
            char_count INTEGER,
            md_path VARCHAR,
            chat_id BIGINT,
            created_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_docs_cat ON documents (category);
        CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents (content_hash);
        """
    )
    if own:
        con.close()


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:24]


def find_by_hash(h: str) -> dict[str, Any] | None:
    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            row = con.execute(
                """
                SELECT doc_id, title, category, source_name, md_path, created_at
                FROM documents WHERE content_hash = ? LIMIT 1
                """,
                [h],
            ).fetchone()
        finally:
            con.close()
    if not row:
        return None
    return {
        "doc_id": row[0],
        "title": row[1],
        "category": row[2],
        "source_name": row[3],
        "md_path": row[4],
        "created_at": row[5],
    }


def commit_document(
    *,
    title: str,
    body: str,
    category: str = "user_docs",
    source_name: str = "",
    chat_id: int | None = None,
) -> str:
    """Write markdown + DuckDB row. Caller must have user confirmation."""
    title = (title or "dokumen").strip()
    body = (body or "").strip()
    category = (category or "user_docs").strip().lower()
    if category not in ALLOWED_CATEGORIES:
        category = "user_docs"
    if not body:
        return "ERROR: isi dokumen kosong"
    if len(body) < 40:
        return "ERROR: isi dokumen terlalu pendek untuk disimpan"

    h = content_hash(body)
    existing = find_by_hash(h)
    if existing:
        return (
            f"SKIP: dokumen identik sudah ada di DB\n"
            f"doc_id={existing['doc_id']}\n"
            f"title={existing['title']}\n"
            f"md={existing['md_path']}"
        )

    stamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", title).strip("_")[:50] or "doc"
    doc_id = f"{stamp}_{slug}"
    md_path = knowledge_docs_dir() / f"{doc_id}.md"
    header = (
        f"# {title}\n\n"
        f"- doc_id: `{doc_id}`\n"
        f"- category: `{category}`\n"
        f"- source: `{source_name or '-'}`\n"
        f"- saved: {datetime.now(ZoneInfo('Asia/Jakarta')).isoformat()}\n"
        f"- hash: `{h}`\n\n"
        f"---\n\n"
    )
    md_path.write_text(header + body + "\n", encoding="utf-8")

    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            con.execute(
                """
                INSERT INTO documents
                (doc_id, title, category, source_name, content, content_hash,
                 char_count, md_path, chat_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    doc_id,
                    title,
                    category,
                    source_name or "",
                    body,
                    h,
                    len(body),
                    str(md_path),
                    chat_id,
                    datetime.now(timezone.utc).replace(tzinfo=None),
                ],
            )
        finally:
            con.close()

    logger.info("Document committed doc_id=%s chars=%s", doc_id, len(body))
    return (
        f"OK: dokumen disimpan (setelah konfirmasi user)\n"
        f"doc_id={doc_id}\n"
        f"category={category}\n"
        f"chars={len(body)}\n"
        f"db={db_path()}\n"
        f"FILE: {md_path}"
    )


def search_documents(query: str, *, limit: int = 5) -> str:
    q = (query or "").strip()
    if not q:
        return "ERROR: query kosong"
    limit = max(1, min(int(limit or 5), 20))
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", q) if len(t) > 2]
    if not tokens:
        tokens = [q.lower()]

    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            rows = con.execute(
                """
                SELECT doc_id, title, category, source_name, char_count,
                       substr(content, 1, 400), created_at
                FROM documents
                ORDER BY created_at DESC
                LIMIT 200
                """
            ).fetchall()
        finally:
            con.close()

    scored: list[tuple[float, tuple]] = []
    for row in rows:
        blob = f"{row[1]} {row[2]} {row[3]} {row[5]}".lower()
        score = sum(1.0 for t in tokens if t in blob)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: (-x[0], str(x[1][6])))

    if not scored:
        return f"(no rows) Tidak ada dokumen untuk: {q}\nDB={db_path()}"

    lines = [f"search_documents q={q!r} hits={min(len(scored), limit)} db={db_path()}", ""]
    for score, row in scored[:limit]:
        lines.append(
            f"- [{row[2]}] {row[1]} (id={row[0]}, chars={row[4]}, score={score:g})\n"
            f"  source={row[3] or '-'} @ {row[6]}\n"
            f"  {(row[5] or '')[:280]}…"
        )
    return "\n".join(lines)


def stats() -> str:
    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            row = con.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(char_count),0),
                       MIN(created_at), MAX(created_at)
                FROM documents
                """
            ).fetchone()
            by_cat = con.execute(
                """
                SELECT category, COUNT(*) FROM documents
                GROUP BY 1 ORDER BY 2 DESC
                """
            ).fetchall()
            recent = con.execute(
                """
                SELECT doc_id, title, category, char_count, created_at
                FROM documents ORDER BY created_at DESC LIMIT 8
                """
            ).fetchall()
        finally:
            con.close()

    lines = [
        f"Document DuckDB: {db_path()}",
        f"docs={row[0]} chars={row[1]} range={row[2]} → {row[3]}",
        "by_category: "
        + (", ".join(f"{c}={n}" for c, n in by_cat) if by_cat else "(empty)"),
        "",
        "Terbaru:",
    ]
    if not recent:
        lines.append("(kosong — stage_document_ingest + konfirmasi ya)")
    for r in recent:
        lines.append(f"- {r[0]} [{r[2]}] {r[1]} ({r[3]} chars) @ {r[4]}")
    return "\n".join(lines)
