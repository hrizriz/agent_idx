from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^## (20\d{2}-\d{2}-\d{2} \d{2}:\d{2}) WIB\s*$")
_LINK_RE = re.compile(r"^link:\s*(\S+)\s*$", re.I)
_META_RE = {
    "source": re.compile(r"^Source:\s*(.+)$", re.I),
    "window": re.compile(r"^Window:\s*(.+)$", re.I),
    "scraped": re.compile(r"^Scraped:\s*(.+)$", re.I),
}
_RUNS_FILE = "scrape_runs.json"
_COLLECTION = "stockbit_reports"


class StockbitReportsStore:
    """ChromaDB vector store for scraped Stockbit Reports stream posts."""

    def __init__(
        self,
        chroma_dir: Path,
        export_dir: Path | None = None,
        *,
        auto_sync: bool = False,
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.export_dir = Path(export_dir) if export_dir else None
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._runs_path = self.chroma_dir / _RUNS_FILE
        self._client = None
        self._collection = None
        if auto_sync and self.export_dir is not None:
            self.sync_from_exports(self.export_dir)

    @property
    def chroma_ready(self) -> bool:
        return self._collection is not None

    def _ensure_client(self):
        if self._collection is not None:
            return self._collection
        self._client = self._open_client()
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    @property
    def store_path(self) -> Path:
        return self.chroma_dir

    # Backward-compatible alias used in tool messages.
    @property
    def db_path(self) -> Path:
        return self.chroma_dir

    def _open_client(self):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb belum terpasang. Jalankan: py -3 -m pip install chromadb"
            ) from exc
        return chromadb.PersistentClient(path=str(self.chroma_dir))

    def _load_runs(self) -> list[dict]:
        if not self._runs_path.is_file():
            return []
        try:
            return json.loads(self._runs_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save_runs(self, runs: list[dict]) -> None:
        self._runs_path.write_text(
            json.dumps(runs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def sync_from_exports(self, export_dir: Path | None = None) -> int:
        base = Path(export_dir) if export_dir is not None else self.export_dir
        folder = (base or Path(".")) / "stockbit"
        if not folder.is_dir():
            return 0
        self._ensure_client()
        total = 0
        for path in sorted(folder.glob("StockbitReports_*.md")):
            total += self.ingest_markdown(path)
        return total

    def ingest_markdown(self, path: Path) -> int:
        path = Path(path)
        if not path.is_file():
            return 0

        source_file = str(path.resolve())
        runs = self._load_runs()
        if any(r.get("source_file") == source_file for r in runs):
            return 0

        text = path.read_text(encoding="utf-8", errors="ignore")
        meta = _parse_markdown_meta(text)
        records = _parse_markdown_posts(text, source_file)
        if not records:
            return 0

        self._ensure_client()
        ingested_at = datetime.now(timezone.utc).isoformat()
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        for rec in records:
            doc_id = _doc_id(rec["posted_at"], rec["title"])
            body = rec.get("body") or ""
            doc_text = rec["title"] if not body else f"{rec['title']}\n{body}"
            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(
                {
                    "posted_at": rec["posted_at"],
                    "posted_date": rec["posted_date"],
                    "title": rec["title"][:500],
                    "url": rec.get("url") or "",
                    "source_file": source_file,
                    "ingested_at": ingested_at,
                    "window": meta.get("window") or "",
                }
            )

        batch = 200
        inserted = 0
        for i in range(0, len(ids), batch):
            chunk_ids = ids[i : i + batch]
            try:
                self._collection.upsert(
                    ids=chunk_ids,
                    documents=documents[i : i + batch],
                    metadatas=metadatas[i : i + batch],
                )
                inserted += len(chunk_ids)
            except Exception:  # noqa: BLE001
                for j, doc_id in enumerate(chunk_ids):
                    try:
                        self._collection.upsert(
                            ids=[doc_id],
                            documents=[documents[i + j]],
                            metadatas=[metadatas[i + j]],
                        )
                        inserted += 1
                    except Exception:  # noqa: BLE001
                        continue

        runs.append(
            {
                "source_file": source_file,
                "source_url": meta.get("source"),
                "window_label": meta.get("window"),
                "scraped_at": meta.get("scraped") or ingested_at,
                "post_count": inserted,
            }
        )
        self._save_runs(runs)
        return inserted

    def stats(self) -> str:
        if self._collection is None:
            folder = (self.export_dir or Path(".")) / "stockbit"
            md_count = (
                len(list(folder.glob("StockbitReports_*.md"))) if folder.is_dir() else 0
            )
            lines = [
                f"stockbit_vector={self.chroma_dir} (ChromaDB, warming up)",
                f"markdown_files={md_count}",
                "hint: bot sudah jalan; ingest Chroma di background (~79MB model sekali saja)",
            ]
            for run in reversed(self._load_runs()[-3:]):
                name = Path(run.get("source_file", "")).name
                lines.append(
                    f"- {name}: {run.get('post_count', 0)} post "
                    f"(scraped {(run.get('scraped_at') or '')[:19]})"
                )
            return "\n".join(lines)

        count = self._collection.count()
        if count == 0:
            return (
                "stockbit_vector=empty (ChromaDB)\n"
                "hint: scrape Stockbit Reports dulu, file .md akan di-ingest otomatis"
            )

        sample = self._collection.get(include=["metadatas"], limit=min(count, 5000))
        dates = sorted(
            {m.get("posted_date", "") for m in sample.get("metadatas") or [] if m}
        )
        date_range = f"{dates[0]} .. {dates[-1]}" if dates else "?"

        lines = [
            f"stockbit_vector={self.chroma_dir} (ChromaDB)",
            f"collection={_COLLECTION}",
            f"posts={count}",
            f"date_range={date_range}",
        ]
        for run in reversed(self._load_runs()[-3:]):
            name = Path(run.get("source_file", "")).name
            lines.append(
                f"- {name}: {run.get('post_count', 0)} post "
                f"(scraped {(run.get('scraped_at') or '')[:19]})"
            )
        return "\n".join(lines)

    def posts_by_date(self, target_date: str, *, limit: int = 50) -> str:
        if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", target_date or ""):
            return f"ERROR: tanggal tidak valid: {target_date!r}"

        if self._collection is None:
            md = self._posts_by_date_markdown(target_date, limit=limit)
            if not md.startswith("ERROR"):
                return md

        try:
            result = self._ensure_client().get(
                where={"posted_date": target_date},
                include=["documents", "metadatas"],
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma posts_by_date failed: %s", exc)
            return self._posts_by_date_markdown(target_date, limit=limit)

        metas = result.get("metadatas") or []
        docs = result.get("documents") or []
        if not metas:
            md = self._posts_by_date_markdown(target_date, limit=limit)
            if not md.startswith("ERROR"):
                return md
            return (
                f"ERROR: tidak ada post Stockbit Reports untuk tanggal {target_date} "
                f"di vector store.\n\n{self.stats()}"
            )

        rows = list(zip(metas, docs))
        rows.sort(key=lambda x: x[0].get("posted_at", ""), reverse=True)

        blocks: list[str] = []
        for meta, doc in rows:
            chunk = f"## {meta.get('posted_at', '?')} WIB\n"
            url = meta.get("url") or ""
            if url:
                chunk += f"link: {url}\n"
            chunk += f"\n{doc or meta.get('title', '')}"
            blocks.append(chunk)

        tail = ""
        if len(rows) >= limit:
            tail = f"\n... (max {limit} post)"
        body = "\n\n---\n\n".join(blocks)
        return (
            f"OK: {len(rows)} post Stockbit Reports tanggal {target_date} (dari ChromaDB)\n"
            f"vector_store={self.chroma_dir}\n\n{body}{tail}"
        )

    def search(self, query: str, *, limit: int = 20) -> str:
        q = (query or "").strip()
        if len(q) < 2:
            return "ERROR: query terlalu pendek (min 2 karakter)"

        if self._collection is None:
            return (
                "ERROR: semantic search belum siap — Chroma sedang download model embedding "
                "pertama kali (~79MB, sekali saja). Bot tetap jalan; coba lagi 1–2 menit, "
                "atau tanya report per tanggal dulu.\n\n"
                f"{self.stats()}"
            )

        if self._collection.count() == 0:
            return f"ERROR: vector store kosong.\n\n{self.stats()}"

        result = self._collection.query(
            query_texts=[q],
            n_results=min(limit, 50),
            include=["documents", "metadatas", "distances"],
        )
        metas = (result.get("metadatas") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        if not metas:
            return f"OK: tidak ada post mirip {q!r}.\n\n{self.stats()}"

        lines = [f"OK: {len(metas)} post relevan untuk {q!r} (semantic search ChromaDB)", ""]
        for meta, doc in zip(metas, docs):
            title = meta.get("title") or (doc or "")[:120]
            line = f"- {meta.get('posted_date', '?')} {meta.get('posted_at', '')}: {title}"
            url = meta.get("url") or ""
            if url:
                line += f" ({url})"
            lines.append(line)
        return "\n".join(lines)

    def _posts_by_date_markdown(self, target_date: str, *, limit: int = 50) -> str:
        folder = (self.export_dir or Path(".")) / "stockbit"
        if not folder.is_dir():
            return (
                "ERROR: belum ada data Stockbit Reports.\n"
                "Jalankan scrape dulu: /ask scrape Stockbit Reports ..."
            )

        rows: list[dict[str, str]] = []
        for path in sorted(folder.glob("StockbitReports_*.md"), reverse=True):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for rec in _parse_markdown_posts(text, str(path.resolve())):
                if rec.get("posted_date") == target_date:
                    rows.append(rec)

        rows.sort(key=lambda r: r.get("posted_at", ""), reverse=True)
        rows = rows[:limit]
        if not rows:
            return (
                f"ERROR: tidak ada post Stockbit Reports untuk tanggal {target_date} "
                f"di file scrape lokal."
            )

        blocks: list[str] = []
        for rec in rows:
            chunk = f"## {rec.get('posted_at', '?')} WIB\n"
            url = rec.get("url") or ""
            if url:
                chunk += f"link: {url}\n"
            body = rec.get("body") or ""
            chunk += f"\n{rec.get('title', '')}"
            if body:
                chunk += f"\n{body}"
            blocks.append(chunk)

        tail = ""
        if len(rows) >= limit:
            tail = f"\n... (max {limit} post)"
        body = "\n\n---\n\n".join(blocks)
        return (
            f"OK: {len(rows)} post Stockbit Reports tanggal {target_date} "
            f"(dari markdown scrape)\n\n{body}{tail}"
        )


def _doc_id(posted_at: str, title: str) -> str:
    raw = f"{posted_at}|{title}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:32]


def _parse_markdown_meta(text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in text.splitlines()[:12]:
        for key, pattern in _META_RE.items():
            match = pattern.match(line.strip())
            if match:
                meta[key] = match.group(1).strip()
    return meta


def _parse_markdown_posts(text: str, source_file: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current_header: str | None = None
    current_url: str | None = None
    body_lines: list[str] = []

    def flush() -> None:
        nonlocal current_header, current_url, body_lines
        if not current_header:
            return
        title = ""
        extra_body: list[str] = []
        for line in body_lines:
            s = line.strip()
            if not s:
                continue
            if not title:
                title = s
            else:
                extra_body.append(s)
        if not title:
            current_header = None
            current_url = None
            body_lines = []
            return
        posted_date = current_header[:10]
        records.append(
            {
                "posted_at": current_header,
                "posted_date": posted_date,
                "title": title,
                "body": "\n".join(extra_body) if extra_body else "",
                "url": current_url or "",
            }
        )
        current_header = None
        current_url = None
        body_lines = []

    for line in text.splitlines():
        header = _HEADER_RE.match(line.strip())
        if header:
            flush()
            current_header = header.group(1)
            continue
        if current_header is None:
            continue
        link = _LINK_RE.match(line.strip())
        if link:
            current_url = link.group(1)
            continue
        body_lines.append(line)

    flush()
    return records
