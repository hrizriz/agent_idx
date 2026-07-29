"""Access to the agent_idx project directory (allowlisted).

Reads are direct. Writes are STAGED and require explicit user confirmation
(via Telegram) before they touch disk — the agent can only *propose* changes.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Never expose these to the LLM / Telegram.
_BLOCKED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "cookie.txt",
    "cookies.txt",
}
_BLOCKED_DIR_PARTS = {
    ".git",
    "__pycache__",
    "node_modules",
    "stockbit_profile",  # browser profile / cookies
    "chroma",  # binary vector db
    ".venv",
    "venv",
}
_BLOCKED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".exe",
    ".dll",
    ".so",
    ".pyd",
    ".db",
    ".sqlite",
    ".sqlite3",
}

_MAX_LIST = 200
_MAX_READ_CHARS = 24_000
_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".env.example",
    ".gitignore",
    ".sql",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".log",
}


def _is_blocked_path(path: Path) -> bool:
    name = path.name.lower()
    if name in {n.lower() for n in _BLOCKED_NAMES}:
        return True
    if name.startswith(".env") and name != ".env.example":
        return True
    parts = {p.lower() for p in path.parts}
    if parts & {d.lower() for d in _BLOCKED_DIR_PARTS}:
        return True
    if path.suffix.lower() in _BLOCKED_SUFFIXES:
        return True
    return False


def resolve_project_path(rel: str | None) -> Path:
    """Resolve a relative path under PROJECT_ROOT. Raises ValueError if outside."""
    raw = (rel or ".").strip().replace("\\", "/")
    if raw.startswith("/"):
        raw = raw.lstrip("/")
    # Disallow absolute Windows paths sneaking in.
    if re.match(r"^[A-Za-z]:", raw):
        raise ValueError("Path absolut tidak diizinkan. Pakai path relatif dari agent_idx.")
    if ".." in Path(raw).parts:
        raise ValueError("Path '..' tidak diizinkan.")

    root = PROJECT_ROOT.resolve()
    target = (root / raw).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Di luar folder agent_idx: {rel!r}") from exc
    if _is_blocked_path(target):
        raise ValueError(f"Path diblokir (sensitif/binary): {rel!r}")
    return target


def list_project_dir(path: str = ".", *, recursive: bool = False, limit: int = 100) -> str:
    try:
        target = resolve_project_path(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not target.exists():
        return f"ERROR: tidak ada: {path}"
    if not target.is_dir():
        return f"ERROR: bukan folder: {path}"

    limit = max(1, min(int(limit), _MAX_LIST))
    root = PROJECT_ROOT.resolve()
    lines = [f"root={root}", f"listing={target.relative_to(root) or '.'}", f"recursive={recursive}"]
    entries: list[str] = []

    if recursive:
        for p in sorted(target.rglob("*")):
            if _is_blocked_path(p):
                continue
            # Skip anything under blocked dir segments
            if any(part.lower() in {d.lower() for d in _BLOCKED_DIR_PARTS} for part in p.parts):
                continue
            rel = p.relative_to(root).as_posix()
            kind = "dir" if p.is_dir() else "file"
            size = "" if p.is_dir() else f" size={p.stat().st_size}"
            entries.append(f"{kind}\t{rel}{size}")
            if len(entries) >= limit:
                break
    else:
        for p in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if _is_blocked_path(p):
                continue
            if p.name.lower() in {d.lower() for d in _BLOCKED_DIR_PARTS}:
                continue
            rel = p.relative_to(root).as_posix()
            kind = "dir" if p.is_dir() else "file"
            size = "" if p.is_dir() else f" size={p.stat().st_size}"
            entries.append(f"{kind}\t{rel}{size}")
            if len(entries) >= limit:
                break

    lines.append(f"count={len(entries)} (limit={limit})")
    lines.extend(entries)
    if len(entries) >= limit:
        lines.append("... truncated")
    return "\n".join(lines)


def read_project_file(path: str, *, max_chars: int = 12_000) -> str:
    try:
        target = resolve_project_path(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not target.exists():
        return f"ERROR: tidak ada: {path}"
    if target.is_dir():
        return (
            f"ERROR: {path} adalah folder. Pakai list_project_dir dulu.\n"
            + list_project_dir(path, recursive=False, limit=50)
        )
    if _is_blocked_path(target):
        return f"ERROR: file diblokir: {path}"

    suffix = target.suffix.lower()
    # Allow extensionless small text files (e.g. LICENSE) and known text types.
    if suffix and suffix not in _TEXT_SUFFIXES and target.name.lower() not in {
        "dockerfile",
        "makefile",
        "license",
        "licence",
    }:
        # Parquet/binary: only report metadata
        if suffix in {".parquet", ".pkl", ".joblib", ".bin", ".pt", ".pth"}:
            st = target.stat()
            return (
                f"FILE binary: {target.relative_to(PROJECT_ROOT).as_posix()}\n"
                f"size={st.st_size}\n"
                "Konten binary tidak ditampilkan. "
                "Untuk parquet saham pakai tools data (run_sql / get_stock_history)."
            )
        return f"ERROR: tipe file tidak didukung untuk dibaca sebagai teks: {suffix}"

    max_chars = max(500, min(int(max_chars), _MAX_READ_CHARS))
    try:
        text = target.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return f"ERROR: file bukan teks UTF-8: {path}"

    # Defense in depth: scrub obvious secrets if somehow allowed through.
    if re.search(r"(?i)(api[_-]?key|token|password|secret)\s*=", text):
        if target.name.lower().startswith(".env") or "secret" in target.name.lower():
            return f"ERROR: file sensitif diblokir: {path}"

    rel = target.relative_to(PROJECT_ROOT).as_posix()
    truncated = len(text) > max_chars
    body = text[:max_chars]
    header = f"FILE: {rel}\nchars={len(text)}"
    if truncated:
        header += f" shown={max_chars} (truncated)"
    return f"{header}\n\n{body}"


# --------------------------------------------------------------------------- #
# Write staging (confirmation-gated)                                          #
# --------------------------------------------------------------------------- #

_MAX_WRITE_CHARS = 200_000
_WRITE_MODES = {"overwrite", "append", "create"}
# Only allow staging writes to text-like files (never binaries/parquet/db).
_WRITABLE_SUFFIXES = _TEXT_SUFFIXES | {".env.example"}
_WRITABLE_NAMES = {"dockerfile", "makefile", "license", "licence"}

# Pending, unconfirmed writes keyed by chat/session. Applied only on approval.
_PENDING_WRITES: dict[int, list[dict]] = {}
_WRITE_SEQ: dict[int, int] = {}


def _check_writable(target: Path) -> None:
    """Raise ValueError if target is not an allowed text write target."""
    suffix = target.suffix.lower()
    name = target.name.lower()
    if suffix:
        if suffix not in _WRITABLE_SUFFIXES:
            raise ValueError(
                f"Tulis hanya untuk file teks (mis. .py .md .txt .json .csv). "
                f"Suffix tidak diizinkan: {suffix}"
            )
    elif name not in _WRITABLE_NAMES:
        raise ValueError(f"File tanpa ekstensi tidak diizinkan untuk ditulis: {name}")


def _diff_preview(old: str, new: str, rel: str, *, max_lines: int = 80) -> str:
    diff = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
            n=2,
        )
    )
    if not diff:
        return "(tidak ada perbedaan konten)"
    shown = diff[:max_lines]
    text = "\n".join(shown)
    if len(diff) > max_lines:
        text += f"\n... (+{len(diff) - max_lines} baris diff lagi)"
    return text


def stage_project_write(
    chat_key: int,
    path: str,
    content: str,
    mode: str = "overwrite",
) -> str:
    """Stage a write for later confirmation. Does NOT touch disk."""
    mode = (mode or "overwrite").strip().lower()
    if mode not in _WRITE_MODES:
        return f"ERROR: mode tidak dikenal: {mode!r} (pakai overwrite|append|create)"
    if content is None:
        return "ERROR: content kosong"
    if len(content) > _MAX_WRITE_CHARS:
        return f"ERROR: konten terlalu besar ({len(content)} char, max {_MAX_WRITE_CHARS})"

    try:
        target = resolve_project_path(path)
        _check_writable(target)
    except ValueError as exc:
        return f"ERROR: {exc}"

    exists = target.exists()
    if exists and target.is_dir():
        return f"ERROR: {path} adalah folder, bukan file."
    if mode == "create" and exists:
        return f"ERROR: file sudah ada, pakai mode overwrite/append: {path}"

    old = ""
    if exists:
        try:
            old = target.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            old = ""

    if mode == "append":
        new_content = (old + ("\n" if old and not old.endswith("\n") else "") + content)
        action = "APPEND"
    else:
        new_content = content
        action = "OVERWRITE" if exists else "CREATE"

    rel = target.relative_to(PROJECT_ROOT).as_posix()
    preview = _diff_preview(old, new_content, rel)

    seq = _WRITE_SEQ.get(chat_key, 0) + 1
    _WRITE_SEQ[chat_key] = seq
    wid = f"w{seq}"
    entry = {
        "id": wid,
        "rel": rel,
        "abspath": target,
        "mode": mode,
        "action": action,
        "content": new_content,
        "exists": exists,
    }
    _PENDING_WRITES.setdefault(chat_key, []).append(entry)

    return (
        f"STAGED (menunggu konfirmasi user) [{wid}]\n"
        f"aksi={action} file={rel} mode={mode} bytes={len(new_content.encode('utf-8'))}\n"
        f"--- preview diff ---\n{preview}\n"
        "CATATAN: perubahan BELUM ditulis. User harus menyetujui (ya/tidak) di Telegram."
    )


def has_pending_writes(chat_key: int) -> bool:
    return bool(_PENDING_WRITES.get(chat_key))


def peek_pending_writes(chat_key: int) -> list[dict]:
    return list(_PENDING_WRITES.get(chat_key, []))


def clear_pending_writes(chat_key: int) -> int:
    entries = _PENDING_WRITES.pop(chat_key, [])
    return len(entries)


def describe_pending_writes(chat_key: int) -> str:
    entries = _PENDING_WRITES.get(chat_key, [])
    if not entries:
        return "(tidak ada perubahan tertunda)"
    lines = [f"{len(entries)} perubahan menunggu konfirmasi:"]
    for e in entries:
        lines.append(f"• [{e['id']}] {e['action']} {e['rel']} (mode={e['mode']})")
        preview = _diff_preview(
            e["abspath"].read_text(encoding="utf-8-sig")
            if e["exists"] and e["abspath"].is_file()
            else "",
            e["content"],
            e["rel"],
            max_lines=30,
        )
        lines.append(preview)
    return "\n".join(lines)


def apply_pending_writes(chat_key: int) -> str:
    """Write all staged changes to disk, then clear the queue."""
    entries = _PENDING_WRITES.pop(chat_key, [])
    if not entries:
        return "Tidak ada perubahan untuk diterapkan."

    results: list[str] = []
    for e in entries:
        target: Path = e["abspath"]
        try:
            # Re-validate at apply time (defense in depth).
            _check_writable(target)
            if _is_blocked_path(target):
                raise ValueError("path diblokir")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(e["content"], encoding="utf-8")
            results.append(f"OK [{e['id']}] {e['action']} {e['rel']}")
        except Exception as exc:  # noqa: BLE001
            results.append(f"GAGAL [{e['id']}] {e['rel']}: {exc}")
    return "\n".join(results)
