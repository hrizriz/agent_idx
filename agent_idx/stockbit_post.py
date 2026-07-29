"""Stage + run Stockbit stream posts (confirmation-gated)."""
from __future__ import annotations

_PENDING_POSTS: dict[int, dict] = {}


def stage_post(chat_key: int, text: str) -> str:
    body = (text or "").strip()
    if not body:
        return "ERROR: teks post kosong"
    if len(body) > 2000:
        return "ERROR: teks post terlalu panjang (max 2000 karakter)"
    _PENDING_POSTS[int(chat_key)] = {"text": body}
    preview = body if len(body) <= 280 else body[:277] + "..."
    return (
        "STAGED (menunggu konfirmasi user)\n"
        "job=stockbit_create_post\n"
        f"teks:\n{preview}\n"
        "CATATAN: post BELUM dikirim. User harus menyetujui (ya/tidak) di Telegram."
    )


def has_pending_post(chat_key: int) -> bool:
    return int(chat_key) in _PENDING_POSTS


def peek_pending_post(chat_key: int) -> dict | None:
    return _PENDING_POSTS.get(int(chat_key))


def describe_pending_post(chat_key: int) -> str:
    job = peek_pending_post(chat_key)
    if not job:
        return "(tidak ada post tertunda)"
    body = job.get("text") or ""
    preview = body if len(body) <= 400 else body[:397] + "..."
    return f"Post Stockbit tertunda:\n\n{preview}"


def clear_pending_post(chat_key: int) -> bool:
    return _PENDING_POSTS.pop(int(chat_key), None) is not None


def run_pending_post(chat_key: int, *, create_fn) -> str:
    """create_fn(text) -> str  (usually browser.create_post)."""
    job = _PENDING_POSTS.pop(int(chat_key), None)
    if not job:
        return "ERROR: tidak ada post tertunda"
    return create_fn(job.get("text") or "")
