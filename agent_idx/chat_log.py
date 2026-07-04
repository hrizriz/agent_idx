from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from telegram import Message


class ChatLog:
    """Persistent Telegram chat collector across multiple groups (SQLite)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id INTEGER PRIMARY KEY,
                    chat_title TEXT,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    chat_title TEXT,
                    user_id INTEGER,
                    username TEXT,
                    full_name TEXT,
                    text TEXT,
                    has_document INTEGER DEFAULT 0,
                    document_name TEXT,
                    reply_to_message_id INTEGER,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (chat_id, message_id)
                )
                """
            )
            cols = {
                row["name"]
                for row in con.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "chat_title" not in cols:
                con.execute("ALTER TABLE messages ADD COLUMN chat_title TEXT")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_created "
                "ON messages(chat_id, created_at DESC)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_text "
                "ON messages(chat_id, text)"
            )

    def touch_chat(self, chat_id: int, chat_title: str | None = None) -> None:
        """Register a group even when the update has no text (e.g. bot joined)."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO chats (chat_id, chat_title, first_seen, last_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    chat_title=COALESCE(excluded.chat_title, chats.chat_title),
                    last_seen=excluded.last_seen
                """,
                (chat_id, chat_title, ts, ts),
            )

    def add_message(
        self,
        *,
        chat_id: int,
        message_id: int,
        chat_title: str | None = None,
        user_id: int | None = None,
        username: str | None = None,
        full_name: str | None = None,
        text: str | None = None,
        has_document: bool = False,
        document_name: str | None = None,
        reply_to_message_id: int | None = None,
        created_at: str | None = None,
    ) -> None:
        body = (text or "").strip()
        ts = created_at or datetime.now(timezone.utc).isoformat()
        self.touch_chat(chat_id, chat_title)
        if not body and not has_document:
            return
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO messages (
                    chat_id, message_id, chat_title, user_id, username, full_name, text,
                    has_document, document_name, reply_to_message_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, message_id) DO UPDATE SET
                    text=excluded.text,
                    chat_title=COALESCE(excluded.chat_title, messages.chat_title),
                    has_document=excluded.has_document,
                    document_name=excluded.document_name
                """,
                (
                    chat_id,
                    message_id,
                    chat_title,
                    user_id,
                    username,
                    full_name,
                    body,
                    1 if has_document else 0,
                    document_name,
                    reply_to_message_id,
                    ts,
                ),
            )

    def add_from_telegram_message(
        self, message: Message, chat_id: int, chat_title: str | None = None
    ) -> None:
        text = (message.text or message.caption or "").strip()
        doc = message.document
        user = message.from_user
        full_name = None
        username = None
        user_id = None
        if user:
            user_id = user.id
            username = user.username
            full_name = " ".join(
                p for p in [user.first_name, user.last_name] if p
            ).strip()

        created = None
        if message.date:
            created = message.date.astimezone(timezone.utc).isoformat()

        self.add_message(
            chat_id=chat_id,
            message_id=message.message_id,
            chat_title=chat_title,
            user_id=user_id,
            username=username,
            full_name=full_name,
            text=text,
            has_document=bool(doc),
            document_name=doc.file_name if doc else None,
            reply_to_message_id=(
                message.reply_to_message.message_id if message.reply_to_message else None
            ),
            created_at=created,
        )

    def list_chats(self) -> str:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT c.chat_id,
                       COALESCE(c.chat_title, m.chat_title) AS chat_title,
                       COALESCE(m.n, 0) AS n,
                       c.first_seen,
                       c.last_seen,
                       m.last_message_at
                FROM chats c
                LEFT JOIN (
                    SELECT chat_id,
                           MAX(chat_title) AS chat_title,
                           COUNT(*) AS n,
                           MAX(created_at) AS last_message_at
                    FROM messages
                    GROUP BY chat_id
                ) m ON m.chat_id = c.chat_id
                ORDER BY c.last_seen DESC
                """
            ).fetchall()
        if not rows:
            return "(no rows) Belum ada group yang terhubung"
        lines = [
            f"connected_chats={len(rows)}",
            "note: messages=0 berarti bot sudah di group tapi belum ada teks/file "
            "yang sempat di-collect (sering karena privacy mode masih ON).",
        ]
        for row in rows:
            title = row["chat_title"] or "(no title)"
            last_msg = row["last_message_at"] or "-"
            lines.append(
                f"- chat_id={row['chat_id']} | {title} | "
                f"messages={row['n']} | last_seen={row['last_seen']} | "
                f"last_message={last_msg}"
            )
        return "\n".join(lines)

    def recent(self, chat_id: int | None = None, limit: int = 30) -> str:
        limit = max(1, min(int(limit), 100))
        with self._connect() as con:
            if chat_id is None:
                rows = con.execute(
                    """
                    SELECT chat_id, chat_title, created_at, username, full_name,
                           text, document_name
                    FROM messages
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [limit],
                ).fetchall()
                scope = "all_chats"
            else:
                rows = con.execute(
                    """
                    SELECT chat_id, chat_title, created_at, username, full_name,
                           text, document_name
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY message_id DESC
                    LIMIT ?
                    """,
                    [chat_id, limit],
                ).fetchall()
                scope = f"chat_id={chat_id}"
        if not rows:
            return f"(no rows) Belum ada chat tersimpan ({scope})"
        lines = [f"{scope}, recent={len(rows)} (terbaru dulu)"]
        for row in rows:
            lines.append(_format_row(row))
        return "\n".join(lines)

    def search(
        self,
        query: str,
        chat_id: int | None = None,
        limit: int = 20,
    ) -> str:
        q = (query or "").strip()
        if not q:
            raise ValueError("query is required")
        limit = max(1, min(int(limit), 50))
        like = f"%{q}%"
        with self._connect() as con:
            if chat_id is None:
                rows = con.execute(
                    """
                    SELECT chat_id, chat_title, created_at, username, full_name,
                           text, document_name
                    FROM messages
                    WHERE (
                        text LIKE ? COLLATE NOCASE
                        OR document_name LIKE ? COLLATE NOCASE
                        OR username LIKE ? COLLATE NOCASE
                        OR full_name LIKE ? COLLATE NOCASE
                        OR chat_title LIKE ? COLLATE NOCASE
                    )
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [like, like, like, like, like, limit],
                ).fetchall()
                scope = "all_chats"
            else:
                rows = con.execute(
                    """
                    SELECT chat_id, chat_title, created_at, username, full_name,
                           text, document_name
                    FROM messages
                    WHERE chat_id = ?
                      AND (
                        text LIKE ? COLLATE NOCASE
                        OR document_name LIKE ? COLLATE NOCASE
                        OR username LIKE ? COLLATE NOCASE
                        OR full_name LIKE ? COLLATE NOCASE
                        OR chat_title LIKE ? COLLATE NOCASE
                      )
                    ORDER BY message_id DESC
                    LIMIT ?
                    """,
                    [chat_id, like, like, like, like, like, limit],
                ).fetchall()
                scope = f"chat_id={chat_id}"
        if not rows:
            return f"(no rows) Tidak ada chat cocok untuk query={q!r} ({scope})"
        lines = [f"{scope}, query={q!r}, hits={len(rows)}"]
        for row in rows:
            lines.append(_format_row(row))
        return "\n".join(lines)

    def stats(self, chat_id: int | None = None) -> str:
        with self._connect() as con:
            if chat_id is None:
                row = con.execute(
                    "SELECT "
                    "(SELECT COUNT(*) FROM messages) AS n, "
                    "(SELECT COUNT(*) FROM chats) AS chats"
                ).fetchone()
                return (
                    f"total_messages={row['n']}, connected_chats={row['chats']}, "
                    f"db={self.db_path}"
                )
            row = con.execute(
                "SELECT COUNT(*) AS n, MAX(chat_title) AS title, "
                "MIN(created_at) AS first_at, MAX(created_at) AS last_at "
                "FROM messages WHERE chat_id = ?",
                [chat_id],
            ).fetchone()
            return (
                f"chat_id={chat_id}, title={row['title'] or '-'}, "
                f"messages={row['n']}, first={row['first_at']}, "
                f"last={row['last_at']}"
            )


def _format_row(row: sqlite3.Row) -> str:
    who = row["username"] or row["full_name"] or "unknown"
    title = row["chat_title"] or str(row["chat_id"])
    doc = f" [file:{row['document_name']}]" if row["document_name"] else ""
    text = (row["text"] or "").replace("\n", " ")
    return f"- [{title}|{row['chat_id']}] {row['created_at']} @{who}: {text}{doc}"
