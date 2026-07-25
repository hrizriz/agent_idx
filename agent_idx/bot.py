from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Sequence

from telegram import Document, Message, MessageEntity, Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from agent_idx.agent import AnalystAgent
from agent_idx.chat_log import ChatLog
from agent_idx.stockbit_store import StockbitReportsStore
from agent_idx.config import Settings
from agent_idx.data import StockDataStore
from agent_idx.format import to_telegram_html
from agent_idx.knowledge import KnowledgeBase
from agent_idx.pdf_util import extract_pdf_text

logger = logging.getLogger(__name__)

_HISTORY: dict[int, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=20))
# Cache PDFs seen in each chat so "data ini" / reply tanpa attachment tetap kebaca.
_RECENT_PDFS: dict[int, deque[Document]] = defaultdict(lambda: deque(maxlen=10))
_MAX_TELEGRAM_CHARS = 4000
_DOC_REF = re.compile(
    r"\b(pdf|dokumen|document|file|data\s+ini|ini\s+gak|baca\s+ini|"
    r"lampiran|attachment)\b",
    re.I,
)

# Interactive Stockbit login: chat_id -> {step, username?, resume?}
# step: "username" | "password" | "otp"
_PENDING_STOCKBIT: dict[int, dict[str, Any]] = {}
# Agent request waiting for Stockbit login before resume.
_STOCKBIT_RESUME: dict[int, str] = {}
_STOCKBIT_DEFAULT_RESUME = (
    "Pakai tool stockbit_read untuk membaca halaman Stockbit yang sudah login, "
    "lalu ringkas isinya. Jangan pakai tools data pasar parquet."
)


def _is_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Only the interaction group (CHAT_ID) may trigger bot replies."""
    allowed: int | None = context.application.bot_data.get("allowed_chat_id")
    if allowed is None:
        return True
    chat = update.effective_chat
    if chat is None or chat.id != allowed:
        return False
    return True


def _should_collect(chat) -> bool:
    """Collect from any group/supergroup the bot is in."""
    return chat is not None and chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}


def _chunk(text: str, size: int = _MAX_TELEGRAM_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    parts: list[str] = []
    while text:
        parts.append(text[:size])
        text = text[size:]
    return parts


async def log_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Collect chat from every group; only interaction group gets replies."""
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return

    if _should_collect(chat):
        _cache_pdfs_from_message(chat.id, msg)
        chat_log: ChatLog | None = context.application.bot_data.get("chat_log")
        pending = _PENDING_STOCKBIT.get(chat.id)
        # Never store username/password/OTP replies from interactive login.
        skip_body = pending is not None and pending.get("step") in {
            "username",
            "password",
            "otp",
        }
        if chat_log is not None:
            try:
                # Always register the group (even service/join events with empty text).
                chat_log.touch_chat(chat.id, chat.title)
                if not skip_body and not (msg.from_user and msg.from_user.is_bot):
                    chat_log.add_from_telegram_message(
                        msg, chat.id, chat_title=chat.title
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to persist chat message")

    pending = _PENDING_STOCKBIT.get(chat.id)
    if pending and pending.get("step") in {"username", "password", "otp"}:
        preview = f"[redacted stockbit {pending.get('step')}]"
    else:
        preview = (msg.text or msg.caption or "").replace("\n", " ")[:120]
    doc = msg.document
    logger.info(
        "Update chat_id=%s title=%r collect=%s msg_id=%s doc=%r text=%r",
        chat.id,
        chat.title,
        _should_collect(chat),
        msg.message_id,
        getattr(doc, "file_name", None),
        preview,
    )


def _is_pdf_document(doc: Document | None) -> bool:
    if not doc:
        return False
    name = (doc.file_name or "").lower()
    mime = (doc.mime_type or "").lower()
    return mime == "application/pdf" or name.endswith(".pdf")


def _cache_pdfs_from_message(chat_id: int, message: Message) -> None:
    for candidate in (message, message.reply_to_message):
        if candidate and _is_pdf_document(candidate.document):
            _push_recent_pdf(chat_id, candidate.document)


def _push_recent_pdf(chat_id: int, doc: Document) -> None:
    recent = _RECENT_PDFS[chat_id]
    # De-dupe by file_id, keep newest first.
    items = [d for d in recent if d.file_id != doc.file_id]
    items.insert(0, doc)
    recent.clear()
    recent.extend(items[:10])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_message is not None
    uname = context.bot.username or "bot"
    await update.effective_message.reply_text(
        "<b>Analis IDX</b> — data transaksi, berita, PDF, laporan.\n\n"
        "<b>Cara panggil di group:</b>\n"
        f"• <code>/ask pertanyaan</code>\n"
        f"• mention: <code>@{uname} pertanyaan</code> (pilih dari autocomplete)\n"
        "• reply pesan bot\n"
        "• kirim/reply PDF + tanya isinya\n\n"
        "Jika mention tidak direspons: @BotFather → /setprivacy → <b>Disable</b>, "
        "lalu kick & add bot lagi ke group.\n\n"
        "<b>Mode group:</b>\n"
        "• Group interaksi (CHAT_ID): tanya jawab di sini\n"
        "• Group lain: bot hanya <b>mengumpulkan</b> chat (diam), "
        "lalu bisa kamu tanya dari group interaksi\n\n"
        "Agar collect penuh di group pantau: @BotFather /setprivacy → <b>Disable</b>, "
        "lalu kick+add bot di group itu.\n\n"
        "<b>Stockbit:</b> browser guest-like (ephemeral).\n"
        "• /stockbit — login interaktif\n"
        "• /stockbit status — cek sudah login atau belum\n"
        "• /stockbit close — tutup sesi\n\n"
        "Perintah: /ask /stockbit /learn /chats /range /reset /help",
        parse_mode=ParseMode.HTML,
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_chat is not None
    assert update.effective_message is not None
    _HISTORY.pop(update.effective_chat.id, None)
    await update.effective_message.reply_text("Riwayat percakapan direset.")


async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_chat is not None
    assert update.effective_message is not None
    chat_id = update.effective_chat.id
    if chat_id in _PENDING_STOCKBIT:
        _PENDING_STOCKBIT.pop(chat_id, None)
        await update.effective_message.reply_text("Login Stockbit dibatalkan.")
        return
    await update.effective_message.reply_text("Tidak ada proses login yang aktif.")


async def range_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_message is not None
    store: StockDataStore = context.application.bot_data["store"]
    await update.effective_message.reply_text(store.list_date_range())


async def learn_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_message is not None
    knowledge: KnowledgeBase = context.application.bot_data["knowledge"]
    settings: Settings = context.application.bot_data["settings"]
    store: StockDataStore = context.application.bot_data["store"]
    args = [a.lower() for a in (context.args or [])]
    job = args[0] if args else "status"

    if job in {"all", "market", "news", "weekly", "run"}:
        status = await update.effective_message.reply_text(
            f"Menjalankan learning job: {job}..."
        )
        from agent_idx.jobs import (
            job_daily_market_digest,
            job_daily_news,
            job_weekly_lesson,
            run_all_learning_jobs,
        )

        which = "all" if job == "run" else job

        def _run():
            if which == "market":
                return [job_daily_market_digest(store, knowledge)]
            if which == "news":
                return [job_daily_news(knowledge)]
            if which == "weekly":
                return [job_weekly_lesson(knowledge)]
            return run_all_learning_jobs(store, knowledge)

        try:
            paths = await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001
            await status.edit_text(f"Gagal learning job: {exc}")
            return
        lines = [f"Selesai: {which}"]
        for path in paths:
            lines.append(f"- {path.name}")
        await status.edit_text("\n".join(lines))
        return

    topics = knowledge.list_topics()
    cron = (
        "Perintah:\n"
        "/learn — status materi\n"
        "/learn all | market | news | weekly — jalankan belajar sekarang\n\n"
        f"cron_enabled={settings.cron_enabled}\n"
        f"market: {settings.cron_market} ({settings.cron_timezone})\n"
        f"news:   {settings.cron_news}\n"
        f"weekly: {settings.cron_weekly}\n\n"
    )
    text = cron + topics
    for part in _chunk(text):
        await update.effective_message.reply_text(part)


async def chats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_message is not None
    chat_log: ChatLog = context.application.bot_data["chat_log"]
    settings: Settings = context.application.bot_data["settings"]
    text = (
        f"interaction_chat_id={settings.telegram_chat_id}\n"
        f"{chat_log.stats()}\n\n"
        f"{chat_log.list_chats()}\n\n"
        f"{chat_log.recent(limit=8)}"
    )
    await update.effective_message.reply_text(text[:4000])


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_message is not None
    question = " ".join(context.args or []).strip()
    if not question:
        await update.effective_message.reply_text(
            "Contoh: /ask Ringkas BBCA sepanjang 2022"
        )
        return
    await _handle_user_message(update, context, question)


async def _stockbit_status_text(chat_id: int) -> str:
    import os

    from agent_idx import browser_stockbit as _sb
    from agent_idx.browser_stockbit import get_browser

    lines: list[str] = []
    pending = _PENDING_STOCKBIT.get(chat_id)
    if pending:
        lines.append(f"pending_login=True step={pending.get('step')}")
    resume = _STOCKBIT_RESUME.get(chat_id)
    if resume:
        preview = resume.replace("\n", " ")[:100]
        lines.append(f"waiting_resume={preview!r}")

    if _sb._BROWSER is None:
        lines.extend(
            [
                "browser_ready=False",
                "login=not_started",
                "mode=ephemeral/guest-like",
                "hint: /stockbit untuk login interaktif",
            ]
        )
    else:
        headless = os.getenv("STOCKBIT_HEADLESS", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        from agent_idx.browser_stockbit import get_browser, run_sync

        lines.append(
            await asyncio.to_thread(
                run_sync, lambda: get_browser(headless=headless).status()
            )
        )
    return "\n".join(lines)


async def stockbit_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stockbit helpers: status | login (default)."""
    if not _is_allowed(update, context):
        return
    assert update.effective_chat is not None
    assert update.effective_message is not None
    chat_id = update.effective_chat.id
    args = context.args or []
    sub = args[0].lower() if args else "login"

    if sub in {"status", "cek", "check"}:
        text = await _stockbit_status_text(chat_id)
        await update.effective_message.reply_text(text)
        return

    if sub in {"close", "logout"}:
        from agent_idx.browser_stockbit import shutdown_browser

        _PENDING_STOCKBIT.pop(chat_id, None)
        await asyncio.to_thread(shutdown_browser)
        await update.effective_message.reply_text(
            "Browser Stockbit ditutup. Sesi login dihapus."
        )
        return

    if sub not in {"login", ""} and sub not in {"status", "cek", "check", "close", "logout"}:
        await update.effective_message.reply_text(
            "Subcommand tidak dikenal.\n"
            "Pakai: /stockbit status | /stockbit | /stockbit close"
        )
        return

    resume = " ".join(args[1:]).strip() if sub == "login" else " ".join(args).strip()
    if not resume:
        resume = _STOCKBIT_RESUME.get(chat_id, "")
    await _start_stockbit_login(update, context, resume_question=resume)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_chat is not None
    assert update.effective_message is not None
    chat_id = update.effective_chat.id
    if chat_id in _PENDING_STOCKBIT:
        await _continue_stockbit_login(update, context)
        return
    await _handle_addressed_message(update, context)


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """PDF dikirim langsung (caption boleh berisi pertanyaan / mention)."""
    await _handle_addressed_message(update, context)


async def _handle_addressed_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_message is not None

    message = update.effective_message
    chat_id = update.effective_chat.id if update.effective_chat else 0
    if chat_id in _PENDING_STOCKBIT:
        await _continue_stockbit_login(update, context)
        return

    text, entities = _text_and_entities(message)
    mentioned = _is_bot_mentioned(text, entities, context)
    replied_bot = _is_reply_to_bot(message, context)
    question = _strip_bot_mentions(text, entities, context)
    has_pdf = _find_pdf_document(message, chat_id, question or "pdf") is not None

    if not question:
        if has_pdf:
            question = "Ringkas dan jelaskan isi dokumen PDF ini."
        elif mentioned or replied_bot:
            await message.reply_text(
                "Ya, saya di sini.\n"
                f"Contoh: @{context.bot.username or 'bot'} ringkas BBCA 2024\n"
                "atau /ask ringkas BBCA 2024\n"
                "atau /stockbit untuk login Stockbit interaktif"
            )
            return
        else:
            return

    await _handle_user_message(update, context, question)


async def _start_stockbit_login(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    resume_question: str,
) -> None:
    assert update.effective_chat is not None
    assert update.effective_message is not None
    chat_id = update.effective_chat.id

    if chat_id in _PENDING_STOCKBIT:
        step = _PENDING_STOCKBIT[chat_id].get("step", "?")
        await update.effective_message.reply_text(
            f"Login Stockbit sudah menunggu input (step: {step}).\n"
            "Lanjutkan kirim data atau ketik /cancel."
        )
        return

    from agent_idx import browser_stockbit as _sb
    from agent_idx.browser_stockbit import get_browser, run_sync

    if _sb._BROWSER is not None and _sb._BROWSER.ready:
        st = await asyncio.to_thread(run_sync, lambda: get_browser().status())
        if "login=logged_in" in st:
            _STOCKBIT_RESUME.pop(chat_id, None)
            await update.effective_message.reply_text(
                "Sudah login ke Stockbit.\nMelanjutkan permintaan..."
            )
            if resume_question:
                await _handle_user_message(update, context, resume_question)
            return
        if "login=needs_otp" in st or "needs_otp=True" in st:
            _PENDING_STOCKBIT[chat_id] = {
                "step": "otp",
                "username": None,
                "resume": resume_question,
            }
            await update.effective_message.reply_text(
                "Stockbit menunggu <b>kode verifikasi</b>.\n\n"
                "Kirim kode OTP sekarang.\n"
                "Ketik /cancel untuk batal.",
                parse_mode=ParseMode.HTML,
            )
            return

    _PENDING_STOCKBIT[chat_id] = {
        "step": "username",
        "username": None,
        "resume": resume_question,
    }
    await update.effective_message.reply_text(
        "Login Stockbit (browser guest/ephemeral, hanya stockbit.com).\n\n"
        "Kirim <b>username / email</b> Stockbit sekarang.\n"
        "Ketik /cancel untuk batal.",
        parse_mode=ParseMode.HTML,
    )


async def _continue_stockbit_login(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    assert update.effective_chat is not None
    assert update.effective_message is not None
    chat_id = update.effective_chat.id
    pending = _PENDING_STOCKBIT.get(chat_id)
    if not pending:
        return

    text = (update.effective_message.text or "").strip()
    if text.lower() in {"/cancel", "cancel", "batal"}:
        _PENDING_STOCKBIT.pop(chat_id, None)
        await update.effective_message.reply_text("Login Stockbit dibatalkan.")
        return

    step = pending.get("step")
    if step == "username":
        pending["username"] = text
        pending["step"] = "password"
        # Best-effort delete username message.
        try:
            await update.effective_message.delete()
        except Exception:  # noqa: BLE001
            pass
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "Username diterima.\n"
                "Kirim <b>password</b> Stockbit sekarang (pesan akan dihapus bila bisa).\n"
                "/cancel untuk batal."
            ),
            parse_mode=ParseMode.HTML,
        )
        return

    if step == "password":
        username = pending.get("username") or ""
        password = text
        resume = pending.get("resume") or ""
        try:
            await update.effective_message.delete()
        except Exception:  # noqa: BLE001
            pass

        status = await context.bot.send_message(
            chat_id=chat_id, text="Menjalankan login Stockbit..."
        )
        import os

        from agent_idx.browser_stockbit import NEED_OTP, get_browser, run_sync

        headless = os.getenv("STOCKBIT_HEADLESS", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        def _login() -> str:
            browser = get_browser(headless=headless)
            browser.start()
            browser.open(url="https://stockbit.com/login")
            return browser.login(username, password)

        try:
            result = await asyncio.to_thread(run_sync, _login)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Stockbit login failed")
            _PENDING_STOCKBIT.pop(chat_id, None)
            await status.edit_text(f"Login gagal: {exc}")
            return

        if NEED_OTP in result:
            pending["step"] = "otp"
            pending["resume"] = resume
            await status.edit_text(
                "Password diterima. Stockbit minta <b>kode verifikasi</b> "
                "(email / SMS / authenticator).\n\n"
                "Kirim kode OTP sekarang.\n"
                "/cancel untuk batal.",
                parse_mode=ParseMode.HTML,
            )
            return

        _PENDING_STOCKBIT.pop(chat_id, None)
        if result.startswith("OK:"):
            _STOCKBIT_RESUME.pop(chat_id, None)
            if resume and resume != _STOCKBIT_DEFAULT_RESUME:
                await status.edit_text(
                    "Login Stockbit berhasil.\nMelanjutkan permintaan sebelumnya..."
                )
                await _handle_user_message(update, context, resume)
            else:
                await status.edit_text(
                    "Login Stockbit berhasil (sesi ephemeral, hilang saat bot restart).\n"
                    "Siap dipakai — coba /ask scrape Stockbit Reports ..."
                )
        else:
            await status.edit_text(
                f"Login gagal.\n{result[:500]}\n\n"
                "Coba /stockbit lagi. Jika ada captcha, set STOCKBIT_HEADLESS=false."
            )
        return

    if step == "otp":
        code = text
        resume = pending.get("resume") or ""
        try:
            await update.effective_message.delete()
        except Exception:  # noqa: BLE001
            pass

        status = await context.bot.send_message(
            chat_id=chat_id, text="Mengirim kode verifikasi..."
        )
        from agent_idx.browser_stockbit import get_browser, run_sync

        def _otp() -> str:
            browser = get_browser()
            if not browser.ready:
                return "ERROR: browser sudah tertutup. Jalankan /stockbit lagi."
            return browser.submit_otp(code)

        try:
            result = await asyncio.to_thread(run_sync, _otp)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Stockbit OTP failed")
            _PENDING_STOCKBIT.pop(chat_id, None)
            await status.edit_text(f"Verifikasi gagal: {exc}")
            return

        if result.startswith("OK:"):
            _PENDING_STOCKBIT.pop(chat_id, None)
            _STOCKBIT_RESUME.pop(chat_id, None)
            if resume and resume != _STOCKBIT_DEFAULT_RESUME:
                await status.edit_text(
                    "Login Stockbit berhasil.\nMelanjutkan permintaan sebelumnya..."
                )
                await _handle_user_message(update, context, resume)
            else:
                await status.edit_text(
                    "Login Stockbit berhasil (sesi ephemeral, hilang saat bot restart).\n"
                    "Siap dipakai — coba /ask scrape Stockbit Reports ..."
                )
            return

        # Stay on OTP step so user can retry another code.
        pending["step"] = "otp"
        await status.edit_text(
            f"Kode belum diterima.\n{result[:400]}\n\n"
            "Kirim kode verifikasi yang baru, atau /cancel lalu /stockbit ulang."
        )


def _text_and_entities(
    message: Message,
) -> tuple[str, Sequence[MessageEntity]]:
    if message.text:
        return message.text, message.entities or []
    if message.caption:
        return message.caption, message.caption_entities or []
    return "", []


def _entity_slice(text: str, entity: MessageEntity) -> str:
    encoded = text.encode("utf-16-le")
    start = entity.offset * 2
    end = (entity.offset + entity.length) * 2
    return encoded[start:end].decode("utf-16-le")


def _is_bot_mentioned(
    text: str,
    entities: Sequence[MessageEntity],
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    bot_username = (context.bot.username or "").lower()
    bot_id = context.bot.id

    for entity in entities:
        if entity.type == MessageEntity.MENTION:
            mention = _entity_slice(text, entity).lstrip("@").lower()
            if bot_username and mention == bot_username:
                return True
        elif entity.type == MessageEntity.TEXT_MENTION:
            if entity.user and entity.user.id == bot_id:
                return True

    if bot_username and re.search(rf"@{re.escape(bot_username)}\b", text or "", re.I):
        return True
    return False


def _is_reply_to_bot(message: Message, context: ContextTypes.DEFAULT_TYPE) -> bool:
    replied = message.reply_to_message
    if not replied or not replied.from_user:
        return False
    return replied.from_user.id == context.bot.id


def _strip_bot_mentions(
    text: str,
    entities: Sequence[MessageEntity],
    context: ContextTypes.DEFAULT_TYPE,
) -> str:
    bot_username = (context.bot.username or "").lower()
    bot_id = context.bot.id
    if not text:
        return ""

    spans: list[tuple[int, int]] = []
    for entity in entities:
        keep = False
        if entity.type == MessageEntity.MENTION:
            mention = _entity_slice(text, entity).lstrip("@").lower()
            keep = bool(bot_username and mention == bot_username)
        elif entity.type == MessageEntity.TEXT_MENTION:
            keep = bool(entity.user and entity.user.id == bot_id)
        if keep:
            spans.append((entity.offset, entity.offset + entity.length))

    if spans:
        encoded = text.encode("utf-16-le")
        for start, end in sorted(spans, reverse=True):
            encoded = encoded[: start * 2] + encoded[end * 2 :]
        text = encoded.decode("utf-16-le")

    if bot_username:
        text = re.sub(rf"@{re.escape(bot_username)}\b", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _find_pdf_document(message: Message, chat_id: int, question: str) -> Document | None:
    # 1) Attachment on this message
    if _is_pdf_document(message.document):
        return message.document

    # 2) Reply-to message attachment
    if message.reply_to_message and _is_pdf_document(message.reply_to_message.document):
        return message.reply_to_message.document

    # 3) User refers to "pdf/dokumen/data ini" -> latest PDF seen in this chat
    if _DOC_REF.search(question or ""):
        from agent_idx.composite_report import is_composite_report_query
        from agent_idx.stockbit_reports import is_stockbit_scrape_query

        if is_composite_report_query(question) or is_stockbit_scrape_query(question):
            return None
        recent = _RECENT_PDFS.get(chat_id)
        if recent:
            logger.info(
                "Using cached PDF %s for chat_id=%s",
                recent[0].file_name,
                chat_id,
            )
            return recent[0]

    return None


async def _attach_pdf_context(
    update: Update, context: ContextTypes.DEFAULT_TYPE, question: str
) -> tuple[str, str | None]:
    """Returns (question_with_pdf, error_message_or_None)."""
    assert update.effective_message is not None
    assert update.effective_chat is not None

    chat_id = update.effective_chat.id
    doc = _find_pdf_document(update.effective_message, chat_id, question)
    if not doc:
        from agent_idx.composite_report import is_composite_report_query
        from agent_idx.stockbit_reports import is_stockbit_scrape_query

        if is_composite_report_query(question) or is_stockbit_scrape_query(question):
            return question, None
        if _DOC_REF.search(question or ""):
            return question, (
                "Saya belum bisa mengakses file PDF-nya.\n\n"
                "Cara yang pasti:\n"
                "1) Kirim ulang PDF dengan caption pertanyaan, atau\n"
                "2) Reply langsung ke pesan PDF, lalu ketik pertanyaan "
                f"(@{context.bot.username or 'bot'} ringkas laporan ini)\n\n"
                "Jangan hanya bilang \"data ini\" tanpa reply ke file-nya."
            )
        return question, None

    export_dir: Path = context.application.bot_data["export_dir"]
    inbox = export_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w.\-]+", "_", doc.file_name or f"{doc.file_id}.pdf")
    path = inbox / f"{doc.file_unique_id}_{safe_name}"

    try:
        logger.info("Downloading PDF file_id=%s name=%s", doc.file_id, doc.file_name)
        file = await context.bot.get_file(doc.file_id)
        await file.download_to_drive(custom_path=str(path))
        pdf_text = await asyncio.to_thread(extract_pdf_text, path)
        logger.info("Extracted PDF chars=%s file=%s", len(pdf_text), path.name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF download/extract failed")
        return question, f"Gagal membaca PDF <b>{doc.file_name}</b>: {exc}"

    if pdf_text.startswith("(PDF tidak berisi teks"):
        return question, (
            f"PDF <b>{doc.file_name}</b> terunduh, tetapi tidak ada teks yang bisa "
            "diekstrak (kemungkinan hasil scan/gambar)."
        )

    enriched = (
        f"{question}\n\n"
        "INSTRUKSI: Jawab HANYA berdasarkan isi PDF di bawah. "
        "Jangan memakai data transaksi saham lain (mis. BBCA) kecuali diminta.\n\n"
        f"--- ISI PDF ({doc.file_name}) ---\n"
        f"{pdf_text}\n"
        f"--- END PDF ---"
    )
    return enriched, None


async def _send_status(
    update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
):
    """Reply if possible; fall back to plain send (e.g. after deleted OTP message)."""
    assert update.effective_chat is not None
    chat_id = update.effective_chat.id
    message = update.effective_message
    if message is not None:
        try:
            return await message.reply_text(text)
        except BadRequest:
            pass
    return await context.bot.send_message(chat_id=chat_id, text=text)


async def _handle_user_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE, question: str
) -> None:
    assert update.effective_chat is not None

    # Ack immediately so user always sees a response.
    status = await _send_status(update, context, "Sedang memproses...")

    question, pdf_error = await _attach_pdf_context(update, context, question)
    if pdf_error:
        await status.edit_text(pdf_error, parse_mode=ParseMode.HTML)
        return

    agent: AnalystAgent = context.application.bot_data["agent"]
    chat_id = update.effective_chat.id
    # Jangan bawa history analisis saham lama saat user minta baca dokumen.
    history: list[dict[str, Any]]
    if "--- ISI PDF" in question:
        history = []
    else:
        history = list(_HISTORY[chat_id])

    logger.info("Question from chat_id=%s: %s", chat_id, question[:300])

    from agent_idx.agent import _REPORTS_URL, _SCRAPE_VERB, extract_stockbit_url

    if (
        _SCRAPE_VERB.search(question)
        and (_REPORTS_URL.search(question) or extract_stockbit_url(question))
    ):
        await status.edit_text(
            "Scrape Stockbit stream sedang berjalan...\n"
            "Scroll otomatis — rentang panjang butuh beberapa menit.\n"
            "Mohon tunggu, jangan kirim perintah lain."
        )

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        result: AgentResult = await asyncio.to_thread(agent.ask, question, history)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent failed")
        await status.edit_text(
            f"Gagal memproses pertanyaan: {exc}\n"
            "Pastikan gemini-web2api jalan di LLM_BASE_URL."
        )
        return

    # Store shorter history text (without huge PDF body).
    history_question = question
    if "--- ISI PDF" in history_question:
        history_question = history_question.split("--- ISI PDF", 1)[0].strip()
        history_question += " [dengan lampiran PDF]"

    if result.need_stockbit_login:
        resume_q = result.resume_question or question
        _STOCKBIT_RESUME[chat_id] = resume_q

        if chat_id in _PENDING_STOCKBIT:
            step = _PENDING_STOCKBIT[chat_id].get("step", "?")
            await status.edit_text(
                f"Stockbit butuh login.\n"
                f"Login sedang berjalan (step: {step}). Lanjutkan atau /cancel.\n"
                "Cek sesi: /stockbit status"
            )
            return

        from agent_idx import browser_stockbit as _sb
        from agent_idx.browser_stockbit import get_browser, run_sync

        if _sb._BROWSER is not None and _sb._BROWSER.ready:
            st = await asyncio.to_thread(run_sync, lambda: get_browser().status())
            if "login=logged_in" in st:
                _STOCKBIT_RESUME.pop(chat_id, None)
                await status.edit_text("Sesi Stockbit masih aktif. Melanjutkan...")
                await _handle_user_message(update, context, resume_q)
                return
            if "login=needs_otp" in st or "needs_otp=True" in st:
                _PENDING_STOCKBIT[chat_id] = {
                    "step": "otp",
                    "username": None,
                    "resume": resume_q,
                }
                await status.edit_text(
                    "Stockbit menunggu kode OTP.\n"
                    "Kirim kode verifikasi di chat, atau /cancel."
                )
                return

        await status.edit_text(
            "Stockbit membutuhkan login.\n\n"
            "Sesi browser guest — hilang saat bot restart atau /stockbit close.\n"
            "• /stockbit status — cek sesi\n"
            "• /stockbit — login interaktif\n\n"
            "Setelah login, ulangi pertanyaan tadi."
        )
        return

    answer = result.text
    _HISTORY[chat_id].append({"role": "user", "content": history_question})
    _HISTORY[chat_id].append({"role": "assistant", "content": answer})

    formatted = to_telegram_html(answer)
    parts = _chunk(formatted)
    await _send_html(status.edit_text, parts[0])
    for part in parts[1:]:
        msg = update.effective_message
        if msg is not None:
            try:
                await _send_html(msg.reply_text, part)
                continue
            except BadRequest:
                pass
        await context.bot.send_message(
            chat_id=chat_id, text=part, parse_mode=None
        )

    for path in result.files:
        await _send_file(update, context, path)


async def _send_file(
    update: Update, context: ContextTypes.DEFAULT_TYPE, path: Path
) -> None:
    assert update.effective_chat is not None
    if not path.is_file():
        return
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(
        chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT
    )
    try:
        with path.open("rb") as fh:
            message = update.effective_message
            if message is not None:
                try:
                    await message.reply_document(
                        document=fh,
                        filename=path.name,
                        caption=path.name,
                    )
                    return
                except BadRequest:
                    fh.seek(0)
            await context.bot.send_document(
                chat_id=chat_id,
                document=fh,
                filename=path.name,
                caption=path.name,
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed sending file %s", path)


async def _send_html(sender, text: str) -> None:
    try:
        await sender(text, parse_mode=ParseMode.HTML)
    except BadRequest:
        await sender(text, parse_mode=None)


async def _warm_stockbit_chroma(app: Application) -> None:
    store: StockbitReportsStore | None = app.bot_data.get("stockbit_store")
    export_dir: Path | None = app.bot_data.get("export_dir")
    if store is None or export_dir is None:
        return

    loop = asyncio.get_running_loop()

    def _sync() -> None:
        try:
            inserted = store.sync_from_exports(export_dir)
            logger.info("Stockbit Chroma background sync: %s new posts ingested", inserted)
        except Exception:
            logger.exception("Stockbit Chroma background sync failed")

    await loop.run_in_executor(None, _sync)


async def _post_init(app: Application) -> None:
    asyncio.create_task(_warm_stockbit_chroma(app))
    settings: Settings = app.bot_data["settings"]
    store: StockDataStore = app.bot_data["store"]
    knowledge: KnowledgeBase = app.bot_data["knowledge"]
    if not settings.cron_enabled:
        logger.info("Cron disabled (CRON_ENABLED=false)")
        return

    async def send_telegram(title: str, path: Path) -> None:
        chat_id = settings.telegram_chat_id
        if not chat_id:
            return
        await app.bot.send_message(
            chat_id=chat_id,
            text=f"<b>{title}</b>\n<code>{path.name}</code>",
            parse_mode=ParseMode.HTML,
        )
        if path.is_file() and path.stat().st_size < 8_000_000:
            with path.open("rb") as fh:
                await app.bot.send_document(
                    chat_id=chat_id,
                    document=fh,
                    filename=path.name,
                )

    from agent_idx.scheduler import build_scheduler

    scheduler = build_scheduler(
        settings, store, knowledge, send_telegram=send_telegram
    )
    scheduler.start()
    app.bot_data["scheduler"] = scheduler
    logger.info(
        "Cron started tz=%s market=%s news=%s weekly=%s",
        settings.cron_timezone,
        settings.cron_market,
        settings.cron_news,
        settings.cron_weekly,
    )


async def _post_shutdown(app: Application) -> None:
    scheduler = app.bot_data.get("scheduler")
    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            logger.exception("Scheduler shutdown failed")
        else:
            logger.info("Cron stopped")
    try:
        from agent_idx.browser_stockbit import shutdown_browser

        shutdown_browser()
    except Exception:  # noqa: BLE001
        logger.exception("Browser shutdown failed")


def build_application(settings: Settings, store: StockDataStore) -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty. Set it in .env")

    knowledge = KnowledgeBase(settings.knowledge_dir)
    chat_log = ChatLog(settings.chat_db_path)
    stockbit_store = StockbitReportsStore(
        settings.chroma_dir,
        export_dir=settings.export_dir,
        auto_sync=False,
    )
    agent = AnalystAgent(
        settings,
        store,
        knowledge=knowledge,
        chat_log=chat_log,
        stockbit_store=stockbit_store,
        chat_id=settings.telegram_chat_id,
    )
    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(True)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.bot_data["agent"] = agent
    app.bot_data["store"] = store
    app.bot_data["knowledge"] = knowledge
    app.bot_data["chat_log"] = chat_log
    app.bot_data["stockbit_store"] = stockbit_store
    app.bot_data["settings"] = settings
    app.bot_data["allowed_chat_id"] = settings.telegram_chat_id
    app.bot_data["export_dir"] = settings.export_dir

    # Collect + log every update first (group -1 runs before other handlers).
    app.add_handler(TypeHandler(Update, log_update), group=-1)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("stockbit", stockbit_cmd))
    app.add_handler(CommandHandler("learn", learn_cmd))
    app.add_handler(CommandHandler("chats", chats_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("range", range_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    return app


def run_bot(settings: Settings, store: StockDataStore) -> None:
    app = build_application(settings, store)
    logger.info(
        "Starting Telegram bot (allowed_chat_id=%s, knowledge=%s, cron=%s)",
        settings.telegram_chat_id,
        settings.knowledge_dir,
        settings.cron_enabled,
    )
    try:
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=False,
            close_loop=False,
        )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — forcing exit")
        raise
