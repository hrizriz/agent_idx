from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Sequence

from telegram import Document, Message, MessageEntity, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from agent_idx.agent import AgentResult, AnalystAgent
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
    r"\b(pdf|dokumen|document|report|laporan|file|data\s+ini|ini\s+gak|baca\s+ini|"
    r"lampiran|attachment)\b",
    re.I,
)


def _is_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    allowed: int | None = context.application.bot_data.get("allowed_chat_id")
    if allowed is None:
        return True
    chat = update.effective_chat
    if chat is None or chat.id != allowed:
        logger.warning("Ignored update from chat_id=%s", getattr(chat, "id", None))
        return False
    return True


def _chunk(text: str, size: int = _MAX_TELEGRAM_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    parts: list[str] = []
    while text:
        parts.append(text[:size])
        text = text[size:]
    return parts


async def log_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Debug + cache any PDF seen in the allowed chat."""
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return

    _cache_pdfs_from_message(chat.id, msg)

    preview = (msg.text or msg.caption or "").replace("\n", " ")[:120]
    doc = msg.document
    reply_doc = msg.reply_to_message.document if msg.reply_to_message else None
    logger.info(
        "Update chat_id=%s msg_id=%s doc=%r reply_doc=%r text=%r",
        chat.id,
        msg.message_id,
        getattr(doc, "file_name", None),
        getattr(reply_doc, "file_name", None),
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
        "<b>Belajar IDX (cron):</b> curriculum fundamental/charting + digest harian otomatis.\n"
        "Perintah: /ask /learn /range /reset /help",
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
    topics = knowledge.list_topics()
    cron = (
        f"cron_enabled={settings.cron_enabled}\n"
        f"market: {settings.cron_market} ({settings.cron_timezone})\n"
        f"news:   {settings.cron_news}\n"
        f"weekly: {settings.cron_weekly}\n\n"
    )
    text = cron + topics
    for part in _chunk(text):
        await update.effective_message.reply_text(part)


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


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
                "atau kirim/reply PDF lalu tanya isinya."
            )
            return
        else:
            return

    await _handle_user_message(update, context, question)


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


async def _handle_user_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE, question: str
) -> None:
    assert update.effective_chat is not None
    assert update.effective_message is not None

    # Ack immediately so user always sees a response.
    status = await update.effective_message.reply_text("Sedang memproses...")

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

    answer = result.text
    _HISTORY[chat_id].append({"role": "user", "content": history_question})
    _HISTORY[chat_id].append({"role": "assistant", "content": answer})

    formatted = to_telegram_html(answer)
    parts = _chunk(formatted)
    await _send_html(status.edit_text, parts[0])
    for part in parts[1:]:
        await _send_html(update.effective_message.reply_text, part)

    for path in result.files:
        await _send_file(update, context, path)


async def _send_file(
    update: Update, context: ContextTypes.DEFAULT_TYPE, path: Path
) -> None:
    assert update.effective_message is not None
    assert update.effective_chat is not None
    if not path.is_file():
        return
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_DOCUMENT
    )
    try:
        with path.open("rb") as fh:
            await update.effective_message.reply_document(
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


async def _post_init(app: Application) -> None:
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


def build_application(settings: Settings, store: StockDataStore) -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty. Set it in .env")

    knowledge = KnowledgeBase(settings.knowledge_dir)
    agent = AnalystAgent(settings, store, knowledge=knowledge)
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
    app.bot_data["settings"] = settings
    app.bot_data["allowed_chat_id"] = settings.telegram_chat_id
    app.bot_data["export_dir"] = settings.export_dir

    # Log every update first (group -1 runs before other handlers).
    app.add_handler(TypeHandler(Update, log_update), group=-1)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("learn", learn_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
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
