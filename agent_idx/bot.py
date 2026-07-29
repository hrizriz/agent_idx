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

from agent_idx.agent import AgentResult, AnalystAgent
from agent_idx.chat_log import ChatLog
from agent_idx.stockbit_store import StockbitReportsStore
from agent_idx.config import Settings
from agent_idx.data import StockDataStore
from agent_idx.format import to_telegram_html
from agent_idx.image_util import describe_image, is_image_path
from agent_idx.knowledge import KnowledgeBase
from agent_idx.pdf_util import extract_pdf_text

logger = logging.getLogger(__name__)

_HISTORY: dict[int, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=20))
# Serialize agent work per chat — avoid overlapping Stockbit/LLM jobs.
_CHAT_LOCKS: dict[int, asyncio.Lock] = {}
# Cache PDFs seen in each chat so "data ini" / reply tanpa attachment tetap kebaca.
_RECENT_PDFS: dict[int, deque[Document]] = defaultdict(lambda: deque(maxlen=10))
# Cache photos: (file_id, file_unique_id, file_name)
_RECENT_IMAGES: dict[int, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=10))
_MAX_TELEGRAM_CHARS = 4000
_DOC_REF = re.compile(
    r"\b(pdf|dokumen|document|file|data\s+ini|ini\s+gak|baca\s+ini|"
    r"lampiran|attachment)\b",
    re.I,
)
# "Jangan buat PDF" / "buatkan PDF" = instruksi output, BUKAN referensi lampiran.
_PDF_OUTPUT_TALK = re.compile(
    r"(?:"
    r"(?:jangan|ga\s*usah|gak\s*usah|tidak|no|stop|jangan\s+lagi)\b.{0,40}\bpdf\b"
    r"|\bpdf\b.{0,40}\b(?:jangan|ga\s*usah|gak\s*usah|tidak|diminta|kalau\s+ga|kalo\s+ga)\b"
    r"|\b(?:buat|buatkan|generate|kirim|export|bikin)\b.{0,30}\bpdf\b"
    r"|\bpdf\b.{0,30}\b(?:buat|buatkan|generate|kirim|export|bikin)\b"
    r")",
    re.I | re.S,
)
_PDF_NEVER_UNLESS_ASKED = re.compile(
    r"(?:"
    r"(?:jangan|ga\s*usah|gak\s*usah|tidak|no|stop)\b.{0,40}\bpdf\b"
    r"|\bpdf\b.{0,40}\b(?:jangan|ga\s*usah|gak\s*usah|tidak|diminta)\b"
    r")",
    re.I | re.S,
)
_IMG_REF = re.compile(
    r"\b(gambar|foto|photo|image|chart|grafik|screenshot|ss\b|png|jpeg|jpg)\b",
    re.I,
)
_TOOLS_LIST_REQUEST = re.compile(
    r"(?:\b(?:list|daftar|sebutkan|apa\s+(?:saja|aja))\b.{0,45}"
    r"\b(?:tools?|kemampuan|akses)\b|"
    r"\b(?:tools?|kemampuan)\b.{0,45}\b(?:list|daftar|apa\s+(?:saja|aja))\b)",
    re.I,
)
_IMAGE_DOC_MIME = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
}

# Interactive Stockbit login: chat_id -> {step, username?, resume?}
# step: "username" | "password" | "otp"
_PENDING_STOCKBIT: dict[int, dict[str, Any]] = {}

# Telegram chat_id -> project_fs staging key that has changes awaiting ya/tidak.
_AWAITING_WRITE_CONFIRM: dict[int, int] = {}
_CONFIRM_YES = {
    "ya", "iya", "y", "yes", "ok", "oke", "sip", "setuju", "lanjut",
    "terapkan", "apply", "gas", "boleh", "yoi",
}
_CONFIRM_NO = {
    "tidak", "no", "n", "batal", "cancel", "jangan", "gak", "ga",
    "nggak", "engga", "enggak", "stop",
}
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


def _should_skip_auto_file(path: Path) -> bool:
    """Don't attach internal knowledge digests (.md) — answer as chat text only."""
    try:
        resolved = path.resolve()
    except Exception:  # noqa: BLE001
        resolved = path
    name = resolved.name.lower()
    parts = {p.lower() for p in resolved.parts}
    if name.endswith(".md") and (
        "knowledge" in parts
        or name.startswith("news_")
        or name.startswith("market_")
        or name.startswith("weekly_")
    ):
        return True
    return False


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
        f"• <code>/council pertanyaan</code> — 3-agent (Researcher/Analyst/Critic)\n"
        f"• mention: <code>@{uname} pertanyaan</code> (pilih dari autocomplete)\n"
        "• reply pesan bot\n"
        "• kirim/reply PDF + tanya isinya\n"
        "• PDF + caption <b>simpan dokumen ini</b> → stage, konfirmasi ya/tidak ke DB\n"
        "• kirim/reply <b>foto / PNG / JPEG</b> (chart/screenshot) + tanya\n"
        "• minta ubah/buat file di project — agent <b>usul dulu</b>, "
        "kamu konfirmasi <b>ya/tidak</b> sebelum ditulis\n\n"
        "Jika mention tidak direspons: @BotFather → /setprivacy → <b>Disable</b>, "
        "lalu kick & add bot lagi ke group.\n\n"
        "<b>Mode group:</b>\n"
        "• Group interaksi (CHAT_ID): tanya jawab di sini\n"
        "• Group lain: bot hanya <b>mengumpulkan</b> chat (diam), "
        "lalu bisa kamu tanya dari group interaksi\n\n"
        "Agar collect penuh di group pantau: @BotFather /setprivacy → <b>Disable</b>, "
        "lalu kick+add bot di group itu.\n\n"
        "<b>Stockbit:</b> browser + profile persistent.\n"
        "• /stockbit — login interaktif\n"
        "• /stockbit status — cek sudah login atau belum\n"
        "• /stockbit close — tutup sesi\n\n"
        "Perintah: /ask /council /tools /stockbit /learn /chats /range /reset /help",
        parse_mode=ParseMode.HTML,
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


_TOOL_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "Data pasar & analisis",
        frozenset(
            {
                "describe_schema",
                "list_date_range",
                "get_stock_history",
                "summarize_period",
                "rank_foreign_flow",
                "run_sql",
                "run_backtest_scan",
            }
        ),
    ),
    (
        "Stockbit & fundamental",
        frozenset(
            {
                "stockbit_status",
                "stockbit_open",
                "stockbit_read",
                "stockbit_scrape",
                "stockbit_scrape_reports",
                "read_stockbit_scrape_date",
                "list_stockbit_scrape_files",
                "search_stockbit_reports",
                "stockbit_create_post",
                "farm_stockbit_fundamentals",
                "get_fundamentals",
            }
        ),
    ),
    (
        "Chart & teknikal",
        frozenset(
            {
                "farm_stockbit_charts",
                "get_stockbit_ohlcv",
                "query_chart_ohlcv",
                "chart_features",
                "sync_chart_db",
                "chart_db_stats",
                "get_technicals",
                "rebuild_market_from_stockbit",
            }
        ),
    ),
    (
        "Berita & knowledge",
        frozenset(
            {
                "search_news",
                "search_news_sentiment",
                "search_knowledge",
                "list_knowledge_topics",
                "save_learning_note",
                "run_learning_job",
            }
        ),
    ),
    (
        "Chat & dokumen",
        frozenset(
            {
                "list_collected_chats",
                "search_chat_history",
                "recent_chat_history",
                "chat_log_stats",
                "stage_document_ingest",
                "search_documents",
                "doc_store_stats",
            }
        ),
    ),
    (
        "File, laporan & transform",
        frozenset(
            {
                "create_pdf_report",
                "export_query_csv",
                "list_project_dir",
                "read_project_file",
                "write_project_file",
                "run_data_transform",
            }
        ),
    ),
)


def _available_tools_text() -> str:
    """Build the user-facing list from the tools actually registered."""
    from agent_idx.tools import TOOL_DEFINITIONS

    registered = {
        str(tool.get("function", {}).get("name", "")).strip()
        for tool in TOOL_DEFINITIONS
    }
    registered.discard("")
    listed: set[str] = set()
    lines = [f"Tools yang bisa saya akses ({len(registered)}):"]
    for title, names in _TOOL_GROUPS:
        available = sorted(registered & names)
        if not available:
            continue
        listed.update(available)
        lines.extend(("", f"{title}:", " • " + "\n • ".join(available)))
    other = sorted(registered - listed)
    if other:
        lines.extend(("", "Lainnya:", " • " + "\n • ".join(other)))
    lines.extend(
        (
            "",
            "Catatan: tool tulis/publish/farming batch tetap meminta konfirmasi "
            "sebelum dijalankan.",
        )
    )
    return "\n".join(lines)


async def tools_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_message is not None
    for part in _chunk(_available_tools_text()):
        await update.effective_message.reply_text(part)


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
    if chat_id in _AWAITING_WRITE_CONFIRM:
        from agent_idx import (
            chart_farm,
            doc_ingest,
            fund_farm,
            project_fs,
            stockbit_post,
            stockbit_reports,
        )

        key = _AWAITING_WRITE_CONFIRM.pop(chat_id)
        if stockbit_post.clear_pending_post(key):
            await update.effective_message.reply_text("Post Stockbit dibatalkan.")
            return
        if chart_farm.clear_pending_job(key):
            await update.effective_message.reply_text("Job farming dibatalkan.")
            return
        if fund_farm.clear_pending_job(key):
            await update.effective_message.reply_text(
                "Job farm fundamentals dibatalkan."
            )
            return
        if stockbit_reports.clear_pending_stream_scrape(key):
            await update.effective_message.reply_text("Stream scrape dibatalkan.")
            return
        if doc_ingest.clear_pending(key):
            await update.effective_message.reply_text(
                "Simpan dokumen dibatalkan — DB tidak diubah."
            )
            return
        n = project_fs.clear_pending_writes(key)
        await update.effective_message.reply_text(
            f"Perubahan file dibatalkan ({n} dibuang, tidak ada yang ditulis)."
        )
        return
    if chat_id in _PENDING_STOCKBIT:
        _PENDING_STOCKBIT.pop(chat_id, None)
        await update.effective_message.reply_text("Login Stockbit dibatalkan.")
        return
    await update.effective_message.reply_text("Tidak ada proses login/perubahan yang aktif.")


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
        f"market:  {settings.cron_market} ({settings.cron_timezone})\n"
        f"reports: {settings.cron_reports} days={settings.cron_reports_days} "
        f"(@StockbitReports + https://stockbit.com/Stockbit)\n"
        f"farm:    {settings.cron_farm} tf={settings.cron_farm_timeframes} "
        f"(Chartbit stockbit.com/symbol/{{T}}/chartbit)\n"
        f"fund:    {settings.cron_fundamentals} "
        f"(Jumat 07:00; semua IDX; financials,keystats,profile)\n"
        f"news:    {settings.cron_news}\n"
        f"weekly:  {settings.cron_weekly}\n\n"
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


async def council_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_message is not None
    question = " ".join(context.args or []).strip()
    if not question:
        await update.effective_message.reply_text(
            "3-agent council (Researcher → Analyst → Critic).\n"
            "Contoh: /council Analisis BBCA YTD — netral atau menarik?"
        )
        return
    await _handle_user_message(update, context, question, use_council=True)


async def _stockbit_status_text(chat_id: int) -> str:
    from agent_idx import browser_stockbit as _sb

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
                "hint: /stockbit untuk login interaktif",
            ]
        )
    else:
        from agent_idx.browser_stockbit import run_sync
        from agent_idx.tools import _stockbit_browser, _stockbit_headless

        lines.append(
            await asyncio.to_thread(
                run_sync, lambda: _stockbit_browser(headless=_stockbit_headless()).status()
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


def _pending_write_keys(update: Update, context: ContextTypes.DEFAULT_TYPE) -> list[int]:
    """Candidate staging keys used by dispatch_tool for this chat."""
    keys: list[int] = []
    if update.effective_chat is not None:
        keys.append(update.effective_chat.id)
    settings: Settings | None = context.application.bot_data.get("settings")
    if settings is not None and settings.telegram_chat_id:
        keys.append(settings.telegram_chat_id)
    # De-dupe, preserve order.
    seen: set[int] = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


async def _prompt_write_confirm(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    """If the agent staged changes/jobs, ask the user to confirm. Returns True if asked."""
    from agent_idx import (
        chart_farm,
        doc_ingest,
        fund_farm,
        project_fs,
        stockbit_post,
        stockbit_reports,
    )

    assert update.effective_chat is not None
    chat_id = update.effective_chat.id
    for key in _pending_write_keys(update, context):
        if stockbit_post.has_pending_post(key):
            _AWAITING_WRITE_CONFIRM[chat_id] = key
            for part in _chunk(stockbit_post.describe_pending_post(key)):
                await context.bot.send_message(chat_id=chat_id, text=part)
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "Kirim post ini ke Stockbit?\n"
                    "• balas <b>ya</b> untuk publish\n"
                    "• balas <b>tidak</b> untuk batal"
                ),
                parse_mode=ParseMode.HTML,
            )
            return True
        if chart_farm.has_pending_job(key):
            _AWAITING_WRITE_CONFIRM[chat_id] = key
            for part in _chunk(chart_farm.describe_pending_job(key)):
                await context.bot.send_message(chat_id=chat_id, text=part)
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "Jalankan farming ini?\n"
                    "• balas <b>ya</b> untuk mulai\n"
                    "• balas <b>tidak</b> untuk batal"
                ),
                parse_mode=ParseMode.HTML,
            )
            return True
        if fund_farm.has_pending_job(key):
            _AWAITING_WRITE_CONFIRM[chat_id] = key
            for part in _chunk(fund_farm.describe_pending_job(key)):
                await context.bot.send_message(chat_id=chat_id, text=part)
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "Jalankan farm fundamentals (pace pelan)?\n"
                    "• balas <b>ya</b> untuk mulai\n"
                    "• balas <b>tidak</b> untuk batal"
                ),
                parse_mode=ParseMode.HTML,
            )
            return True
        if stockbit_reports.has_pending_stream_scrape(key):
            _AWAITING_WRITE_CONFIRM[chat_id] = key
            for part in _chunk(
                stockbit_reports.describe_pending_stream_scrape(key)
            ):
                await context.bot.send_message(chat_id=chat_id, text=part)
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "Jalankan Stream scrape ini?\n"
                    "• balas <b>ya</b> untuk mulai\n"
                    "• balas <b>tidak</b> untuk batal"
                ),
                parse_mode=ParseMode.HTML,
            )
            return True
        if doc_ingest.has_pending(key):
            _AWAITING_WRITE_CONFIRM[chat_id] = key
            for part in _chunk(doc_ingest.describe_pending(key)):
                await context.bot.send_message(chat_id=chat_id, text=part)
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "Simpan dokumen ini ke DuckDB + knowledge/docs?\n"
                    "• balas <b>ya</b> untuk commit\n"
                    "• balas <b>tidak</b> untuk batal\n"
                    "(tanpa konfirmasi = tidak ada write ke DB)"
                ),
                parse_mode=ParseMode.HTML,
            )
            return True
        if project_fs.has_pending_writes(key):
            _AWAITING_WRITE_CONFIRM[chat_id] = key
            desc = project_fs.describe_pending_writes(key)
            header = (
                "Agent ingin MENGUBAH file berikut (belum ditulis):\n\n"
            )
            footer = (
                "\n\nSetujui perubahan ini?\n"
                "• balas <b>ya</b> untuk menerapkan\n"
                "• balas <b>tidak</b> untuk membatalkan"
            )
            body = header + desc
            parts = _chunk(body)
            for part in parts:
                await context.bot.send_message(chat_id=chat_id, text=part)
            await context.bot.send_message(
                chat_id=chat_id, text=footer, parse_mode=ParseMode.HTML
            )
            return True
    return False


async def _handle_write_confirm(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    """Handle a ya/tidak reply for staged changes/jobs. Returns True if consumed."""
    from agent_idx import (
        chart_farm,
        doc_ingest,
        fund_farm,
        project_fs,
        stockbit_post,
        stockbit_reports,
    )

    assert update.effective_chat is not None
    assert update.effective_message is not None
    chat_id = update.effective_chat.id
    key = _AWAITING_WRITE_CONFIRM.get(chat_id)
    if key is None:
        return False

    is_post = stockbit_post.has_pending_post(key)
    is_job = chart_farm.has_pending_job(key)
    is_fund = fund_farm.has_pending_job(key)
    is_stream = stockbit_reports.has_pending_stream_scrape(key)
    is_doc = doc_ingest.has_pending(key)
    # Nothing staged anymore (e.g. cleared elsewhere) — drop the flag.
    if (
        not is_post
        and not is_job
        and not is_fund
        and not is_stream
        and not is_doc
        and not project_fs.has_pending_writes(key)
    ):
        _AWAITING_WRITE_CONFIRM.pop(chat_id, None)
        return False

    answer = (update.effective_message.text or "").strip().lower()
    token = re.sub(r"[^\w]", "", answer.split()[0]) if answer.split() else ""

    if token in _CONFIRM_YES:
        _AWAITING_WRITE_CONFIRM.pop(chat_id, None)
        if is_post:
            await _run_stockbit_post(update, context, key)
            return True
        if is_job:
            await _run_farm_job(update, context, key)
            return True
        if is_fund:
            await _run_fund_job(update, context, key)
            return True
        if is_stream:
            await _run_stream_scrape(update, context, key)
            return True
        if is_doc:
            result = await asyncio.to_thread(doc_ingest.apply_pending, key)
            await update.effective_message.reply_text(
                "Dokumen di-commit:\n" + result
            )
            return True
        result = await asyncio.to_thread(project_fs.apply_pending_writes, key)
        await update.effective_message.reply_text(
            "Perubahan diterapkan:\n" + result
        )
        return True
    if token in _CONFIRM_NO:
        _AWAITING_WRITE_CONFIRM.pop(chat_id, None)
        if is_post:
            stockbit_post.clear_pending_post(key)
            await update.effective_message.reply_text("Post Stockbit dibatalkan.")
            return True
        if is_job:
            chart_farm.clear_pending_job(key)
            await update.effective_message.reply_text("Farming dibatalkan.")
            return True
        if is_fund:
            fund_farm.clear_pending_job(key)
            await update.effective_message.reply_text(
                "Farm fundamentals dibatalkan."
            )
            return True
        if is_stream:
            stockbit_reports.clear_pending_stream_scrape(key)
            await update.effective_message.reply_text("Stream scrape dibatalkan.")
            return True
        if is_doc:
            doc_ingest.clear_pending(key)
            await update.effective_message.reply_text(
                "Simpan dokumen dibatalkan — DB tidak diubah."
            )
            return True
        n = project_fs.clear_pending_writes(key)
        await update.effective_message.reply_text(
            f"Perubahan dibatalkan ({n} file)."
        )
        return True

    if is_post:
        what = "post Stockbit"
    elif is_job:
        what = "farming chart"
    elif is_fund:
        what = "farm fundamentals"
    elif is_stream:
        what = "Stream scrape"
    elif is_doc:
        what = "simpan dokumen ke DB"
    else:
        what = "perubahan file"
    await update.effective_message.reply_text(
        f"Menunggu konfirmasi {what}.\n"
        "Balas 'ya' untuk lanjut atau 'tidak' untuk batal."
    )
    return True


async def _run_stockbit_post(
    update: Update, context: ContextTypes.DEFAULT_TYPE, key: int
) -> None:
    """Publish an approved Stockbit post."""
    from agent_idx import stockbit_post

    assert update.effective_chat is not None
    chat_id = update.effective_chat.id
    job = stockbit_post.peek_pending_post(key)
    preview = (job or {}).get("text") or ""
    if len(preview) > 80:
        preview = preview[:77] + "..."

    status = await context.bot.send_message(
        chat_id=chat_id,
        text=f"Mengirim post ke Stockbit: {preview!r} ...",
    )

    def _do() -> str:
        from agent_idx.browser_stockbit import run_sync
        from agent_idx.tools import _stockbit_browser, _stockbit_headless

        def _post() -> str:
            headless = _stockbit_headless()
            browser = _stockbit_browser(headless)
            if not browser.ready:
                browser.start()
            return stockbit_post.run_pending_post(key, create_fn=browser.create_post)

        # MUST stay on stockbit-pw thread (Playwright affinity).
        return run_sync(_post)

    try:
        result = await asyncio.to_thread(_do)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stockbit post failed")
        stockbit_post.clear_pending_post(key)
        await _safe_edit(status, f"Post gagal: {exc}")
        return

    if "NEED_STOCKBIT_CREDENTIALS" in (result or ""):
        stockbit_post.clear_pending_post(key)
        await _safe_edit(
            status,
            "Post gagal: sesi Stockbit belum login. Jalankan /stockbit dulu.",
        )
        return

    parts = _chunk(result)
    await _safe_edit(status, parts[0])
    for part in parts[1:]:
        await context.bot.send_message(chat_id=chat_id, text=part)


async def _run_stream_scrape(
    update: Update, context: ContextTypes.DEFAULT_TYPE, key: int
) -> None:
    """Execute a confirmed Stream scrape."""
    from agent_idx import stockbit_reports

    assert update.effective_chat is not None
    chat_id = update.effective_chat.id
    status = await context.bot.send_message(
        chat_id=chat_id,
        text="Stream scrape dimulai...",
    )
    store = context.application.bot_data.get("stockbit_store")
    try:
        result = await asyncio.to_thread(
            stockbit_reports.run_pending_stream_scrape,
            key,
            stockbit_store=store,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stream scrape failed")
        stockbit_reports.clear_pending_stream_scrape(key)
        await _safe_edit(status, f"Stream scrape gagal: {exc}")
        return

    parts = _chunk(result)
    await _safe_edit(status, parts[0])
    for part in parts[1:]:
        await context.bot.send_message(chat_id=chat_id, text=part)


async def _run_fund_job(
    update: Update, context: ContextTypes.DEFAULT_TYPE, key: int
) -> None:
    """Execute an approved fundamentals-farming job."""
    from agent_idx import fund_farm

    assert update.effective_chat is not None
    chat_id = update.effective_chat.id
    job = fund_farm.peek_pending_job(key)
    est = job.get("est_min") if job else None

    status = await context.bot.send_message(
        chat_id=chat_id,
        text=(
            "Farm fundamentals Stockbit dimulai"
            + (f" (estimasi ~{est} menit)" if est else "")
            + "...\nPace pelan — jangan kirim perintah lain sampai selesai."
        ),
    )

    loop = asyncio.get_running_loop()
    last_sent = 0.0

    def on_progress(msg: str) -> None:
        nonlocal last_sent
        now = loop.time()
        if now - last_sent < 20:
            return
        last_sent = now
        asyncio.run_coroutine_threadsafe(
            _safe_edit(status, f"Fund farm berjalan...\n{msg}"), loop
        )

    try:
        summary = await asyncio.to_thread(
            fund_farm.run_pending_job, key, on_progress=on_progress
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fund farming failed")
        fund_farm.clear_pending_job(key)
        await _safe_edit(status, f"Fund farm gagal: {exc}")
        return

    parts = _chunk(summary)
    await _safe_edit(status, parts[0])
    for part in parts[1:]:
        await context.bot.send_message(chat_id=chat_id, text=part)


async def _run_farm_job(
    update: Update, context: ContextTypes.DEFAULT_TYPE, key: int
) -> None:
    """Execute an approved chart-farming job, streaming progress to Telegram."""
    from agent_idx import chart_farm

    assert update.effective_chat is not None
    chat_id = update.effective_chat.id
    job = chart_farm.peek_pending_job(key)
    est = job.get("est_min") if job else None

    status = await context.bot.send_message(
        chat_id=chat_id,
        text=(
            "Farming chart Stockbit dimulai"
            + (f" (estimasi ~{est} menit)" if est else "")
            + "...\nJangan kirim perintah lain sampai selesai."
        ),
    )

    loop = asyncio.get_running_loop()
    last_sent = 0.0

    def on_progress(msg: str) -> None:
        nonlocal last_sent
        now = loop.time()
        if now - last_sent < 20:  # throttle Telegram edits
            return
        last_sent = now
        asyncio.run_coroutine_threadsafe(
            _safe_edit(status, f"Farming berjalan...\n{msg}"), loop
        )

    try:
        summary = await asyncio.to_thread(
            chart_farm.run_pending_job, key, on_progress=on_progress
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chart farming failed")
        chart_farm.clear_pending_job(key)
        await _safe_edit(status, f"Farming gagal: {exc}")
        return

    parts = _chunk(summary)
    await _safe_edit(status, parts[0])
    for part in parts[1:]:
        await context.bot.send_message(chat_id=chat_id, text=part)


async def _safe_edit(message, text: str) -> None:
    try:
        await message.edit_text(text)
    except BadRequest:
        pass


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, context):
        return
    assert update.effective_chat is not None
    assert update.effective_message is not None
    chat_id = update.effective_chat.id
    if chat_id in _AWAITING_WRITE_CONFIRM:
        if await _handle_write_confirm(update, context):
            return
    if chat_id in _PENDING_STOCKBIT:
        await _continue_stockbit_login(update, context)
        return
    await _handle_addressed_message(update, context)


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """PDF / image document dikirim langsung (caption boleh berisi pertanyaan)."""
    await _handle_addressed_message(update, context)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Compressed Telegram photo (JPEG)."""
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

    # Cache attachments even before we decide to answer.
    if _is_pdf_document(message.document):
        _push_recent_pdf(chat_id, message.document)
    img_meta = _image_meta_from_message(message)
    if img_meta:
        _push_recent_image(chat_id, img_meta)

    text, entities = _text_and_entities(message)
    mentioned = _is_bot_mentioned(text, entities, context)
    replied_bot = _is_reply_to_bot(message, context)
    question = _strip_bot_mentions(text, entities, context)
    has_pdf = _find_pdf_document(message, chat_id, question or "pdf") is not None
    has_image = _find_image_ref(message, chat_id, question or "gambar") is not None

    if not question:
        if has_pdf:
            question = "Ringkas dan jelaskan isi dokumen PDF ini."
        elif has_image:
            question = (
                "Analisis gambar ini untuk konteks saham/IDX "
                "(chart, indikator, level, dan insight singkat)."
            )
        elif mentioned or replied_bot:
            await message.reply_text(
                "Ya, saya di sini.\n"
                f"Contoh: @{context.bot.username or 'bot'} ringkas BBCA 2024\n"
                "atau /ask ringkas BBCA 2024\n"
                "atau kirim foto chart + caption pertanyaan\n"
                "atau /stockbit untuk login Stockbit interaktif"
            )
            return
        else:
            return

    if _TOOLS_LIST_REQUEST.search(question):
        for part in _chunk(_available_tools_text()):
            await message.reply_text(part)
        return

    # Preferensi "jangan buat PDF": ack + simpan, jangan salah jadi baca lampiran.
    if (
        _PDF_NEVER_UNLESS_ASKED.search(question)
        and len(question) < 120
        and not has_pdf
        and not has_image
    ):
        await _save_no_pdf_preference(update, context, question)
        return

    await _handle_user_message(update, context, question)


async def _save_no_pdf_preference(
    update: Update, context: ContextTypes.DEFAULT_TYPE, question: str
) -> None:
    """Ack user preference and persist as a learning note."""
    assert update.effective_message is not None
    knowledge = context.application.bot_data.get("knowledge")
    if knowledge is not None:
        try:
            await asyncio.to_thread(
                knowledge.save_note,
                "Preferensi user: jangan buat PDF",
                (
                    "User meminta: JANGAN membuat/mengirim file PDF kecuali "
                    "diminta secara eksplisit. Default output = teks di chat Telegram.\n\n"
                    f"Pesan user: {question}"
                ),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed saving no-PDF preference")

    await update.effective_message.reply_text(
        "Oke, dicatat.\n"
        "Mulai sekarang saya <b>tidak membuat PDF</b> kecuali kamu minta "
        "secara eksplisit (mis. \"buat PDF laporan BBCA\").\n"
        "Jawaban default = teks di chat.",
        parse_mode=ParseMode.HTML,
    )


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
        "Login Stockbit (browser + profile persistent, hanya stockbit.com).\n\n"
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
        from agent_idx.browser_stockbit import NEED_OTP, run_sync
        from agent_idx.tools import _stockbit_browser, _stockbit_headless

        def _login() -> str:
            browser = _stockbit_browser(headless=_stockbit_headless())
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
                    "Login Stockbit berhasil (sesi terverifikasi).\n"
                    "Melanjutkan permintaan sebelumnya..."
                )
                await _handle_user_message(update, context, resume)
            else:
                await status.edit_text(
                    "Login Stockbit berhasil (sesi persistent / profile).\n"
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
            from agent_idx.tools import _stockbit_browser

            browser = _stockbit_browser()
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
                    "Login Stockbit berhasil (sesi terverifikasi).\n"
                    "Melanjutkan permintaan sebelumnya..."
                )
                await _handle_user_message(update, context, resume)
            else:
                await status.edit_text(
                    "Login Stockbit berhasil (sesi persistent / profile).\n"
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
    #    Skip if they're talking ABOUT making/not making a PDF (output instruction).
    if _DOC_REF.search(question or "") and not _PDF_OUTPUT_TALK.search(question or ""):
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
        # "jangan buat pdf" / "buatkan pdf" = instruksi, bukan minta baca lampiran.
        if _PDF_OUTPUT_TALK.search(question or ""):
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


def _is_image_document(doc: Document | None) -> bool:
    if doc is None:
        return False
    mime = (doc.mime_type or "").lower()
    name = (doc.file_name or "").lower()
    if mime in _IMAGE_DOC_MIME:
        return True
    return name.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))


def _image_meta_from_message(message: Message) -> dict[str, str] | None:
    if message.photo:
        photo = message.photo[-1]  # largest
        return {
            "file_id": photo.file_id,
            "file_unique_id": photo.file_unique_id,
            "file_name": f"{photo.file_unique_id}.jpg",
            "kind": "photo",
        }
    if _is_image_document(message.document):
        doc = message.document
        assert doc is not None
        name = doc.file_name or f"{doc.file_unique_id}.png"
        return {
            "file_id": doc.file_id,
            "file_unique_id": doc.file_unique_id,
            "file_name": name,
            "kind": "document",
        }
    return None


def _push_recent_image(chat_id: int, meta: dict[str, str]) -> None:
    recent = _RECENT_IMAGES[chat_id]
    items = [d for d in recent if d["file_id"] != meta["file_id"]]
    items.insert(0, meta)
    recent.clear()
    recent.extend(items[:10])


def _find_image_ref(
    message: Message, chat_id: int, question: str
) -> dict[str, str] | None:
    direct = _image_meta_from_message(message)
    if direct:
        return direct

    if message.reply_to_message:
        replied = _image_meta_from_message(message.reply_to_message)
        if replied:
            return replied

    if _IMG_REF.search(question or ""):
        from agent_idx.composite_report import is_composite_report_query
        from agent_idx.stockbit_reports import is_stockbit_scrape_query

        if is_composite_report_query(question) or is_stockbit_scrape_query(question):
            return None
        recent = _RECENT_IMAGES.get(chat_id)
        if recent:
            return recent[0]
    return None


async def _attach_image_context(
    update: Update, context: ContextTypes.DEFAULT_TYPE, question: str
) -> tuple[str, str | None]:
    """Returns (question_with_image_description, error_or_None)."""
    assert update.effective_message is not None
    assert update.effective_chat is not None

    chat_id = update.effective_chat.id
    meta = _find_image_ref(update.effective_message, chat_id, question)
    if not meta:
        return question, None

    export_dir: Path = context.application.bot_data["export_dir"]
    inbox = export_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w.\-]+", "_", meta["file_name"])
    path = inbox / f"{meta['file_unique_id']}_{safe_name}"

    try:
        logger.info("Downloading image file_id=%s name=%s", meta["file_id"], safe_name)
        file = await context.bot.get_file(meta["file_id"])
        await file.download_to_drive(custom_path=str(path))
        if not is_image_path(path) and path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
        }:
            # Telegram photo often has no extension
            path = path.with_suffix(".jpg")
            await file.download_to_drive(custom_path=str(path))

        settings: Settings = context.application.bot_data["settings"]
        await status_typing(context, chat_id)
        description = await asyncio.to_thread(
            describe_image, path, settings, user_hint=question
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Image download/describe failed")
        return question, (
            f"Gagal membaca gambar <b>{meta['file_name']}</b>: {exc}\n"
            "Pastikan gemini-web2api jalan dengan dukungan multimodal "
            "(<code>py -3 -m gemini_web2api</code>)."
        )

    enriched = (
        f"{question}\n\n"
        "INSTRUKSI: Jawab berdasarkan deskripsi gambar di bawah "
        "(boleh gabungkan dengan data tools jika relevan).\n\n"
        f"--- ISI GAMBAR ({meta['file_name']}) ---\n"
        f"{description}\n"
        f"--- END GAMBAR ---"
    )
    return enriched, None


async def status_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except Exception:  # noqa: BLE001
        pass


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
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    question: str,
    *,
    use_council: bool = False,
) -> None:
    assert update.effective_chat is not None
    chat_id = update.effective_chat.id
    lock = _CHAT_LOCKS.setdefault(chat_id, asyncio.Lock())
    if lock.locked():
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                "Masih memproses pesan sebelumnya — antrian, tunggu sebentar..."
            )
    async with lock:
        await _handle_user_message_locked(
            update, context, question, use_council=use_council
        )


async def _handle_user_message_locked(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    question: str,
    *,
    use_council: bool = False,
) -> None:
    assert update.effective_chat is not None

    # Preferensi singkat "jangan buat PDF" via /ask juga.
    if (
        _PDF_NEVER_UNLESS_ASKED.search(question or "")
        and len(question or "") < 120
        and not (update.effective_message and update.effective_message.document)
    ):
        await _save_no_pdf_preference(update, context, question)
        return

    # Ack immediately so user always sees a response.
    status_msg = (
        "Council 3-agent sedang berpikir (Researcher → Analyst → Critic)..."
        if use_council
        else "Sedang memproses..."
    )
    status = await _send_status(update, context, status_msg)

    question, pdf_error = await _attach_pdf_context(update, context, question)
    if pdf_error:
        await status.edit_text(pdf_error, parse_mode=ParseMode.HTML)
        return

    msg = update.effective_message
    chat_id_pre = update.effective_chat.id
    maybe_img = _find_image_ref(msg, chat_id_pre, question) if msg else None
    if maybe_img:
        await status.edit_text("Membaca gambar (vision)...")

    question, img_error = await _attach_image_context(update, context, question)
    if img_error:
        await status.edit_text(img_error, parse_mode=ParseMode.HTML)
        return

    if maybe_img:
        await status.edit_text(status_msg)

    agent: AnalystAgent = context.application.bot_data["agent"]
    chat_id = update.effective_chat.id

    from agent_idx.context import select_history, should_drop_history, summarize_history_for_log

    raw_history = list(_HISTORY[chat_id])
    drop, drop_reason = should_drop_history(question, raw_history)
    history = select_history(question, raw_history)
    if drop:
        logger.info(
            "Dropping history for chat_id=%s reason=%s (%s)",
            chat_id,
            drop_reason,
            summarize_history_for_log(raw_history),
        )

    logger.info(
        "Question from chat_id=%s council=%s history_used=%s: %s",
        chat_id,
        use_council,
        len(history),
        question[:300],
    )

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
        if use_council:
            from agent_idx.multi_agent import MultiAgentCouncil

            settings: Settings = context.application.bot_data["settings"]
            council = MultiAgentCouncil(
                settings,
                context.application.bot_data["store"],
                knowledge=context.application.bot_data["knowledge"],
                chat_log=context.application.bot_data.get("chat_log"),
                stockbit_store=context.application.bot_data.get("stockbit_store"),
                chat_id=chat_id,
            )
            result: AgentResult = await asyncio.to_thread(council.ask, question, history)
        else:
            result = await asyncio.to_thread(agent.ask, question, history)
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
    if "--- ISI GAMBAR" in history_question:
        history_question = history_question.split("--- ISI GAMBAR", 1)[0].strip()
        history_question += " [dengan lampiran gambar]"

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
            "Sesi belum terverifikasi (atau expired).\n"
            "• /stockbit status — cek sesi\n"
            "• /stockbit — login interaktif (+ OTP jika diminta)\n\n"
            "Setelah login=logged_in di status, ulangi pertanyaan tadi.\n"
            "Jika Chartbit tetap kosong: set STOCKBIT_HEADLESS=false, restart bot."
        )
        return

    answer = result.text
    # Keep history lean so old long analyses don't dominate later turns.
    history_answer = answer if len(answer) <= 900 else answer[:900] + "…"
    _HISTORY[chat_id].append({"role": "user", "content": history_question})
    _HISTORY[chat_id].append({"role": "assistant", "content": history_answer})

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
        if _should_skip_auto_file(path):
            logger.info("Skip auto-send knowledge digest file: %s", path)
            continue
        await _send_file(update, context, path)

    # If the agent staged any file changes, ask the user to confirm before writing.
    await _prompt_write_confirm(update, context)


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
        # If HTML still invalid, strip tags and send as plain text.
        from agent_idx.format import to_plain_text

        await sender(to_plain_text(text), parse_mode=None)


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
        """Notify chat with digest text — no file attachment (teks saja)."""
        chat_id = settings.telegram_chat_id
        if not chat_id:
            return
        body = ""
        if path.is_file():
            try:
                body = path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:  # noqa: BLE001
                body = ""
        # Strip markdown fences / heavy headers for readable chat text.
        if body:
            body = re.sub(r"(?m)^```.*$", "", body)
            body = re.sub(r"(?m)^#+\s*", "", body)
            body = re.sub(r"\n{3,}", "\n\n", body).strip()
        header = f"{title}"
        text = f"{header}\n\n{body}" if body else header
        # Keep under Telegram limits; prefer content over attaching .md
        for part in _chunk(text):
            await app.bot.send_message(chat_id=chat_id, text=part)

    async def send_text(text: str) -> None:
        chat_id = settings.telegram_chat_id
        if not chat_id:
            return
        for part in _chunk(text):
            await app.bot.send_message(chat_id=chat_id, text=part)

    from agent_idx.scheduler import build_scheduler

    scheduler = build_scheduler(
        settings,
        store,
        knowledge,
        send_telegram=send_telegram,
        send_text=send_text,
        stockbit_store=app.bot_data.get("stockbit_store"),
    )
    scheduler.start()
    app.bot_data["scheduler"] = scheduler
    logger.info(
        "Cron started tz=%s market=%s reports=%s(days=%s) farm=%s(%s) news=%s weekly=%s",
        settings.cron_timezone,
        settings.cron_market,
        settings.cron_reports,
        settings.cron_reports_days,
        settings.cron_farm,
        settings.cron_farm_timeframes,
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
    app.add_handler(CommandHandler("tools", tools_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("council", council_cmd))
    app.add_handler(CommandHandler("stockbit", stockbit_cmd))
    app.add_handler(CommandHandler("learn", learn_cmd))
    app.add_handler(CommandHandler("chats", chats_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("range", range_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
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
