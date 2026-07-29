from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from agent_idx.config import Settings
from agent_idx.data import StockDataStore
from agent_idx.jobs import (
    job_daily_chart_farm,
    job_daily_market_digest,
    job_daily_news,
    job_daily_stockbit_reports,
    job_weekly_lesson,
    job_weekly_stockbit_fundamentals,
)
from agent_idx.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

_CRON_OFF = frozenset({"", "off", "false", "0", "disable", "disabled"})


def _cron_on(value: str | None) -> bool:
    return (value or "").strip().lower() not in _CRON_OFF


def build_scheduler(
    settings: Settings,
    store: StockDataStore,
    knowledge: KnowledgeBase,
    *,
    send_telegram=None,
    send_text=None,
    stockbit_store=None,
) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=settings.cron_timezone)

    async def _run_market() -> None:
        path = await asyncio.to_thread(job_daily_market_digest, store, knowledge)
        await _notify(send_telegram, "Daily market digest siap", path)

    async def _run_news() -> None:
        path = await asyncio.to_thread(job_daily_news, knowledge)
        await _notify(send_telegram, "Daily news digest siap", path)

    async def _run_weekly() -> None:
        path = await asyncio.to_thread(job_weekly_lesson, knowledge)
        await _notify(send_telegram, "Weekly IDX study sheet siap", path)

    async def _run_reports() -> None:
        """Weekday farm: @StockbitReports + official @Stockbit."""
        days = settings.cron_reports_days
        await _notify_text(
            send_text,
            (
                "Mulai farming Stockbit stream (cron weekday)\n"
                "urls=@StockbitReports + @Stockbit "
                f"days={days}\n"
                "Hasil di-ingest ke Chroma bila login OK."
            ),
        )
        try:
            path = await asyncio.to_thread(
                job_daily_stockbit_reports,
                settings.export_dir,
                days=days,
                stockbit_store=stockbit_store,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Cron Stockbit stream farm failed")
            await _notify_text(send_text, f"Farming Stockbit stream GAGAL: {exc}")
            return
        await _notify(send_telegram, "Farming Stockbit stream selesai", path)

    async def _run_farm() -> None:
        """Weekday Chartbit farm — https://stockbit.com/symbol/{T}/chartbit"""
        tfs = settings.cron_farm_timeframes
        await _notify_text(
            send_text,
            (
                "Mulai farming Chartbit (cron weekday 17:00 WIB)\n"
                f"timeframes={tfs} sync_daily={settings.cron_farm_sync_daily}\n"
                "Sumber: stockbit.com/symbol/{TICKER}/chartbit\n"
                "Ini bisa berjalan lama; progress dikirim berkala."
            ),
        )

        loop = asyncio.get_running_loop()
        last_sent = [0.0]

        def on_progress(msg: str) -> None:
            now = time.monotonic()
            if now - last_sent[0] < 120:
                return
            last_sent[0] = now
            asyncio.run_coroutine_threadsafe(
                _notify_text(send_text, f"Farming Chartbit berjalan...\n{msg}"),
                loop,
            )

        try:
            path = await asyncio.to_thread(
                job_daily_chart_farm,
                knowledge,
                timeframes=tfs,
                sync_daily=settings.cron_farm_sync_daily,
                on_progress=on_progress,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Cron chart farm failed")
            await _notify_text(send_text, f"Farming Chartbit GAGAL: {exc}")
            return

        await _notify(send_telegram, "Farming Chartbit selesai", path)

    async def _run_fundamentals(tier: str) -> None:
        """Weekly deep fundamentals for one tier of the universe."""
        from agent_idx import universe

        sections = ["financials", "keystats", "profile"]
        tier_name, symbols = await asyncio.to_thread(
            universe.resolve_tier, tier, top_n=settings.fundamentals_top_n
        )
        label = (
            f"top {len(symbols)} emiten"
            if tier_name == universe.TIER_TOP
            else f"sisa {len(symbols)} emiten"
        )
        await _notify_text(
            send_text,
            (
                "Mulai farming fundamental Stockbit mingguan\n"
                f"tier={tier_name} ({label})\n"
                f"sections={','.join(sections)}\n"
                "Proses berurutan + jitter dan bisa berjalan sangat lama. "
                "Jangan tutup Chromium; pastikan login/OTP masih aktif."
            ),
        )

        loop = asyncio.get_running_loop()
        last_sent = [0.0]

        def on_progress(msg: str) -> None:
            now = time.monotonic()
            # Long-running by design; avoid Telegram progress spam.
            if now - last_sent[0] < 900:
                return
            last_sent[0] = now
            asyncio.run_coroutine_threadsafe(
                _notify_text(
                    send_text, f"Fundamental farm ({tier_name}) berjalan...\n{msg}"
                ),
                loop,
            )

        try:
            path = await asyncio.to_thread(
                job_weekly_stockbit_fundamentals,
                settings.export_dir,
                sections=sections,
                tier=tier_name,
                on_progress=on_progress,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Weekly Stockbit fundamentals farm failed (%s)", tier_name)
            await _notify_text(
                send_text,
                f"Farming fundamental Stockbit ({tier_name}) GAGAL: {exc}",
            )
            return
        await _notify(
            send_telegram,
            f"Farming fundamental Stockbit selesai (tier={tier_name})",
            path,
        )

    async def _run_fundamentals_top() -> None:
        await _run_fundamentals("top")

    async def _run_fundamentals_rest() -> None:
        await _run_fundamentals("rest")

    scheduler.add_job(
        _run_market,
        CronTrigger.from_crontab(settings.cron_market, timezone=settings.cron_timezone),
        id="daily_market_digest",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    if _cron_on(settings.cron_reports):
        scheduler.add_job(
            _run_reports,
            CronTrigger.from_crontab(
                settings.cron_reports, timezone=settings.cron_timezone
            ),
            id="daily_stockbit_reports_farm",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
    if _cron_on(settings.cron_farm):
        scheduler.add_job(
            _run_farm,
            CronTrigger.from_crontab(settings.cron_farm, timezone=settings.cron_timezone),
            id="daily_chartbit_farm",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
    if _cron_on(settings.cron_fundamentals):
        scheduler.add_job(
            _run_fundamentals_top,
            CronTrigger.from_crontab(
                settings.cron_fundamentals,
                timezone=settings.cron_timezone,
            ),
            id="weekly_stockbit_fundamentals_farm",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
    if _cron_on(settings.cron_fundamentals_rest):
        scheduler.add_job(
            _run_fundamentals_rest,
            CronTrigger.from_crontab(
                settings.cron_fundamentals_rest,
                timezone=settings.cron_timezone,
            ),
            id="weekly_stockbit_fundamentals_farm_rest",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
    scheduler.add_job(
        _run_news,
        CronTrigger.from_crontab(settings.cron_news, timezone=settings.cron_timezone),
        id="daily_news_digest",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _run_weekly,
        CronTrigger.from_crontab(settings.cron_weekly, timezone=settings.cron_timezone),
        id="weekly_lesson",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    return scheduler


async def _notify(send_telegram, title: str, path: Path) -> None:
    if send_telegram is None:
        logger.info("%s: %s", title, path)
        return
    try:
        await send_telegram(title, path)
    except Exception:  # noqa: BLE001
        logger.exception("Failed telegram notify for %s", path)


async def _notify_text(send_text, text: str) -> None:
    if send_text is None:
        logger.info("%s", text)
        return
    try:
        await send_text(text)
    except Exception:  # noqa: BLE001
        logger.exception("Failed telegram text notify")
