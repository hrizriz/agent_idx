from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from agent_idx.config import Settings
from agent_idx.data import StockDataStore
from agent_idx.jobs import (
    job_daily_market_digest,
    job_daily_news,
    job_weekly_lesson,
)
from agent_idx.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


def build_scheduler(
    settings: Settings,
    store: StockDataStore,
    knowledge: KnowledgeBase,
    *,
    send_telegram=None,
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

    # Weekdays after market: market digest then news.
    scheduler.add_job(
        _run_market,
        CronTrigger.from_crontab(settings.cron_market, timezone=settings.cron_timezone),
        id="daily_market_digest",
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
