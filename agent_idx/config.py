from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    parquet_dir: Path
    export_dir: Path
    knowledge_dir: Path
    chat_db_path: Path
    chroma_dir: Path
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    telegram_bot_token: str
    telegram_chat_id: int | None
    max_sql_rows: int
    cron_enabled: bool
    cron_timezone: str
    cron_market: str
    cron_news: str
    cron_weekly: str
    cron_farm: str
    cron_farm_timeframes: str
    cron_farm_sync_daily: bool
    cron_reports: str
    cron_reports_days: int
    cron_fundamentals: str
    cron_fundamentals_rest: str
    fundamentals_top_n: int


def load_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    parquet_dir = Path(
        os.getenv(
            "PARQUET_DIR",
            r"D:\MLOps - Project\stock_idx\airflow\data\parquet",
        )
    )
    export_dir = Path(os.getenv("EXPORT_DIR", str(root / "exports")))
    knowledge_dir = Path(os.getenv("KNOWLEDGE_DIR", str(root / "knowledge")))
    chat_db_path = Path(
        os.getenv("CHAT_DB_PATH", str(root / "data" / "telegram_chat.db"))
    )
    chroma_dir = Path(os.getenv("CHROMA_DIR", str(root / "data" / "chroma")))
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_raw = (
        os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("CHAT_ID")
        or ""
    ).strip()
    chat_id = int(chat_raw) if chat_raw else None
    cron_enabled = os.getenv("CRON_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    farm_sync = os.getenv("CRON_FARM_SYNC_DAILY", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    reports_days = int(os.getenv("CRON_REPORTS_DAYS", "3"))
    return Settings(
        parquet_dir=parquet_dir,
        export_dir=export_dir,
        knowledge_dir=knowledge_dir,
        chat_db_path=chat_db_path,
        chroma_dir=chroma_dir,
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:18765/v1").rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", "sk-local"),
        llm_model=os.getenv("LLM_MODEL", "gemini-3.6-flash-thinking"),
        telegram_bot_token=token,
        telegram_chat_id=chat_id,
        max_sql_rows=int(os.getenv("MAX_SQL_ROWS", "100")),
        cron_enabled=cron_enabled,
        cron_timezone=os.getenv("CRON_TIMEZONE", "Asia/Jakarta"),
        # Weekday fields use day names: APScheduler counts 0=Monday, so numeric
        # crontab ranges written for 0=Sunday land one day late.
        # Mon-Fri: market 16:45, Stockbit streams 16:50, Chartbit 17:00, news 17:15
        cron_market=os.getenv("CRON_MARKET", "45 16 * * mon-fri"),
        cron_news=os.getenv("CRON_NEWS", "15 17 * * mon-fri"),
        cron_weekly=os.getenv("CRON_WEEKLY", "0 9 * * sun"),
        cron_farm=os.getenv("CRON_FARM", "0 17 * * mon-fri"),
        cron_farm_timeframes=os.getenv("CRON_FARM_TIMEFRAMES", "1H,1D"),
        cron_farm_sync_daily=farm_sync,
        cron_reports=os.getenv("CRON_REPORTS", "50 16 * * mon-fri"),
        cron_reports_days=max(1, min(reports_days, 90)),
        # The full universe takes most of a day, so it is split in two:
        # Friday 07:00 the top tier, Saturday 07:00 everything else.
        cron_fundamentals=os.getenv("CRON_FUNDAMENTALS", "0 7 * * fri"),
        cron_fundamentals_rest=os.getenv("CRON_FUNDAMENTALS_REST", "0 7 * * sat"),
        fundamentals_top_n=max(1, int(os.getenv("FUNDAMENTALS_TOP_N", "200"))),
    )
