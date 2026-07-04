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
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    telegram_bot_token: str
    telegram_chat_id: int | None
    max_sql_rows: int
    max_agent_steps: int
    cron_enabled: bool
    cron_timezone: str
    cron_market: str
    cron_news: str
    cron_weekly: str


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
    return Settings(
        parquet_dir=parquet_dir,
        export_dir=export_dir,
        knowledge_dir=knowledge_dir,
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:18765/v1").rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", "sk-local"),
        llm_model=os.getenv("LLM_MODEL", "gemini-3.5-flash-thinking"),
        telegram_bot_token=token,
        telegram_chat_id=chat_id,
        max_sql_rows=int(os.getenv("MAX_SQL_ROWS", "100")),
        max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "10")),
        cron_enabled=cron_enabled,
        cron_timezone=os.getenv("CRON_TIMEZONE", "Asia/Jakarta"),
        # Mon-Fri 16:45 WIB market digest, 17:15 news, Sun 09:00 weekly lesson
        cron_market=os.getenv("CRON_MARKET", "45 16 * * 1-5"),
        cron_news=os.getenv("CRON_NEWS", "15 17 * * 1-5"),
        cron_weekly=os.getenv("CRON_WEEKLY", "0 9 * * 0"),
    )
