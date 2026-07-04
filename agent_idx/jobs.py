from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent_idx.data import StockDataStore
from agent_idx.knowledge import KnowledgeBase
from agent_idx.news import search_news

logger = logging.getLogger(__name__)
WIB = ZoneInfo("Asia/Jakarta")


def _today_stamp() -> str:
    return datetime.now(WIB).strftime("%Y%m%d")


def job_daily_market_digest(store: StockDataStore, knowledge: KnowledgeBase) -> Path:
    """Build a markdown digest from latest trading day in parquet."""
    row = store._con.execute(
        "SELECT MAX(date) FROM daily_stock"
    ).fetchone()
    latest = int(row[0]) if row and row[0] is not None else None
    if latest is None:
        raise RuntimeError("No trading data available for digest")

    # Use last ~5 sessions ending at latest.
    stats = store._con.execute(
        """
        SELECT
            COUNT(DISTINCT symbol) AS symbols,
            COUNT(DISTINCT date) AS days,
            MIN(date) AS d0,
            MAX(date) AS d1,
            SUM(value) AS total_value,
            SUM(volume) AS total_volume
        FROM daily_stock
        WHERE date BETWEEN ? AND ?
        """,
        [latest, latest],
    ).fetchone()

    top_value = store._con.execute(
        """
        SELECT symbol, name, value, volume, close, change,
               foreign_buy, foreign_sell,
               foreign_buy - foreign_sell AS net_foreign
        FROM daily_stock
        WHERE date = ?
        ORDER BY value DESC
        LIMIT 10
        """,
        [latest],
    ).fetchdf()

    top_sell = store._con.execute(
        """
        SELECT symbol, name, foreign_buy AS fb, foreign_sell AS fs,
               foreign_buy - foreign_sell AS net_foreign
        FROM daily_stock
        WHERE date = ? AND (foreign_buy - foreign_sell) < 0
        ORDER BY net_foreign ASC
        LIMIT 10
        """,
        [latest],
    ).fetchdf()

    top_buy = store._con.execute(
        """
        SELECT symbol, name, foreign_buy AS fb, foreign_sell AS fs,
               foreign_buy - foreign_sell AS net_foreign
        FROM daily_stock
        WHERE date = ? AND (foreign_buy - foreign_sell) > 0
        ORDER BY net_foreign DESC
        LIMIT 10
        """,
        [latest],
    ).fetchdf()

    movers = store._con.execute(
        """
        SELECT symbol, name, close, change, value,
               CASE WHEN prev_close IS NULL OR prev_close = 0 THEN NULL
                    ELSE (close - prev_close) / prev_close END AS ret
        FROM daily_stock
        WHERE date = ? AND value > 10000000000
        ORDER BY ret DESC NULLS LAST
        LIMIT 10
        """,
        [latest],
    ).fetchdf()

    now = datetime.now(WIB).strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        f"# Daily Market Digest — {latest}",
        "",
        f"Generated: {now}",
        "",
        "## Ringkasan sesi",
        f"- Trading date: **{latest}**",
        f"- Symbols: **{stats[0]}**",
        f"- Total value: **{stats[4]:,.0f}**",
        f"- Total volume: **{stats[5]:,.0f}**",
        "",
        "## Top value",
        "```",
        top_value.to_string(index=False),
        "```",
        "",
        "## Top net foreign sell",
        "```",
        top_sell.to_string(index=False),
        "```",
        "",
        "## Top net foreign buy",
        "```",
        top_buy.to_string(index=False),
        "```",
        "",
        "## Top return (value > 10M)",
        "```",
        movers.to_string(index=False),
        "```",
        "",
        "## Catatan belajar",
        "- Bandingkan foreign flow vs pergerakan harga (lihat curriculum 05).",
        "- Value tinggi + return ekstrem = kandidat review likuiditas/katalis.",
        "- Lanjut cek berita harian di folder `knowledge/news/`.",
    ]
    path = knowledge.root / "daily" / f"market_{latest}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote market digest %s", path)
    return path


def job_daily_news(knowledge: KnowledgeBase) -> Path:
    queries = [
        "IHSG IDX bursa",
        "saham bank BBCA BMRI BBRI BBNI",
        "foreign flow asing IDX",
        "BI rate suku bunga saham",
    ]
    now = datetime.now(WIB)
    stamp = now.strftime("%Y%m%d")
    blocks = [
        f"# Daily News Digest — {stamp}",
        "",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "Sumber: Google News RSS (judul + link). Verifikasi sebelum dipakai keputusan.",
        "",
    ]
    for q in queries:
        try:
            blocks.append(f"## Query: {q}")
            blocks.append("```")
            blocks.append(search_news(q, limit=5))
            blocks.append("```")
            blocks.append("")
        except Exception as exc:  # noqa: BLE001
            blocks.append(f"## Query: {q}")
            blocks.append(f"ERROR: {exc}")
            blocks.append("")

    blocks.extend(
        [
            "## Kaitan ke fundamental",
            "- Berita suku bunga → valuasi bank & properti.",
            "- Berita komoditas → emiten batubara/CPO/nikel.",
            "- Berita regulasi/OJK/BEI → risiko sektoral.",
            "- Gunakan curriculum `02_fundamental_analysis.md` untuk kerangka.",
        ]
    )
    path = knowledge.root / "news" / f"news_{stamp}.md"
    path.write_text("\n".join(blocks), encoding="utf-8")
    logger.info("Wrote news digest %s", path)
    return path


def job_weekly_lesson(knowledge: KnowledgeBase) -> Path:
    """Compile a weekly study sheet from curriculum + latest digests."""
    now = datetime.now(WIB)
    stamp = now.strftime("%Y%m%d")
    parts = [
        f"# Weekly IDX Study Sheet — {stamp}",
        "",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "Lembar belajar mingguan: gabungan kurikulum + digest terbaru.",
        "",
    ]

    curriculum = sorted((knowledge.root / "curriculum").glob("*.md"))
    for path in curriculum:
        parts.append(f"## Curriculum: {path.name}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Keep it compact.
        parts.append(text[:2500])
        parts.append("")

    daily_files = sorted((knowledge.root / "daily").glob("market_*.md"))
    if daily_files:
        latest_daily = daily_files[-1]
        parts.append(f"## Latest market digest: {latest_daily.name}")
        parts.append(latest_daily.read_text(encoding="utf-8", errors="ignore")[:3000])
        parts.append("")

    news_files = sorted((knowledge.root / "news").glob("news_*.md"))
    if news_files:
        latest_news = news_files[-1]
        parts.append(f"## Latest news digest: {latest_news.name}")
        parts.append(latest_news.read_text(encoding="utf-8", errors="ignore")[:3000])
        parts.append("")

    parts.extend(
        [
            "## Latihan mandiri",
            "1. Pilih 1 emiten liquid; ringkas fundamental dari laporan/berita.",
            "2. Ambil return + foreign flow 20 hari dari data transaksi.",
            "3. Tulis thesis: menarik / netral / hati-hati + invalidasi.",
            "4. Simpan kesimpulan; bandingkan minggu depan.",
        ]
    )

    out = knowledge.root / "lessons" / f"weekly_{stamp}.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Wrote weekly lesson %s", out)
    return out


def run_all_learning_jobs(store: StockDataStore, knowledge: KnowledgeBase) -> list[Path]:
    paths = [
        job_daily_market_digest(store, knowledge),
        job_daily_news(knowledge),
        job_weekly_lesson(knowledge),
    ]
    return paths
