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
        SELECT symbol, name, value, volume, close, change
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
    top_sell_text = (
        top_sell.to_string(index=False)
        if not top_sell.empty
        else "GAP_DATA: sesi ini tidak memiliki foreign buy/sell"
    )
    top_buy_text = (
        top_buy.to_string(index=False)
        if not top_buy.empty
        else "GAP_DATA: sesi ini tidak memiliki foreign buy/sell"
    )
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
        top_sell_text,
        "```",
        "",
        "## Top net foreign buy",
        "```",
        top_buy_text,
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


def job_daily_chart_farm(
    knowledge: KnowledgeBase,
    *,
    timeframes: str | list[str] = "1H,1D",
    sync_daily: bool = True,
    on_progress=None,
) -> Path:
    """Farm OHLCV via tvkit for all symbols; write a text summary."""
    from agent_idx import chart_farm

    if isinstance(timeframes, str):
        tfs = [t.strip().upper() for t in timeframes.split(",") if t.strip()]
    else:
        tfs = [str(t).strip().upper() for t in timeframes if str(t).strip()]
    if not tfs:
        tfs = ["1H", "1D"]

    symbols = chart_farm.load_symbols()
    now = datetime.now(WIB)
    stamp = now.strftime("%Y%m%d")
    header = [
        f"# Daily Chart Farm (tvkit) — {stamp}",
        "",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M %Z')}",
        f"symbols={len(symbols)} timeframes={','.join(tfs)} sync_daily={sync_daily}",
        "source=tvkit IDX:{TICKER}",
        "",
    ]

    summary = chart_farm.run_farm(
        symbols,
        tfs,
        sync_daily=bool(sync_daily),
        on_progress=on_progress,
    )
    out_dir = knowledge.root / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"farm_{stamp}.md"
    path.write_text("\n".join(header) + summary + "\n", encoding="utf-8")
    logger.info("Wrote chart farm summary %s", path)
    return path


def job_weekly_stockbit_fundamentals(
    export_dir: Path,
    *,
    sections: list[str] | None = None,
    tier: str | None = None,
    on_progress=None,
) -> Path:
    """Farm deep fundamentals for one tier of the IDX universe."""
    from agent_idx import fund_farm, universe

    secs = fund_farm.normalize_sections(
        sections or ["financials", "keystats", "profile"]
    )
    tier_name, symbols = universe.resolve_tier(tier)
    tiers = universe.load_tiers()
    now = datetime.now(WIB)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(export_dir) / "stockbit"
    out_dir.mkdir(parents=True, exist_ok=True)

    header = [
        f"# Weekly Stockbit Fundamentals Farm — {stamp}",
        "",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M %Z')}",
        f"tier={tier_name} symbols={len(symbols)} sections={','.join(secs)}",
        f"ranking={tiers.get('basis')} ({tiers.get('basis_note')})",
        "source=https://stockbit.com/symbol/{TICKER}/"
        "{financials|keystats|profile}",
        "pace=sequential + jitter; stop on login/OTP",
        "",
    ]
    summary = fund_farm.run_fund_farm(
        symbols,
        secs,
        on_progress=on_progress,
    )
    try:
        from agent_idx.pg_export import export_fundamentals
        from agent_idx.transforms import run_data_transform

        summary += "\n\n" + run_data_transform(scope="fundamentals")
        summary += "\n\n" + export_fundamentals()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Postgres export after fundamentals farm failed")
        summary += f"\n\nexport fundamentals GAGAL: {exc}"

    path = out_dir / f"fundamentals_farm_{tier_name}_{stamp}.md"
    path.write_text("\n".join(header) + summary + "\n", encoding="utf-8")
    logger.info("Wrote weekly Stockbit fundamentals summary %s", path)
    return path


def job_daily_stockbit_reports(
    export_dir: Path,
    *,
    days: int = 3,
    url: str | None = None,
    urls: list[str] | None = None,
    stockbit_store=None,
) -> Path:
    """Scrape Stockbit stream profiles and optionally ingest into Chroma.

    Default farms both @StockbitReports and official @Stockbit
    (https://stockbit.com/Stockbit — often mid-day foreign flow).
    """
    import os
    import re

    from agent_idx.browser_stockbit import OFFICIAL_URL, REPORTS_URL, get_browser, run_sync

    days = max(1, min(int(days), 90))
    if urls:
        targets = [u.strip() for u in urls if u and str(u).strip()]
    elif url:
        targets = [url.strip()]
    else:
        targets = [REPORTS_URL, OFFICIAL_URL]
    export_dir = Path(export_dir)
    stockbit_export = export_dir / "stockbit"
    stockbit_export.mkdir(parents=True, exist_ok=True)

    headless = os.getenv("STOCKBIT_HEADLESS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    profile = os.getenv("STOCKBIT_PROFILE_DIR", "").strip() or None
    if not profile:
        root = Path(__file__).resolve().parent.parent
        profile = str(root / "data" / "stockbit_profile")

    def _scrape_all() -> str:
        browser = get_browser(headless=headless, user_data_dir=profile)
        if not browser.ready:
            browser.start()
        parts: list[str] = []
        for target in targets:
            part = browser.scrape_reports_stream(days=days, url=target)
            parts.append(part)
        return "\n\n".join(parts)

    result = run_sync(_scrape_all)
    now = datetime.now(WIB)
    stamp = now.strftime("%Y%m%d")
    summary_lines = [
        f"# Daily Stockbit Stream Farm — {stamp}",
        "",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M %Z')}",
        f"urls={targets}",
        f"days={days}",
        "",
        result[:20000],
        "",
    ]

    inserted = 0
    file_re = re.compile(r"^FILE:\s*(.+)\s*$", re.M)
    if stockbit_store is not None and "NEED_STOCKBIT" not in result:
        for match in file_re.finditer(result or ""):
            path = Path(match.group(1).strip().strip('"'))
            if path.is_file():
                try:
                    inserted += int(stockbit_store.ingest_markdown(path) or 0)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Ingest failed for %s", path)
                    summary_lines.append(f"ingest_error {path.name}: {exc}")
        summary_lines.append(f"chroma_ingested={inserted}")

    out = stockbit_export / f"reports_farm_{stamp}.md"
    out.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Wrote Stockbit stream farm summary %s", out)
    if "NEED_STOCKBIT" in result:
        raise RuntimeError(result.split("\n", 1)[0])
    return out


def run_all_learning_jobs(store: StockDataStore, knowledge: KnowledgeBase) -> list[Path]:
    paths = [
        job_daily_market_digest(store, knowledge),
        job_daily_news(knowledge),
        job_weekly_lesson(knowledge),
    ]
    return paths
