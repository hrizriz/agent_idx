"""IDX daily panel sync: Google Drive Ringkasan Saham ∪ tvkit OHLCV gaps.

Combine sources (only these two when ``MARKET_SOURCE=idx``):
- Drive day panels — foreign_buy/sell (and listed_shares)
- tvkit chart 1D — OHLCV for every date Drive does not cover (pre-Drive
  history and any missed day); foreign stays NULL/GAP_DATA

Yahoo / Stockbit rebuild parquets are not part of this combine.

Published into ``data/parquet/`` as:
- ``daily_stock_summary_YYYYMMDD.parquet`` — from Drive
- ``daily_stock_summary_tvkit_gap.parquet`` — OHLCV-only gap fill
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"
DRIVE_DIR = PARQUET_DIR / "drive_idx"
CHARTS_DIR = ROOT / "data" / "charts"

_DAY_FILE = re.compile(r"^daily_stock_summary_(\d{8})\.parquet$", re.I)
_DAY_FILE_DASH = re.compile(
    r"^daily_stock_summary_(\d{4})-(\d{2})-(\d{2})\.parquet$", re.I
)


def normalize_panel_filename(name: str) -> str | None:
    m = _DAY_FILE.match(name)
    if m:
        return f"daily_stock_summary_{m.group(1)}.parquet"
    m = _DAY_FILE_DASH.match(name)
    if m:
        return f"daily_stock_summary_{m.group(1)}{m.group(2)}{m.group(3)}.parquet"
    return None


def publish_drive_panels(
    src_dir: Path = DRIVE_DIR, dest_dir: Path = PARQUET_DIR
) -> dict:
    """Copy Drive panels into parquet dir with normalized YYYYMMDD names."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(src_dir)
    copied = skipped = bad = 0
    dates: list[int] = []
    for src in sorted(src_dir.glob("daily_stock_summary_*.parquet")):
        norm = normalize_panel_filename(src.name)
        if not norm:
            bad += 1
            continue
        dest = dest_dir / norm
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            skipped += 1
        else:
            dest.write_bytes(src.read_bytes())
            copied += 1
        m = _DAY_FILE.match(norm)
        if m:
            dates.append(int(m.group(1)))
    return {
        "copied": copied,
        "skipped": skipped,
        "bad": bad,
        "dates_n": len(dates),
        "min_date": min(dates) if dates else None,
        "max_date": max(dates) if dates else None,
    }


def drive_date_set(parquet_dir: Path = PARQUET_DIR) -> set[int]:
    out: set[int] = set()
    for p in Path(parquet_dir).glob("daily_stock_summary_*.parquet"):
        if "_stockbit_" in p.name.lower() or "_yf_" in p.name.lower():
            continue
        if "_tvkit_" in p.name.lower():
            continue
        norm = normalize_panel_filename(p.name)
        if not norm:
            continue
        m = _DAY_FILE.match(norm)
        if m:
            out.add(int(m.group(1)))
    return out


def _chart_1d_paths(symbols: set[str] | None = None) -> list[str]:
    paths: list[str] = []
    if not CHARTS_DIR.is_dir():
        return paths
    for sym_dir in sorted(CHARTS_DIR.iterdir()):
        if not sym_dir.is_dir() or sym_dir.name.startswith("_"):
            continue
        sym = sym_dir.name.upper()
        if symbols and sym not in symbols:
            continue
        p1d = sym_dir / "1D.parquet"
        if p1d.is_file():
            paths.append(p1d.as_posix())
    return paths


def build_tvkit_gap_panel(
    *,
    after_date: int | None = None,
    parquet_dir: Path = PARQUET_DIR,
    out_name: str = "daily_stock_summary_tvkit_gap.parquet",
) -> Path | None:
    """Write tvkit 1D OHLCV for dates not covered by the Drive day panels.

    Combined panel = Drive Ringkasan (1091 day files, with foreign flow) ∪ this
    gap file. Yahoo/Stockbit rebuilds are not part of the combine.

    By default every tvkit date outside the Drive set is included (history
    before Drive starts, plus any day Drive missed). Pass ``after_date`` to
    restrict to ``date > after_date`` (incremental / recent-only fill).
    """
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    covered = drive_date_set(parquet_dir)

    paths = _chart_1d_paths()
    if not paths:
        logger.warning("No tvkit/chart 1D parquet for gap fill")
        return None

    listed = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)
    date_filter = ""
    if after_date is not None:
        date_filter = f"AND date_i > {int(after_date)}"

    con = duckdb.connect(":memory:")
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE gap AS
            WITH raw AS (
                SELECT
                    UPPER(CAST(symbol AS VARCHAR)) AS symbol,
                    UPPER(CAST(symbol AS VARCHAR)) AS name,
                    CAST(NULL AS DOUBLE) AS prev_close,
                    CAST(open AS DOUBLE) AS open_price,
                    CAST(high AS DOUBLE) AS high,
                    CAST(low AS DOUBLE) AS low,
                    CAST(close AS DOUBLE) AS close,
                    CAST(NULL AS DOUBLE) AS change,
                    CAST(volume AS DOUBLE) AS volume,
                    CAST(close AS DOUBLE) * CAST(volume AS DOUBLE) AS value,
                    0.0 AS frequency,
                    CAST(NULL AS DOUBLE) AS index_individual,
                    CAST(NULL AS DOUBLE) AS weight_for_index,
                    CAST(NULL AS DOUBLE) AS foreign_buy,
                    CAST(NULL AS DOUBLE) AS foreign_sell,
                    CAST(NULL AS DOUBLE) AS offer,
                    CAST(NULL AS DOUBLE) AS offer_volume,
                    CAST(NULL AS DOUBLE) AS bid,
                    CAST(NULL AS DOUBLE) AS bid_volume,
                    CAST(NULL AS DOUBLE) AS listed_shares,
                    CAST(NULL AS DOUBLE) AS tradable_shares,
                    0.0 AS non_regular_volume,
                    0.0 AS non_regular_value,
                    0.0 AS non_regular_frequency,
                    CAST(replace(CAST(date AS VARCHAR), '-', '') AS INTEGER) AS date_i,
                    'tvkit_1d_gap' AS upload_file
                FROM read_parquet([{listed}], union_by_name=true)
                WHERE close IS NOT NULL AND date IS NOT NULL
            )
            SELECT
                symbol, name, prev_close, open_price, high, low, close, change,
                volume, value, frequency, index_individual, weight_for_index,
                foreign_buy, foreign_sell, offer, offer_volume, bid, bid_volume,
                listed_shares, tradable_shares,
                non_regular_volume, non_regular_value, non_regular_frequency,
                date_i AS date,
                upload_file
            FROM raw
            WHERE date_i IS NOT NULL {date_filter}
            """
        )
        # Drop every date already covered by a Drive day file (no double rows).
        if covered:
            covered_list = ", ".join(str(d) for d in sorted(covered))
            con.execute(
                f"DELETE FROM gap WHERE date IN ({covered_list})"
            )

        # Fill prev_close / change
        con.execute(
            """
            CREATE OR REPLACE TABLE gap2 AS
            SELECT
                * EXCLUDE (prev_close, change),
                lag(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close,
                close - lag(close) OVER (PARTITION BY symbol ORDER BY date) AS change
            FROM gap
            ORDER BY date, symbol
            """
        )
        n = con.execute("SELECT COUNT(*), COUNT(DISTINCT date), COUNT(DISTINCT symbol) FROM gap2").fetchone()
        if not n or n[0] == 0:
            logger.info(
                "No tvkit gap rows (Drive days=%s after_date=%s)",
                len(covered),
                after_date,
            )
            out = parquet_dir / out_name
            if out.exists():
                out.unlink()
            return None
        out = parquet_dir / out_name
        con.execute(f"COPY gap2 TO '{out.as_posix()}' (FORMAT PARQUET)")
        d_range = con.execute("SELECT MIN(date), MAX(date) FROM gap2").fetchone()
        logger.info(
            "Wrote %s rows=%s dates=%s symbols=%s range=%s..%s after_date=%s",
            out.name,
            n[0],
            n[1],
            n[2],
            d_range[0] if d_range else None,
            d_range[1] if d_range else None,
            after_date,
        )
        return out
    finally:
        con.close()


def combine_panels(*, publish_drive: bool = True) -> str:
    """Publish Drive panels + rebuild tvkit gap; return a text summary."""
    lines: list[str] = []
    if publish_drive:
        if not DRIVE_DIR.is_dir() or not any(DRIVE_DIR.glob("*.parquet")):
            lines.append(f"WARN: no Drive files in {DRIVE_DIR}")
        else:
            pub = publish_drive_panels()
            lines.append(
                f"Drive publish: copied={pub['copied']} skipped={pub['skipped']} "
                f"dates={pub['dates_n']} range={pub['min_date']}..{pub['max_date']}"
            )
    gap = build_tvkit_gap_panel()
    if gap:
        lines.append(f"tvkit gap: {gap.relative_to(ROOT).as_posix()}")
    else:
        lines.append("tvkit gap: (none / empty)")
    covered = drive_date_set()
    lines.append(
        f"Drive day-files in parquet/: {len(covered)} "
        f"max={max(covered) if covered else None}"
    )
    lines.append("Set MARKET_SOURCE=idx then restart bot to query the combined panel.")
    return "\n".join(lines)
