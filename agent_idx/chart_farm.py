"""Farm IDX OHLCV via tvkit (TradingView) into parquet + indicators.

Shared by scripts/farm_stockbit_charts.py and the agent tool `farm_stockbit_charts`.
Bulk runs are staged and require explicit user confirmation before they start.

Bars are written with ``source = 'tvkit'``. Project Timeframe codes (``1M`` =
1 minute) are mapped to tvkit intervals; tvkit's own ``1M`` means monthly and
must never be passed through unchanged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = ROOT / "data" / "charts"
PARQUET_DIR = ROOT / "data" / "parquet"
SYMBOLS_FILE = ROOT / "data" / "symbols_list.txt"

DEFAULT_TIMEFRAMES = ["1H", "1D", "1W"]
SUPPORTED_TIMEFRAMES = frozenset(
    {"1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W"}
)
# Project TF → tvkit interval string. Critical: project 1M is 1-minute ("1"),
# not tvkit's monthly "1M".
TVKIT_INTERVAL = {
    "1M": "1",
    "5M": "5",
    "15M": "15",
    "30M": "30",
    "1H": "1H",
    "4H": "4H",
    "1D": "1D",
    "1W": "1W",
}
DEFAULT_BARS_COUNT = {
    "1M": 5000,
    "5M": 5000,
    "15M": 5000,
    "30M": 5000,
    "1H": 5000,
    "4H": 3000,
    "1D": 5000,
    "1W": 800,
}
BAR_SOURCE = "tvkit"
# More than this many symbols is a "bulk" job and needs user confirmation.
BULK_THRESHOLD = 3
MAX_SYMBOLS_PER_JOB = 1000

ProgressFn = Callable[[str], None]


def parse_timeframe(text: str, *, default: str = "1H") -> str:
    """Parse user TF like '5 menit', '5m', '1H', 'daily' → canonical code."""
    raw = (text or "").strip().lower()
    if not raw:
        return default

    # Explicit codes first
    m = re.search(r"\b(1m|5m|15m|30m|1h|4h|1d|1w)\b", raw, re.I)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b(\d+)\s*menit\b", raw, re.I)
    if m:
        n = int(m.group(1))
        code = f"{n}M"
        return code if code in SUPPORTED_TIMEFRAMES else default

    m = re.search(r"\b(\d+)\s*min(?:ute)?s?\b", raw, re.I)
    if m:
        n = int(m.group(1))
        code = f"{n}M"
        return code if code in SUPPORTED_TIMEFRAMES else default

    m = re.search(r"\b(\d+)\s*jam\b", raw, re.I)
    if m:
        n = int(m.group(1))
        if n == 1:
            return "1H"
        if n == 4:
            return "4H"
        return default

    if re.search(r"\b(harian|daily|1\s*hari)\b", raw, re.I):
        return "1D"
    if re.search(r"\b(mingguan|weekly)\b", raw, re.I):
        return "1W"

    return default


def normalize_timeframes(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_TIMEFRAMES)
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in raw if str(p).strip()]
    out: list[str] = []
    for p in parts:
        tf = parse_timeframe(p, default="")
        if not tf:
            tf = p.upper()
        if tf in SUPPORTED_TIMEFRAMES and tf not in out:
            out.append(tf)
    return out or list(DEFAULT_TIMEFRAMES)


def load_symbols(path: Path | None = None) -> list[str]:
    path = path or SYMBOLS_FILE
    symbols: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sym = line.split(",", 1)[0].split("\t", 1)[0].strip().upper().replace(".JK", "")
        if not sym.isalnum() or sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)
    return symbols


def resolve_symbols(
    symbols: str | list[str] | None = None,
    *,
    limit: int = 0,
    offset: int = 0,
) -> list[str]:
    """Explicit list wins; otherwise read symbols_list.txt with offset/limit."""
    if isinstance(symbols, str):
        picked = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif symbols:
        picked = [str(s).strip().upper() for s in symbols if str(s).strip()]
    else:
        picked = []

    if not picked:
        picked = load_symbols()
        picked = picked[max(0, int(offset)) :]
    if limit and int(limit) > 0:
        picked = picked[: int(limit)]
    return picked[:MAX_SYMBOLS_PER_JOB]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators from OHLCV (Stockbit-independent transform)."""
    out = df.sort_values("ts").copy()
    close = out["close"]
    vol = out["volume"] if "volume" in out.columns else None

    # Moving averages — explicit MA ladder for technical analysis tools
    out["sma_5"] = close.rolling(5, min_periods=1).mean()
    out["sma_20"] = close.rolling(20, min_periods=1).mean()
    out["sma_50"] = close.rolling(50, min_periods=1).mean()
    out["sma_200"] = close.rolling(200, min_periods=1).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, pd.NA)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    out["rsi_14"] = out["rsi_14"].fillna(50)

    if vol is not None:
        out["vol_sma_20"] = vol.rolling(20, min_periods=1).mean()
    return out


def bars_to_df(symbol: str, timeframe: str, bars: list[dict]) -> pd.DataFrame:
    rows = []
    for b in bars:
        ts = int(b["ts"])
        local = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "ts": ts,
                "datetime": local.strftime("%Y-%m-%d %H:%M:%S"),
                "date": int(local.strftime("%Y%m%d")),
                "open": b.get("open"),
                "high": b.get("high"),
                "low": b.get("low"),
                "close": b.get("close"),
                "volume": b.get("volume") or 0,
                "source": BAR_SOURCE,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return add_indicators(df)


def save_tf_parquet(symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
    out_dir = CHARTS_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{timeframe}.parquet"
    con = duckdb.connect(":memory:")
    con.register("df", df)
    con.execute(f"COPY (SELECT * FROM df) TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()
    # Keep DuckDB query store in sync for structured tools / pipelines.
    try:
        from agent_idx import chart_db

        chart_db.upsert_ohlcv(df)
    except Exception:  # noqa: BLE001
        logger.exception("DuckDB upsert failed for %s/%s", symbol, timeframe)
    return path


def format_latest_bars(symbol: str, timeframe: str, *, n: int = 12) -> str:
    """Human-readable latest OHLCV rows from farmed parquet."""
    sym = (symbol or "").strip().upper()
    tf = (timeframe or "1H").strip().upper()
    path = CHARTS_DIR / sym / f"{tf}.parquet"
    if not path.is_file():
        return f"(belum ada data chart {sym}/{tf} di {path})"
    try:
        con = duckdb.connect(":memory:")
        df = con.execute(
            f"SELECT * FROM read_parquet('{path.as_posix()}') ORDER BY 1"
        ).df()
        con.close()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR baca parquet {path.name}: {exc}"
    if df.empty:
        return f"(parquet {sym}/{tf} kosong)"
    # Prefer datetime-ish index/column
    time_col = None
    for c in ("datetime", "ts", "time", "date", "timestamp"):
        if c in df.columns:
            time_col = c
            break
    work = df.copy()
    if time_col:
        try:
            work = work.sort_values(time_col)
        except Exception:  # noqa: BLE001
            pass
    tail = work.tail(n)
    src = ""
    if "source" in work.columns and len(work):
        try:
            src = str(work["source"].iloc[-1])
        except Exception:  # noqa: BLE001
            src = ""
    lines = [
        f"Chart {sym} timeframe {tf} — {len(df)} bar tersimpan, {n} terakhir:",
        f"sumber: {path.as_posix()}" + (f" ({src})" if src else ""),
        "",
    ]
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in tail.columns]
    for _, row in tail.iterrows():
        when = ""
        if time_col:
            when = str(row.get(time_col))
            if "T" in when:
                when = when.replace("T", " ")[:16]
            else:
                when = when[:19]
        ohlc = " ".join(f"{c[0].upper()}={row.get(c)}" for c in cols)
        lines.append(f"- {when}  {ohlc}".rstrip())
    # Last close change if possible
    if "close" in work.columns and len(work) >= 2:
        try:
            last = float(work["close"].iloc[-1])
            prev = float(work["close"].iloc[-2])
            chg = last - prev
            pct = (chg / prev * 100) if prev else 0.0
            lines.append("")
            lines.append(f"Last close={last}  delta={chg:+.4g} ({pct:+.2f}%) vs bar sebelumnya")
        except Exception:  # noqa: BLE001
            pass
    return "\n".join(lines)


def quick_chart_analysis(symbol: str, timeframe: str, *, lookback: int = 40) -> str:
    """Deterministic short technical summary from farmed parquet (no LLM)."""
    sym = (symbol or "").strip().upper()
    tf = (timeframe or "1H").strip().upper()
    path = CHARTS_DIR / sym / f"{tf}.parquet"
    if not path.is_file():
        return f"(belum ada data chart {sym}/{tf})"
    try:
        con = duckdb.connect(":memory:")
        df = con.execute(
            f"SELECT * FROM read_parquet('{path.as_posix()}') ORDER BY 1"
        ).df()
        con.close()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR baca parquet: {exc}"
    if df.empty or "close" not in df.columns:
        return f"(parquet {sym}/{tf} kosong / tanpa close)"

    work = df.tail(max(lookback, 5)).copy()
    closes = pd.to_numeric(work["close"], errors="coerce").dropna()
    if closes.empty:
        return f"(close tidak valid untuk {sym}/{tf})"
    last = float(closes.iloc[-1])
    first = float(closes.iloc[0])
    chg = last - first
    pct = (chg / first * 100.0) if first else 0.0
    hi = float(pd.to_numeric(work["high"], errors="coerce").max()) if "high" in work else last
    lo = float(pd.to_numeric(work["low"], errors="coerce").min()) if "low" in work else last
    if len(closes) >= 2:
        prev = float(closes.iloc[-2])
        bar_chg = last - prev
        bar_pct = (bar_chg / prev * 100.0) if prev else 0.0
    else:
        bar_chg, bar_pct = 0.0, 0.0

    if pct > 0.4:
        bias = "naik / bullish singkat"
    elif pct < -0.4:
        bias = "turun / bearish singkat"
    else:
        bias = "sideways / konsolidasi"

    vol_note = ""
    if "volume" in work.columns:
        vols = pd.to_numeric(work["volume"], errors="coerce").dropna()
        if len(vols) >= 5:
            last_v = float(vols.iloc[-1])
            avg_v = float(vols.iloc[:-1].mean()) or 1.0
            vol_note = f"Volume bar terakhir {last_v:,.0f} vs avg {avg_v:,.0f} ({last_v / avg_v:.2f}x)."

    return (
        f"Ringkasan teknikal cepat {sym} TF {tf} (tanpa LLM):\n"
        f"• Last={last:g} | bar chg={bar_chg:+.4g} ({bar_pct:+.2f}%)\n"
        f"• Range {lookback} bar: High={hi:g} Low={lo:g}\n"
        f"• Arah {lookback} bar: {chg:+.4g} ({pct:+.2f}%) → {bias}\n"
        f"• Support kasar ≈ {lo:g}, resistance kasar ≈ {hi:g}\n"
        + (f"• {vol_note}\n" if vol_note else "")
        + "Catatan: bukan saran investasi; verifikasi di sumber chart."
    )


def sync_daily_to_summary(symbols: list[str], out_dir: Path = PARQUET_DIR) -> Path | None:
    """Rebuild Stockbit daily panel (prefer 1D, else aggregate 1H) for given symbols.

    Always scans available Chartbit parquet under data/charts — source=Stockbit, not Yahoo.
    """
    return rebuild_stockbit_daily_summary(out_dir=out_dir, symbols=symbols)


def rebuild_stockbit_daily_summary(
    *,
    out_dir: Path = PARQUET_DIR,
    symbols: list[str] | None = None,
) -> Path | None:
    """Build daily_stock_summary_stockbit_*.parquet from Chartbit farms (all ~900 emiten).

    Priority per symbol:
    1) data/charts/{SYM}/1D.parquet
    2) else aggregate data/charts/{SYM}/1H.parquet by date
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sym_filter = {s.strip().upper() for s in (symbols or []) if str(s).strip()} or None

    paths_1d: list[str] = []
    paths_1h: list[str] = []
    if not CHARTS_DIR.is_dir():
        logger.warning("No charts dir at %s", CHARTS_DIR)
        return None

    for sym_dir in sorted(CHARTS_DIR.iterdir()):
        if not sym_dir.is_dir() or sym_dir.name.startswith("_"):
            continue
        sym = sym_dir.name.upper()
        if sym_filter and sym not in sym_filter:
            continue
        p1d = sym_dir / "1D.parquet"
        p1h = sym_dir / "1H.parquet"
        if p1d.is_file():
            paths_1d.append(p1d.as_posix())
        elif p1h.is_file():
            paths_1h.append(p1h.as_posix())

    if not paths_1d and not paths_1h:
        logger.warning("No 1D/1H Chartbit parquet to rebuild daily summary")
        return None

    con = duckdb.connect(":memory:")
    try:
        frames_sql: list[str] = []
        if paths_1d:
            # DuckDB list of files
            listed = ", ".join("'" + p.replace("'", "''") + "'" for p in paths_1d)
            frames_sql.append(
                f"""
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
                    CAST(replace(CAST(date AS VARCHAR), '-', '') AS INTEGER) AS date,
                    'tvkit_1d' AS upload_file
                FROM read_parquet([{listed}], union_by_name=true)
                WHERE close IS NOT NULL
                """
            )
        if paths_1h:
            listed = ", ".join("'" + p.replace("'", "''") + "'" for p in paths_1h)
            # Aggregate intraday/daily-like 1H bars → one row per symbol+date.
            frames_sql.append(
                f"""
                SELECT
                    UPPER(CAST(symbol AS VARCHAR)) AS symbol,
                    UPPER(CAST(symbol AS VARCHAR)) AS name,
                    CAST(NULL AS DOUBLE) AS prev_close,
                    CAST(arg_min(open, ts) AS DOUBLE) AS open_price,
                    CAST(max(high) AS DOUBLE) AS high,
                    CAST(min(low) AS DOUBLE) AS low,
                    CAST(arg_max(close, ts) AS DOUBLE) AS close,
                    CAST(NULL AS DOUBLE) AS change,
                    CAST(sum(volume) AS DOUBLE) AS volume,
                    CAST(arg_max(close, ts) AS DOUBLE) * CAST(sum(volume) AS DOUBLE) AS value,
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
                    CAST(replace(CAST(date AS VARCHAR), '-', '') AS INTEGER) AS date,
                    'tvkit_1h' AS upload_file
                FROM read_parquet([{listed}], union_by_name=true)
                WHERE close IS NOT NULL AND date IS NOT NULL
                GROUP BY 1, 2, date
                """
            )

        union = " UNION ALL BY NAME ".join(f"({q})" for q in frames_sql)
        # Prefer 1D rows when both exist for same symbol+date.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE panel AS
            SELECT * EXCLUDE (rn) FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol, date
                        ORDER BY CASE WHEN upload_file = 'tvkit_1d' THEN 0 ELSE 1 END
                    ) AS rn
                FROM ({union})
            ) WHERE rn = 1
            """
        )
        # Fill prev_close / change from prior trading day per symbol.
        con.execute(
            """
            CREATE OR REPLACE TABLE panel2 AS
            SELECT
                * EXCLUDE (prev_close, change),
                lag(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close,
                close - lag(close) OVER (PARTITION BY symbol ORDER BY date) AS change
            FROM panel
            ORDER BY date, symbol
            """
        )
        stamp = datetime.now().strftime("%Y%m%d")
        out_file = out_dir / f"daily_stock_summary_stockbit_{stamp}.parquet"
        # Remove older stockbit daily panels to avoid duplicate rows in glob view.
        for old in out_dir.glob("daily_stock_summary_stockbit_*.parquet"):
            if old.resolve() != out_file.resolve():
                try:
                    old.unlink()
                except OSError:
                    pass
        con.execute(
            f"COPY panel2 TO '{out_file.as_posix()}' (FORMAT PARQUET)"
        )
        stats = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date) FROM panel2"
        ).fetchone()
        logger.info(
            "Rebuilt Stockbit daily summary %s rows=%s symbols=%s range=%s..%s",
            out_file.name,
            stats[0],
            stats[1],
            stats[2],
            stats[3],
        )
        return out_file
    finally:
        con.close()


def tvkit_exchange_symbol(symbol: str) -> str:
    """Map project Symbol / IHSG pseudo-symbol to a tvkit exchange:symbol."""
    sym = (symbol or "").strip().upper()
    if sym in {"IHSG", "COMPOSITE", "JKSE"}:
        return "INDEX:JKSE"
    return f"IDX:{sym}"


def _bar_to_dict(bar) -> dict:
    return {
        "ts": int(bar.timestamp),
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(bar.volume or 0),
    }


async def _fetch_one_async(client, symbol: str, timeframes: list[str]) -> dict:
    """Pull OHLCV for one Symbol via an open tvkit OHLCV client."""
    sym = (symbol or "").strip().upper()
    if not sym or not sym.replace(".", "").isalnum() or len(sym) > 8:
        return {
            "symbol": sym,
            "timeframes": {},
            "intercepted_urls": [],
            "unauthorized_urls": [],
            "error": "invalid symbol",
        }

    tv_sym = tvkit_exchange_symbol(sym)
    result_tfs: dict = {}
    fetch_errors: list[str] = []

    for tf in timeframes:
        interval = TVKIT_INTERVAL.get(tf)
        if not interval:
            result_tfs[tf] = {"bars": [], "source_urls": [], "error": "unsupported tf"}
            fetch_errors.append(f"{tf}:unsupported")
            continue
        bars_count = DEFAULT_BARS_COUNT.get(tf, 3000)
        source_url = f"tvkit://{tv_sym}/{interval}"
        try:
            bars = await client.get_historical_ohlcv(
                exchange_symbol=tv_sym,
                interval=interval,
                bars_count=bars_count,
            )
            result_tfs[tf] = {
                "bars": [_bar_to_dict(b) for b in (bars or [])],
                "source_urls": [source_url],
            }
            if not bars:
                fetch_errors.append(f"{tf}:empty")
        except Exception as exc:  # noqa: BLE001
            logger.warning("tvkit fetch failed %s %s: %s", sym, tf, exc)
            result_tfs[tf] = {
                "bars": [],
                "source_urls": [source_url],
                "error": f"{type(exc).__name__}: {exc}",
            }
            fetch_errors.append(f"{tf}:{type(exc).__name__}")

    any_bars = any((p.get("bars") or []) for p in result_tfs.values())
    err = None if any_bars else ("NO_OHLCV" if fetch_errors else "NO_OHLCV")
    return {
        "symbol": sym,
        "timeframes": result_tfs,
        "intercepted_urls": [f"tvkit://{tv_sym}"],
        "unauthorized_urls": [],
        "error": err,
        "fetch_errors": fetch_errors,
    }


def farm_one(
    symbol: str,
    timeframes: list[str],
    *,
    headless: bool | None = None,
    wait_login_sec: int = 0,
) -> dict:
    """Farm one Symbol via tvkit. ``headless`` / ``wait_login_sec`` kept for callers."""
    del headless, wait_login_sec  # unused — no browser for chart farm
    tfs = [t.strip().upper() for t in timeframes if str(t).strip()]

    async def _run():
        from tvkit.api.chart.ohlcv import OHLCV

        async with OHLCV() as client:
            return await _fetch_one_async(client, symbol, tfs)

    return asyncio.run(_run())


def run_farm(
    symbols: list[str],
    timeframes: list[str] | None = None,
    *,
    headless: bool | None = None,
    sync_daily: bool = False,
    sleep_sec: float = 0.25,
    on_progress: ProgressFn | None = None,
) -> str:
    """Farm OHLCV for each symbol via tvkit and write parquet. Returns a text summary."""
    del headless  # unused — no browser for chart farm
    tfs = [t.strip().upper() for t in (timeframes or DEFAULT_TIMEFRAMES) if t.strip()]
    if not symbols:
        return "ERROR: tidak ada simbol untuk di-farm."

    meta_dir = CHARTS_DIR / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    def progress(msg: str) -> None:
        logger.info("[farm] %s", msg)
        if on_progress:
            try:
                on_progress(msg)
            except Exception:  # noqa: BLE001
                pass

    ok_symbols: list[str] = []
    failed: list[str] = []
    lines: list[str] = []
    total = len(symbols)
    progress(f"Mulai farming tvkit {total} simbol x {tfs}")

    async def _run_all() -> None:
        from tvkit.api.chart.ohlcv import OHLCV

        async with OHLCV() as client:
            for i, sym in enumerate(symbols, start=1):
                try:
                    result = await _fetch_one_async(client, sym, tfs)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Farm failed for %s", sym)
                    failed.append(sym)
                    lines.append(f"{sym}: GAGAL ({exc})")
                    continue

                err = result.get("error")
                meta = {
                    "symbol": sym,
                    "source": BAR_SOURCE,
                    "error": err,
                    "intercepted_urls": result.get("intercepted_urls") or [],
                    "unauthorized_urls": result.get("unauthorized_urls") or [],
                    "fetch_errors": result.get("fetch_errors") or [],
                    "timeframes": {},
                }
                counts: list[str] = []
                any_bars = False
                for tf, payload in (result.get("timeframes") or {}).items():
                    bars = payload.get("bars") or []
                    meta["timeframes"][tf] = {
                        "n": len(bars),
                        "source_urls": payload.get("source_urls") or [],
                        "error": payload.get("error"),
                    }
                    if not bars:
                        counts.append(f"{tf}=0")
                        continue
                    df = bars_to_df(sym, tf, bars)
                    save_tf_parquet(sym, tf, df)
                    counts.append(f"{tf}={len(df)}")
                    any_bars = True

                (meta_dir / f"{sym}.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )

                if any_bars:
                    ok_symbols.append(sym)
                    lines.append(f"{sym}: OK ({', '.join(counts)})")
                else:
                    failed.append(sym)
                    hint = "tvkit kosong / symbol tidak ada di TradingView"
                    if result.get("fetch_errors"):
                        hint = ", ".join(result["fetch_errors"][:4])
                    lines.append(f"{sym}: 0 bars ({hint})")

                if i % 5 == 0 or i == total:
                    progress(
                        f"{i}/{total} selesai — OK={len(ok_symbols)} gagal={len(failed)}"
                    )
                if sleep_sec > 0:
                    await asyncio.sleep(sleep_sec)

    asyncio.run(_run_all())

    summary = [
        f"Farming selesai: OK={len(ok_symbols)} gagal={len(failed)} dari {total} simbol",
        f"source={BAR_SOURCE} timeframe={','.join(tfs)}",
        f"output={CHARTS_DIR.relative_to(ROOT).as_posix()}/{{SYMBOL}}/{{TF}}.parquet",
    ]
    summary.extend(lines[:40])
    if len(lines) > 40:
        summary.append(f"... (+{len(lines) - 40} simbol lagi)")

    if sync_daily and ok_symbols:
        out = rebuild_stockbit_daily_summary()
        if out:
            summary.append(f"Sync daily panel -> {out.relative_to(ROOT).as_posix()}")

    return "\n".join(summary)


# --------------------------------------------------------------------------- #
# Job staging (confirmation-gated, mirrors project_fs write staging)          #
# --------------------------------------------------------------------------- #

_PENDING_JOBS: dict[int, dict] = {}


def stage_farm_job(
    chat_key: int,
    symbols: list[str],
    timeframes: list[str],
    *,
    sync_daily: bool = False,
) -> str:
    # ~1.5s per symbol×TF with tvkit (vs ~12s Chartbit scrape).
    est_min = max(1, round(len(symbols) * len(timeframes) * 1.5 / 60))
    job = {
        "symbols": symbols,
        "timeframes": timeframes,
        "sync_daily": bool(sync_daily),
        "est_min": est_min,
    }
    _PENDING_JOBS[chat_key] = job
    preview = ", ".join(symbols[:15]) + ("..." if len(symbols) > 15 else "")
    return (
        "STAGED (menunggu konfirmasi user)\n"
        f"job=farm_stockbit_charts simbol={len(symbols)} timeframe={','.join(timeframes)}\n"
        f"source={BAR_SOURCE} sync_daily={sync_daily} estimasi={est_min} menit\n"
        f"daftar: {preview}\n"
        "CATATAN: farming BELUM jalan. User harus menyetujui (ya/tidak) di Telegram."
    )


def has_pending_job(chat_key: int) -> bool:
    return chat_key in _PENDING_JOBS


def peek_pending_job(chat_key: int) -> dict | None:
    return _PENDING_JOBS.get(chat_key)


def describe_pending_job(chat_key: int) -> str:
    job = _PENDING_JOBS.get(chat_key)
    if not job:
        return "(tidak ada job tertunda)"
    symbols = job["symbols"]
    preview = ", ".join(symbols[:20]) + ("..." if len(symbols) > 20 else "")
    return (
        "Job farming chart (tvkit) menunggu konfirmasi:\n"
        f"• simbol: {len(symbols)} ({preview})\n"
        f"• timeframe: {', '.join(job['timeframes'])}\n"
        f"• sync ke parquet harian: {job['sync_daily']}\n"
        f"• estimasi durasi: ~{job['est_min']} menit\n"
        f"• output: data/charts/{{SYMBOL}}/{{TF}}.parquet"
    )


def clear_pending_job(chat_key: int) -> bool:
    return _PENDING_JOBS.pop(chat_key, None) is not None


def run_pending_job(chat_key: int, *, on_progress: ProgressFn | None = None) -> str:
    job = _PENDING_JOBS.pop(chat_key, None)
    if not job:
        return "Tidak ada job farming untuk dijalankan."
    return run_farm(
        job["symbols"],
        job["timeframes"],
        sync_daily=job["sync_daily"],
        on_progress=on_progress,
    )
