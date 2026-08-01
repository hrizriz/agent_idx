"""DuckDB store for Chartbit OHLCV + features (queryable by bot tools / pipelines)."""
from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = ROOT / "data" / "charts"
DEFAULT_DB = ROOT / "data" / "charts.duckdb"

_LOCK = threading.RLock()


def db_path() -> Path:
    raw = (os.getenv("CHART_DB_PATH") or "").strip()
    path = Path(raw) if raw else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path()))


def init_schema(con: duckdb.DuckDBPyConnection | None = None) -> None:
    own = con is None
    if own:
        con = connect()
    assert con is not None
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol VARCHAR NOT NULL,
            timeframe VARCHAR NOT NULL,
            ts BIGINT NOT NULL,
            datetime VARCHAR,
            date INTEGER,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            sma_5 DOUBLE,
            sma_20 DOUBLE,
            sma_50 DOUBLE,
            sma_200 DOUBLE,
            ema_12 DOUBLE,
            ema_26 DOUBLE,
            macd DOUBLE,
            macd_signal DOUBLE,
            macd_hist DOUBLE,
            rsi_14 DOUBLE,
            vol_sma_20 DOUBLE,
            source VARCHAR,
            updated_at TIMESTAMP
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_pk
            ON ohlcv (symbol, timeframe, ts);
        CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_tf
            ON ohlcv (symbol, timeframe);
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id VARCHAR PRIMARY KEY,
            intent VARCHAR,
            pipeline VARCHAR,
            tools_json VARCHAR,
            status VARCHAR,
            detail VARCHAR,
            created_at TIMESTAMP
        );
        """
    )
    # Migrate older DBs that predate MA5/MA200.
    for col in ("sma_5", "sma_200", "vol_sma_20"):
        try:
            con.execute(f"ALTER TABLE ohlcv ADD COLUMN {col} DOUBLE")
        except Exception:  # noqa: BLE001
            pass
    if own:
        con.close()


def upsert_ohlcv(df: pd.DataFrame) -> int:
    """Replace rows for (symbol, timeframe) pairs present in df. Returns row count."""
    if df is None or df.empty:
        return 0
    work = df.copy()
    if "symbol" not in work.columns or "timeframe" not in work.columns:
        raise ValueError("df must have symbol and timeframe columns")
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["timeframe"] = work["timeframe"].astype(str).str.upper()
    work["updated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

    cols = [
        "symbol",
        "timeframe",
        "ts",
        "datetime",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_5",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "vol_sma_20",
        "source",
        "updated_at",
    ]
    for c in cols:
        if c not in work.columns:
            work[c] = None
    work = work[cols]

    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            pairs = (
                work[["symbol", "timeframe"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            for sym, tf in pairs:
                con.execute(
                    "DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                    [sym, tf],
                )
            con.register("_ohlcv_upsert", work)
            col_list = ", ".join(cols)
            con.execute(
                f"INSERT INTO ohlcv ({col_list}) SELECT {col_list} FROM _ohlcv_upsert"
            )
            n = len(work)
            logger.info(
                "DuckDB upsert ohlcv rows=%s pairs=%s",
                n,
                work[["symbol", "timeframe"]].drop_duplicates().shape[0],
            )
            return n
        finally:
            con.close()


def _read_parquet_df(path: Path) -> pd.DataFrame:
    """Read parquet via DuckDB (no pyarrow required)."""
    con = duckdb.connect(":memory:")
    try:
        return con.execute(
            f"SELECT * FROM read_parquet('{path.as_posix()}')"
        ).df()
    finally:
        con.close()


def sync_symbol_tf(symbol: str, timeframe: str, parquet_path: Path | None = None) -> int:
    """Load one parquet into DuckDB."""
    sym = (symbol or "").strip().upper()
    tf = (timeframe or "").strip().upper()
    path = parquet_path or (CHARTS_DIR / sym / f"{tf}.parquet")
    if not path.is_file():
        return 0
    df = _read_parquet_df(path)
    if df.empty:
        return 0
    if "symbol" not in df.columns:
        df["symbol"] = sym
    if "timeframe" not in df.columns:
        df["timeframe"] = tf
    return upsert_ohlcv(df)


def sync_from_parquet(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> str:
    """Bulk sync data/charts/{SYM}/{TF}.parquet → DuckDB."""
    init_schema()
    sym_filter = {s.strip().upper() for s in (symbols or []) if s.strip()} or None
    tf_filter = {t.strip().upper() for t in (timeframes or []) if t.strip()} or None

    total_files = 0
    total_rows = 0
    skipped = 0
    if not CHARTS_DIR.is_dir():
        return f"ERROR: folder chart tidak ada: {CHARTS_DIR}"

    for sym_dir in sorted(CHARTS_DIR.iterdir()):
        if not sym_dir.is_dir() or sym_dir.name.startswith("_"):
            continue
        sym = sym_dir.name.upper()
        if sym_filter and sym not in sym_filter:
            continue
        for pq in sorted(sym_dir.glob("*.parquet")):
            tf = pq.stem.upper()
            if tf_filter and tf not in tf_filter:
                continue
            try:
                n = sync_symbol_tf(sym, tf, pq)
                total_files += 1
                total_rows += n
            except Exception as exc:  # noqa: BLE001
                logger.warning("sync %s/%s gagal: %s", sym, tf, exc)
                skipped += 1

    return (
        f"OK: sync DuckDB selesai\n"
        f"db={db_path()}\n"
        f"files={total_files} rows={total_rows} skipped={skipped}"
    )


def query_ohlcv(
    symbol: str,
    timeframe: str,
    *,
    n: int = 50,
    ascending: bool = False,
) -> str:
    """Return latest (or oldest) OHLCV rows as text table."""
    sym = (symbol or "").strip().upper()
    tf = (timeframe or "1H").strip().upper()
    limit = max(1, min(int(n or 50), 500))
    order = "ASC" if ascending else "DESC"

    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            count = con.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                [sym, tf],
            ).fetchone()[0]
            if not count:
                # Lazy sync from parquet if present
                path = CHARTS_DIR / sym / f"{tf}.parquet"
                if path.is_file():
                    con.close()
                    sync_symbol_tf(sym, tf, path)
                    con = connect()
                    init_schema(con)
                    count = con.execute(
                        "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                        [sym, tf],
                    ).fetchone()[0]
            if not count:
                return (
                    f"ERROR: belum ada OHLCV {sym}/{tf} di DuckDB maupun parquet.\n"
                    f"Farm dulu: farm_stockbit_charts symbols={sym} timeframes={tf}"
                )
            df = con.execute(
                f"""
                SELECT datetime, open, high, low, close, volume,
                       sma_5, sma_20, sma_50, sma_200, ema_12, macd, rsi_14, ts
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                ORDER BY ts {order}
                LIMIT ?
                """,
                [sym, tf, limit],
            ).df()
        finally:
            con.close()

    if not ascending:
        df = df.iloc[::-1].reset_index(drop=True)

    lines = [
        f"DuckDB OHLCV {sym} TF {tf} — total={count} bar, tampil={len(df)}",
        f"db={db_path()}",
        "",
    ]
    for _, row in df.iterrows():
        when = str(row.get("datetime") or "")[:19]
        lines.append(
            f"- {when}  O={row.get('open')} H={row.get('high')} "
            f"L={row.get('low')} C={row.get('close')} V={row.get('volume')} "
            f"MA5={_fmt(row.get('sma_5'))} MA20={_fmt(row.get('sma_20'))} "
            f"MA50={_fmt(row.get('sma_50'))} MA200={_fmt(row.get('sma_200'))} "
            f"RSI={_fmt(row.get('rsi_14'))} MACD={_fmt(row.get('macd'))}"
        )
    return "\n".join(lines)


def chart_features(
    symbol: str,
    timeframes: str | list[str] | None = None,
    *,
    lookback: int = 40,
) -> str:
    """Multi-TF feature summary from DuckDB for LLM analysis."""
    sym = (symbol or "").strip().upper()
    if isinstance(timeframes, str):
        tfs = [t.strip().upper() for t in timeframes.split(",") if t.strip()]
    elif timeframes:
        tfs = [str(t).strip().upper() for t in timeframes if str(t).strip()]
    else:
        tfs = ["5M", "1H", "1D"]
    lookback = max(5, min(int(lookback or 40), 500))

    blocks: list[str] = [f"Chart features {sym} (DuckDB)", f"db={db_path()}", ""]
    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            for tf in tfs:
                # Lazy parquet sync
                n = con.execute(
                    "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                    [sym, tf],
                ).fetchone()[0]
                if not n:
                    path = CHARTS_DIR / sym / f"{tf}.parquet"
                    if path.is_file():
                        con.close()
                        sync_symbol_tf(sym, tf, path)
                        con = connect()
                        init_schema(con)

                df = con.execute(
                    """
                    SELECT datetime, open, high, low, close, volume,
                           sma_5, sma_20, sma_50, sma_200,
                           ema_12, macd, macd_signal, rsi_14, ts
                    FROM ohlcv
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    [sym, tf, lookback],
                ).df()
                if df.empty:
                    blocks.append(f"### {tf}\n(belum ada data — farm dulu)\n")
                    continue
                df = df.iloc[::-1].reset_index(drop=True)
                closes = pd.to_numeric(df["close"], errors="coerce").dropna()
                last = float(closes.iloc[-1])
                first = float(closes.iloc[0])
                chg = last - first
                pct = (chg / first * 100.0) if first else 0.0
                hi = float(pd.to_numeric(df["high"], errors="coerce").max())
                lo = float(pd.to_numeric(df["low"], errors="coerce").min())
                rsi = df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else None
                macd = df["macd"].iloc[-1] if "macd" in df.columns else None
                signal = (
                    df["macd_signal"].iloc[-1] if "macd_signal" in df.columns else None
                )
                sma5 = df["sma_5"].iloc[-1] if "sma_5" in df.columns else None
                sma20 = df["sma_20"].iloc[-1] if "sma_20" in df.columns else None
                sma50 = df["sma_50"].iloc[-1] if "sma_50" in df.columns else None
                sma200 = df["sma_200"].iloc[-1] if "sma_200" in df.columns else None
                bar_chg = float(closes.iloc[-1] - closes.iloc[-2]) if len(closes) > 1 else 0.0
                if pct > 0.4:
                    bias = "bullish singkat"
                elif pct < -0.4:
                    bias = "bearish singkat"
                else:
                    bias = "sideways"

                vol_note = ""
                if "volume" in df.columns and len(df) >= 5:
                    vols = pd.to_numeric(df["volume"], errors="coerce").dropna()
                    if len(vols) >= 5:
                        last_v = float(vols.iloc[-1])
                        avg_v = float(vols.iloc[:-1].mean()) or 1.0
                        vol_note = (
                            f"vol_last={last_v:,.0f} vs avg={avg_v:,.0f} "
                            f"({last_v / avg_v:.2f}x)"
                        )

                # Price vs MA stack
                ma_pos = []
                for label, val in (
                    ("MA5", sma5),
                    ("MA20", sma20),
                    ("MA50", sma50),
                    ("MA200", sma200),
                ):
                    try:
                        if val is not None and float(val) == float(val):
                            ma_pos.append(
                                f"{label}={_fmt(val)}({'di atas' if last >= float(val) else 'di bawah'})"
                            )
                    except (TypeError, ValueError):
                        pass

                blocks.append(
                    f"### {tf} (n={len(df)} lookback)\n"
                    f"- last={last:g} bar_chg={bar_chg:+.4g}\n"
                    f"- range: high={hi:g} low={lo:g}\n"
                    f"- change_{lookback}={chg:+.4g} ({pct:+.2f}%) → {bias}\n"
                    f"- RSI14={_fmt(rsi)} MACD={_fmt(macd)} signal={_fmt(signal)}\n"
                    f"- MA: {', '.join(ma_pos) or 'n/a'}\n"
                    f"- support≈{lo:g} resist≈{hi:g}\n"
                    + (f"- {vol_note}\n" if vol_note else "")
                    + f"- window={df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}\n"
                )
        finally:
            con.close()

    blocks.append("Catatan: fitur dari DuckDB/Chartbit — bukan saran investasi.")
    return "\n".join(blocks)


def stats() -> str:
    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            row = con.execute(
                """
                SELECT COUNT(*) AS bars,
                       COUNT(DISTINCT symbol) AS symbols,
                       COUNT(DISTINCT timeframe) AS tfs,
                       MIN(datetime) AS min_dt,
                       MAX(datetime) AS max_dt
                FROM ohlcv
                """
            ).fetchone()
            top = con.execute(
                """
                SELECT symbol, timeframe, COUNT(*) AS n, MAX(datetime) AS last_dt
                FROM ohlcv
                GROUP BY 1, 2
                ORDER BY last_dt DESC NULLS LAST
                LIMIT 15
                """
            ).fetchall()
        finally:
            con.close()

    lines = [
        f"DuckDB chart store: {db_path()}",
        f"bars={row[0]} symbols={row[1]} timeframes={row[2]}",
        f"range={row[3]} → {row[4]}",
        "",
        "Terbaru:",
    ]
    for sym, tf, n, last in top:
        lines.append(f"- {sym}/{tf}: {n} bars last={last}")
    if not top:
        lines.append("(kosong — sync_chart_db atau farm chart dulu)")
    return "\n".join(lines)


def get_stockbit_ohlcv(
    symbol: str,
    timeframes: str | list[str] | None = None,
    *,
    n: int = 40,
    refresh: bool = False,
) -> str:
    """Primary backend tool helper: OHLCV for 5M/30M/1H/1D/1W.

    Reads DuckDB/parquet. If refresh=True or TF missing, farms via tvkit first.
    """
    from agent_idx import chart_farm

    sym = (symbol or "").strip().upper()
    if not sym or not sym.isalnum() or len(sym) > 5:
        return "ERROR: symbol tidak valid (contoh: BBCA, DSSA)"

    if isinstance(timeframes, str):
        raw_list = [
            t.strip() for t in timeframes.replace(";", ",").split(",") if t.strip()
        ]
    elif timeframes:
        raw_list = [str(t).strip() for t in timeframes if str(t).strip()]
    else:
        raw_list = ["1D"]

    tfs: list[str] = []
    for raw in raw_list:
        compact = raw.replace(" ", "")
        if re.fullmatch(r"(?i)1m|5m|15m|30m|1h|4h|1d|1w", compact):
            tf = compact.upper()
        else:
            tf = chart_farm.parse_timeframe(raw, default="")
        if not tf or tf not in chart_farm.SUPPORTED_TIMEFRAMES:
            return (
                f"ERROR: timeframe tidak didukung: {raw!r}. "
                "Pakai: 5M, 30M, 1H, 1D, 1W (atau 1M/15M/4H)."
            )
        if tf not in tfs:
            tfs.append(tf)

    n = max(1, min(int(n or 40), 500))
    missing = [
        tf for tf in tfs if not (CHARTS_DIR / sym / f"{tf}.parquet").is_file()
    ]
    farmed = False
    farm_note = ""
    if refresh or missing:
        need = tfs if refresh else missing
        try:
            result = chart_farm.farm_one(
                sym, need, wait_login_sec=90 if (refresh or missing) else 0
            )
            err = result.get("error")
            parts: list[str] = []
            any_saved = False
            for tf in need:
                payload = (result.get("timeframes") or {}).get(tf) or {}
                bars = payload.get("bars") or []
                if not bars:
                    parts.append(f"{tf}=0")
                    continue
                df = chart_farm.bars_to_df(sym, tf, bars)
                chart_farm.save_tf_parquet(sym, tf, df)
                sync_symbol_tf(sym, tf)
                parts.append(f"{tf}={len(df)}")
                any_saved = True
            farmed = any_saved
            if err and not any_saved:
                farm_note = f"farm_error={err}"
            elif err:
                farm_note = f"farm_partial error={err} " + ", ".join(parts)
            else:
                farm_note = "farm_ok " + ", ".join(parts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_stockbit_ohlcv farm failed")
            farm_note = f"farm_exception={exc}"

    blocks = [
        f"get_stockbit_ohlcv symbol={sym} timeframes={','.join(tfs)} n={n}",
        (
            f"source=tvkit (parquet+DuckDB) "
            f"refresh={bool(refresh)} farmed={farmed}"
        ),
    ]
    if farm_note:
        blocks.append(farm_note)
    blocks.append("")

    for tf in tfs:
        blocks.append(query_ohlcv(sym, tf, n=n))
        blocks.append("")
        blocks.append(chart_features(sym, [tf], lookback=min(n, 40)))
        blocks.append("")

    return "\n".join(blocks).strip()


def log_pipeline_run(
    run_id: str,
    *,
    intent: str,
    pipeline: str,
    tools: list[str],
    status: str,
    detail: str = "",
) -> None:
    import json

    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            con.execute(
                """
                INSERT OR REPLACE INTO pipeline_runs
                (run_id, intent, pipeline, tools_json, status, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    intent,
                    pipeline,
                    json.dumps(tools, ensure_ascii=False),
                    status,
                    (detail or "")[:4000],
                    datetime.now(timezone.utc).replace(tzinfo=None),
                ],
            )
        finally:
            con.close()


def _fmt(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "n/a"
    try:
        return f"{float(val):.4g}"
    except (TypeError, ValueError):
        return str(val)
