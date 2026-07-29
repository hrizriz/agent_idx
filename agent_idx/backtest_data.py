from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _yyyymmdd_to_date(n: int) -> str:
    s = str(int(n))
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def fetch_yfinance_symbol(symbol: str, start_date: int, end_date: int) -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance for an IDX Symbol (`SYMBOL.JK`)."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance belum terpasang. Jalankan: py -3 -m pip install yfinance"
        ) from exc

    sym = symbol.strip().upper()
    yahoo_symbol = f"{sym}.JK"
    start = _yyyymmdd_to_date(start_date)
    end = _yyyymmdd_to_date(end_date)
    raw = yf.download(
        yahoo_symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    out = raw.reset_index()
    rename = {c: str(c).lower() for c in out.columns}
    out = out.rename(columns=rename)
    date_col = "date" if "date" in out.columns else out.columns[0]
    out["date"] = pd.to_datetime(out[date_col]).dt.strftime("%Y%m%d").astype(int)
    out["symbol"] = sym
    out["open_price"] = out.get("open", out.get("Open"))
    out["high"] = out.get("high", out.get("High"))
    out["low"] = out.get("low", out.get("Low"))
    out["close"] = out.get("close", out.get("Close"))
    out["volume"] = out.get("volume", out.get("Volume", 0)).fillna(0)
    out["prev_close"] = out.groupby("symbol")["close"].shift(1)
    out["change"] = (out["close"] / out["prev_close"] - 1) * 100
    # yfinance has no IDX foreign flow — leave NaN (never invent 0).
    out["foreign_buy"] = np.nan
    out["foreign_sell"] = np.nan
    out["listed_shares"] = None
    out["value"] = out["close"] * out["volume"]
    out["market_cap"] = None
    out["source"] = "yfinance"
    cols = [
        "symbol",
        "date",
        "open_price",
        "high",
        "low",
        "close",
        "prev_close",
        "change",
        "volume",
        "value",
        "foreign_buy",
        "foreign_sell",
        "listed_shares",
        "market_cap",
        "source",
    ]
    return out[cols].dropna(subset=["close"])


def panel_coverage_report(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"symbols": 0, "parquet_rows": 0, "yfinance_rows": 0, "total_rows": 0}
    src = df.get("source", "parquet")
    if isinstance(src, pd.Series):
        parquet_rows = int((src == "parquet").sum())
        yfinance_rows = int((src == "yfinance").sum())
    else:
        parquet_rows = len(df)
        yfinance_rows = 0
    return {
        "symbols": int(df["symbol"].nunique()),
        "parquet_rows": parquet_rows,
        "yfinance_rows": yfinance_rows,
        "total_rows": int(len(df)),
        "date_min": int(df["date"].min()),
        "date_max": int(df["date"].max()),
    }


def supplement_panel(
    df: pd.DataFrame,
    symbols: list[str],
    *,
    start_date: int,
    end_date: int,
    min_coverage_ratio: float = 0.85,
) -> tuple[pd.DataFrame, dict]:
    """
    Fill missing symbol-date rows using yfinance when parquet coverage is low.
    Returns merged panel + stats dict.
    """
    if df.empty or not symbols:
        return df, {"fetched_symbols": 0, "added_rows": 0}

    expected_days = df["date"].nunique()
    stats = {"fetched_symbols": 0, "added_rows": 0, "symbols_checked": len(symbols)}
    extras: list[pd.DataFrame] = []

    existing = df.groupby("symbol")["date"].nunique()
    for i, sym in enumerate(symbols):
        have = int(existing.get(sym, 0))
        if expected_days > 0 and have / expected_days >= min_coverage_ratio:
            continue
        if i > 0 and i % 10 == 0:
            time.sleep(0.5)
        try:
            ext = fetch_yfinance_symbol(sym, start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance %s failed: %s", sym, exc)
            continue
        if ext.empty:
            continue
        parquet_dates = set(df.loc[df["symbol"] == sym, "date"].tolist())
        ext = ext[~ext["date"].isin(parquet_dates)]
        if ext.empty:
            continue
        # Carry forward listed_shares / market_cap from parquet if available.
        ref = df.loc[df["symbol"] == sym].sort_values("date").tail(1)
        if not ref.empty and ref["listed_shares"].notna().any():
            ls = float(ref["listed_shares"].dropna().iloc[-1])
            ext["listed_shares"] = ls
            ext["market_cap"] = ext["close"] * ls
        extras.append(ext)
        stats["fetched_symbols"] += 1
        stats["added_rows"] += len(ext)

    if not extras:
        out = df.copy()
        if "source" not in out.columns:
            out["source"] = "parquet"
        return out, stats

    merged = pd.concat([df.assign(source=df.get("source", "parquet")), *extras], ignore_index=True)
    merged.sort_values(["symbol", "date"], inplace=True)
    merged.drop_duplicates(subset=["symbol", "date"], keep="first", inplace=True)
    return merged.reset_index(drop=True), stats
