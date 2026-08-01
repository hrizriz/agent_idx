"""Unit tests for Stock Analysis (analyze_stock)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from agent_idx.stock_analysis import (
    analyze_stock_dict,
    detect_conflicts,
    detect_sector,
    evaluate_fundamentals,
    evaluate_technicals,
    resolve_preset,
    Signal,
)

WIB = ZoneInfo("Asia/Jakarta")


def _ohlcv_rows(n: int = 40, *, trend_up: bool = True) -> pd.DataFrame:
    now = datetime.now(WIB).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    price = 1000.0
    for i in range(n):
        if trend_up:
            price += 5
        else:
            price -= 5
        close = price
        high = close + 10
        low = close - 10
        open_ = close - 2
        ts = int((now - timedelta(days=n - i)).timestamp())
        rows.append(
            {
                "datetime": (now - timedelta(days=n - i)).strftime("%Y-%m-%d %H:%M:%S"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1_000_000 + (200_000 if i == n - 1 else 0),
                "sma_5": close - (5 if trend_up else -5),
                "sma_20": close - (20 if trend_up else -20),
                "sma_50": close - (40 if trend_up else -40),
                "sma_200": close - (80 if trend_up else -80),
                "ema_12": close,
                "ema_26": close - 5,
                "macd": 5 if trend_up else -5,
                "macd_signal": 2 if trend_up else -2,
                "macd_hist": 3 if trend_up else -3,
                "rsi_14": 58 if trend_up else 42,
                "vol_sma_20": 1_000_000,
                "ts": ts,
            }
        )
    # Align MA ladder for stacked up/down
    last = rows[-1]
    if trend_up:
        last["sma_5"] = last["close"] - 5
        last["sma_20"] = last["close"] - 20
        last["sma_50"] = last["close"] - 40
        last["close"] = last["sma_5"] + 10  # above all
    else:
        last["sma_5"] = last["close"] + 5
        last["sma_20"] = last["close"] + 20
        last["sma_50"] = last["close"] + 40
        last["close"] = last["sma_5"] - 10  # below all
    return pd.DataFrame(rows)


def _fund_rows(bank: bool = True) -> pd.DataFrame:
    as_of = datetime.now(WIB).strftime("%Y-%m-%d")
    rows = [
        {
            "section": "company background",
            "metric": "background",
            "period": "",
            "value_raw": (
                "PT Bank Contoh Tbk bergerak di bidang usaha bank umum."
                if bank
                else "PT Contoh Consumer Tbk bergerak di industri barang konsumsi."
            ),
            "value_num": None,
            "as_of": as_of,
            "updated_at": as_of,
        },
        {
            "section": "metrics (structured)",
            "metric": "Current PE Ratio (TTM)",
            "period": "",
            "value_raw": "13.3",
            "value_num": 13.3,
            "as_of": as_of,
            "updated_at": as_of,
        },
        {
            "section": "metrics (structured)",
            "metric": "Rank (Current PE Ratio TTM)",
            "period": "",
            "value_raw": "25.00%",
            "value_num": 25.0,
            "as_of": as_of,
            "updated_at": as_of,
        },
        {
            "section": "metrics (structured)",
            "metric": "Return on Equity (TTM)",
            "period": "",
            "value_raw": "18.00%",
            "value_num": 18.0,
            "as_of": as_of,
            "updated_at": as_of,
        },
        {
            "section": "metrics (structured)",
            "metric": "Net Interest Margin (NIM)",
            "period": "",
            "value_raw": "5.10%",
            "value_num": 5.1,
            "as_of": as_of,
            "updated_at": as_of,
        },
        {
            "section": "metrics (structured)",
            "metric": "NPL - Gross",
            "period": "",
            "value_raw": "1.50%",
            "value_num": 1.5,
            "as_of": as_of,
            "updated_at": as_of,
        },
        {
            "section": "metrics (structured)",
            "metric": "Capital Adequacy Ratio",
            "period": "",
            "value_raw": "22.00%",
            "value_num": 22.0,
            "as_of": as_of,
            "updated_at": as_of,
        },
        {
            "section": "metrics (structured)",
            "metric": "Loan to Deposit Ratio",
            "period": "",
            "value_raw": "82.00%",
            "value_num": 82.0,
            "as_of": as_of,
            "updated_at": as_of,
        },
        {
            "section": "metrics (structured)",
            "metric": "Dividend Yield",
            "period": "",
            "value_raw": "4.50%",
            "value_num": 4.5,
            "as_of": as_of,
            "updated_at": as_of,
        },
        {
            "section": "metrics (structured)",
            "metric": "Revenue (Quarter YoY Growth)",
            "period": "",
            "value_raw": "8.00%",
            "value_num": 8.0,
            "as_of": as_of,
            "updated_at": as_of,
        },
        {
            "section": "metrics (structured)",
            "metric": "Net Income (Quarter YoY Growth)",
            "period": "",
            "value_raw": "6.00%",
            "value_num": 6.0,
            "as_of": as_of,
            "updated_at": as_of,
        },
    ]
    return pd.DataFrame(rows)


def test_resolve_preset_aliases():
    assert resolve_preset("swing").name == "swing"
    assert resolve_preset("investment").name == "position"
    assert resolve_preset("nope").name == "swing"


def test_detect_sector_bank_from_background():
    info = detect_sector(_fund_rows(bank=True))
    assert info["sector"] == "bank"
    assert info["adapter"] == "bank"


def test_detect_sector_general():
    info = detect_sector(_fund_rows(bank=False))
    assert info["sector"] == "general"
    assert info["adapter"] == "general"


def test_technical_uptrend_labels():
    preset = resolve_preset("swing")
    signals, gaps = evaluate_technicals(_ohlcv_rows(trend_up=True), preset)
    by_dim = {s.dimension: s for s in signals}
    assert by_dim["trend"].label == "positive"
    assert by_dim["momentum"].label in {"positive", "neutral"}
    assert not any("no OHLCV" in g for g in gaps)


def test_fundamental_bank_adapter_emits_bank_quality():
    signals, gaps = evaluate_fundamentals(_fund_rows(bank=True), adapter="bank")
    dims = {s.dimension for s in signals}
    assert "bank_quality" in dims
    by_dim = {s.dimension: s for s in signals}
    assert by_dim["valuation"].label == "positive"  # PE rank 25
    assert by_dim["profitability"].label == "positive"
    assert by_dim["bank_quality"].label == "positive"


def test_missing_fundamentals_are_gap_data():
    signals, gaps = evaluate_fundamentals(pd.DataFrame(), adapter="general")
    assert gaps and gaps[0].startswith("GAP_DATA")
    assert all(s.label == "unknown" for s in signals)


def test_conflicts_when_trend_and_profitability_oppose():
    preset = resolve_preset("swing")
    tech = [Signal("trend", "negative", 0.8, [], ["down"])]
    fund = [Signal("profitability", "positive", 0.8, [], ["roe"])]
    conflicts = detect_conflicts(tech, fund, preset)
    assert any(c["id"] == "trend_vs_profitability" for c in conflicts)


def test_analyze_stock_dict_json_shape_no_buysell():
    payload = analyze_stock_dict(
        "TEST",
        "swing",
        ohlcv=_ohlcv_rows(trend_up=True),
        fundamentals=_fund_rows(bank=True),
    )
    assert payload["symbol"] == "TEST"
    assert payload["horizon"] == "swing"
    assert "technical_signals" in payload
    assert "fundamental_signals" in payload
    assert "conflicts" in payload
    assert "freshness" in payload
    blob = json.dumps(payload).lower()
    assert "buy" not in blob or "buy/sell" in blob  # notes may mention absence
    assert payload["context"]["adapter"] == "bank"
    # Explicitly no action verdict key
    assert "verdict" not in payload
    assert "score" not in payload


def test_stale_technical_reduces_confidence(monkeypatch):
    # Make bars look old
    df = _ohlcv_rows(trend_up=True)
    old = (datetime.now(WIB) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    df.loc[df.index[-1], "datetime"] = old
    # Tighten swing technical max age via env
    monkeypatch.setenv("ANALYSIS_SWING_TECH_HOURS", "24")
    payload = analyze_stock_dict(
        "TEST",
        "swing",
        ohlcv=df,
        fundamentals=_fund_rows(bank=True),
    )
    assert payload["freshness"]["technical"]["status"] == "STALE_DATA"
    trend = next(s for s in payload["technical_signals"] if s["dimension"] == "trend")
    # Fresh uptrend confidence ~0.82; stale scales by 0.55 → ~0.45
    assert trend["confidence"] < 0.6
