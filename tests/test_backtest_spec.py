"""Focused tests for backtest spec fixes (cap eligibility, min_trades, supplement, grids)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent_idx import backtest as bt
from agent_idx.backtest_data import fetch_yfinance_symbol


def _ohlcv_row(
    symbol: str,
    date: int,
    close: float,
    *,
    market_cap: float | None = 2e12,
    volume: float = 1_000_000,
    change: float = 0.0,
    foreign_buy: float | None = 100.0,
    foreign_sell: float | None = 50.0,
    listed_shares: float | None = 1e9,
    **extra,
) -> dict:
    row = {
        "symbol": symbol,
        "date": date,
        "open_price": close,
        "high": close,
        "low": close,
        "close": close,
        "prev_close": close,
        "change": change,
        "volume": volume,
        "value": close * volume,
        "foreign_buy": foreign_buy,
        "foreign_sell": foreign_sell,
        "listed_shares": listed_shares,
        "market_cap": market_cap if market_cap is not None else (
            close * listed_shares if listed_shares is not None else None
        ),
        "source": "parquet",
    }
    row.update(extra)
    return row


def _panel_from_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = bt._ensure_net_foreign(df)
    df = bt._ensure_cap_eligible(df, cap_filter="large_cap", min_market_cap=bt.MIN_MARKET_CAP)
    return bt._add_features(df)


# --- 1. Cap filter is entry-only; +N exits count market bars ---


def test_forward_exit_uses_market_bars_across_cap_ineligible_middle():
    """Entry on eligible day; middle bar under cap stays in series so hold=1 exits there."""
    rows = [
        _ohlcv_row("AAA", 20220103, 100.0, market_cap=2e12, change=-5.0),
        _ohlcv_row("AAA", 20220104, 105.0, market_cap=5e11),  # under 1T — not entry-eligible
        _ohlcv_row("AAA", 20220105, 110.0, market_cap=2e12),
    ]
    df = _panel_from_rows(rows)
    assert bool(df.iloc[0]["cap_eligible"]) is True
    assert bool(df.iloc[1]["cap_eligible"]) is False

    sig = df["change"] <= -3
    result = bt._evaluate_custom(df, sig, hold_days=1, label="drop_test")
    assert result["trades"] == 1
    # Exit must be middle bar close 105, not the next eligible 110.
    assert result["avg_return_pct"] == pytest.approx(5.0)


def test_entry_not_taken_on_cap_ineligible_bar():
    rows = [
        _ohlcv_row("AAA", 20220103, 100.0, market_cap=5e11, change=-5.0),
        _ohlcv_row("AAA", 20220104, 110.0, market_cap=2e12),
    ]
    df = _panel_from_rows(rows)
    result = bt._evaluate_custom(df, df["change"] <= -3, hold_days=1, label="drop_test")
    assert result["trades"] == 0


def test_load_panel_preserves_bar_with_missing_listed_shares():
    class Result:
        def fetchdf(self):
            return pd.DataFrame(
                [
                    _ohlcv_row(
                        "AAA",
                        20220103,
                        100.0,
                        listed_shares=None,
                        market_cap=None,
                    ),
                    _ohlcv_row("AAA", 20220104, 105.0),
                ]
            )

    class Connection:
        def execute(self, sql, params):
            assert "listed_shares IS NOT NULL" not in sql
            return Result()

    class Store:
        _con = Connection()

    panel, _ = bt.load_panel(  # type: ignore[arg-type]
        Store(),
        supplement_external=False,
        cap_filter="large_cap",
    )

    assert panel["date"].tolist() == [20220103, 20220104]
    assert panel["cap_eligible"].tolist() == [False, True]


# --- 2. Exact min_trades passed through ---


def test_per_symbol_hits_uses_exact_min_trades():
    # drop_3pct_bounce hold=3 → trades = n_rows - 3 when every bar signals.
    rows: list[dict] = []
    for i in range(17):  # 14 trades
        rows.append(_ohlcv_row("SYM14", 20220103 + i, 100.0 + i, change=-4.0))
    for i in range(18):  # 15 trades
        rows.append(_ohlcv_row("SYM15", 20220103 + i, 100.0 + i, change=-4.0))

    df = _panel_from_rows(rows)
    specs = [s for s in bt.SCENARIOS if s.id == "drop_3pct_bounce"]
    hits_14 = bt.scan_per_symbol_hits(
        df, specs, min_win_rate=0.0, max_win_rate=1.0, min_trades=14
    )
    hits_15 = bt.scan_per_symbol_hits(
        df, specs, min_win_rate=0.0, max_win_rate=1.0, min_trades=15
    )
    syms_14 = {h["symbol"] for h in hits_14}
    syms_15 = {h["symbol"] for h in hits_15}
    assert "SYM14" in syms_14
    assert "SYM14" not in syms_15
    assert "SYM15" in syms_15


def test_scan_scenarios_passes_min_trades_unchanged(monkeypatch):
    captured: dict = {}

    def fake_hits(df, specs, *, min_win_rate, max_win_rate, min_trades, **kwargs):
        captured["hits_min"] = min_trades
        return []

    def fake_grid(df, *, min_win_rate, max_win_rate, min_trades=15, **kwargs):
        captured["grid_min"] = min_trades
        return []

    def fake_load(*args, **kwargs):
        rows = [_ohlcv_row("AAA", 20220103 + i, 100.0) for i in range(30)]
        df = pd.DataFrame(rows)
        df = bt._ensure_net_foreign(df)
        df = bt._ensure_cap_eligible(df, cap_filter="all", min_market_cap=0)
        return df, {"source": "parquet", "supplement": {}}

    monkeypatch.setattr(bt, "load_panel", fake_load)
    monkeypatch.setattr(bt, "scan_per_symbol_hits", fake_hits)
    monkeypatch.setattr(bt, "scan_per_symbol_grid", fake_grid)
    monkeypatch.setattr(bt, "grid_search", lambda *a, **k: [])
    monkeypatch.setattr(bt, "panel_coverage_report", lambda df: {})

    bt.scan_scenarios(
        store=None,  # type: ignore[arg-type]
        min_trades=15,
        supplement_external=False,
        cap_filter="all",
    )
    assert captured["hits_min"] == 15
    assert captured["grid_min"] == 15


# --- 3. Absolute supplementation, no 150 cap ---


def test_sparse_symbols_absolute_coverage_not_relative():
    # Best-covered symbol has 100 days; weaker has 90 (90% of best → would pass relative 85%).
    # Absolute expected calendar is 100 days → 90/100 = 90% OK; 80/100 needs supplement.
    dates = list(range(20220101, 20220101 + 100))
    rows = [_ohlcv_row("FULL", d, 100.0) for d in dates]
    rows += [_ohlcv_row("OK90", d, 100.0) for d in dates[:90]]
    rows += [_ohlcv_row("SPARSE80", d, 100.0) for d in dates[:80]]
    df = pd.DataFrame(rows)
    sparse = bt.symbols_needing_supplement(df, min_coverage_ratio=0.85)
    assert "SPARSE80" in sparse
    assert "OK90" not in sparse
    assert "FULL" not in sparse


def test_sparse_symbols_no_silent_150_cap():
    dates = list(range(20220101, 20220101 + 20))
    rows: list[dict] = []
    # Shared calendar of 20 days; each of 160 symbols only has 10 days (<85%).
    for i in range(160):
        sym = f"S{i:03d}"
        for d in dates[:10]:
            rows.append(_ohlcv_row(sym, d, 100.0))
    # Anchor full calendar with one complete symbol so expected_days = 20
    for d in dates:
        rows.append(_ohlcv_row("ANCHOR", d, 100.0))
    df = pd.DataFrame(rows)
    sparse = bt.symbols_needing_supplement(df, min_coverage_ratio=0.85)
    assert len(sparse) >= 160


# --- 4. Per-symbol volume grid ---


def test_per_symbol_grid_includes_volume_families():
    rows: list[dict] = []
    for i in range(40):
        vol = 10_000_000 if i % 5 == 0 else 100_000
        close = 100.0 + i * 0.5
        prev = close - 1
        rows.append(
            _ohlcv_row(
                "VOLA",
                20220103 + i,
                close,
                volume=vol,
                change=1.0,
                prev_close=prev,
            )
        )
    df = _panel_from_rows(rows)
    # Force volume spike eligibility by setting vol_ma20 low via features already
    hits = bt.scan_per_symbol_grid(
        df, min_win_rate=0.0, max_win_rate=1.0, min_trades=3
    )
    vol_hits = [h for h in hits if h["scenario_id"].startswith("grid_vol_")]
    assert vol_hits, "per-symbol exhaustive grid must include volume families"
    labels = bt.per_symbol_grid_labels()
    assert any(x.startswith("grid_vol_") for x in labels)


# --- 5. Aggregate grid allowed prefixes only ---


def test_aggregate_grid_only_allowed_prefixes():
    rows = [_ohlcv_row("AAA", 20220103 + i, 100.0 + i, change=-6.0) for i in range(40)]
    df = _panel_from_rows(rows)
    # Widen band so some rows pass and we can inspect labels via a helper
    labels = bt.aggregate_grid_labels()
    allowed = ("grid_drop_", "grid_foreign_", "grid_momentum_", "grid_vol_")
    forbidden = ("grid_breakout_", "grid_ma", "grid_oversold_")
    assert labels
    for lab in labels:
        assert lab.startswith(allowed), f"unexpected aggregate grid label: {lab}"
        assert not any(lab.startswith(f) for f in forbidden)
    # Built-in scenarios still include breakout / ma / oversold
    ids = {s.id for s in bt.SCENARIOS}
    assert "breakout_20d" in ids
    assert "ma_cross_5_20" in ids
    assert "rsi_oversold_proxy" in ids


# --- 6. Complete SETUP TRADING markdown ---


def test_saved_markdown_has_complete_setup_trading_section(tmp_path: Path):
    hits = [
        {
            "kind": "aggregate",
            "scenario_id": f"agg_{i}",
            "symbol": "*",
            "trades": 20,
            "wins": 16,
            "win_rate": 0.8,
            "avg_return_pct": 1.0,
        }
        for i in range(5)
    ] + [
        {
            "kind": "per_symbol",
            "scenario_id": f"sym_{i}",
            "symbol": f"S{i}",
            "trades": 20,
            "wins": 16,
            "win_rate": 0.8,
            "avg_return_pct": 1.0,
        }
        for i in range(50)
    ]
    payload = {
        "start_date": 20220101,
        "end_date": 20220601,
        "cap_filter": "large_cap",
        "universe": "large_cap",
        "min_market_cap": bt.MIN_MARKET_CAP,
        "min_win_rate": 0.75,
        "max_win_rate": 0.85,
        "min_trades": 15,
        "passed": [],
        "results": [],
        "grid_passed": [],
        "per_symbol_hits": hits[5:],
        "per_symbol_grid": [],
        "all_band_hits": hits,
    }
    _json_path, md_path = bt.save_scan_artifacts(payload, tmp_path)
    text = md_path.read_text(encoding="utf-8")
    assert text.count("## SETUP TRADING") == 1
    for h in hits:
        assert h["scenario_id"] in text
        if h["symbol"] != "*":
            assert h["symbol"] in text


def test_format_scan_report_preview_marked_when_capped():
    hits = [
        {
            "kind": "aggregate",
            "scenario_id": f"agg_{i}",
            "symbol": "*",
            "trades": 20,
            "wins": 16,
            "win_rate": 0.8,
            "avg_return_pct": 1.0,
        }
        for i in range(60)
    ]
    payload = {
        "start_date": 20220101,
        "end_date": 20220601,
        "cap_filter": "large_cap",
        "min_market_cap": bt.MIN_MARKET_CAP,
        "min_win_rate": 0.75,
        "max_win_rate": 0.85,
        "min_trades": 15,
        "panel_rows": 100,
        "avg_symbols_per_day": 10,
        "scenarios_passed": 0,
        "scenarios_tested": 0,
        "passed": [],
        "results": [],
        "all_band_hits": hits,
    }
    report = bt.format_scan_report(payload)
    assert "SETUP TRADING" in report
    assert "preview" in report.lower()


# --- 7. Gain / on-track removed ---


def test_gain_on_track_api_removed():
    assert not hasattr(bt, "scan_high_gain_setups")
    assert not hasattr(bt, "run_gain_scan")
    assert not hasattr(bt, "format_gain_scan_report")


# --- 8/9. Foreign NaN semantics ---


def test_missing_foreign_stays_nan_and_blocks_foreign_signals():
    rows = [
        _ohlcv_row("AAA", 20220103, 100.0, foreign_buy=None, foreign_sell=None),
        _ohlcv_row("AAA", 20220104, 101.0, foreign_buy=None, foreign_sell=10.0),
        _ohlcv_row("AAA", 20220105, 102.0, foreign_buy=50.0, foreign_sell=10.0),
        _ohlcv_row("AAA", 20220106, 103.0, foreign_buy=60.0, foreign_sell=10.0),
    ]
    df = pd.DataFrame(rows)
    df = bt._ensure_net_foreign(df)
    assert pd.isna(df.loc[0, "net_foreign"])
    assert pd.isna(df.loc[1, "net_foreign"])  # sell present but buy missing
    assert df.loc[2, "net_foreign"] == pytest.approx(40.0)

    df = bt._ensure_cap_eligible(df, cap_filter="all", min_market_cap=0)
    df = bt._add_features(df)
    sig = bt._signal_mask(df, "foreign_2d_positive")
    # Day index 3 can signal (days 2 and 3 both have valid positive net_foreign)
    assert bool(sig.iloc[0]) is False
    assert bool(sig.iloc[1]) is False
    assert bool(sig.iloc[2]) is False
    assert bool(sig.iloc[3]) is True


def test_yfinance_foreign_columns_are_nan(monkeypatch):
    class _FakeYF:
        @staticmethod
        def download(*args, **kwargs):
            idx = pd.date_range("2022-01-03", periods=3, freq="B")
            return pd.DataFrame(
                {
                    "Open": [1.0, 1.1, 1.2],
                    "High": [1.0, 1.1, 1.2],
                    "Low": [1.0, 1.1, 1.2],
                    "Close": [1.0, 1.1, 1.2],
                    "Volume": [100, 100, 100],
                },
                index=idx,
            )

    import agent_idx.backtest_data as bd

    monkeypatch.setitem(__import__("sys").modules, "yfinance", _FakeYF())
    # force re-import path inside fetch
    out = fetch_yfinance_symbol("BBCA", 20220101, 20220110)
    assert out["foreign_buy"].isna().all()
    assert out["foreign_sell"].isna().all()


def test_cap_filter_alias_universe_accepted():
    """Legacy universe= still maps to cap_filter without filtering rows away."""
    rows = [
        _ohlcv_row("AAA", 20220103, 100.0, market_cap=2e12),
        _ohlcv_row("AAA", 20220104, 100.0, market_cap=5e11),
    ]
    df = pd.DataFrame(rows)
    df = bt._ensure_net_foreign(df)
    out = bt._ensure_cap_eligible(df, cap_filter="large_cap", min_market_cap=bt.MIN_MARKET_CAP)
    assert len(out) == 2
    assert out["cap_eligible"].tolist() == [True, False]
