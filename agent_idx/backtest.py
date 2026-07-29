from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from agent_idx.backtest_data import panel_coverage_report, supplement_panel
from agent_idx.data import StockDataStore

WIB = ZoneInfo("Asia/Jakarta")
DEFAULT_START = 20220101
MIN_MARKET_CAP = 1_000_000_000_000  # 1 Triliun IDR
DEFAULT_MIN_TRADES = 15
DEFAULT_MIN_WIN_RATE = 0.75
DEFAULT_MAX_WIN_RATE = 0.85
REPORT_PREVIEW_HITS = 40


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    name: str
    description: str
    hold_days: int
    entry: str
    exit_rule: str


SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec(
        id="foreign_2d_positive",
        name="Foreign flow 2 hari positif",
        description=(
            "Beli close jika net foreign (buy-sell) positif 2 hari berturut-turut. "
            "Cap filter: market cap >= 1T pada hari sinyal (entry only)."
        ),
        hold_days=5,
        entry="net_foreign > 0 pada hari t dan t-1",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="foreign_3d_streak",
        name="Foreign buy streak 3 hari",
        description="Beli close setelah net foreign positif 3 hari berturut.",
        hold_days=5,
        entry="net_foreign > 0 untuk 3 hari berturut",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="foreign_reversal",
        name="Foreign reversal",
        description=(
            "Net foreign 3 hari sebelumnya negatif (total), hari ini net foreign positif "
            "dan > volume rata-rata 20 hari."
        ),
        hold_days=5,
        entry="sum(net_foreign,3)<0 AND net_foreign>0 AND net_foreign>avg20",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="drop_3pct_bounce",
        name="Drop >=3% mean reversion",
        description="Beli close jika perubahan harga harian <= -3% (change column).",
        hold_days=3,
        entry="change <= -3",
        exit_rule="Jual close +3 hari kerja",
    ),
    ScenarioSpec(
        id="drop_5pct_bounce",
        name="Drop >=5% mean reversion",
        description="Beli close jika perubahan harian <= -5%.",
        hold_days=5,
        entry="change <= -5",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="breakout_20d",
        name="Breakout 20 hari",
        description="Close menembus high 20 hari sebelumnya (exclude hari ini).",
        hold_days=10,
        entry="close > max(close, 20 hari sebelumnya)",
        exit_rule="Jual close +10 hari kerja",
    ),
    ScenarioSpec(
        id="volume_spike_up",
        name="Volume spike + harga naik",
        description="Volume > 2x rata-rata 20 hari dan close > prev_close.",
        hold_days=5,
        entry="volume > 2*vol_ma20 AND close > prev_close",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="gap_down_open",
        name="Gap down open recovery",
        description="Open <= 98% prev_close, beli close hari yang sama.",
        hold_days=5,
        entry="open_price <= prev_close * 0.98",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="net_foreign_top_day",
        name="Net foreign ekstrem vs 20 hari",
        description="Net foreign hari ini >= 95th percentile rolling 20 hari (per emiten).",
        hold_days=5,
        entry="net_foreign >= quantile_95 rolling 20d",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="rsi_oversold_proxy",
        name="RSI proxy oversold",
        description="5-day return <= -8% (proxy oversold), beli close.",
        hold_days=5,
        entry="return_5d <= -8%",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="ma_cross_5_20",
        name="MA cross 5/20",
        description="MA5 cross above MA20 (golden cross short-term).",
        hold_days=7,
        entry="ma5 > ma20 AND ma5_prev <= ma20_prev",
        exit_rule="Jual close +7 hari kerja",
    ),
    ScenarioSpec(
        id="momentum_5d",
        name="Momentum 5 hari",
        description="Return 5 hari >= +5%, beli close (trend follow).",
        hold_days=5,
        entry="ret_5d >= 5",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="pullback_ma20",
        name="Pullback ke MA20",
        description="Close <= MA20 dan change <= -2% (pullback in uptrend proxy).",
        hold_days=5,
        entry="close <= ma20 AND change <= -2 AND ma20 > ma50",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="foreign_intensity",
        name="Foreign intensity",
        description="Net foreign > 2x abs(MA20) dan close naik.",
        hold_days=5,
        entry="net_foreign > 2*abs(nf_ma20) AND close > prev_close",
        exit_rule="Jual close +5 hari kerja",
    ),
    ScenarioSpec(
        id="breakout_10d",
        name="Breakout 10 hari",
        description="Close > high 10 hari sebelumnya.",
        hold_days=7,
        entry="close > high_10_prev",
        exit_rule="Jual close +7 hari kerja",
    ),
    ScenarioSpec(
        id="volume_spike_3x",
        name="Volume spike 3x",
        description="Volume > 3x MA20 dan close > prev_close.",
        hold_days=5,
        entry="volume > 3*vol_ma20 AND close > prev_close",
        exit_rule="Jual close +5 hari kerja",
    ),
]


def _resolve_cap_filter(
    cap_filter: str = "large_cap",
    universe: str | None = None,
) -> str:
    """Prefer explicit legacy `universe=` when provided; otherwise use `cap_filter`."""
    mode = universe if universe is not None else cap_filter
    if mode in {"none", "None", ""}:
        mode = "all"
    if mode not in {"large_cap", "all"}:
        raise ValueError(f"cap_filter tidak valid: {mode}")
    return mode


def _in_win_band(win_rate: float, min_wr: float, max_wr: float) -> bool:
    return min_wr <= win_rate <= max_wr


def _passes_filter(
    win_rate: float,
    trades: int,
    *,
    min_win_rate: float,
    max_win_rate: float,
    min_trades: int,
) -> bool:
    return trades >= min_trades and _in_win_band(win_rate, min_win_rate, max_win_rate)


def _ensure_net_foreign(df: pd.DataFrame) -> pd.DataFrame:
    """net_foreign only when both foreign_buy and foreign_sell exist; else NaN."""
    out = df.copy()
    out["net_foreign"] = out["foreign_buy"] - out["foreign_sell"]
    return out


def _ensure_cap_eligible(
    df: pd.DataFrame,
    *,
    cap_filter: str = "large_cap",
    min_market_cap: float = MIN_MARKET_CAP,
    universe: str | None = None,
) -> pd.DataFrame:
    """Mark entry eligibility; never drop bars from the chronological series."""
    out = df.copy()
    mode = _resolve_cap_filter(cap_filter, universe)
    if mode == "all":
        out["cap_eligible"] = True
    else:
        out["cap_eligible"] = out["market_cap"].notna() & (out["market_cap"] >= min_market_cap)
    return out


def symbols_needing_supplement(
    df: pd.DataFrame,
    *,
    min_coverage_ratio: float = 0.85,
) -> list[str]:
    """Symbols whose absolute trading-date coverage is below the threshold."""
    if df.empty:
        return []
    expected_days = int(df["date"].nunique())
    if expected_days <= 0:
        return []
    counts = df.groupby("symbol")["date"].nunique()
    sparse = counts[counts < expected_days * min_coverage_ratio]
    return sparse.index.astype(str).tolist()


def load_panel(
    store: StockDataStore,
    *,
    start_date: int = DEFAULT_START,
    min_market_cap: float = MIN_MARKET_CAP,
    cap_filter: str = "large_cap",
    universe: str | None = None,
    supplement_external: bool = True,
) -> tuple[pd.DataFrame, dict]:
    mode = _resolve_cap_filter(cap_filter, universe)
    df = store._con.execute(
        """
        SELECT
            symbol,
            date,
            open_price,
            high,
            low,
            close,
            prev_close,
            change,
            volume,
            value,
            foreign_buy,
            foreign_sell,
            listed_shares,
            close * listed_shares AS market_cap
        FROM daily_stock
        WHERE date >= ?
          AND close IS NOT NULL
        ORDER BY symbol, date
        """,
        [start_date],
    ).fetchdf()

    if df.empty:
        raise RuntimeError(f"Tidak ada data parquet sejak {start_date}")

    df = _ensure_net_foreign(df)
    df["source"] = "parquet"
    meta: dict = {"source": "parquet", "supplement": {}, "cap_filter": mode}

    max_d = int(df["date"].max())

    if supplement_external and not df.empty:
        sparse = symbols_needing_supplement(df, min_coverage_ratio=0.85)
        if sparse:
            df, sup = supplement_panel(
                df,
                sparse,
                start_date=start_date,
                end_date=max_d,
            )
            meta["supplement"] = sup
            meta["sparse_symbols"] = len(sparse)
            df = _ensure_net_foreign(df)

    df = _ensure_cap_eligible(df, cap_filter=mode, min_market_cap=min_market_cap)
    df.sort_values(["symbol", "date"], inplace=True)
    return df.reset_index(drop=True), meta


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("symbol", group_keys=False)

    df["nf_lag1"] = g["net_foreign"].shift(1)
    df["nf_lag2"] = g["net_foreign"].shift(2)
    df["nf_lag3"] = g["net_foreign"].shift(3)
    df["nf_prev3_sum"] = df["nf_lag1"] + df["nf_lag2"] + df["nf_lag3"]
    df["nf_ma20"] = g["net_foreign"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["nf_q95_20"] = g["net_foreign"].transform(
        lambda s: s.rolling(20, min_periods=10).quantile(0.95)
    )
    df["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["high_20_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).max())
    df["high_10_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).max())
    df["high_30_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(30, min_periods=15).max())
    df["ret_5d"] = g["close"].pct_change(5) * 100
    df["ma5"] = g["close"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["ma10"] = g["close"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    df["ma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["ma50"] = g["close"].transform(lambda s: s.rolling(50, min_periods=20).mean())
    df["ma5_prev"] = g["ma5"].shift(1)
    df["ma20_prev"] = g["ma20"].shift(1)
    return df


def _foreign_streak_mask(df: pd.DataFrame, streak: int) -> pd.Series:
    """Net-foreign positive for `streak` consecutive sessions; requires notna each day."""
    if streak < 1:
        raise ValueError("streak must be >= 1")
    sig = df["net_foreign"].notna() & (df["net_foreign"] > 0)
    g = df.groupby("symbol")["net_foreign"]
    for lag in range(1, streak):
        lagged = g.shift(lag)
        sig = sig & lagged.notna() & (lagged > 0)
    return sig


def _entry_eligible(df: pd.DataFrame) -> pd.Series:
    if "cap_eligible" in df.columns:
        return df["cap_eligible"].fillna(False).astype(bool)
    return pd.Series(True, index=df.index)


def _signal_mask(df: pd.DataFrame, scenario_id: str) -> pd.Series:
    dispatch = {
        "foreign_2d_positive": lambda: _foreign_streak_mask(df, 2),
        "foreign_3d_streak": lambda: _foreign_streak_mask(df, 3),
        "foreign_reversal": lambda: (
            df["nf_prev3_sum"].notna()
            & df["net_foreign"].notna()
            & df["nf_ma20"].notna()
            & (df["nf_prev3_sum"] < 0)
            & (df["net_foreign"] > 0)
            & (df["net_foreign"] > df["nf_ma20"])
        ),
        "drop_3pct_bounce": lambda: df["change"] <= -3,
        "drop_5pct_bounce": lambda: df["change"] <= -5,
        "breakout_20d": lambda: df["close"] > df["high_20_prev"],
        "volume_spike_up": lambda: (df["volume"] > 2 * df["vol_ma20"])
        & (df["close"] > df["prev_close"]),
        "gap_down_open": lambda: df["open_price"] <= df["prev_close"] * 0.98,
        "net_foreign_top_day": lambda: df["net_foreign"].notna()
        & df["nf_q95_20"].notna()
        & (df["net_foreign"] >= df["nf_q95_20"]),
        "rsi_oversold_proxy": lambda: df["ret_5d"] <= -8,
        "ma_cross_5_20": lambda: (df["ma5"] > df["ma20"]) & (df["ma5_prev"] <= df["ma20_prev"]),
        "momentum_5d": lambda: df["ret_5d"] >= 5,
        "pullback_ma20": lambda: (df["close"] <= df["ma20"])
        & (df["change"] <= -2)
        & (df["ma20"] > df["ma50"]),
        "foreign_intensity": lambda: df["net_foreign"].notna()
        & df["nf_ma20"].notna()
        & (df["net_foreign"] > 2 * df["nf_ma20"].abs())
        & (df["close"] > df["prev_close"]),
        "breakout_10d": lambda: df["close"] > df["high_10_prev"],
        "volume_spike_3x": lambda: (df["volume"] > 3 * df["vol_ma20"])
        & (df["close"] > df["prev_close"]),
    }
    if scenario_id not in dispatch:
        raise ValueError(f"Skenario tidak dikenal: {scenario_id}")
    return dispatch[scenario_id]()


def _empty_eval(label: str, hold_days: int, name: str | None = None) -> dict:
    return {
        "scenario_id": label,
        "name": name or label,
        "hold_days": hold_days,
        "trades": 0,
        "wins": 0,
        "win_rate": 0.0,
        "avg_return_pct": 0.0,
        "median_return_pct": 0.0,
        "symbols": 0,
    }


def _evaluate_scenario(df: pd.DataFrame, spec: ScenarioSpec) -> dict:
    hold = spec.hold_days
    work = df.copy()
    work["signal"] = _signal_mask(work, spec.id) & _entry_eligible(work)
    work["exit_close"] = work.groupby("symbol")["close"].shift(-hold)
    trades = work[work["signal"] & work["exit_close"].notna()].copy()
    if trades.empty:
        return _empty_eval(spec.id, hold, spec.name)

    trades["return_pct"] = (trades["exit_close"] / trades["close"] - 1) * 100
    trades["win"] = trades["return_pct"] > 0
    return {
        "scenario_id": spec.id,
        "name": spec.name,
        "hold_days": hold,
        "trades": int(len(trades)),
        "wins": int(trades["win"].sum()),
        "win_rate": float(trades["win"].mean()),
        "avg_return_pct": float(trades["return_pct"].mean()),
        "median_return_pct": float(trades["return_pct"].median()),
        "symbols": int(trades["symbol"].nunique()),
        "date_from": int(trades["date"].min()),
        "date_to": int(trades["date"].max()),
    }


def _evaluate_custom(df: pd.DataFrame, signal: pd.Series, hold_days: int, label: str) -> dict:
    work = df.copy()
    work["signal"] = signal.reindex(work.index, fill_value=False).fillna(False) & _entry_eligible(
        work
    )
    work["exit_close"] = work.groupby("symbol")["close"].shift(-hold_days)
    trades = work[work["signal"] & work["exit_close"].notna()]
    if trades.empty:
        return _empty_eval(label, hold_days)
    ret = (trades["exit_close"] / trades["close"] - 1) * 100
    wins = int((ret > 0).sum())
    return {
        "scenario_id": label,
        "name": label,
        "hold_days": hold_days,
        "trades": int(len(trades)),
        "wins": wins,
        "win_rate": float(wins / len(trades)),
        "avg_return_pct": float(ret.mean()),
        "median_return_pct": float(ret.median()),
        "symbols": int(trades["symbol"].nunique()),
    }


def aggregate_grid_defs() -> list[tuple[str, str]]:
    """
    Extra aggregate grid families (built-in SCENARIOS keep breakout/MA/oversold).
    Returns (label_template_prefix_key, family) descriptors for docs/tests.
    """
    labels: list[tuple[str, str]] = []
    for threshold in (-2, -3, -4, -5, -6, -7, -8, -9, -10):
        for hold in (1, 2, 3, 5, 7, 10, 14):
            labels.append((f"grid_drop_{abs(threshold)}pct_hold{hold}d", "drop"))
    for streak in (2, 3, 4, 5):
        for hold in (3, 5, 7, 10, 14):
            labels.append((f"grid_foreign_{streak}d_hold{hold}d", "foreign"))
    for mom in (3, 5, 7, 10):
        for hold in (3, 5, 7, 10):
            labels.append((f"grid_momentum_{mom}pct_hold{hold}d", "momentum"))
    for mult in (2.0, 2.5, 3.0, 4.0):
        for hold in (3, 5, 7, 10):
            labels.append((f"grid_vol_{str(mult).replace('.', 'p')}x_hold{hold}d", "volume"))
    return labels


def aggregate_grid_labels() -> list[str]:
    return [lab for lab, _ in aggregate_grid_defs()]


def per_symbol_grid_defs() -> list[tuple[str, str, int]]:
    """(label, family, hold_days) for exhaustive per-symbol grid including volume."""
    defs: list[tuple[str, str, int]] = []
    for threshold in (-3, -4, -5, -6, -7, -8):
        for hold in (3, 5, 7):
            defs.append((f"grid_drop_{abs(threshold)}pct_hold{hold}d", "drop", hold))
    for streak in (2, 3, 4):
        for hold in (5, 7, 10):
            defs.append((f"grid_foreign_{streak}d_hold{hold}d", "foreign", hold))
    for threshold in (5, 7, 10):
        for hold in (5, 7):
            defs.append((f"grid_momentum_{threshold}pct_hold{hold}d", "momentum", hold))
    for mult in (2.0, 2.5, 3.0, 4.0):
        for hold in (3, 5, 7, 10):
            defs.append(
                (f"grid_vol_{str(mult).replace('.', 'p')}x_hold{hold}d", "volume", hold)
            )
    return defs


def per_symbol_grid_labels() -> list[str]:
    return [lab for lab, _, _ in per_symbol_grid_defs()]


def _grid_signal(df: pd.DataFrame, label: str) -> pd.Series:
    if label.startswith("grid_drop_"):
        # grid_drop_{n}pct_hold{h}d
        pct = int(label.split("_")[2].replace("pct", ""))
        return df["change"] <= -pct
    if label.startswith("grid_foreign_"):
        streak = int(label.split("_")[2].replace("d", ""))
        return _foreign_streak_mask(df, streak)
    if label.startswith("grid_momentum_"):
        mom = int(label.split("_")[2].replace("pct", ""))
        return df["ret_5d"] >= mom
    if label.startswith("grid_vol_"):
        # grid_vol_2p5x_hold3d
        mult_token = label.split("_")[2].replace("x", "").replace("p", ".")
        mult = float(mult_token)
        return (df["volume"] > mult * df["vol_ma20"]) & (df["close"] > df["prev_close"])
    raise ValueError(f"Grid label tidak dikenal: {label}")


def grid_search(
    df: pd.DataFrame,
    *,
    min_win_rate: float,
    max_win_rate: float,
    min_trades: int,
) -> list[dict]:
    """Parameter grid: drop-bounce, foreign streak, momentum, volume spike only."""
    results: list[dict] = []
    for label, _family in aggregate_grid_defs():
        hold = int(label.rsplit("hold", 1)[1].replace("d", ""))
        sig = _grid_signal(df, label)
        row = _evaluate_custom(df, sig, hold, label)
        row["description"] = label
        row["passes_filter"] = _passes_filter(
            row["win_rate"],
            row["trades"],
            min_win_rate=min_win_rate,
            max_win_rate=max_win_rate,
            min_trades=min_trades,
        )
        results.append(row)

    return sorted(
        [r for r in results if r["passes_filter"]],
        key=lambda x: (x["win_rate"], x["trades"]),
        reverse=True,
    )


def scan_per_symbol_hits(
    df: pd.DataFrame,
    specs: list[ScenarioSpec],
    *,
    min_win_rate: float,
    max_win_rate: float,
    min_trades: int,
) -> list[dict]:
    hits: list[dict] = []
    for spec in specs:
        hold = spec.hold_days
        work = df.copy()
        work["signal"] = _signal_mask(work, spec.id) & _entry_eligible(work)
        work["exit_close"] = work.groupby("symbol")["close"].shift(-hold)
        trades = work[work["signal"] & work["exit_close"].notna()].copy()
        if trades.empty:
            continue
        trades["return_pct"] = (trades["exit_close"] / trades["close"] - 1) * 100
        trades["win"] = trades["return_pct"] > 0
        agg = (
            trades.groupby("symbol")
            .agg(trades=("win", "count"), wins=("win", "sum"), avg_return=("return_pct", "mean"))
            .reset_index()
        )
        agg["win_rate"] = agg["wins"] / agg["trades"]
        filt = agg[
            (agg["trades"] >= min_trades)
            & (agg["win_rate"] >= min_win_rate)
            & (agg["win_rate"] <= max_win_rate)
        ]
        for _, row in filt.iterrows():
            hits.append(
                {
                    "scenario_id": spec.id,
                    "symbol": row["symbol"],
                    "trades": int(row["trades"]),
                    "wins": int(row["wins"]),
                    "win_rate": float(row["win_rate"]),
                    "avg_return_pct": float(row["avg_return"]),
                }
            )
    hits.sort(key=lambda x: (x["win_rate"], x["trades"]), reverse=True)
    return hits


def scan_per_symbol_grid(
    df: pd.DataFrame,
    *,
    min_win_rate: float,
    max_win_rate: float,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> list[dict]:
    """Exhaustive per-symbol grid (drop/foreign/momentum/volume) — complete results."""
    hits: list[dict] = []
    symbols = df["symbol"].unique()

    # Precompute base signals on the full panel once; slice per symbol.
    labeled_sigs: list[tuple[str, pd.Series, int]] = []
    for label, _family, hold in per_symbol_grid_defs():
        labeled_sigs.append((label, _grid_signal(df, label), hold))

    for sym in symbols:
        sub = df[df["symbol"] == sym]
        if len(sub) < min_trades + 10:
            continue
        eligible = _entry_eligible(sub)
        for label, base_sig, hold in labeled_sigs:
            sig = base_sig.reindex(sub.index, fill_value=False).fillna(False) & eligible
            work = sub.copy()
            work["signal"] = sig
            work["exit_close"] = work["close"].shift(-hold)
            trades = work[work["signal"] & work["exit_close"].notna()]
            n = len(trades)
            if n < min_trades:
                continue
            ret = (trades["exit_close"] / trades["close"] - 1) * 100
            wins = int((ret > 0).sum())
            wr = wins / n
            if not _in_win_band(wr, min_win_rate, max_win_rate):
                continue
            hits.append(
                {
                    "scenario_id": label,
                    "symbol": sym,
                    "trades": n,
                    "wins": wins,
                    "win_rate": float(wr),
                    "avg_return_pct": float(ret.mean()),
                }
            )

    hits.sort(key=lambda x: (x["win_rate"], x["trades"]), reverse=True)
    return hits


def _merge_band_hits(
    passed: list[dict],
    grid_passed: list[dict],
    per_symbol: list[dict],
    per_symbol_grid: list[dict],
) -> list[dict]:
    """Unified list of all setups in the win-rate band (complete, unsorted truncations)."""
    merged: list[dict] = []
    for r in passed:
        merged.append(
            {
                "kind": "aggregate",
                "scenario_id": r["scenario_id"],
                "symbol": "*",
                "trades": r["trades"],
                "wins": r["wins"],
                "win_rate": r["win_rate"],
                "avg_return_pct": r.get("avg_return_pct", 0),
            }
        )
    for r in grid_passed:
        merged.append(
            {
                "kind": "grid",
                "scenario_id": r["scenario_id"],
                "symbol": "*",
                "trades": r["trades"],
                "wins": r["wins"],
                "win_rate": r["win_rate"],
                "avg_return_pct": r.get("avg_return_pct", 0),
            }
        )
    for h in per_symbol:
        merged.append({**h, "kind": "per_symbol"})
    for h in per_symbol_grid:
        merged.append({**h, "kind": "per_symbol_grid"})
    merged.sort(key=lambda x: (x["win_rate"], x["trades"]), reverse=True)
    return merged


def scan_scenarios(
    store: StockDataStore,
    *,
    start_date: int = DEFAULT_START,
    min_market_cap: float = MIN_MARKET_CAP,
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
    max_win_rate: float = DEFAULT_MAX_WIN_RATE,
    min_trades: int = DEFAULT_MIN_TRADES,
    scenario_ids: list[str] | None = None,
    supplement_external: bool = True,
    cap_filter: str = "large_cap",
    universe: str | None = None,
) -> dict:
    mode = _resolve_cap_filter(cap_filter, universe)
    specs = SCENARIOS
    if scenario_ids:
        wanted = set(scenario_ids)
        specs = [s for s in SCENARIOS if s.id in wanted]
        if not specs:
            raise ValueError(f"Tidak ada skenario valid: {scenario_ids}")

    panel, panel_meta = load_panel(
        store,
        start_date=start_date,
        min_market_cap=min_market_cap,
        cap_filter=mode,
        supplement_external=supplement_external,
    )
    panel = _add_features(panel)
    coverage = panel_coverage_report(panel)

    min_d, max_d = int(panel["date"].min()), int(panel["date"].max())
    # Avg eligible symbols/day (cap filter is entry-side only).
    eligible = panel[panel["cap_eligible"]] if "cap_eligible" in panel.columns else panel
    avg_symbols = float(eligible.groupby("date")["symbol"].nunique().mean()) if not eligible.empty else 0.0

    results: list[dict] = []
    for spec in specs:
        row = _evaluate_scenario(panel, spec)
        row["description"] = spec.description
        row["entry"] = spec.entry
        row["exit_rule"] = spec.exit_rule
        row["passes_filter"] = _passes_filter(
            row["win_rate"],
            row["trades"],
            min_win_rate=min_win_rate,
            max_win_rate=max_win_rate,
            min_trades=min_trades,
        )
        results.append(row)

    results.sort(key=lambda r: (r["passes_filter"], r["win_rate"], r["trades"]), reverse=True)
    passed = [r for r in results if r["passes_filter"]]

    grid_passed = grid_search(
        panel,
        min_win_rate=min_win_rate,
        max_win_rate=max_win_rate,
        min_trades=min_trades,
    )
    per_symbol = scan_per_symbol_hits(
        panel,
        specs,
        min_win_rate=min_win_rate,
        max_win_rate=max_win_rate,
        min_trades=min_trades,
    )
    per_symbol_grid = scan_per_symbol_grid(
        panel,
        min_win_rate=min_win_rate,
        max_win_rate=max_win_rate,
        min_trades=min_trades,
    )

    all_band_hits = _merge_band_hits(passed, grid_passed, per_symbol, per_symbol_grid)

    return {
        "start_date": start_date,
        "end_date": max_d,
        "cap_filter": mode,
        "universe": mode,  # legacy mirror for older callers/docs
        "min_market_cap": min_market_cap,
        "min_win_rate": min_win_rate,
        "max_win_rate": max_win_rate,
        "min_trades": min_trades,
        "panel_rows": int(len(panel)),
        "panel_meta": panel_meta,
        "data_supplement": panel_meta.get("supplement", {}),
        "coverage": coverage,
        "avg_symbols_per_day": avg_symbols,
        "scenarios_tested": len(results),
        "scenarios_passed": len(passed),
        "grid_passed": grid_passed,
        "per_symbol_hits": per_symbol,
        "per_symbol_grid": per_symbol_grid,
        "all_band_hits": all_band_hits,
        "results": results,
        "passed": passed,
    }


def format_scan_report(payload: dict) -> str:
    max_wr = payload.get("max_win_rate", 1.0)
    band = f"{payload['min_win_rate']*100:.0f}–{max_wr*100:.0f}%"
    mode = payload.get("cap_filter") or payload.get("universe", "large_cap")
    uni_line = (
        f"Cap filter: {mode} (market cap >= {payload['min_market_cap']:,.0f} IDR entry-only, "
        f"~{payload['avg_symbols_per_day']:.0f} emiten eligible/hari)"
        if mode == "large_cap"
        else f"Cap filter: {mode} (~{payload['avg_symbols_per_day']:.0f} emiten/hari)"
    )
    lines = [
        "OK: backtest scan selesai",
        f"Periode: {payload['start_date']} .. {payload['end_date']}",
        uni_line,
        f"Baris panel: {payload['panel_rows']:,}",
        f"Filter: win_rate {band}, trades >= {payload['min_trades']}",
        f"Skenario lulus: {payload['scenarios_passed']} / {payload['scenarios_tested']}",
    ]
    sup = payload.get("data_supplement") or {}
    if sup.get("fetched_symbols"):
        lines.append(
            f"Data eksternal (yfinance): +{sup.get('added_rows', 0)} baris, "
            f"{sup.get('fetched_symbols', 0)} emiten"
        )
    cov = payload.get("coverage") or {}
    if cov:
        lines.append(
            f"Coverage: {cov.get('symbols', 0)} emiten, "
            f"parquet={cov.get('parquet_rows', 0)}, yfinance={cov.get('yfinance_rows', 0)}"
        )
    lines.append("")

    band_hits = payload.get("all_band_hits") or []
    if band_hits:
        preview = band_hits[:REPORT_PREVIEW_HITS]
        capped = len(band_hits) > len(preview)
        header = f"=== SETUP TRADING (win {band}, total {len(band_hits)}) ==="
        if capped:
            header = (
                f"=== SETUP TRADING (win {band}, total {len(band_hits)}; "
                f"preview first {len(preview)}) ==="
            )
        lines.append(header)
        for h in preview:
            sym = h.get("symbol", "*")
            kind = h.get("kind", "?")
            lines.append(
                f"- [{kind}] {sym} + {h['scenario_id']}: win={h['win_rate']*100:.1f}% "
                f"({h['wins']}/{h['trades']}) avg={h.get('avg_return_pct', 0):.2f}%"
            )
        lines.append("")

    if payload["passed"]:
        lines.append("=== SKENARIO LULUS ===")
        for r in payload["passed"]:
            lines.append(
                f"- {r['scenario_id']}: {r['name']} | "
                f"win={r['win_rate']*100:.1f}% ({r['wins']}/{r['trades']}) | "
                f"avg_ret={r['avg_return_pct']:.2f}% | median={r['median_return_pct']:.2f}% | "
                f"symbols={r['symbols']}"
            )
    else:
        lines.append("Tidak ada skenario agregat yang memenuhi filter.")

    if payload.get("grid_passed"):
        grid = payload["grid_passed"]
        preview = grid[:15]
        note = f" (preview first {len(preview)} of {len(grid)})" if len(grid) > 15 else ""
        lines.append("")
        lines.append(f"=== GRID SEARCH LULUS ({len(grid)}){note} ===")
        for r in preview:
            lines.append(
                f"- {r['scenario_id']}: win={r['win_rate']*100:.1f}% "
                f"({r['wins']}/{r['trades']}) avg={r['avg_return_pct']:.2f}%"
            )

    if payload.get("per_symbol_hits"):
        hits = payload["per_symbol_hits"]
        preview = hits[:25]
        note = f" (preview first {len(preview)} of {len(hits)})" if len(hits) > 25 else ""
        lines.append("")
        lines.append(f"=== PER EMITEN (skenario bawaan, win {band}){note} ===")
        for h in preview:
            lines.append(
                f"- {h['symbol']} + {h['scenario_id']}: win={h['win_rate']*100:.1f}% "
                f"({h['wins']}/{h['trades']}) avg={h['avg_return_pct']:.2f}%"
            )

    if payload.get("per_symbol_grid"):
        hits = payload["per_symbol_grid"]
        preview = hits[:25]
        note = f" (preview first {len(preview)} of {len(hits)})" if len(hits) > 25 else ""
        lines.append("")
        lines.append(f"=== PER EMITEN GRID (win {band}){note} ===")
        for h in preview:
            lines.append(
                f"- {h['symbol']} + {h['scenario_id']}: win={h['win_rate']*100:.1f}% "
                f"({h['wins']}/{h['trades']}) avg={h['avg_return_pct']:.2f}%"
            )
    elif not payload["passed"] and not payload.get("grid_passed") and not payload.get("per_symbol_hits"):
        lines.append("")
        lines.append("Tidak ada kombinasi grid/per-emiten yang lulus filter either.")

    lines.append("")
    lines.append("=== SEMUA SKENARIO (sorted by win_rate) ===")
    for r in sorted(payload["results"], key=lambda x: x["win_rate"], reverse=True):
        flag = "PASS" if r["passes_filter"] else "----"
        lines.append(
            f"[{flag}] {r['scenario_id']}: win={r['win_rate']*100:.1f}% "
            f"({r['wins']}/{r['trades']}) avg={r['avg_return_pct']:.2f}% hold={r['hold_days']}d"
        )

    lines.append("")
    lines.append(
        "CATATAN: win rate tinggi bisa karena sample kecil / overfitting. "
        "Validasi out-of-sample sebelum dipakai live. Dokumentasi: docs/BACKTEST_SCENARIOS.md"
    )
    return "\n".join(lines)


def save_scan_artifacts(payload: dict, export_dir: Path) -> tuple[Path, Path]:
    out = Path(export_dir) / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
    json_path = out / f"scan_{stamp}.json"
    md_path = out / f"scan_{stamp}.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    max_wr = payload.get("max_win_rate", 1.0)
    mode = payload.get("cap_filter") or payload.get("universe", "large_cap")
    band = f"{payload['min_win_rate']*100:.0f}–{max_wr*100:.0f}%"
    band_hits = payload.get("all_band_hits") or []

    md_lines = [
        "# Backtest scan — IDX",
        "",
        f"- Generated: {datetime.now(WIB).isoformat()}",
        f"- Period: {payload['start_date']} .. {payload['end_date']}",
        f"- Cap filter: {mode}",
        f"- Min market cap: IDR {payload['min_market_cap']:,.0f} (entry eligibility only)",
        f"- Filter: win rate {band}, min trades {payload['min_trades']}",
        "",
        f"## SETUP TRADING",
        "",
        f"Win band {band}. Every aggregate / grid / per-symbol hit ({len(band_hits)} total):",
        "",
    ]
    if band_hits:
        md_lines.append("| Kind | Symbol | Scenario | Win% | Trades | Avg ret% |")
        md_lines.append("|---|---|---|---:|---:|---:|")
        for h in band_hits:
            md_lines.append(
                f"| {h.get('kind', '')} | {h.get('symbol', '*')} | {h['scenario_id']} | "
                f"{h['win_rate']*100:.1f} | {h['trades']} | {h.get('avg_return_pct', 0):.2f} |"
            )
        md_lines.append("")
    else:
        md_lines.append("_Tidak ada setup dalam band win rate._\n")

    md_lines.append("## Semua skenario bawaan\n")
    md_lines.append("| ID | Win% | Trades | Avg ret% | Hold | Pass |")
    md_lines.append("|---|---:|---:|---:|---:|:---:|")
    for r in sorted(payload.get("results") or [], key=lambda x: x["win_rate"], reverse=True):
        md_lines.append(
            f"| {r['scenario_id']} | {r['win_rate']*100:.1f} | {r['trades']} | "
            f"{r['avg_return_pct']:.2f} | {r['hold_days']} | {'Y' if r['passes_filter'] else 'N'} |"
        )

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def run_backtest_scan(
    store: StockDataStore,
    export_dir: Path,
    *,
    start_date: str | int = DEFAULT_START,
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
    max_win_rate: float = DEFAULT_MAX_WIN_RATE,
    min_trades: int = DEFAULT_MIN_TRADES,
    min_market_cap: float = MIN_MARKET_CAP,
    supplement_external: bool = True,
    cap_filter: str = "large_cap",
    universe: str | None = None,
) -> str:
    start = int(str(start_date).replace("-", "")[:8])
    mode = _resolve_cap_filter(cap_filter, universe)
    payload = scan_scenarios(
        store,
        start_date=start,
        min_market_cap=min_market_cap,
        min_win_rate=min_win_rate,
        max_win_rate=max_win_rate,
        min_trades=min_trades,
        supplement_external=supplement_external,
        cap_filter=mode,
    )
    json_path, md_path = save_scan_artifacts(payload, export_dir)
    report = format_scan_report(payload)
    return f"{report}\n\nFILE: {json_path}\nFILE: {md_path}"


def is_backtest_query(text: str) -> bool:
    raw = text or ""
    return bool(
        re.search(
            r"\b("
            r"backtest|win\s*rate|skenario|scenario|setup\s+trading|"
            r"trading\s+setup|scan\s+trading|uji\s+strategi"
            r")\b",
            raw,
            re.I,
        )
    )
