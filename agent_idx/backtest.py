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
MAX_HOLD_DAYS = 63  # ~3 bulan hari bursa
MIN_GAIN_WIN_PCT = 20.0
MIN_GAIN_AVG_PCT = 20.0
MAX_GAIN_AVG_PCT = 50.0
MAX_GAIN_TP_PCT = 50.0


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
            "Universe: market cap >= 1T pada hari sinyal."
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


def load_panel(
    store: StockDataStore,
    *,
    start_date: int = DEFAULT_START,
    min_market_cap: float = MIN_MARKET_CAP,
    universe: str = "large_cap",
    supplement_external: bool = True,
) -> tuple[pd.DataFrame, dict]:
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
          AND listed_shares IS NOT NULL
          AND listed_shares > 0
        ORDER BY symbol, date
        """,
        [start_date],
    ).fetchdf()

    if df.empty:
        raise RuntimeError(f"Tidak ada data parquet sejak {start_date}")

    df["net_foreign"] = df["foreign_buy"].fillna(0) - df["foreign_sell"].fillna(0)
    df["source"] = "parquet"
    meta: dict = {"source": "parquet", "supplement": {}}

    max_d = int(df["date"].max())
    if universe == "large_cap":
        df = df[df["market_cap"] >= min_market_cap].copy()
    elif universe != "all":
        raise ValueError(f"universe tidak valid: {universe}")

    if supplement_external and not df.empty:
        sym_counts = df.groupby("symbol")["date"].count()
        sparse = sym_counts[sym_counts < sym_counts.max() * 0.85].index.tolist()
        if sparse:
            df, sup = supplement_panel(
                df,
                sparse[:150],
                start_date=start_date,
                end_date=max_d,
            )
            meta["supplement"] = sup
            meta["sparse_symbols"] = len(sparse)

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


def _signal_mask(df: pd.DataFrame, scenario_id: str) -> pd.Series:
    if scenario_id == "foreign_2d_positive":
        return (df["net_foreign"] > 0) & (df["nf_lag1"] > 0)
    if scenario_id == "foreign_3d_streak":
        return (df["net_foreign"] > 0) & (df["nf_lag1"] > 0) & (df["nf_lag2"] > 0)
    if scenario_id == "foreign_reversal":
        return (df["nf_prev3_sum"] < 0) & (df["net_foreign"] > 0) & (df["net_foreign"] > df["nf_ma20"])
    if scenario_id == "drop_3pct_bounce":
        return df["change"] <= -3
    if scenario_id == "drop_5pct_bounce":
        return df["change"] <= -5
    if scenario_id == "breakout_20d":
        return df["close"] > df["high_20_prev"]
    if scenario_id == "volume_spike_up":
        return (df["volume"] > 2 * df["vol_ma20"]) & (df["close"] > df["prev_close"])
    if scenario_id == "gap_down_open":
        return df["open_price"] <= df["prev_close"] * 0.98
    if scenario_id == "net_foreign_top_day":
        return df["net_foreign"] >= df["nf_q95_20"]
    if scenario_id == "rsi_oversold_proxy":
        return df["ret_5d"] <= -8
    if scenario_id == "ma_cross_5_20":
        return (df["ma5"] > df["ma20"]) & (df["ma5_prev"] <= df["ma20_prev"])
    if scenario_id == "momentum_5d":
        return df["ret_5d"] >= 5
    if scenario_id == "pullback_ma20":
        return (df["close"] <= df["ma20"]) & (df["change"] <= -2) & (df["ma20"] > df["ma50"])
    if scenario_id == "foreign_intensity":
        return (df["net_foreign"] > 2 * df["nf_ma20"].abs()) & (df["close"] > df["prev_close"])
    if scenario_id == "breakout_10d":
        return df["close"] > df["high_10_prev"]
    if scenario_id == "volume_spike_3x":
        return (df["volume"] > 3 * df["vol_ma20"]) & (df["close"] > df["prev_close"])
    raise ValueError(f"Skenario tidak dikenal: {scenario_id}")


def _track_mode_for_scenario(scenario_id: str) -> str:
    sid = scenario_id.lower()
    if "foreign" in sid:
        return "foreign_positive"
    if "ma_cross" in sid or "pullback" in sid or "ma5_20" in sid:
        return "ma5_above_ma20"
    if "drop" in sid or "oversold" in sid or "gap" in sid or "rsi" in sid:
        return "recovery_ma10"
    return "trend_ma10"


def _still_on_track(row: pd.Series, entry_price: float, mode: str) -> bool:
    if mode == "foreign_positive":
        return float(row["net_foreign"]) > 0
    if mode == "ma5_above_ma20":
        return float(row["ma5"]) > float(row["ma20"])
    if mode == "recovery_ma10":
        return float(row["close"]) > float(row["ma10"]) or float(row["close"]) >= entry_price * 1.05
    return float(row["close"]) > float(row["ma10"]) and float(row["close"]) >= entry_price * 0.90


def _still_on_track_np(
    close: float,
    net_foreign: float,
    ma10: float,
    ma5: float,
    ma20: float,
    entry_price: float,
    mode: str,
) -> bool:
    if mode == "foreign_positive":
        return net_foreign > 0
    if mode == "ma5_above_ma20":
        return ma5 > ma20
    if mode == "recovery_ma10":
        return close > ma10 or close >= entry_price * 1.05
    return close > ma10 and close >= entry_price * 0.90


def _simulate_symbol_numpy(
    full: pd.DataFrame,
    entry_indices: list[int],
    track_mode: str,
    *,
    max_hold: int = MAX_HOLD_DAYS,
    take_profit: float = MAX_GAIN_TP_PCT,
    stop_loss: float = 10.0,
) -> list[float]:
    close = full["close"].to_numpy(dtype=float)
    nf = full["net_foreign"].to_numpy(dtype=float)
    ma10 = full["ma10"].to_numpy(dtype=float)
    ma5 = full["ma5"].to_numpy(dtype=float)
    ma20 = full["ma20"].to_numpy(dtype=float)
    n = len(close)
    returns: list[float] = []
    for i in entry_indices:
        entry = close[i]
        if entry <= 0:
            continue
        exit_ret: float | None = None
        end = min(i + max_hold, n - 1)
        for j in range(i + 1, end + 1):
            ret = (close[j] / entry - 1) * 100
            if ret >= take_profit:
                exit_ret = take_profit
                break
            if ret <= -stop_loss:
                exit_ret = ret
                break
            if not _still_on_track_np(close[j], nf[j], ma10[j], ma5[j], ma20[j], entry, track_mode):
                exit_ret = ret
                break
        if exit_ret is None:
            if end <= i:
                continue
            exit_ret = (close[end] / entry - 1) * 100
        returns.append(exit_ret)
    return returns


def _entry_indices(full: pd.DataFrame, signal: pd.Series) -> list[int]:
    sig = signal.reindex(full.index, fill_value=False).fillna(False)
    return [int(i) for i in full.index[sig.to_numpy(dtype=bool)]]


def _simulate_on_track_returns(
    df: pd.DataFrame,
    signal: pd.Series,
    track_mode: str,
    *,
    max_hold: int = MAX_HOLD_DAYS,
    take_profit: float = MAX_GAIN_TP_PCT,
    stop_loss: float = 10.0,
    symbol_frames: dict[str, pd.DataFrame] | None = None,
) -> list[float]:
    """Simulasi hold fleksibel per emiten (numpy-accelerated)."""
    if symbol_frames is None:
        symbol_frames = {
            sym: g.sort_values("date").reset_index(drop=True)
            for sym, g in df.groupby("symbol")
        }

    returns: list[float] = []
    sig_df = df[signal.fillna(False)]
    if sig_df.empty:
        return returns

    for sym, grp in sig_df.groupby("symbol"):
        full = symbol_frames.get(sym)
        if full is None or full.empty:
            continue
        pos = {int(d): i for i, d in enumerate(full["date"])}
        indices = [pos[int(d)] for d in grp["date"] if int(d) in pos]
        if not indices:
            continue
        returns.extend(
            _simulate_symbol_numpy(
                full, indices, track_mode,
                max_hold=max_hold, take_profit=take_profit, stop_loss=stop_loss,
            )
        )
    return returns


def _stats_from_returns(
    returns: list[float],
    *,
    label: str,
    min_gain_win: float = MIN_GAIN_WIN_PCT,
) -> dict:
    if not returns:
        return {
            "scenario_id": label,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
        }
    s = pd.Series(returns)
    wins = int((s >= min_gain_win).sum())
    return {
        "scenario_id": label,
        "trades": int(len(s)),
        "wins": wins,
        "win_rate": float(wins / len(s)),
        "avg_return_pct": float(s.mean()),
        "median_return_pct": float(s.median()),
        "avg_hold_proxy": None,
    }


def _passes_gain_filter(
    row: dict,
    *,
    min_win_rate: float,
    max_win_rate: float,
    min_trades: int,
    min_gain_win: float = MIN_GAIN_WIN_PCT,
    min_avg_gain: float = MIN_GAIN_AVG_PCT,
    max_avg_gain: float = MAX_GAIN_AVG_PCT,
) -> bool:
    if row["trades"] < min_trades:
        return False
    if not _in_win_band(row["win_rate"], min_win_rate, max_win_rate):
        return False
    if row["avg_return_pct"] < min_avg_gain or row["avg_return_pct"] > max_avg_gain:
        return False
    return True


def _evaluate_on_track(
    df: pd.DataFrame,
    scenario_id: str,
    signal: pd.Series | None = None,
    *,
    max_hold: int = MAX_HOLD_DAYS,
    min_gain_win: float = MIN_GAIN_WIN_PCT,
    symbol_frames: dict[str, pd.DataFrame] | None = None,
) -> dict:
    track = _track_mode_for_scenario(scenario_id)
    sig = signal if signal is not None else _signal_mask(df, scenario_id)
    rets = _simulate_on_track_returns(
        df, sig, track, max_hold=max_hold, symbol_frames=symbol_frames
    )
    row = _stats_from_returns(rets, label=scenario_id, min_gain_win=min_gain_win)
    row["exit_mode"] = f"on_track({track}) max {max_hold}d TP {MAX_GAIN_TP_PCT}%"
    row["min_gain_win_pct"] = min_gain_win
    return row


def _scan_per_symbol_on_track(
    panel: pd.DataFrame,
    scenarios: list[tuple[str, pd.Series | None]],
    symbol_frames: dict[str, pd.DataFrame],
    *,
    min_win_rate: float,
    max_win_rate: float,
    min_trades: int,
    min_gain_win: float,
    min_avg_gain: float,
    max_avg_gain: float,
    max_hold: int,
) -> tuple[list[dict], list[dict]]:
    hits: list[dict] = []
    near: list[dict] = []
    for sym, full in symbol_frames.items():
        if len(full) < min_trades + max_hold:
            continue
        sub = panel[panel["symbol"] == sym]
        for scenario_id, sig_override in scenarios:
            if sig_override is not None:
                sig = sig_override.reindex(sub.index, fill_value=False)
            else:
                sig = _signal_mask(sub, scenario_id)
            if not sig.any():
                continue
            track = _track_mode_for_scenario(scenario_id)
            pos = {int(d): i for i, d in enumerate(full["date"])}
            indices = [pos[int(d)] for d in sub.loc[sig, "date"] if int(d) in pos]
            if len(indices) < min_trades:
                continue
            rets = _simulate_symbol_numpy(full, indices, track, max_hold=max_hold)
            if len(rets) < min_trades:
                continue
            row = _stats_from_returns(rets, label=scenario_id, min_gain_win=min_gain_win)
            entry = {
                **row,
                "symbol": sym,
                "exit_mode": f"on_track({track}) max {max_hold}d",
            }
            if _passes_gain_filter(
                row,
                min_win_rate=min_win_rate,
                max_win_rate=max_win_rate,
                min_trades=min_trades,
                min_gain_win=min_gain_win,
                min_avg_gain=min_avg_gain,
                max_avg_gain=max_avg_gain,
            ):
                hits.append(entry)
            elif row["avg_return_pct"] >= min_avg_gain * 0.75 and row["win_rate"] >= 0.50:
                near.append(entry)
    hits.sort(key=lambda x: (x["win_rate"], x["avg_return_pct"], x["trades"]), reverse=True)
    near.sort(key=lambda x: (x["win_rate"], x["avg_return_pct"]), reverse=True)
    return hits, near


def scan_high_gain_setups(
    store: StockDataStore,
    *,
    start_date: int = DEFAULT_START,
    min_market_cap: float = MIN_MARKET_CAP,
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
    max_win_rate: float = DEFAULT_MAX_WIN_RATE,
    min_trades: int = 10,
    min_gain_win: float = MIN_GAIN_WIN_PCT,
    min_avg_gain: float = MIN_GAIN_AVG_PCT,
    max_avg_gain: float = MAX_GAIN_AVG_PCT,
    max_hold: int = MAX_HOLD_DAYS,
    supplement_external: bool = True,
    universe: str = "large_cap",
) -> dict:
    """Scan setup: win >= min_gain_win%, avg gain min_avg–max_avg%, hold on-track up to max_hold."""
    panel, panel_meta = load_panel(
        store,
        start_date=start_date,
        min_market_cap=min_market_cap,
        universe=universe,
        supplement_external=supplement_external,
    )
    panel = _add_features(panel)
    min_d, max_d = int(panel["date"].min()), int(panel["date"].max())
    symbol_frames = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in panel.groupby("symbol")
    }

    aggregate: list[dict] = []
    for spec in SCENARIOS:
        row = _evaluate_on_track(
            panel, spec.id, max_hold=max_hold, min_gain_win=min_gain_win, symbol_frames=symbol_frames
        )
        row["name"] = spec.name
        row["passes"] = _passes_gain_filter(
            row,
            min_win_rate=min_win_rate,
            max_win_rate=max_win_rate,
            min_trades=min_trades,
            min_gain_win=min_gain_win,
            min_avg_gain=min_avg_gain,
            max_avg_gain=max_avg_gain,
        )
        aggregate.append(row)

    grid_defs: list[tuple[str, pd.Series]] = []
    for threshold in (-5, -8, -10, -12, -15):
        grid_defs.append((f"grid_drop_{abs(threshold)}pct", panel["change"] <= threshold))
    for streak in (2, 3, 4, 5):
        sig = panel["net_foreign"] > 0
        for lag in range(1, streak):
            sig &= panel.groupby("symbol")["net_foreign"].shift(lag) > 0
        grid_defs.append((f"grid_foreign_{streak}d", sig))
    for mom in (5, 7, 10, 15):
        grid_defs.append((f"grid_momentum_{mom}pct", panel["ret_5d"] >= mom))
    for n, col in ((10, "high_10_prev"), (20, "high_20_prev"), (30, "high_30_prev")):
        grid_defs.append((f"grid_breakout_{n}d", panel["close"] > panel[col]))
    for threshold in (-8, -10, -15, -20):
        grid_defs.append((f"grid_oversold_{abs(threshold)}pct", panel["ret_5d"] <= threshold))

    grid_hits: list[dict] = []
    for label, sig in grid_defs:
        row = _evaluate_on_track(
            panel, label, signal=sig, max_hold=max_hold, min_gain_win=min_gain_win, symbol_frames=symbol_frames
        )
        if _passes_gain_filter(
            row,
            min_win_rate=min_win_rate,
            max_win_rate=max_win_rate,
            min_trades=min_trades,
            min_gain_win=min_gain_win,
            min_avg_gain=min_avg_gain,
            max_avg_gain=max_avg_gain,
        ):
            grid_hits.append(row)

    ranked = sorted(aggregate, key=lambda x: x["avg_return_pct"], reverse=True)
    top_ids = {r["scenario_id"] for r in ranked[:10]}
    per_scenarios: list[tuple[str, pd.Series | None]] = [
        (s.id, None) for s in SCENARIOS if s.id in top_ids
    ]
    seen = {x[0] for x in per_scenarios}
    for label, sig in grid_defs:
        if label not in seen:
            per_scenarios.append((label, sig))
            seen.add(label)

    per_symbol, near_miss = _scan_per_symbol_on_track(
        panel,
        per_scenarios,
        symbol_frames,
        min_win_rate=min_win_rate,
        max_win_rate=max_win_rate,
        min_trades=min_trades,
        min_gain_win=min_gain_win,
        min_avg_gain=min_avg_gain,
        max_avg_gain=max_avg_gain,
        max_hold=max_hold,
    )

    aggregate_pass = [r for r in aggregate if r["passes"]]
    grid_hits.sort(key=lambda x: (x["win_rate"], x["avg_return_pct"]), reverse=True)

    return {
        "start_date": start_date,
        "end_date": max_d,
        "universe": universe,
        "min_market_cap": min_market_cap,
        "min_win_rate": min_win_rate,
        "max_win_rate": max_win_rate,
        "min_trades": min_trades,
        "min_gain_win_pct": min_gain_win,
        "min_avg_gain_pct": min_avg_gain,
        "max_avg_gain_pct": max_avg_gain,
        "max_hold_days": max_hold,
        "exit_rule": "on_track + TP 50% + SL 10% + max hold",
        "panel_rows": int(len(panel)),
        "panel_meta": panel_meta,
        "aggregate": aggregate,
        "aggregate_pass": aggregate_pass,
        "grid_hits": grid_hits,
        "per_symbol_hits": per_symbol[:80],
        "near_miss": near_miss[:30],
        "total_per_symbol_hits": len(per_symbol),
    }


def format_gain_scan_report(payload: dict) -> str:
    band = f"{payload['min_win_rate']*100:.0f}–{payload['max_win_rate']*100:.0f}%"
    gain_band = f"{payload['min_avg_gain_pct']:.0f}–{payload['max_avg_gain_pct']:.0f}%"
    lines = [
        "OK: high-gain on-track scan selesai",
        f"Periode: {payload['start_date']} .. {payload['end_date']}",
        f"Hold: on-track (max {payload['max_hold_days']} hari bursa ~3 bln), TP 50%, SL 10%",
        f"Win = gain >= {payload['min_gain_win_pct']:.0f}% per transaksi",
        f"Filter: win_rate {band}, avg gain {gain_band}, trades >= {payload['min_trades']}",
        f"Panel: {payload['panel_rows']:,} baris",
        "",
    ]

    hits = payload.get("per_symbol_hits") or []
    if hits:
        lines.append(f"=== SETUP LULUS ({len(hits)} dari {payload['total_per_symbol_hits']}) ===")
        for h in hits[:35]:
            lines.append(
                f"- {h['symbol']} + {h['scenario_id']}: win>={payload['min_gain_win_pct']:.0f}% "
                f"= {h['win_rate']*100:.1f}% ({h['wins']}/{h['trades']}) "
                f"avg={h['avg_return_pct']:.1f}% med={h['median_return_pct']:.1f}%"
            )
        lines.append("")
    else:
        lines.append("=== TIDAK ADA setup yang lulus filter ketat ===")
        lines.append("")

    agg_pass = payload.get("aggregate_pass") or []
    if agg_pass:
        lines.append("=== AGREGAT LULUS ===")
        for r in agg_pass:
            lines.append(
                f"- {r['scenario_id']}: win={r['win_rate']*100:.1f}% ({r['wins']}/{r['trades']}) "
                f"avg={r['avg_return_pct']:.1f}%"
            )
        lines.append("")

    grid = payload.get("grid_hits") or []
    if grid:
        lines.append("=== GRID AGREGAT LULUS ===")
        for r in grid[:15]:
            lines.append(
                f"- {r['scenario_id']}: win={r['win_rate']*100:.1f}% ({r['wins']}/{r['trades']}) "
                f"avg={r['avg_return_pct']:.1f}%"
            )
        lines.append("")

    near = payload.get("near_miss") or []
    if near and not hits:
        lines.append("=== NEAR MISS (avg gain OK, win rate 60%+, belum 75%) ===")
        for h in near[:20]:
            lines.append(
                f"- {h['symbol']} + {h['scenario_id']}: win>={payload['min_gain_win_pct']:.0f}% "
                f"= {h['win_rate']*100:.1f}% ({h['wins']}/{h['trades']}) avg={h['avg_return_pct']:.1f}%"
            )
        lines.append("")

    lines.append("=== TOP AGREGAT (semua skenario, sorted avg gain) ===")
    for r in sorted(payload.get("aggregate") or [], key=lambda x: x["avg_return_pct"], reverse=True)[:12]:
        lines.append(
            f"- {r['scenario_id']}: win>={payload['min_gain_win_pct']:.0f}% "
            f"= {r['win_rate']*100:.1f}% ({r['wins']}/{r['trades']}) "
            f"avg={r['avg_return_pct']:.1f}% med={r['median_return_pct']:.1f}%"
        )
    lines.append("")
    lines.append(
        "CATATAN: win 20%+ dengan hold panjang jauh lebih sulit dari win >0%. "
        "Hasil per emiten — bukan universal. Validasi out-of-sample wajib."
    )
    return "\n".join(lines)


def run_gain_scan(
    store: StockDataStore,
    export_dir: Path,
    **kwargs,
) -> str:
    payload = scan_high_gain_setups(store, **kwargs)
    export_dir = Path(export_dir) / "backtest"
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
    json_path = export_dir / f"gain_scan_{ts}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report = format_gain_scan_report(payload)
    md_path = export_dir / f"gain_scan_{ts}.md"
    md_path.write_text(report, encoding="utf-8")
    return f"{report}\n\nFILE: {json_path}\nFILE: {md_path}"


def _evaluate_scenario(df: pd.DataFrame, spec: ScenarioSpec) -> dict:
    hold = spec.hold_days
    work = df.copy()
    work["signal"] = _signal_mask(work, spec.id)
    work["exit_close"] = work.groupby("symbol")["close"].shift(-hold)
    trades = work[work["signal"] & work["exit_close"].notna()].copy()
    if trades.empty:
        return {
            "scenario_id": spec.id,
            "name": spec.name,
            "hold_days": hold,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "symbols": 0,
        }

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
    work["signal"] = signal
    work["exit_close"] = work.groupby("symbol")["close"].shift(-hold_days)
    trades = work[work["signal"] & work["exit_close"].notna()]
    if trades.empty:
        return {
            "scenario_id": label,
            "name": label,
            "hold_days": hold_days,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "symbols": 0,
        }
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


def grid_search(
    df: pd.DataFrame,
    *,
    min_win_rate: float,
    max_win_rate: float,
    min_trades: int,
) -> list[dict]:
    """Parameter grid: drop-bounce, foreign streak, momentum, volume spike."""
    results: list[dict] = []

    for threshold in (-2, -3, -4, -5, -6, -7, -8, -9, -10):
        for hold in (1, 2, 3, 5, 7, 10, 14):
            sig = df["change"] <= threshold
            label = f"grid_drop_{abs(threshold)}pct_hold{hold}d"
            row = _evaluate_custom(df, sig, hold, label)
            row["description"] = f"Beli close jika change<={threshold}, hold {hold}d"
            row["passes_filter"] = _passes_filter(
                row["win_rate"], row["trades"],
                min_win_rate=min_win_rate, max_win_rate=max_win_rate, min_trades=min_trades,
            )
            results.append(row)

    for streak in (2, 3, 4, 5):
        for hold in (3, 5, 7, 10, 14):
            sig = df["net_foreign"] > 0
            for lag in range(1, streak):
                sig &= df.groupby("symbol")["net_foreign"].shift(lag) > 0
            label = f"grid_foreign_{streak}d_hold{hold}d"
            row = _evaluate_custom(df, sig, hold, label)
            row["description"] = f"Net foreign positif {streak} hari berturut, hold {hold}d"
            row["passes_filter"] = _passes_filter(
                row["win_rate"], row["trades"],
                min_win_rate=min_win_rate, max_win_rate=max_win_rate, min_trades=min_trades,
            )
            results.append(row)

    for mom in (3, 5, 7, 10):
        for hold in (3, 5, 7, 10):
            sig = df["ret_5d"] >= mom
            label = f"grid_momentum_{mom}pct_hold{hold}d"
            row = _evaluate_custom(df, sig, hold, label)
            row["description"] = f"Return 5d >= {mom}%, hold {hold}d"
            row["passes_filter"] = _passes_filter(
                row["win_rate"], row["trades"],
                min_win_rate=min_win_rate, max_win_rate=max_win_rate, min_trades=min_trades,
            )
            results.append(row)

    for mult in (2.0, 2.5, 3.0, 4.0):
        for hold in (3, 5, 7, 10):
            sig = (df["volume"] > mult * df["vol_ma20"]) & (df["close"] > df["prev_close"])
            label = f"grid_vol_{str(mult).replace('.', 'p')}x_hold{hold}d"
            row = _evaluate_custom(df, sig, hold, label)
            row["description"] = f"Volume > {mult}x MA20 & close up, hold {hold}d"
            row["passes_filter"] = _passes_filter(
                row["win_rate"], row["trades"],
                min_win_rate=min_win_rate, max_win_rate=max_win_rate, min_trades=min_trades,
            )
            results.append(row)

    for n, col in ((10, "high_10_prev"), (20, "high_20_prev"), (30, "high_30_prev")):
        for hold in (5, 7, 10, 15):
            sig = df["close"] > df[col]
            label = f"grid_breakout_{n}d_hold{hold}d"
            row = _evaluate_custom(df, sig, hold, label)
            row["description"] = f"Close > high {n}d, hold {hold}d"
            row["passes_filter"] = _passes_filter(
                row["win_rate"], row["trades"],
                min_win_rate=min_win_rate, max_win_rate=max_win_rate, min_trades=min_trades,
            )
            results.append(row)

    for hold in (3, 5, 7, 10, 14):
        sig = (df["ma5"] > df["ma20"]) & (df["ma5_prev"] <= df["ma20_prev"])
        label = f"grid_ma5_20_cross_hold{hold}d"
        row = _evaluate_custom(df, sig, hold, label)
        row["description"] = f"MA5 cross MA20, hold {hold}d"
        row["passes_filter"] = _passes_filter(
            row["win_rate"], row["trades"],
            min_win_rate=min_win_rate, max_win_rate=max_win_rate, min_trades=min_trades,
        )
        results.append(row)

    for threshold in (-5, -8, -10, -12):
        for hold in (3, 5, 7, 10):
            sig = df["ret_5d"] <= threshold
            label = f"grid_oversold_{abs(threshold)}pct_hold{hold}d"
            row = _evaluate_custom(df, sig, hold, label)
            row["description"] = f"Return 5d <= {threshold}%, hold {hold}d"
            row["passes_filter"] = _passes_filter(
                row["win_rate"], row["trades"],
                min_win_rate=min_win_rate, max_win_rate=max_win_rate, min_trades=min_trades,
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
    top_n: int = 50,
) -> list[dict]:
    hits: list[dict] = []
    for spec in specs:
        hold = spec.hold_days
        work = df.copy()
        work["signal"] = _signal_mask(work, spec.id)
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
    return hits[:top_n]


def scan_per_symbol_grid(
    df: pd.DataFrame,
    *,
    min_win_rate: float,
    max_win_rate: float,
    min_trades: int = 12,
    top_n: int = 80,
) -> list[dict]:
    """Exhaustive per-symbol grid for setups in the win-rate band."""
    hits: list[dict] = []
    symbols = df["symbol"].unique()

    grid_defs: list[tuple[str, pd.Series, int]] = []
    for threshold in (-3, -4, -5, -6, -7, -8):
        for hold in (3, 5, 7):
            grid_defs.append(
                (f"grid_drop_{abs(threshold)}pct_hold{hold}d", df["change"] <= threshold, hold)
            )
    for streak in (2, 3, 4):
        for hold in (5, 7, 10):
            sig = df["net_foreign"] > 0
            for lag in range(1, streak):
                sig &= df.groupby("symbol")["net_foreign"].shift(lag) > 0
            grid_defs.append((f"grid_foreign_{streak}d_hold{hold}d", sig, hold))
    for threshold in (5, 7, 10):
        for hold in (5, 7):
            grid_defs.append(
                (f"grid_momentum_{threshold}pct_hold{hold}d", df["ret_5d"] >= threshold, hold)
            )

    for sym in symbols:
        sub = df[df["symbol"] == sym]
        if len(sub) < min_trades + 10:
            continue
        for label, base_sig, hold in grid_defs:
            sig = base_sig.reindex(sub.index, fill_value=False)
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
    return hits[:top_n]


def _merge_band_hits(
    passed: list[dict],
    grid_passed: list[dict],
    per_symbol: list[dict],
    per_symbol_grid: list[dict],
    *,
    min_trades: int,
) -> list[dict]:
    """Unified list of all setups in the win-rate band."""
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
    universe: str = "large_cap",
) -> dict:
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
        universe=universe,
        supplement_external=supplement_external,
    )
    panel = _add_features(panel)
    coverage = panel_coverage_report(panel)

    min_d, max_d = int(panel["date"].min()), int(panel["date"].max())
    universe_avg = panel.groupby("date")["symbol"].nunique().mean()

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
    per_symbol_min = max(12, min_trades - 3)
    per_symbol = scan_per_symbol_hits(
        panel,
        specs,
        min_win_rate=min_win_rate,
        max_win_rate=max_win_rate,
        min_trades=per_symbol_min,
    )
    per_symbol_grid = scan_per_symbol_grid(
        panel,
        min_win_rate=min_win_rate,
        max_win_rate=max_win_rate,
        min_trades=per_symbol_min,
        top_n=120,
    )

    all_band_hits = _merge_band_hits(
        passed, grid_passed, per_symbol, per_symbol_grid, min_trades=min_trades
    )

    return {
        "start_date": start_date,
        "end_date": max_d,
        "universe": universe,
        "min_market_cap": min_market_cap,
        "min_win_rate": min_win_rate,
        "max_win_rate": max_win_rate,
        "min_trades": min_trades,
        "panel_rows": int(len(panel)),
        "panel_meta": panel_meta,
        "data_supplement": panel_meta.get("supplement", {}),
        "coverage": coverage,
        "avg_symbols_per_day": float(universe_avg),
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
    uni = payload.get("universe", "large_cap")
    uni_line = (
        f"Universe: {uni} (market cap >= {payload['min_market_cap']:,.0f} IDR, "
        f"~{payload['avg_symbols_per_day']:.0f} emiten/hari)"
        if uni == "large_cap"
        else f"Universe: {uni} (~{payload['avg_symbols_per_day']:.0f} emiten/hari)"
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
        lines.append(f"=== SETUP TRADING (win {band}, total {len(band_hits)}) ===")
        for h in band_hits[:40]:
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
        lines.append("")
        lines.append(f"=== GRID SEARCH LULUS ({len(payload['grid_passed'])}) ===")
        for r in payload["grid_passed"][:15]:
            lines.append(
                f"- {r['scenario_id']}: win={r['win_rate']*100:.1f}% "
                f"({r['wins']}/{r['trades']}) avg={r['avg_return_pct']:.2f}%"
            )

    if payload.get("per_symbol_hits"):
        lines.append("")
        lines.append(f"=== PER EMITEN (skenario bawaan, win {band}) ===")
        for h in payload["per_symbol_hits"][:25]:
            lines.append(
                f"- {h['symbol']} + {h['scenario_id']}: win={h['win_rate']*100:.1f}% "
                f"({h['wins']}/{h['trades']}) avg={h['avg_return_pct']:.2f}%"
            )

    if payload.get("per_symbol_grid"):
        lines.append("")
        lines.append(f"=== PER EMITEN GRID (win {band}, top) ===")
        for h in payload["per_symbol_grid"][:25]:
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
    md_lines = [
        "# Backtest scan — IDX",
        "",
        f"- Generated: {datetime.now(WIB).isoformat()}",
        f"- Period: {payload['start_date']} .. {payload['end_date']}",
        f"- Universe: {payload.get('universe', 'large_cap')}",
        f"- Min market cap: IDR {payload['min_market_cap']:,.0f}",
        f"- Filter: win rate {payload['min_win_rate']*100:.0f}–{max_wr*100:.0f}%, min trades {payload['min_trades']}",
        "",
        "## Skenario lulus",
        "",
    ]
    if payload["passed"]:
        for r in payload["passed"]:
            md_lines.extend(
                [
                    f"### {r['scenario_id']} — {r['name']}",
                    "",
                    r["description"],
                    "",
                    f"- Entry: {r['entry']}",
                    f"- Exit: {r['exit_rule']}",
                    f"- Win rate: **{r['win_rate']*100:.1f}%** ({r['wins']}/{r['trades']})",
                    f"- Avg return: {r['avg_return_pct']:.2f}%",
                    f"- Median return: {r['median_return_pct']:.2f}%",
                    f"- Symbols traded: {r['symbols']}",
                    "",
                ]
            )
    else:
        md_lines.append("_Tidak ada skenario agregat yang lulus filter._\n")

    if payload.get("grid_passed"):
        md_lines.append("## Grid search lulus\n")
        for r in payload["grid_passed"][:20]:
            md_lines.append(
                f"- **{r['scenario_id']}**: win {r['win_rate']*100:.1f}% "
                f"({r['wins']}/{r['trades']}), avg {r['avg_return_pct']:.2f}%"
            )
        md_lines.append("")

    if payload.get("per_symbol_hits"):
        md_lines.append("## Per emiten (win rate tinggi)\n")
        md_lines.append("| Symbol | Scenario | Win% | Trades | Avg ret% |")
        md_lines.append("|---|---|---:|---:|---:|")
        for h in payload["per_symbol_hits"][:30]:
            md_lines.append(
                f"| {h['symbol']} | {h['scenario_id']} | {h['win_rate']*100:.1f} | "
                f"{h['trades']} | {h['avg_return_pct']:.2f} |"
            )
        md_lines.append("")

    if payload.get("per_symbol_grid"):
        md_lines.append("## Per emiten — grid search\n")
        md_lines.append("| Symbol | Scenario | Win% | Trades | Avg ret% |")
        md_lines.append("|---|---|---:|---:|---:|")
        for h in payload["per_symbol_grid"][:40]:
            md_lines.append(
                f"| {h['symbol']} | {h['scenario_id']} | {h['win_rate']*100:.1f} | "
                f"{h['trades']} | {h['avg_return_pct']:.2f} |"
            )
        md_lines.append("")

    md_lines.append("## Semua skenario\n")
    md_lines.append("| ID | Win% | Trades | Avg ret% | Hold | Pass |")
    md_lines.append("|---|---:|---:|---:|---:|:---:|")
    for r in sorted(payload["results"], key=lambda x: x["win_rate"], reverse=True):
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
    universe: str = "large_cap",
) -> str:
    start = int(str(start_date).replace("-", "")[:8])
    payload = scan_scenarios(
        store,
        start_date=start,
        min_market_cap=min_market_cap,
        min_win_rate=min_win_rate,
        max_win_rate=max_win_rate,
        min_trades=min_trades,
        supplement_external=supplement_external,
        universe=universe,
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
