"""Investigate pre-signs before price rises along stacked MA5/20/50/200.

Question: before a stock rides the MAs upward, do the MA lines narrow (squeeze)?
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from agent_idx.backtest import DEFAULT_START, MIN_MARKET_CAP, load_panel
from agent_idx.data import StockDataStore

WIB = ZoneInfo("Asia/Jakarta")

# Lookback windows (trading days) to measure squeeze before the event
LOOKBACKS = (5, 10, 20)
# Forward return window to confirm "naik ikut MA"
FORWARD_DAYS = 20
# Min forward gain to count as a "rise along MA" event
MIN_FORWARD_GAIN = 10.0


def _add_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("symbol", group_keys=False)
    df["ma5"] = g["close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    df["ma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["ma50"] = g["close"].transform(lambda s: s.rolling(50, min_periods=50).mean())
    df["ma200"] = g["close"].transform(lambda s: s.rolling(200, min_periods=200).mean())
    df["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    foreign_known = df["foreign_buy"].notna() & df["foreign_sell"].notna()
    df["nf"] = (df["foreign_buy"] - df["foreign_sell"]).where(foreign_known)
    df["nf_pos"] = (df["nf"] > 0).where(foreign_known)

    mas = df[["ma5", "ma20", "ma50", "ma200"]]
    df["ma_max"] = mas.max(axis=1)
    df["ma_min"] = mas.min(axis=1)
    df["ma_mid"] = mas.mean(axis=1)
    df["ma_bandwidth"] = (df["ma_max"] - df["ma_min"]) / df["ma_mid"] * 100

    df["spread_5_20"] = (df["ma5"] - df["ma20"]).abs() / df["ma_mid"] * 100
    df["spread_20_50"] = (df["ma20"] - df["ma50"]).abs() / df["ma_mid"] * 100
    df["spread_50_200"] = (df["ma50"] - df["ma200"]).abs() / df["ma_mid"] * 100

    df["stacked_bull"] = (
        (df["ma5"] > df["ma20"])
        & (df["ma20"] > df["ma50"])
        & (df["ma50"] > df["ma200"])
    )
    df["price_above_all"] = (
        (df["close"] > df["ma5"])
        & (df["close"] > df["ma20"])
        & (df["close"] > df["ma50"])
        & (df["close"] > df["ma200"])
    )
    df["close_vs_ma5_pct"] = (df["close"] / df["ma5"] - 1) * 100

    g = df.groupby("symbol", group_keys=False)
    for lb in LOOKBACKS:
        prev = g["ma_bandwidth"].shift(lb)
        df[f"bw_chg_{lb}d"] = np.where(
            prev.abs() > 0.05,
            (df["ma_bandwidth"] - prev) / prev * 100,
            np.nan,
        )
        df[f"bw_prev_{lb}d"] = prev
        df[f"bw_delta_{lb}d"] = df["ma_bandwidth"] - prev
        df[f"bw_min_{lb}d"] = g["ma_bandwidth"].transform(
            lambda s: s.rolling(lb, min_periods=max(3, lb // 2)).min()
        )
        df[f"squeeze_{lb}d"] = (
            (df["ma_bandwidth"] <= df[f"bw_min_{lb}d"] * 1.15)
            & (df[f"bw_chg_{lb}d"] < 0)
        )

    g = df.groupby("symbol", group_keys=False)
    for lb in LOOKBACKS:
        # State N days BEFORE the event day
        df[f"pre_bw_{lb}d"] = g["ma_bandwidth"].shift(lb)
        df[f"pre_squeeze_{lb}d"] = g[f"squeeze_{lb}d"].shift(lb)
        df[f"pre_vol_ratio_{lb}d"] = g["vol_ratio"].shift(lb)
        df[f"pre_nf_pos_{lb}d"] = g["nf_pos"].shift(lb)

    df["fwd_ret"] = g["close"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS) * 100
    stacked_prev = g["stacked_bull"].shift(1).fillna(False)
    above_prev = g["price_above_all"].shift(1).fillna(False)
    df["ma_ride_onset"] = df["stacked_bull"] & df["price_above_all"] & ~(stacked_prev & above_prev)
    return df


def _safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) else float("nan")


def _safe_pct(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean() * 100) if len(s) else float("nan")


def _summarize_group(sub: pd.DataFrame, label: str) -> dict:
    row: dict = {
        "group": label,
        "n": int(len(sub)),
        "avg_fwd_ret_pct": _safe_mean(sub["fwd_ret"]),
        "pct_fwd_gt_10": _safe_pct(sub["fwd_ret"] >= MIN_FORWARD_GAIN),
        "avg_ma_bandwidth": _safe_mean(sub["ma_bandwidth"]),
        "avg_spread_5_20": _safe_mean(sub["spread_5_20"]),
        "avg_spread_20_50": _safe_mean(sub["spread_20_50"]),
        "avg_spread_50_200": _safe_mean(sub["spread_50_200"]),
        "avg_vol_ratio": _safe_mean(sub["vol_ratio"]),
        "pct_nf_positive": _safe_pct(sub["nf_pos"]),
        "avg_close_vs_ma5": _safe_mean(sub["close_vs_ma5_pct"]),
    }
    for lb in LOOKBACKS:
        row[f"avg_bw_chg_{lb}d"] = _safe_mean(sub[f"bw_chg_{lb}d"])
        row[f"avg_bw_delta_{lb}d"] = _safe_mean(sub[f"bw_delta_{lb}d"])
        row[f"pct_squeeze_{lb}d"] = _safe_pct(sub[f"squeeze_{lb}d"])
        row[f"pct_bw_narrowed_{lb}d"] = _safe_pct(sub[f"bw_chg_{lb}d"] < 0)
        row[f"pre_bw_{lb}d"] = _safe_mean(sub[f"pre_bw_{lb}d"])
        row[f"pre_pct_squeeze_{lb}d"] = _safe_pct(sub[f"pre_squeeze_{lb}d"].fillna(False))
        row[f"pre_vol_ratio_{lb}d"] = _safe_mean(sub[f"pre_vol_ratio_{lb}d"])
        row[f"pre_pct_nf_pos_{lb}d"] = _safe_pct(sub[f"pre_nf_pos_{lb}d"].fillna(False))
    return row


def investigate_ma_squeeze(
    store: StockDataStore,
    *,
    start_date: int = DEFAULT_START,
    min_market_cap: float = MIN_MARKET_CAP,
    cap_filter: str = "large_cap",
    min_forward_gain: float = MIN_FORWARD_GAIN,
    sample_control: int = 50_000,
) -> dict:
    panel, meta = load_panel(
        store,
        start_date=start_date,
        min_market_cap=min_market_cap,
        cap_filter=cap_filter,
        supplement_external=False,
    )
    panel = _add_ma_features(panel)
    # Need MA200 ready
    ready = panel.dropna(subset=["ma5", "ma20", "ma50", "ma200", "fwd_ret", "ma_bandwidth"])
    ready = ready[ready["ma_mid"] > 0].copy()

    # Event A: onset of riding stacked MAs AND forward gain >= threshold
    onset = ready[ready["ma_ride_onset"]].copy()
    successful = onset[onset["fwd_ret"] >= min_forward_gain].copy()
    failed = onset[onset["fwd_ret"] < min_forward_gain].copy()

    # Event B: any day already stacked + above, then big rise (not just onset)
    riding = ready[ready["stacked_bull"] & ready["price_above_all"]].copy()
    ride_success = riding[riding["fwd_ret"] >= min_forward_gain].copy()

    # Control: random days with valid MAs (not successful ride days)
    ctrl_pool = ready[~ready.index.isin(successful.index)]
    n_ctrl = min(sample_control, len(ctrl_pool))
    control = ctrl_pool.sample(n=n_ctrl, random_state=42) if n_ctrl else ctrl_pool

    comparisons = [
        _summarize_group(successful, f"onset+gain>={min_forward_gain}%"),
        _summarize_group(failed, f"onset+gain<{min_forward_gain}%"),
        _summarize_group(ride_success, f"any_ride+gain>={min_forward_gain}%"),
        _summarize_group(control, "control_random"),
    ]

    # Squeeze → then rise: among days where squeeze_10d, what is forward return?
    squeeze_stats = []
    for lb in LOOKBACKS:
        col = f"squeeze_{lb}d"
        sq = ready[ready[col]]
        non = ready[~ready[col]]
        squeeze_stats.append(
            {
                "lookback": lb,
                "n_squeeze": int(len(sq)),
                "avg_fwd_ret_squeeze": _safe_mean(sq["fwd_ret"]),
                "pct_gain_ge_10_squeeze": _safe_pct(sq["fwd_ret"] >= min_forward_gain),
                "n_no_squeeze": int(len(non)),
                "avg_fwd_ret_no_squeeze": _safe_mean(non["fwd_ret"]),
                "pct_gain_ge_10_no_squeeze": _safe_pct(non["fwd_ret"] >= min_forward_gain),
            }
        )

    # Top symbols with successful onset + prior squeeze
    top_syms = []
    if not successful.empty:
        for lb in LOOKBACKS:
            successful[f"_sq{lb}"] = successful[f"squeeze_{lb}d"].astype(int)
        agg = (
            successful.groupby("symbol")
            .agg(
                events=("fwd_ret", "count"),
                avg_fwd=("fwd_ret", "mean"),
                avg_bw=("ma_bandwidth", "mean"),
                pct_sq10=("squeeze_10d", "mean"),
                pct_sq20=("squeeze_20d", "mean"),
                avg_bw_chg_10=("bw_chg_10d", "mean"),
            )
            .reset_index()
        )
        agg = agg[agg["events"] >= 3].sort_values("avg_fwd", ascending=False)
        for _, r in agg.head(25).iterrows():
            top_syms.append(
                {
                    "symbol": r["symbol"],
                    "events": int(r["events"]),
                    "avg_fwd_ret_pct": float(r["avg_fwd"]),
                    "avg_bandwidth": float(r["avg_bw"]),
                    "pct_squeeze_10d": float(r["pct_sq10"] * 100),
                    "pct_squeeze_20d": float(r["pct_sq20"] * 100),
                    "avg_bw_chg_10d": float(r["avg_bw_chg_10"]),
                }
            )

    # Verdict: compare successful onset vs control on PRE-event squeeze
    s_ok = comparisons[0]
    s_ctrl = comparisons[3]
    pre_sq_ok = s_ok.get("pre_pct_squeeze_10d", 0) or 0
    pre_sq_ctrl = s_ctrl.get("pre_pct_squeeze_10d", 0) or 0
    bw_ok = s_ok.get("avg_ma_bandwidth", 0) or 0
    bw_ctrl = s_ctrl.get("avg_ma_bandwidth", 0) or 0
    # Narrowing hypothesis: successful should have MORE pre-squeeze and LOWER/narrowing BW
    squeeze_more = pre_sq_ok > pre_sq_ctrl + 5 and (s_ok.get("avg_bw_delta_10d") or 0) < 0

    return {
        "start_date": start_date,
        "end_date": int(ready["date"].max()) if len(ready) else None,
        "cap_filter": cap_filter,
        "panel_rows": int(len(panel)),
        "ready_rows": int(len(ready)),
        "n_onset": int(len(onset)),
        "n_successful_onset": int(len(successful)),
        "n_failed_onset": int(len(failed)),
        "min_forward_gain_pct": min_forward_gain,
        "forward_days": FORWARD_DAYS,
        "comparisons": comparisons,
        "squeeze_predictive": squeeze_stats,
        "top_symbols": top_syms,
        "verdict": {
            "ma_narrows_before_rise": bool(squeeze_more),
            "note": (
                "Jika bandwidth onset LEBIH LEBAR dari kontrol dan squeeze LEBIH JARANG, "
                "hipotesis 'MA menyempit dulu' TIDAK didukung."
            ),
            "successful_avg_bw": bw_ok,
            "control_avg_bw": bw_ctrl,
            "successful_pre_squeeze_10d": pre_sq_ok,
            "control_pre_squeeze_10d": pre_sq_ctrl,
            "successful_bw_delta_10d": s_ok.get("avg_bw_delta_10d"),
            "control_bw_delta_10d": s_ctrl.get("avg_bw_delta_10d"),
            "successful_vol_ratio": s_ok.get("avg_vol_ratio"),
            "control_vol_ratio": s_ctrl.get("avg_vol_ratio"),
            "successful_nf_pos": s_ok.get("pct_nf_positive"),
            "control_nf_pos": s_ctrl.get("pct_nf_positive"),
        },
        "panel_meta": meta,
    }


def format_ma_squeeze_report(payload: dict) -> str:
    v = payload["verdict"]
    lines = [
        "OK: investigasi MA squeeze sebelum naik ikut MA selesai",
        f"Periode: {payload['start_date']} .. {payload['end_date']}",
        f"Cap filter: {payload['cap_filter']} | ready rows: {payload['ready_rows']:,}",
        f"Event: onset stacked MA5>20>50>200 + close di atas semua MA",
        f"Sukses = return +{payload['forward_days']}d >= {payload['min_forward_gain_pct']}%",
        f"Onset total: {payload['n_onset']:,} | sukses: {payload['n_successful_onset']:,} | gagal: {payload['n_failed_onset']:,}",
        "",
        "=== VERDICT: apakah MA menyempit sebelum naik? ===",
        f"Jawaban: {'YA — lebih sering menyempit vs kontrol' if v['ma_narrows_before_rise'] else 'TIDAK — data tidak mendukung hipotesis squeeze'}",
        f"- Bandwidth saat onset: sukses={v['successful_avg_bw']:.2f}% | kontrol={v['control_avg_bw']:.2f}%",
        f"- Delta bandwidth 10d (negatif=menyempit): sukses={v['successful_bw_delta_10d']:.2f}pp | kontrol={v['control_bw_delta_10d']:.2f}pp",
        f"- Squeeze 10d SEBELUM onset: sukses={v['successful_pre_squeeze_10d']:.1f}% | kontrol={v['control_pre_squeeze_10d']:.1f}%",
        f"- Volume ratio onset: sukses={v['successful_vol_ratio']:.2f}x | kontrol={v['control_vol_ratio']:.2f}x",
        f"- Net foreign positif onset: sukses={v['successful_nf_pos']:.1f}% | kontrol={v['control_nf_pos']:.1f}%",
        f"- Note: {v['note']}",
        "",
        "=== PERBANDINGAN GRUP ===",
    ]
    for c in payload["comparisons"]:
        lines.append(
            f"- {c['group']} (n={c['n']:,}): "
            f"fwd={c['avg_fwd_ret_pct']:.2f}% | bw={c['avg_ma_bandwidth']:.2f}% | "
            f"bw_delta10={c.get('avg_bw_delta_10d', float('nan')):.2f}pp | "
            f"pre_sq10={c.get('pre_pct_squeeze_10d', float('nan')):.1f}% | "
            f"volx={c['avg_vol_ratio']:.2f} | nf+={c['pct_nf_positive']:.1f}%"
        )
    lines.append("")
    lines.append("=== APAKAH SQUEEZE MEMPREDIKSI NAIK? ===")
    for s in payload["squeeze_predictive"]:
        lines.append(
            f"- Squeeze {s['lookback']}d: fwd={s['avg_fwd_ret_squeeze']:.2f}% "
            f"(gain>={payload['min_forward_gain_pct']}%: {s['pct_gain_ge_10_squeeze']:.1f}%, n={s['n_squeeze']:,}) "
            f"| tanpa squeeze: fwd={s['avg_fwd_ret_no_squeeze']:.2f}% "
            f"({s['pct_gain_ge_10_no_squeeze']:.1f}%, n={s['n_no_squeeze']:,})"
        )
    lines.append("")
    if payload.get("top_symbols"):
        lines.append("=== EMITEN: onset sukses (sample) ===")
        for h in payload["top_symbols"][:15]:
            lines.append(
                f"- {h['symbol']}: events={h['events']} avg_fwd={h['avg_fwd_ret_pct']:.1f}% "
                f"bw={h['avg_bandwidth']:.1f}% sq10={h['pct_squeeze_10d']:.0f}%"
            )
        lines.append("")
    lines.append(
        "CATATAN: bandwidth = (max-min MA5/20/50/200)/mid*100. "
        "Ini research pada large-cap IDX, bukan saran investasi."
    )
    return "\n".join(lines)


def run_ma_squeeze_study(
    store: StockDataStore,
    export_dir: Path,
    **kwargs,
) -> str:
    payload = investigate_ma_squeeze(store, **kwargs)
    out = Path(export_dir) / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
    json_path = out / f"ma_squeeze_{ts}.json"
    md_path = out / f"ma_squeeze_{ts}.md"
    # Drop non-serializable if any
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report = format_ma_squeeze_report(payload)
    md_path.write_text(report, encoding="utf-8")
    return f"{report}\n\nFILE: {json_path}\nFILE: {md_path}"
