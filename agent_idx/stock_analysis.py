"""Combined technical + fundamental Stock Analysis for one Symbol.

Read-only: never scrapes or writes. Returns a JSON evidence pack with
dimension labels (positive/neutral/negative/unknown), freshness flags,
explicit conflicts, and GAP_DATA where figures are missing. No buy/sell.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

WIB = ZoneInfo("Asia/Jakarta")

HORIZONS = frozenset({"intraday", "swing", "position"})

Label = str  # positive | neutral | negative | unknown


@dataclass(frozen=True)
class HorizonPreset:
    name: str
    timeframe: str
    lookback: int
    technical_max_age: timedelta
    fundamental_max_age: timedelta
    technical_weight: str  # dominant | balanced | secondary
    fundamental_weight: str


PRESETS: dict[str, HorizonPreset] = {
    "intraday": HorizonPreset(
        name="intraday",
        timeframe="15M",
        lookback=80,
        technical_max_age=timedelta(hours=8),
        fundamental_max_age=timedelta(days=21),
        technical_weight="dominant",
        fundamental_weight="secondary",
    ),
    "swing": HorizonPreset(
        name="swing",
        timeframe="1D",
        lookback=60,
        technical_max_age=timedelta(days=3),
        fundamental_max_age=timedelta(days=14),
        technical_weight="balanced",
        fundamental_weight="balanced",
    ),
    "position": HorizonPreset(
        name="position",
        timeframe="1D",
        lookback=120,
        technical_max_age=timedelta(days=7),
        fundamental_max_age=timedelta(days=45),
        technical_weight="secondary",
        fundamental_weight="dominant",
    ),
}

_BANK_RE = re.compile(r"\b(bank|perbankan|banking|syariah)\b", re.I)

# Preferred metric names → aliases (lowered, substring match order matters).
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "pe_ttm": ("current pe ratio (ttm)", "pe ratio (ttm)", "current pe ratio"),
    "pb": ("current price to book value", "price to book value", "p/b"),
    "roe": ("return on equity (ttm)", "return on equity"),
    "roa": ("return on assets (ttm)", "return on assets"),
    "div_yield": ("dividend yield",),
    "nim": ("net interest margin (nim)", "net interest margin"),
    "npl_gross": ("npl - gross", "npl gross"),
    "npl_coverage": ("npl - coverage", "npl coverage"),
    "car": ("capital adequacy ratio",),
    "ldr": ("loan to deposit ratio",),
    "der": ("debt to equity ratio (quarter)", "debt to equity ratio"),
    "current_ratio": ("current ratio (quarter)", "current ratio"),
    "rev_yoy": ("revenue (quarter yoy growth)", "revenue growth"),
    "ni_yoy": ("net income (quarter yoy growth)", "net income growth"),
    "fcf_ttm": ("free cash flow (ttm)",),
    "cfo_ttm": ("cash from operations (ttm)",),
    "rank_pe": ("rank (current pe ratio ttm)",),
    "rank_pb": ("rank (p/b)",),
    "ihsg_pe": ("ihsg pe ratio ttm (median)",),
}


def _env_hours(name: str, default_hours: float) -> timedelta:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return timedelta(hours=default_hours)
    try:
        return timedelta(hours=float(raw))
    except ValueError:
        return timedelta(hours=default_hours)


def _env_days(name: str, default_days: float) -> timedelta:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return timedelta(days=default_days)
    try:
        return timedelta(days=float(raw))
    except ValueError:
        return timedelta(days=default_days)


def resolve_preset(horizon: str) -> HorizonPreset:
    key = (horizon or "swing").strip().lower()
    if key in {"investment", "invest", "long"}:
        key = "position"
    if key not in PRESETS:
        key = "swing"
    base = PRESETS[key]
    # Optional overrides: ANALYSIS_{HORIZON}_TECH_HOURS / _FUND_DAYS
    prefix = f"ANALYSIS_{key.upper()}"
    tech = _env_hours(
        f"{prefix}_TECH_HOURS",
        base.technical_max_age.total_seconds() / 3600.0,
    )
    fund = _env_days(
        f"{prefix}_FUND_DAYS",
        base.fundamental_max_age.total_seconds() / 86400.0,
    )
    return HorizonPreset(
        name=base.name,
        timeframe=base.timeframe,
        lookback=base.lookback,
        technical_max_age=tech,
        fundamental_max_age=fund,
        technical_weight=base.technical_weight,
        fundamental_weight=base.fundamental_weight,
    )


def _finite(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _round(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    return round(float(x), nd)


def _now_wib() -> datetime:
    return datetime.now(WIB)


def _parse_as_of(raw: Any) -> datetime | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y%m%d"):
        try:
            dt = datetime.strptime(s[:19] if " " in s else s[:10], fmt)
            return dt.replace(tzinfo=WIB)
        except ValueError:
            continue
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        py = dt.to_pydatetime()
        if py.tzinfo is None:
            return py.replace(tzinfo=WIB)
        return py.astimezone(WIB)
    except Exception:  # noqa: BLE001
        return None


def _parse_bar_time(row: pd.Series) -> datetime | None:
    raw = row.get("datetime")
    dt = _parse_as_of(raw)
    if dt is not None:
        return dt
    ts = _finite(row.get("ts"))
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=WIB)


@dataclass
class Signal:
    dimension: str
    label: Label
    confidence: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "label": self.label,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 3),
            "evidence": self.evidence,
            "reasons": self.reasons,
        }


def _signal(
    dimension: str,
    label: Label,
    confidence: float,
    *,
    evidence: list[dict[str, Any]] | None = None,
    reasons: list[str] | None = None,
) -> Signal:
    return Signal(
        dimension=dimension,
        label=label,
        confidence=confidence,
        evidence=evidence or [],
        reasons=reasons or [],
    )


def _unknown(dimension: str, reason: str) -> Signal:
    return _signal(dimension, "unknown", 0.0, reasons=[reason])


# --------------------------------------------------------------------------- #
# Data loaders (read-only)
# --------------------------------------------------------------------------- #


def _load_ohlcv(symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
    from agent_idx import chart_db

    sym = symbol.upper()
    tf = timeframe.upper()
    n = max(20, min(int(lookback), 500))
    with chart_db._LOCK:
        con = chart_db.connect()
        try:
            chart_db.init_schema(con)
            count = con.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                [sym, tf],
            ).fetchone()[0]
            if not count:
                path = chart_db.CHARTS_DIR / sym / f"{tf}.parquet"
                if path.is_file():
                    con.close()
                    chart_db.sync_symbol_tf(sym, tf, path)
                    con = chart_db.connect()
                    chart_db.init_schema(con)
            df = con.execute(
                """
                SELECT datetime, open, high, low, close, volume,
                       sma_5, sma_20, sma_50, sma_200,
                       ema_12, ema_26, macd, macd_signal, macd_hist,
                       rsi_14, vol_sma_20, ts, source, updated_at
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                [sym, tf, n],
            ).df()
        finally:
            con.close()
    if df is None or df.empty:
        return pd.DataFrame()
    return df.iloc[::-1].reset_index(drop=True)


def _load_fundamentals(symbol: str) -> pd.DataFrame:
    from agent_idx.transforms import _LOCK, connect, init_schema

    sym = symbol.upper()
    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            df = con.execute(
                """
                SELECT section, metric, period, value_raw, value_num, as_of, updated_at
                FROM fundamental_metrics
                WHERE symbol = ?
                """,
                [sym],
            ).df()
        finally:
            con.close()
    return df if df is not None else pd.DataFrame()


def _section_trust(section: str) -> int:
    s = (section or "").lower()
    if s == "metrics (structured)":
        return 0
    if s.startswith("statement:"):
        return 1
    if s.startswith("series:"):
        return 2
    if s in {"company background", "company info", "shareholders"}:
        return 3
    if s == "raw page text":
        return 8
    return 5


def _pick_metric(
    df: pd.DataFrame,
    key: str,
    *,
    require_num: bool = True,
) -> dict[str, Any] | None:
    aliases = _METRIC_ALIASES.get(key, (key,))
    if df is None or df.empty:
        return None
    work = df.copy()
    work["_m"] = work["metric"].astype(str).str.lower().str.strip()
    work["_trust"] = [_section_trust(s) for s in work["section"].astype(str)]
    candidates: list[pd.Series] = []
    for alias in aliases:
        hit = work[work["_m"] == alias]
        if hit.empty:
            hit = work[work["_m"].str.contains(re.escape(alias), regex=True, na=False)]
        if hit.empty:
            continue
        hit = hit.sort_values(["_trust", "period"], ascending=[True, False])
        for _, row in hit.iterrows():
            num = _finite(row.get("value_num"))
            if require_num and num is None:
                raw = str(row.get("value_raw") or "").strip()
                if not raw or raw in {"-", "—", "–"}:
                    continue
            candidates.append(row)
            break
        if candidates:
            break
    if not candidates:
        return None
    row = candidates[0]
    return {
        "metric": str(row["metric"]),
        "section": str(row["section"]),
        "period": str(row.get("period") or "") or None,
        "value_raw": str(row.get("value_raw") or ""),
        "value_num": _finite(row.get("value_num")),
        "as_of": str(row.get("as_of") or "") or None,
    }


def _background(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    hit = df[
        (df["section"].astype(str).str.lower() == "company background")
        & (df["metric"].astype(str).str.lower() == "background")
    ]
    if hit.empty:
        return ""
    return str(hit.iloc[0].get("value_raw") or "")


def detect_sector(df: pd.DataFrame) -> dict[str, Any]:
    text = _background(df)
    if not text:
        return {
            "sector": "GAP_DATA",
            "adapter": "general",
            "sector_source": "GAP_DATA",
            "background_excerpt": None,
        }
    if _BANK_RE.search(text):
        return {
            "sector": "bank",
            "adapter": "bank",
            "sector_source": "company background",
            "background_excerpt": text[:180],
        }
    return {
        "sector": "general",
        "adapter": "general",
        "sector_source": "company background",
        "background_excerpt": text[:180],
    }


# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #


def _freshness_block(
    *,
    kind: str,
    when: datetime | None,
    max_age: timedelta,
    detail: dict[str, Any],
) -> dict[str, Any]:
    now = _now_wib()
    if when is None:
        return {
            "source": kind,
            "status": "GAP_DATA",
            "last": None,
            "age_hours": None,
            "max_age_hours": max_age.total_seconds() / 3600.0,
            **detail,
        }
    age = now - when
    stale = age > max_age
    return {
        "source": kind,
        "status": "STALE_DATA" if stale else "ok",
        "last": when.isoformat(),
        "age_hours": round(age.total_seconds() / 3600.0, 2),
        "max_age_hours": round(max_age.total_seconds() / 3600.0, 2),
        **detail,
    }


def _confidence_penalty(fresh: dict[str, Any]) -> float:
    status = fresh.get("status")
    if status == "GAP_DATA":
        return 0.0
    if status == "STALE_DATA":
        return 0.55
    return 1.0


# --------------------------------------------------------------------------- #
# Technical signals
# --------------------------------------------------------------------------- #


def _atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if df is None or len(df) < 2:
        return None
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    val = tr.rolling(period, min_periods=max(3, period // 2)).mean().iloc[-1]
    return _finite(val)


def evaluate_technicals(df: pd.DataFrame, preset: HorizonPreset) -> tuple[list[Signal], list[str]]:
    gaps: list[str] = []
    if df is None or df.empty:
        gaps.append(f"GAP_DATA: no OHLCV for timeframe {preset.timeframe}")
        dims = ["trend", "momentum", "volume", "volatility", "structure"]
        return [_unknown(d, gaps[0]) for d in dims], gaps

    last = df.iloc[-1]
    close = _finite(last.get("close"))
    if close is None:
        gaps.append("GAP_DATA: last close missing")
        return [_unknown(d, gaps[0]) for d in ("trend", "momentum", "volume", "volatility", "structure")], gaps

    sma5 = _finite(last.get("sma_5"))
    sma20 = _finite(last.get("sma_20"))
    sma50 = _finite(last.get("sma_50"))
    sma200 = _finite(last.get("sma_200"))
    rsi = _finite(last.get("rsi_14"))
    macd = _finite(last.get("macd"))
    macd_sig = _finite(last.get("macd_signal"))
    macd_hist = _finite(last.get("macd_hist"))
    vol = _finite(last.get("volume"))
    vol_sma = _finite(last.get("vol_sma_20"))
    atr = _atr(df)

    # --- trend ---
    above = 0
    below = 0
    ma_ev: list[dict[str, Any]] = []
    for name, val in (("sma_5", sma5), ("sma_20", sma20), ("sma_50", sma50), ("sma_200", sma200)):
        if val is None:
            continue
        side = "above" if close >= val else "below"
        if side == "above":
            above += 1
        else:
            below += 1
        ma_ev.append({"metric": name, "value": _round(val), "relation": side})
    stacked_up = (
        sma5 is not None
        and sma20 is not None
        and sma50 is not None
        and sma5 >= sma20 >= sma50
    )
    stacked_dn = (
        sma5 is not None
        and sma20 is not None
        and sma50 is not None
        and sma5 <= sma20 <= sma50
    )
    if above >= 3 and stacked_up:
        trend = _signal(
            "trend",
            "positive",
            0.82,
            evidence=ma_ev + [{"metric": "close", "value": _round(close)}],
            reasons=["price above most of MA ladder; short MAs stacked up"],
        )
    elif below >= 3 and stacked_dn:
        trend = _signal(
            "trend",
            "negative",
            0.82,
            evidence=ma_ev + [{"metric": "close", "value": _round(close)}],
            reasons=["price below most of MA ladder; short MAs stacked down"],
        )
    elif above > below:
        trend = _signal(
            "trend",
            "positive",
            0.55,
            evidence=ma_ev + [{"metric": "close", "value": _round(close)}],
            reasons=["price above more MAs than below"],
        )
    elif below > above:
        trend = _signal(
            "trend",
            "negative",
            0.55,
            evidence=ma_ev + [{"metric": "close", "value": _round(close)}],
            reasons=["price below more MAs than above"],
        )
    else:
        trend = _signal(
            "trend",
            "neutral",
            0.45 if ma_ev else 0.2,
            evidence=ma_ev + [{"metric": "close", "value": _round(close)}],
            reasons=["mixed MA ladder / sideways"] if ma_ev else ["GAP_DATA: MA incomplete"],
        )
        if not ma_ev:
            gaps.append("GAP_DATA: MA ladder incomplete")

    # --- momentum ---
    mom_ev: list[dict[str, Any]] = []
    if rsi is not None:
        mom_ev.append({"metric": "rsi_14", "value": _round(rsi)})
    if macd is not None:
        mom_ev.append({"metric": "macd", "value": _round(macd)})
    if macd_sig is not None:
        mom_ev.append({"metric": "macd_signal", "value": _round(macd_sig)})
    if macd_hist is not None:
        mom_ev.append({"metric": "macd_hist", "value": _round(macd_hist)})

    if rsi is None and macd is None:
        gaps.append("GAP_DATA: RSI/MACD missing")
        momentum = _unknown("momentum", gaps[-1])
    elif rsi is not None and rsi >= 70:
        momentum = _signal(
            "momentum",
            "negative" if (macd_hist is not None and macd_hist < 0) else "neutral",
            0.7,
            evidence=mom_ev,
            reasons=["RSI overbought (≥70); exhaustion risk"],
        )
    elif rsi is not None and rsi <= 30:
        momentum = _signal(
            "momentum",
            "positive" if (macd_hist is None or macd_hist >= 0) else "neutral",
            0.7,
            evidence=mom_ev,
            reasons=["RSI oversold (≤30); rebound bias if MACD not worsening"],
        )
    elif macd is not None and macd_sig is not None and macd > macd_sig and (macd_hist or 0) > 0:
        momentum = _signal(
            "momentum",
            "positive",
            0.65,
            evidence=mom_ev,
            reasons=["MACD above signal with positive histogram"],
        )
    elif macd is not None and macd_sig is not None and macd < macd_sig and (macd_hist or 0) < 0:
        momentum = _signal(
            "momentum",
            "negative",
            0.65,
            evidence=mom_ev,
            reasons=["MACD below signal with negative histogram"],
        )
    else:
        momentum = _signal(
            "momentum",
            "neutral",
            0.5,
            evidence=mom_ev,
            reasons=["momentum mixed / mid-range RSI"],
        )

    # --- volume ---
    if vol is None or vol_sma is None or vol_sma <= 0:
        gaps.append("GAP_DATA: volume or vol_sma_20 missing")
        volume = _unknown("volume", gaps[-1])
    else:
        ratio = vol / vol_sma
        vol_ev = [
            {"metric": "volume", "value": _round(vol, 0)},
            {"metric": "vol_sma_20", "value": _round(vol_sma, 0)},
            {"metric": "volume_ratio", "value": _round(ratio, 3)},
        ]
        bar_chg = None
        if len(df) >= 2:
            prev_c = _finite(df.iloc[-2].get("close"))
            if prev_c and prev_c != 0:
                bar_chg = (close - prev_c) / prev_c
                vol_ev.append({"metric": "bar_change_pct", "value": _round(bar_chg * 100, 3)})
        if ratio >= 1.4 and bar_chg is not None and bar_chg > 0:
            volume = _signal(
                "volume",
                "positive",
                0.7,
                evidence=vol_ev,
                reasons=["volume elevated with rising bar"],
            )
        elif ratio >= 1.4 and bar_chg is not None and bar_chg < 0:
            volume = _signal(
                "volume",
                "negative",
                0.7,
                evidence=vol_ev,
                reasons=["volume elevated with falling bar"],
            )
        elif ratio < 0.7:
            volume = _signal(
                "volume",
                "neutral",
                0.45,
                evidence=vol_ev,
                reasons=["volume light vs 20-bar average"],
            )
        else:
            volume = _signal(
                "volume",
                "neutral",
                0.5,
                evidence=vol_ev,
                reasons=["volume near average"],
            )

    # --- volatility ---
    if atr is None or close <= 0:
        gaps.append("GAP_DATA: ATR unavailable")
        volatility = _unknown("volatility", gaps[-1])
    else:
        atr_pct = atr / close * 100.0
        vol_ev2 = [
            {"metric": "atr_14", "value": _round(atr)},
            {"metric": "atr_pct", "value": _round(atr_pct, 3)},
            {"metric": "close", "value": _round(close)},
        ]
        # Horizon-sensitive soft bands on daily-ish bars; intraday naturally higher %.
        hi = 4.0 if preset.name == "intraday" else 3.0
        lo = 0.6 if preset.name == "intraday" else 0.8
        if atr_pct >= hi:
            volatility = _signal(
                "volatility",
                "negative",
                0.65,
                evidence=vol_ev2,
                reasons=[f"ATR elevated ({atr_pct:.2f}% of price) — wider risk"],
            )
        elif atr_pct <= lo:
            volatility = _signal(
                "volatility",
                "neutral",
                0.55,
                evidence=vol_ev2,
                reasons=[f"ATR compressed ({atr_pct:.2f}% of price)"],
            )
        else:
            volatility = _signal(
                "volatility",
                "neutral",
                0.55,
                evidence=vol_ev2,
                reasons=[f"ATR moderate ({atr_pct:.2f}% of price)"],
            )

    # --- structure (support / resistance from lookback window) ---
    highs = pd.to_numeric(df["high"], errors="coerce")
    lows = pd.to_numeric(df["low"], errors="coerce")
    resist = _finite(highs.max())
    support = _finite(lows.min())
    if resist is None or support is None or resist <= support:
        gaps.append("GAP_DATA: support/resistance window incomplete")
        structure = _unknown("structure", gaps[-1])
    else:
        span = resist - support
        pos = (close - support) / span if span else 0.5
        st_ev = [
            {"metric": "support", "value": _round(support)},
            {"metric": "resistance", "value": _round(resist)},
            {"metric": "close", "value": _round(close)},
            {"metric": "range_position", "value": _round(pos, 3)},
        ]
        if pos >= 0.85:
            structure = _signal(
                "structure",
                "negative",
                0.6,
                evidence=st_ev,
                reasons=["price near lookback resistance"],
            )
        elif pos <= 0.15:
            structure = _signal(
                "structure",
                "positive",
                0.6,
                evidence=st_ev,
                reasons=["price near lookback support"],
            )
        else:
            structure = _signal(
                "structure",
                "neutral",
                0.5,
                evidence=st_ev,
                reasons=["price mid-range of lookback window"],
            )

    return [trend, momentum, volume, volatility, structure], gaps


# --------------------------------------------------------------------------- #
# Fundamental signals
# --------------------------------------------------------------------------- #


def _rank_label(rank_pct: float | None, *, higher_is_better: bool) -> tuple[Label, str] | None:
    """Stockbit Rank is a percentile (0–100). For PE/PB, lower rank ≈ cheaper."""
    if rank_pct is None:
        return None
    if higher_is_better:
        if rank_pct >= 70:
            return "positive", f"peer rank {rank_pct:.0f}th (strong)"
        if rank_pct <= 30:
            return "negative", f"peer rank {rank_pct:.0f}th (weak)"
        return "neutral", f"peer rank {rank_pct:.0f}th (mid)"
    # cheaper / safer when rank low for valuation multiples
    if rank_pct <= 30:
        return "positive", f"peer rank {rank_pct:.0f}th (relatively cheap)"
    if rank_pct >= 70:
        return "negative", f"peer rank {rank_pct:.0f}th (relatively expensive)"
    return "neutral", f"peer rank {rank_pct:.0f}th (mid)"


def evaluate_fundamentals(
    df: pd.DataFrame,
    *,
    adapter: str,
) -> tuple[list[Signal], list[str]]:
    gaps: list[str] = []
    if df is None or df.empty:
        gaps.append("GAP_DATA: no fundamental_metrics for symbol")
        dims = [
            "valuation",
            "profitability",
            "growth",
            "solvency",
            "cash_flow",
            "dividend",
        ]
        if adapter == "bank":
            dims.append("bank_quality")
        return [_unknown(d, gaps[0]) for d in dims], gaps

    def ev(row: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not row:
            return []
        return [
            {
                "metric": row["metric"],
                "value_raw": row["value_raw"],
                "value_num": row["value_num"],
                "period": row.get("period"),
                "section": row.get("section"),
                "as_of": row.get("as_of"),
            }
        ]

    pe = _pick_metric(df, "pe_ttm")
    pb = _pick_metric(df, "pb")
    rank_pe = _pick_metric(df, "rank_pe")
    rank_pb = _pick_metric(df, "rank_pb")
    ihsg_pe = _pick_metric(df, "ihsg_pe")
    roe = _pick_metric(df, "roe")
    roa = _pick_metric(df, "roa")
    div_y = _pick_metric(df, "div_yield")
    rev_yoy = _pick_metric(df, "rev_yoy")
    ni_yoy = _pick_metric(df, "ni_yoy")
    der = _pick_metric(df, "der")
    cur = _pick_metric(df, "current_ratio")
    fcf = _pick_metric(df, "fcf_ttm")
    cfo = _pick_metric(df, "cfo_ttm")
    nim = _pick_metric(df, "nim")
    npl = _pick_metric(df, "npl_gross")
    npl_cov = _pick_metric(df, "npl_coverage")
    car = _pick_metric(df, "car")
    ldr = _pick_metric(df, "ldr")

    # --- valuation ---
    val_ev = ev(pe) + ev(pb) + ev(rank_pe) + ev(rank_pb) + ev(ihsg_pe)
    rank_hit = _rank_label(
        rank_pe["value_num"] if rank_pe else None, higher_is_better=False
    )
    if rank_hit:
        valuation = _signal(
            "valuation",
            rank_hit[0],
            0.75,
            evidence=val_ev,
            reasons=[rank_hit[1] + " on PE; sector/peer-aware via Stockbit Rank"],
        )
    elif pe and pe["value_num"] is not None and ihsg_pe and ihsg_pe["value_num"]:
        cheap = pe["value_num"] < ihsg_pe["value_num"]
        valuation = _signal(
            "valuation",
            "positive" if cheap else "negative",
            0.55,
            evidence=val_ev,
            reasons=[
                f"PE {pe['value_num']:.2f} vs IHSG median PE {ihsg_pe['value_num']:.2f} "
                "(absolute market compare; sector peer GAP_DATA)"
            ],
        )
    elif pe and pe["value_num"] is not None:
        valuation = _signal(
            "valuation",
            "neutral",
            0.35,
            evidence=val_ev,
            reasons=[
                "PE present but peer/sector benchmark GAP_DATA — absolute only, low confidence"
            ],
        )
    else:
        gaps.append("GAP_DATA: valuation multiples missing")
        valuation = _unknown("valuation", gaps[-1])

    # --- profitability ---
    if adapter == "bank" and (roe or nim):
        reasons = []
        label: Label = "neutral"
        conf = 0.5
        if roe and roe["value_num"] is not None:
            if roe["value_num"] >= 15:
                label, conf = "positive", 0.75
                reasons.append(f"ROE {roe['value_num']:.2f}% strong for bank")
            elif roe["value_num"] < 8:
                label, conf = "negative", 0.7
                reasons.append(f"ROE {roe['value_num']:.2f}% weak for bank")
            else:
                reasons.append(f"ROE {roe['value_num']:.2f}% mid for bank")
        if nim and nim["value_num"] is not None:
            if nim["value_num"] >= 4.5 and label != "negative":
                label = "positive" if label == "neutral" else label
                conf = max(conf, 0.7)
                reasons.append(f"NIM {nim['value_num']:.2f}% supportive")
            elif nim["value_num"] < 3.0:
                if label == "positive":
                    label = "neutral"
                else:
                    label = "negative"
                reasons.append(f"NIM {nim['value_num']:.2f}% soft")
        profitability = _signal(
            "profitability",
            label,
            conf,
            evidence=ev(roe) + ev(roa) + ev(nim),
            reasons=reasons or ["bank profitability mixed"],
        )
    elif roe and roe["value_num"] is not None:
        if roe["value_num"] >= 15:
            profitability = _signal(
                "profitability",
                "positive",
                0.7,
                evidence=ev(roe) + ev(roa),
                reasons=[f"ROE {roe['value_num']:.2f}%"],
            )
        elif roe["value_num"] < 8:
            profitability = _signal(
                "profitability",
                "negative",
                0.7,
                evidence=ev(roe) + ev(roa),
                reasons=[f"ROE {roe['value_num']:.2f}%"],
            )
        else:
            profitability = _signal(
                "profitability",
                "neutral",
                0.55,
                evidence=ev(roe) + ev(roa),
                reasons=[f"ROE {roe['value_num']:.2f}% mid"],
            )
    else:
        gaps.append("GAP_DATA: profitability metrics missing")
        profitability = _unknown("profitability", gaps[-1])

    # --- growth ---
    g_rows = [r for r in (rev_yoy, ni_yoy) if r and r.get("value_num") is not None]
    if not g_rows:
        gaps.append("GAP_DATA: growth YoY metrics missing")
        growth = _unknown("growth", gaps[-1])
    else:
        nums = [r["value_num"] for r in g_rows]
        avg = sum(nums) / len(nums)
        if avg >= 10:
            growth = _signal(
                "growth",
                "positive",
                0.7,
                evidence=ev(rev_yoy) + ev(ni_yoy),
                reasons=[f"avg YoY growth ~{avg:.2f}%"],
            )
        elif avg <= -5:
            growth = _signal(
                "growth",
                "negative",
                0.7,
                evidence=ev(rev_yoy) + ev(ni_yoy),
                reasons=[f"avg YoY growth ~{avg:.2f}%"],
            )
        else:
            growth = _signal(
                "growth",
                "neutral",
                0.55,
                evidence=ev(rev_yoy) + ev(ni_yoy),
                reasons=[f"avg YoY growth ~{avg:.2f}%"],
            )

    # --- solvency ---
    if adapter == "bank":
        reasons = []
        label = "neutral"
        conf = 0.5
        if car and car["value_num"] is not None:
            if car["value_num"] >= 17:
                label, conf = "positive", 0.75
                reasons.append(f"CAR {car['value_num']:.2f}% comfortable")
            elif car["value_num"] < 12:
                label, conf = "negative", 0.8
                reasons.append(f"CAR {car['value_num']:.2f}% thin")
            else:
                reasons.append(f"CAR {car['value_num']:.2f}% adequate")
        if ldr and ldr["value_num"] is not None:
            if 78 <= ldr["value_num"] <= 92:
                reasons.append(f"LDR {ldr['value_num']:.2f}% balanced")
            elif ldr["value_num"] > 100:
                label = "negative" if label != "positive" else "neutral"
                reasons.append(f"LDR {ldr['value_num']:.2f}% stretched")
            else:
                reasons.append(f"LDR {ldr['value_num']:.2f}%")
        if not reasons:
            gaps.append("GAP_DATA: bank solvency (CAR/LDR) missing")
            solvency = _unknown("solvency", gaps[-1])
        else:
            solvency = _signal(
                "solvency",
                label,
                conf,
                evidence=ev(car) + ev(ldr),
                reasons=reasons,
            )
    else:
        if der is None and cur is None:
            gaps.append("GAP_DATA: solvency ratios missing")
            solvency = _unknown("solvency", gaps[-1])
        else:
            reasons = []
            label = "neutral"
            conf = 0.5
            if der and der["value_num"] is not None:
                if der["value_num"] <= 1.0:
                    label, conf = "positive", 0.7
                    reasons.append(f"DER {der['value_num']:.2f} low")
                elif der["value_num"] >= 2.5:
                    label, conf = "negative", 0.75
                    reasons.append(f"DER {der['value_num']:.2f} high")
                else:
                    reasons.append(f"DER {der['value_num']:.2f}")
            if cur and cur["value_num"] is not None:
                if cur["value_num"] >= 1.5:
                    if label != "negative":
                        label = "positive"
                    conf = max(conf, 0.65)
                    reasons.append(f"current ratio {cur['value_num']:.2f}")
                elif cur["value_num"] < 1.0:
                    label = "negative"
                    conf = max(conf, 0.7)
                    reasons.append(f"current ratio {cur['value_num']:.2f} < 1")
            solvency = _signal(
                "solvency",
                label,
                conf,
                evidence=ev(der) + ev(cur),
                reasons=reasons or ["solvency mixed"],
            )

    # --- cash flow ---
    if adapter == "bank":
        # Banks: FCF less meaningful; mark as context-limited rather than inventing.
        cash_flow = _signal(
            "cash_flow",
            "unknown",
            0.2,
            evidence=ev(fcf) + ev(cfo),
            reasons=[
                "bank cash-flow lines are less diagnostic than CAR/NIM/NPL — treated as GAP-ish"
            ],
        )
    elif fcf is None and cfo is None:
        gaps.append("GAP_DATA: cash flow metrics missing")
        cash_flow = _unknown("cash_flow", gaps[-1])
    else:
        reasons = []
        label = "neutral"
        conf = 0.55
        if fcf and fcf["value_num"] is not None:
            if fcf["value_num"] > 0:
                label, conf = "positive", 0.7
                reasons.append("FCF TTM positive")
            else:
                label, conf = "negative", 0.7
                reasons.append("FCF TTM negative")
        if cfo and cfo["value_num"] is not None:
            reasons.append(
                "CFO TTM positive" if cfo["value_num"] > 0 else "CFO TTM negative"
            )
            if cfo["value_num"] <= 0 and label == "positive":
                label = "neutral"
        cash_flow = _signal(
            "cash_flow",
            label,
            conf,
            evidence=ev(fcf) + ev(cfo),
            reasons=reasons or ["cash flow mixed"],
        )

    # --- dividend ---
    if div_y and div_y["value_num"] is not None:
        y = div_y["value_num"]
        if y >= 4.0:
            dividend = _signal(
                "dividend",
                "positive",
                0.65,
                evidence=ev(div_y),
                reasons=[f"dividend yield {y:.2f}%"],
            )
        elif y <= 0.5:
            dividend = _signal(
                "dividend",
                "neutral",
                0.45,
                evidence=ev(div_y),
                reasons=[f"dividend yield {y:.2f}% (low / growth-style)"],
            )
        else:
            dividend = _signal(
                "dividend",
                "neutral",
                0.55,
                evidence=ev(div_y),
                reasons=[f"dividend yield {y:.2f}%"],
            )
    else:
        gaps.append("GAP_DATA: dividend yield missing")
        dividend = _unknown("dividend", gaps[-1])

    signals = [valuation, profitability, growth, solvency, cash_flow, dividend]

    # --- bank quality adapter ---
    if adapter == "bank":
        reasons = []
        label = "neutral"
        conf = 0.5
        bank_ev = ev(npl) + ev(npl_cov) + ev(nim) + ev(car)
        if npl and npl["value_num"] is not None:
            if npl["value_num"] <= 2.0:
                label, conf = "positive", 0.75
                reasons.append(f"NPL gross {npl['value_num']:.2f}% low")
            elif npl["value_num"] >= 4.0:
                label, conf = "negative", 0.8
                reasons.append(f"NPL gross {npl['value_num']:.2f}% elevated")
            else:
                reasons.append(f"NPL gross {npl['value_num']:.2f}%")
        if npl_cov and npl_cov["value_num"] is not None:
            if npl_cov["value_num"] >= 100:
                reasons.append(f"NPL coverage {npl_cov['value_num']:.1f}%")
            else:
                if label == "positive":
                    label = "neutral"
                reasons.append(f"NPL coverage {npl_cov['value_num']:.1f}% thin")
        if not reasons:
            gaps.append("GAP_DATA: bank quality (NPL/NIM) missing")
            signals.append(_unknown("bank_quality", gaps[-1]))
        else:
            signals.append(
                _signal(
                    "bank_quality",
                    label,
                    conf,
                    evidence=bank_ev,
                    reasons=reasons,
                )
            )

    return signals, gaps


# --------------------------------------------------------------------------- #
# Conflicts
# --------------------------------------------------------------------------- #


def detect_conflicts(
    technical: list[Signal],
    fundamental: list[Signal],
    preset: HorizonPreset,
) -> list[dict[str, Any]]:
    tech_map = {s.dimension: s for s in technical}
    fund_map = {s.dimension: s for s in fundamental}
    out: list[dict[str, Any]] = []

    t_trend = tech_map.get("trend")
    f_prof = fund_map.get("profitability")
    f_val = fund_map.get("valuation")
    f_growth = fund_map.get("growth")

    def opposing(a: Label, b: Label) -> bool:
        return {a, b} == {"positive", "negative"}

    if t_trend and f_prof and opposing(t_trend.label, f_prof.label):
        out.append(
            {
                "id": "trend_vs_profitability",
                "technical": t_trend.label,
                "fundamental": f_prof.label,
                "horizon_relevance": (
                    "technical_weight="
                    + preset.technical_weight
                    + "; fundamental_weight="
                    + preset.fundamental_weight
                ),
                "note": (
                    "Trend and profitability disagree — do not average; "
                    "state the trade-off for this horizon."
                ),
            }
        )
    if t_trend and f_val and opposing(t_trend.label, f_val.label):
        out.append(
            {
                "id": "trend_vs_valuation",
                "technical": t_trend.label,
                "fundamental": f_val.label,
                "horizon_relevance": (
                    "technical_weight="
                    + preset.technical_weight
                    + "; fundamental_weight="
                    + preset.fundamental_weight
                ),
                "note": "Price structure vs valuation disagree.",
            }
        )
    if t_trend and f_growth and opposing(t_trend.label, f_growth.label):
        out.append(
            {
                "id": "trend_vs_growth",
                "technical": t_trend.label,
                "fundamental": f_growth.label,
                "horizon_relevance": (
                    "technical_weight="
                    + preset.technical_weight
                    + "; fundamental_weight="
                    + preset.fundamental_weight
                ),
                "note": "Trend vs growth disagree.",
            }
        )
    return out


def _scale_signals(signals: list[Signal], factor: float) -> list[Signal]:
    if factor >= 0.999:
        return signals
    out: list[Signal] = []
    for s in signals:
        out.append(
            Signal(
                dimension=s.dimension,
                label=s.label,
                confidence=s.confidence * factor,
                evidence=s.evidence,
                reasons=list(s.reasons)
                + (
                    ["confidence reduced: STALE_DATA or weak freshness"]
                    if factor < 1.0 and s.label != "unknown"
                    else []
                ),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #


def analyze_stock_dict(
    symbol: str,
    horizon: str = "swing",
    *,
    ohlcv: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Structured Stock Analysis. Optional frames let tests inject fixtures."""
    sym = (symbol or "").strip().upper()
    preset = resolve_preset(horizon)
    now = _now_wib()

    if not sym:
        return {
            "symbol": "",
            "horizon": preset.name,
            "error": "GAP_DATA: symbol required",
            "technical_signals": [],
            "fundamental_signals": [],
            "conflicts": [],
            "gaps": ["GAP_DATA: symbol required"],
        }

    tf = preset.timeframe
    # Fallback chain for intraday if 15M missing
    df_ohlcv = ohlcv
    used_tf = tf
    if df_ohlcv is None:
        df_ohlcv = _load_ohlcv(sym, tf, preset.lookback)
        if df_ohlcv.empty and tf == "15M":
            for alt in ("5M", "1H", "1D"):
                alt_df = _load_ohlcv(sym, alt, preset.lookback)
                if not alt_df.empty:
                    df_ohlcv = alt_df
                    used_tf = alt
                    break

    df_fund = fundamentals if fundamentals is not None else _load_fundamentals(sym)
    sector = detect_sector(df_fund)

    last_bar_dt = None
    if df_ohlcv is not None and not df_ohlcv.empty:
        last_bar_dt = _parse_bar_time(df_ohlcv.iloc[-1])

    fund_as_of = None
    if df_fund is not None and not df_fund.empty:
        dates = [_parse_as_of(v) for v in df_fund["as_of"].dropna().unique()]
        dates = [d for d in dates if d is not None]
        if dates:
            fund_as_of = max(dates)

    tech_fresh = _freshness_block(
        kind="technical",
        when=last_bar_dt,
        max_age=preset.technical_max_age,
        detail={"timeframe": used_tf, "bars": int(len(df_ohlcv) if df_ohlcv is not None else 0)},
    )
    fund_fresh = _freshness_block(
        kind="fundamental",
        when=fund_as_of,
        max_age=preset.fundamental_max_age,
        detail={"rows": int(len(df_fund) if df_fund is not None else 0)},
    )

    tech_signals, tech_gaps = evaluate_technicals(df_ohlcv, preset)
    fund_signals, fund_gaps = evaluate_fundamentals(
        df_fund, adapter=str(sector.get("adapter") or "general")
    )

    tech_signals = _scale_signals(tech_signals, _confidence_penalty(tech_fresh))
    fund_signals = _scale_signals(fund_signals, _confidence_penalty(fund_fresh))

    # For intraday, fundamental is secondary: keep labels but note relevance.
    if preset.fundamental_weight == "secondary":
        fund_signals = [
            Signal(
                dimension=s.dimension,
                label=s.label,
                confidence=s.confidence * 0.85,
                evidence=s.evidence,
                reasons=list(s.reasons)
                + ["horizon=intraday: fundamental used as risk/quality filter only"],
            )
            for s in fund_signals
        ]

    conflicts = detect_conflicts(tech_signals, fund_signals, preset)
    gaps = tech_gaps + fund_gaps
    if used_tf != tf:
        gaps.append(f"GAP_DATA: preferred timeframe {tf} missing; used {used_tf}")

    last_close = None
    if df_ohlcv is not None and not df_ohlcv.empty:
        last_close = _round(_finite(df_ohlcv.iloc[-1].get("close")))

    return {
        "symbol": sym,
        "horizon": preset.name,
        "as_of_wib": now.isoformat(),
        "price": {
            "last": last_close,
            "timeframe": used_tf,
            "last_bar": last_bar_dt.isoformat() if last_bar_dt else None,
        },
        "freshness": {
            "technical": tech_fresh,
            "fundamental": fund_fresh,
        },
        "context": {
            **sector,
            "weights": {
                "technical": preset.technical_weight,
                "fundamental": preset.fundamental_weight,
            },
        },
        "technical_signals": [s.to_dict() for s in tech_signals],
        "fundamental_signals": [s.to_dict() for s in fund_signals],
        "conflicts": conflicts,
        "gaps": gaps,
        "notes": [
            "Dimension labels only — no buy/sell/hold verdict from this tool.",
            "Agent synthesizes prose; keep conflicts and GAP_DATA / STALE_DATA visible.",
            "Drill-down: get_technicals / get_fundamentals.",
        ],
    }


def analyze_stock(symbol: str, horizon: str = "swing") -> str:
    """Tool entry: JSON Stock Analysis for one Symbol and horizon preset."""
    payload = analyze_stock_dict(symbol, horizon)
    return json.dumps(payload, ensure_ascii=False, indent=2)
