"""Split the IDX universe into a priority tier and the tail.

Farming fundamentals for all ~980 symbols runs for the better part of a day, so
the farm works through the biggest names first and picks up the rest later.

Market cap is the ranking we want, but it only exists after keystats have been
scraped — a chicken-and-egg problem on a fresh install. Until coverage is high
enough the ranking falls back to median traded value, which tracks company size
loosely and, unlike market cap, is already in the daily parquet. The basis used
is recorded in the tier file so a surprising ordering can be explained.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TIER_FILE = ROOT / "data" / "universe_tiers.json"

DEFAULT_TOP_N = 200
# Sessions of traded value behind the fallback ranking.
TURNOVER_SESSIONS = 90
TURNOVER_MIN_SESSIONS = 10
# Below this share of the universe, scraped market caps are too sparse to rank
# with — a symbol missing from the table would be pushed down for no reason.
MARKET_CAP_MIN_COVERAGE = 0.9

TIER_TOP = "top"
TIER_REST = "rest"
TIER_ALL = "all"
_TIER_ALIASES = {
    "top": TIER_TOP,
    "top200": TIER_TOP,
    "tier1": TIER_TOP,
    "big": TIER_TOP,
    "large": TIER_TOP,
    "rest": TIER_REST,
    "sisa": TIER_REST,
    "tier2": TIER_REST,
    "lainlain": TIER_REST,
    "tail": TIER_REST,
    "all": TIER_ALL,
    "semua": TIER_ALL,
}


def normalize_tier(raw: str | None) -> str:
    key = (raw or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return _TIER_ALIASES.get(key, TIER_ALL)


def _parquet_dir() -> Path:
    raw = (os.getenv("PARQUET_DIR") or "").strip()
    return Path(raw) if raw else ROOT / "data" / "parquet"


def _parquet_files() -> list[Path]:
    """Mirror StockDataStore's source preference: Stockbit panel wins."""
    directory = _parquet_dir()
    if not directory.is_dir():
        return []
    files = sorted(directory.glob("daily_stock_summary_*.parquet"))
    stockbit = [p for p in files if "_stockbit_" in p.name.lower()]
    source = (os.getenv("MARKET_SOURCE") or "stockbit").strip().lower()
    if source in {"yahoo", "yf", "yfinance"}:
        return [p for p in files if "_stockbit_" not in p.name.lower()] or files
    if source in {"all", "both", "*"}:
        return files
    return stockbit or files


def turnover_scores() -> dict[str, float]:
    """Median daily traded value per symbol over the recent sessions."""
    files = _parquet_files()
    if not files:
        return {}
    import duckdb

    listed = ", ".join("'" + p.as_posix().replace("'", "''") + "'" for p in files)
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""
            WITH recent AS (
                SELECT
                    upper(symbol) AS symbol,
                    value,
                    ROW_NUMBER() OVER (
                        PARTITION BY upper(symbol)
                        ORDER BY CAST(replace(CAST(date AS VARCHAR), '-', '') AS BIGINT) DESC
                    ) AS rn
                FROM read_parquet([{listed}], union_by_name=true)
                WHERE value IS NOT NULL AND value > 0
            )
            SELECT symbol, MEDIAN(value)
            FROM recent
            WHERE rn <= {int(TURNOVER_SESSIONS)}
            GROUP BY symbol
            HAVING COUNT(*) >= {int(TURNOVER_MIN_SESSIONS)}
            """
        ).fetchall()
    finally:
        con.close()
    return {str(sym): float(val) for sym, val in rows if val is not None}


def market_cap_scores() -> dict[str, float]:
    """Market cap per symbol from already-scraped keystats, in rupiah."""
    from agent_idx import transforms

    if not transforms.db_path().is_file():
        return {}
    con = transforms.connect()
    try:
        rows = con.execute(
            """
            SELECT upper(symbol), MAX(value_num)
            FROM fundamental_metrics
            WHERE lower(metric) = 'market cap' AND value_num IS NOT NULL
              AND value_num > 0
            GROUP BY 1
            """
        ).fetchall()
    except Exception as exc:  # noqa: BLE001 - table may not exist yet
        logger.debug("market_cap_scores unavailable: %s", exc)
        return {}
    finally:
        con.close()
    return {str(sym): float(val) for sym, val in rows if val is not None}


def build_tiers(top_n: int = DEFAULT_TOP_N) -> dict[str, Any]:
    from agent_idx.chart_farm import load_symbols

    symbols = load_symbols()
    caps = market_cap_scores()
    turnover = turnover_scores()
    coverage = len(set(caps) & set(symbols)) / len(symbols) if symbols else 0.0

    if coverage >= MARKET_CAP_MIN_COVERAGE:
        basis = "market_cap"
        primary, secondary = caps, turnover
    else:
        basis = "turnover_proxy"
        primary, secondary = turnover, {}

    # Symbols the primary source knows nothing about sort last, but keep a
    # secondary key so the tail is still ordered by something meaningful.
    ranked = sorted(
        symbols,
        key=lambda s: (
            -(primary.get(s) or 0.0),
            -(secondary.get(s) or 0.0),
            s,
        ),
    )
    n = max(1, min(int(top_n or DEFAULT_TOP_N), len(ranked)))
    return {
        "generated_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "basis": basis,
        "basis_note": (
            "ranked by scraped Market Cap"
            if basis == "market_cap"
            else "market cap not scraped yet; ranked by median daily traded value"
        ),
        "market_cap_coverage": round(coverage, 4),
        "top_n": n,
        "universe": len(symbols),
        "top": ranked[:n],
        "rest": ranked[n:],
    }


def save_tiers(tiers: dict[str, Any]) -> Path:
    TIER_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIER_FILE.write_text(
        json.dumps(tiers, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return TIER_FILE


def load_tiers(
    top_n: int = DEFAULT_TOP_N, *, refresh: bool = False
) -> dict[str, Any]:
    """Cached tiers, rebuilt when missing, stale, or a different size."""
    if not refresh and TIER_FILE.is_file():
        try:
            cached = json.loads(TIER_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cached = None
        if isinstance(cached, dict) and cached.get("top") and cached.get("top_n") == top_n:
            return cached
    tiers = build_tiers(top_n)
    save_tiers(tiers)
    return tiers


def resolve_tier(
    tier: str | None,
    *,
    top_n: int = DEFAULT_TOP_N,
    refresh: bool = False,
) -> tuple[str, list[str]]:
    """Return (normalized tier, symbols) for 'top' / 'rest' / 'all'."""
    name = normalize_tier(tier)
    tiers = load_tiers(top_n, refresh=refresh)
    if name == TIER_TOP:
        return name, list(tiers.get("top") or [])
    if name == TIER_REST:
        return name, list(tiers.get("rest") or [])
    return name, list(tiers.get("top") or []) + list(tiers.get("rest") or [])


def describe_tiers(top_n: int = DEFAULT_TOP_N, *, refresh: bool = False) -> str:
    tiers = load_tiers(top_n, refresh=refresh)
    top = tiers.get("top") or []
    rest = tiers.get("rest") or []
    preview = ", ".join(top[:15]) + ("..." if len(top) > 15 else "")
    return "\n".join(
        (
            f"universe={tiers.get('universe')} basis={tiers.get('basis')}",
            f"({tiers.get('basis_note')}; "
            f"market cap coverage={tiers.get('market_cap_coverage')})",
            f"top={len(top)} rest={len(rest)}",
            f"top preview: {preview}",
            f"file={TIER_FILE}",
        )
    )
