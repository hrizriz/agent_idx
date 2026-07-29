"""Post-scrape / post-farm transforms: raw → structured queryable data.

Pipeline roles
--------------
scrape/farm  →  raw (parquet / markdown / RSS)
transform    →  technicals, fundamentals metrics, news sentiment, foreign snapshots
tools        →  get_technicals / get_fundamentals / search_news_sentiment / …

Domain map (product language)
-----------------------------
data pasar (OHLCV)     → technical (MA5/20/50/200, RSI, MACD, …)
data keystats/profile  → fundamental metrics
data news              → sentiment pos/neg/neutral
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = ROOT / "data" / "charts"
FUND_MD_DIR = ROOT / "exports" / "stockbit"
FUND_JSON_DIR = ROOT / "data" / "fundamentals"
TRANSFORM_DB = ROOT / "data" / "transforms.duckdb"

_LOCK = threading.RLock()

# --- News lexicon (ID + EN), lightweight rule-based ---
_POS = frozenset(
    """
    naik melonjak menguat rebound rally bullish positif untung laba surplus
    ekspansi akuisisi kontrak menang award upgrade outperform buy akumulasi
    pertumbuhan rekor tertinggi breakout recovery stimulus insentif
    gain surge rise soar jump boost profit growth strong positive upgrade
    """.split()
)
_NEG = frozenset(
    """
    turun anjlok melemah koreksi bearish negatif rugi defisit PHK layoff
    downgrade underperform sell distribusi gagal default sanksi fraud skandal
    terendah breakdown recession panic selloff cut loss
    drop fall plunge crash decline loss weak negative downgrade lawsuit
    """.split()
)


def db_path() -> Path:
    TRANSFORM_DB.parent.mkdir(parents=True, exist_ok=True)
    return TRANSFORM_DB


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path()))


def _migrate_fundamental_metrics(con: duckdb.DuckDBPyConnection) -> None:
    """Rebuild fundamental_metrics when it predates the `period` column.

    Rows here are derived from the markdown in exports/stockbit, so dropping
    them costs nothing: run_data_transform(scope='fundamentals') rebuilds the
    table. Keeping them would mean period-less rows sitting alongside the new
    period-keyed ones forever, since the primary key no longer matches.
    """
    try:
        cols = {r[0] for r in con.execute("DESCRIBE fundamental_metrics").fetchall()}
    except duckdb.CatalogException:
        return
    if "period" in cols:
        return
    n = con.execute("SELECT COUNT(1) FROM fundamental_metrics").fetchone()[0]
    logger.warning(
        "fundamental_metrics: dropping %s pre-period rows; re-run "
        "run_data_transform(scope='fundamentals') to rebuild from markdown",
        n,
    )
    con.execute("DROP TABLE fundamental_metrics")


def init_schema(con: duckdb.DuckDBPyConnection | None = None) -> None:
    own = con is None
    if own:
        con = connect()
    assert con is not None
    _migrate_fundamental_metrics(con)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id VARCHAR PRIMARY KEY,
            query VARCHAR,
            title VARCHAR,
            source VARCHAR,
            published_at VARCHAR,
            url VARCHAR,
            sentiment VARCHAR,
            score DOUBLE,
            created_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS fundamental_metrics (
            symbol VARCHAR,
            as_of VARCHAR,
            section VARCHAR,
            metric VARCHAR,
            period VARCHAR,
            value_raw VARCHAR,
            value_num DOUBLE,
            source_file VARCHAR,
            updated_at TIMESTAMP,
            PRIMARY KEY (symbol, section, metric, period)
        );
        CREATE TABLE IF NOT EXISTS foreign_flow_daily (
            symbol VARCHAR,
            as_of_date INTEGER,
            foreign_buy DOUBLE,
            foreign_sell DOUBLE,
            net_foreign DOUBLE,
            last_price DOUBLE,
            source VARCHAR,
            updated_at TIMESTAMP,
            PRIMARY KEY (symbol, as_of_date)
        );
        """
    )
    if own:
        con.close()


# --------------------------------------------------------------------------- #
# Technicals (pasar → indikator)
# --------------------------------------------------------------------------- #


def recompute_chart_indicators(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    limit: int = 0,
) -> str:
    """Re-run add_indicators on farmed parquet → rewrite + DuckDB sync."""
    from agent_idx.chart_farm import add_indicators, save_tf_parquet

    tfs_filter = {t.upper() for t in (timeframes or []) if t}
    syms = [s.upper() for s in (symbols or []) if s]
    paths: list[Path] = []
    if not CHARTS_DIR.is_dir():
        return f"ERROR: {CHARTS_DIR} tidak ada — farm chart dulu"

    for sym_dir in sorted(CHARTS_DIR.iterdir()):
        if not sym_dir.is_dir() or sym_dir.name.startswith("_"):
            continue
        sym = sym_dir.name.upper()
        if syms and sym not in syms:
            continue
        for pq in sorted(sym_dir.glob("*.parquet")):
            tf = pq.stem.upper()
            if tfs_filter and tf not in tfs_filter:
                continue
            paths.append(pq)

    if limit and limit > 0:
        paths = paths[: int(limit)]

    ok = 0
    fail = 0
    lines: list[str] = []
    for path in paths:
        try:
            con = duckdb.connect(":memory:")
            df = con.execute(
                f"SELECT * FROM read_parquet('{path.as_posix()}')"
            ).df()
            con.close()
            if df.empty:
                fail += 1
                continue
            # Drop old indicator cols then recompute
            drop_cols = [
                c
                for c in df.columns
                if c.startswith("sma_")
                or c.startswith("ema_")
                or c.startswith("macd")
                or c.startswith("rsi_")
                or c.startswith("vol_sma")
            ]
            base = df.drop(columns=drop_cols, errors="ignore")
            out = add_indicators(base)
            sym = str(out["symbol"].iloc[0]).upper()
            tf = str(out["timeframe"].iloc[0]).upper()
            save_tf_parquet(sym, tf, out)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            logger.exception("recompute failed %s", path)
            lines.append(f"{path.name}: {exc}")

    return (
        f"transform technicals: OK={ok} gagal={fail} files={len(paths)}\n"
        f"MA ladder=SMA5/20/50/200 + EMA12/26 + MACD + RSI14 + vol_sma_20\n"
        + ("\n".join(lines[:15]) if lines else "")
    )


def get_technicals(symbol: str, timeframe: str = "1D", *, n: int = 5) -> str:
    """Human-readable technical snapshot (pasar = teknikal)."""
    from agent_idx import chart_db

    sym = (symbol or "").strip().upper()
    tf = (timeframe or "1D").strip().upper()
    body = chart_db.query_ohlcv(sym, tf, n=max(1, min(int(n or 5), 50)))
    feats = chart_db.chart_features(sym, [tf], lookback=max(40, int(n or 5) * 8))
    return (
        f"TECHNICALS (data pasar → indikator) {sym}/{tf}\n"
        f"{body}\n\n{feats}"
    )


# --------------------------------------------------------------------------- #
# News sentiment
# --------------------------------------------------------------------------- #


def _parse_magnitude(token: str) -> float | None:
    """Parse Stockbit numeric tokens: 13.10, 2,193.77, 14,684 B, (16,348 B), 7.51%."""
    s = (token or "").strip()
    if not s or s in {"-", "—", "–"}:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace("%", "").strip().upper().replace("−", "-")
    m = re.match(
        r"^([+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)\s*([KMBT])?$",
        s,
    )
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(m.group(2) or "", 1.0)
    val = num * mult
    return -val if neg else val


def score_headline(title: str) -> tuple[str, float]:
    """Return (label, score) score in [-1,1]."""
    words = re.findall(r"[A-Za-zÀ-ÿ]+", (title or "").lower())
    pos = sum(1 for w in words if w in _POS)
    neg = sum(1 for w in words if w in _NEG)
    total = pos + neg
    if total == 0:
        return "neutral", 0.0
    score = (pos - neg) / total
    if score >= 0.25:
        return "positif", score
    if score <= -0.25:
        return "negatif", score
    return "neutral", score


def search_news_sentiment(query: str, limit: int = 8) -> str:
    """Fetch news then transform to sentiment buckets + persist."""
    from agent_idx.news import search_news_items

    items = search_news_items(query, limit=limit)
    if not items:
        return f"(no rows) Tidak ada berita untuk query: {query}"

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    pos_n = neg_n = neu_n = 0
    lines = [
        f"NEWS SENTIMENT query={query!r} n={len(items)}",
        "(transform rule-based ID/EN lexicon — bukan model ML)",
        "",
    ]
    rows: list[dict[str, Any]] = []
    for i, it in enumerate(items, 1):
        label, score = score_headline(it["title"])
        if label == "positif":
            pos_n += 1
        elif label == "negatif":
            neg_n += 1
        else:
            neu_n += 1
        rid = f"{hash((it['title'], it.get('url') or '')) & 0xFFFFFFFF:08x}"
        rows.append(
            {
                "id": rid,
                "query": query,
                "title": it["title"],
                "source": it.get("source") or "",
                "published_at": it.get("when") or "",
                "url": it.get("url") or "",
                "sentiment": label,
                "score": float(score),
                "created_at": now,
            }
        )
        lines.append(
            f"{i}. [{label} {score:+.2f}] {it['title']}\n"
            f"   sumber: {it.get('source') or '-'} | {it.get('when') or '-'}\n"
            f"   link: {it.get('url') or '-'}"
        )

    lines.insert(
        2,
        f"ringkas: positif={pos_n} negatif={neg_n} neutral={neu_n}",
    )

    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            df = pd.DataFrame(rows)
            con.register("news_df", df)
            con.execute(
                """
                INSERT OR REPLACE INTO news_sentiment
                SELECT * FROM news_df
                """
            )
        finally:
            con.close()

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Fundamentals (keystats markdown → metrics)
# --------------------------------------------------------------------------- #


_METRIC_LINE = re.compile(
    r"^[\-\*\u2022]?\s*(?:\[[^\]]+\]\s*)?"
    r"([A-Za-z(][A-Za-z0-9 %/().\-]{1,80}?)\s*[:=]\s*(.+)$"
)
_METRIC_LINE2 = re.compile(
    r"^([A-Za-z(][A-Za-z0-9 %/().\-]{1,60})\s{2,}([-+(]?\d[\d,.]*(?:\s*[KMBT%])?\)?)$"
)
_SECTION_HEADERS = {
    "current valuation",
    "per share",
    "solvency",
    "management effectiveness",
    "profitability",
    "growth",
    "dividend",
    "market rank",
    "income statement",
    "balance sheet",
    "cash flow statement",
    "price performance",
    "dividend history",
    "metrics (structured)",
    "profile fields",
    "profile structured",
    "company background",
    "company info",
    "shareholder > 1%",
    "shareholder composition",
    "holding composition",
    "shareholders",
    "directors and commissioners ownership",
    "number of shareholders",
    "subsidiary companies",
    "other tables",
    "ultimate beneficiary owner",
    "board of directors",
    "board of commissioners",
    "address",
    "company history",
}
_SERIES_HEADER = re.compile(r"^###\s+Series\s*[—\-]\s*(.+)$", re.I)
_STATEMENT_HEADER = re.compile(r"^###\s+Statement\s*[—\-]\s*(.+)$", re.I)
_RATIO_HEADER = re.compile(r"^###\s+Key Ratio\s*[—\-]\s*(.+)$", re.I)
# Financial statements expose ~74 periods; keep only the most recent columns.
_MAX_FIN_COLUMNS = 16
_MD_ROW = re.compile(r"^\|(.+)\|$")
# Column labels Stockbit uses for periods: "Q2 2026", "12M 2025", "2026", "TTM".
_PERIOD_CELL = re.compile(
    r"^(?:(?:Q[1-4]|\d{1,2}M)\s+)?(?:19|20)\d{2}$|^(?:TTM|LTM)$", re.I
)
_QUARTER_LABEL = re.compile(r"^Q[1-4]$", re.I)
_NUMERIC_CELL = re.compile(r"^[<>]?[-+(]?[\d.,]+\s*[%KMBT]?\)?$")


def _is_period_header(header: list[str]) -> bool:
    """True when a table's header row carries period labels rather than data."""
    return sum(1 for c in header[1:] if _PERIOD_CELL.match(c or "")) >= 2


def _looks_like_header(cells: list[str]) -> bool:
    """A header row names its columns; a data row is mostly figures.

    Several profile tables ship no header at all, so without this check their
    first shareholder gets consumed as the column names.
    """
    vals = [c for c in cells[1:] if c]
    if not vals:
        return False
    return sum(1 for c in vals if _NUMERIC_CELL.match(c)) * 2 <= len(vals)


def parse_keystats_markdown(text: str) -> list[dict[str, Any]]:
    """Parse scraped Stockbit keystats/profile markdown into metric rows.

    Handles:
    - `Label: value` (structured extractor output)
    - alternate-line label/value in raw text
    - markdown tables under `### Series — Net Income|EPS|Revenue`
    """
    rows: list[dict[str, Any]] = []
    section = "overview"
    series_name = ""
    table_header: list[str] = []
    headerless = False
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        # A heading can hold several tables in a row (a series table followed by
        # Dividend History, a statement followed by its ratio table). Each one
        # needs its own header, so drop the previous one as soon as the table
        # ends rather than only when the next heading arrives.
        if not line.startswith("|"):
            table_header = []
            headerless = False
        if not line:
            continue
        if line.startswith("## "):
            section = line[3:].strip().lower() or "overview"
            series_name = ""
            table_header = []
            continue
        sm = _SERIES_HEADER.match(line)
        if sm:
            series_name = sm.group(1).strip()
            section = f"series:{series_name.lower()}"
            table_header = []
            continue
        stm = _STATEMENT_HEADER.match(line)
        if stm:
            series_name = ""
            section = f"statement:{stm.group(1).strip().lower()}"
            table_header = []
            continue
        rm = _RATIO_HEADER.match(line)
        if rm:
            series_name = ""
            section = f"ratio:{rm.group(1).strip().lower()}"
            table_header = []
            continue
        if line.startswith("### "):
            title = line[4:].strip()
            low = title.lower()
            if low in _SECTION_HEADERS or low.startswith("raw") or low.startswith("series"):
                section = low
            else:
                section = low or section
            series_name = ""
            table_header = []
            continue
        if line.startswith("url:") or line.startswith("clicked=") or line.startswith("#"):
            continue
        if line.startswith("| ---") or re.match(r"^\|\s*:?---", line):
            continue

        # UBO / list items under profile boards
        if line.startswith("- ") and section in {
            "ultimate beneficiary owner",
            "board of directors",
            "board of commissioners",
        }:
            name = line[2:].strip()
            if name:
                rows.append(
                    {
                        "section": section,
                        "metric": name[:80],
                        "period": "",
                        "value_raw": "listed",
                        "value_num": None,
                    }
                )
            continue

        # Company background prose → one metric blob (first paragraph kept)
        if section == "company background" and len(line) > 40 and ":" not in line[:40]:
            rows.append(
                {
                    "section": section,
                    "metric": "background",
                    "period": "",
                    "value_raw": line[:500],
                    "value_num": None,
                }
            )
            continue

        # Markdown table rows for series / profile tables
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not table_header:
                table_header = cells
                if _is_period_header(cells) or _looks_like_header(cells):
                    joined = " ".join(cells)
                    if re.search(r"ex[- ]?date", joined, re.I) and re.search(
                        r"dividend", joined, re.I
                    ):
                        # Dividend History sits under the series headings but is
                        # a different table with a different subject.
                        section = "dividend history"
                        series_name = ""
                    continue
                headerless = True
            if not cells:
                continue
            row_label = cells[0]
            if headerless:
                # No column names to qualify the value with, so take the row's
                # last figure — the share/percentage these tables end on.
                vals = [c for c in cells[1:] if c and c not in {"-", "—", "–"}]
                if row_label and vals:
                    rows.append(
                        {
                            "section": section,
                            "metric": re.sub(r"\s+", " ", row_label)[:80],
                            "period": "",
                            "value_raw": vals[-1][:200],
                            "value_num": _parse_magnitude(vals[-1]),
                        }
                    )
                continue
            period_header = _is_period_header(table_header)
            is_financial = section.startswith("statement:") or section.startswith("ratio:")
            max_col = (
                min(len(table_header), _MAX_FIN_COLUMNS + 1)
                if is_financial
                else len(table_header)
            )
            # Profile tables: prefer Percentage / last numeric column as primary metric
            preferred_idx = None
            if not is_financial:
                for hi, h in enumerate(table_header):
                    if hi == 0:
                        continue
                    if re.search(r"percent|percentage|%", h, re.I):
                        preferred_idx = hi
                        break
                if preferred_idx is None and section in {
                    "shareholder > 1%",
                    "shareholder composition",
                    "shareholders",
                    "subsidiary companies",
                    "directors and commissioners ownership",
                }:
                    preferred_idx = len(table_header) - 1

            for col_i, val in enumerate(cells[1:], start=1):
                if col_i >= max_col:
                    break
                col_name = table_header[col_i]
                if not val or val in {"-", "—"}:
                    continue
                if is_financial and not row_label:
                    continue
                if preferred_idx is not None and col_i != preferred_idx:
                    # Still keep Total Shares / Shares columns when useful
                    if not re.search(r"share|amount|count|asset", col_name, re.I):
                        continue
                if period_header:
                    # Series tables put the quarter in the row and the year in
                    # the column, so the cell's period is the two combined.
                    if series_name and _QUARTER_LABEL.match(row_label):
                        metric = series_name
                        period = f"{row_label.upper()} {col_name}"
                    else:
                        metric = row_label
                        period = col_name
                else:
                    metric = (
                        f"{series_name} {row_label} {col_name}".strip()
                        if series_name
                        else f"{row_label} {col_name}".strip()
                    )
                    period = ""
                metric = re.sub(r"\s+", " ", metric)[:80]
                rows.append(
                    {
                        "section": section,
                        "metric": metric,
                        "period": re.sub(r"\s+", " ", period)[:20],
                        "value_raw": val[:200],
                        "value_num": _parse_magnitude(val),
                    }
                )
            continue

        m = _METRIC_LINE.match(line) or _METRIC_LINE2.match(line)
        if m:
            metric = re.sub(r"\s+", " ", m.group(1)).strip(" :-")
            value_raw = m.group(2).strip()
            rows.append(
                {
                    "section": section,
                    "metric": metric[:80],
                    "period": "",
                    "value_raw": value_raw[:200],
                    "value_num": _parse_magnitude(value_raw),
                }
            )
            continue

        # Alternate-line label / value (raw Stockbit text dump)
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            nxt = lines[j].strip()
            if (
                line.lower() not in _SECTION_HEADERS
                and not line.lower().startswith("raw ")
                and re.match(r"^[A-Za-z(].{1,80}$", line)
                and re.match(
                    r"^(?:[-—–]|\(?[0-9].*|\d{1,2}\s+[A-Za-z]{3}\s+\d{2})$",
                    nxt,
                )
                and not re.match(r"^(Period|Q[1-4]|Annualised|TTM|Div\b)", line, re.I)
                and not re.match(r"^20\d{2}$", line)
            ):
                rows.append(
                    {
                        "section": section,
                        "metric": re.sub(r"\s+", " ", line)[:80],
                        "period": "",
                        "value_raw": nxt[:200],
                        "value_num": _parse_magnitude(nxt),
                    }
                )
                i = j + 1
    return rows


def parse_foreign_from_text(text: str) -> dict[str, float]:
    """Extract F Buy / F Sell from Stockbit overview/orderbook text."""
    out: dict[str, float] = {}
    patterns = (
        (r"F\s*Buy[:\s]+([0-9]+(?:[.,][0-9]+)?\s*[KMBT]?)", "foreign_buy"),
        (r"F\s*Sell[:\s]+([0-9]+(?:[.,][0-9]+)?\s*[KMBT]?)", "foreign_sell"),
        (r"Foreign\s*Buy[:\s]+([0-9]+(?:[.,][0-9]+)?\s*[KMBT]?)", "foreign_buy"),
        (r"Foreign\s*Sell[:\s]+([0-9]+(?:[.,][0-9]+)?\s*[KMBT]?)", "foreign_sell"),
    )
    for pat, key in patterns:
        m = re.search(pat, text or "", re.I)
        if m:
            val = _parse_magnitude(m.group(1))
            if val is not None:
                out[key] = val
    if "foreign_buy" in out and "foreign_sell" in out:
        out["net_foreign"] = out["foreign_buy"] - out["foreign_sell"]
    return out


def ingest_fundamental_file(path: Path) -> int:
    """Parse one exports/stockbit markdown into DuckDB + JSON sidecar."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m_sym = re.search(r"Stockbit scrape\s*[—\-]\s*([A-Z]{3,5})", text)
    if not m_sym:
        m_sym = re.search(r"/symbol/([A-Z]{3,5})", text)
    if not m_sym:
        # filename SYMBOL_timestamp.md
        m_sym = re.match(r"^([A-Z]{3,5})_", path.name)
    if not m_sym:
        return 0
    sym = m_sym.group(1).upper()
    metrics = parse_keystats_markdown(text)
    foreign = parse_foreign_from_text(text)
    quote = {}
    m_last = re.search(
        r"\b([0-9]{2,6}(?:\.[0-9]+)?)\s*[+-]\s*[0-9]",
        text,
    )
    if m_last:
        try:
            quote["last"] = float(m_last.group(1))
        except ValueError:
            pass

    as_of = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [
        {
            "symbol": sym,
            "as_of": as_of,
            "section": r["section"],
            "metric": r["metric"],
            "period": r.get("period", ""),
            "value_raw": r["value_raw"],
            "value_num": r["value_num"],
            "source_file": str(path),
            "updated_at": now,
        }
        for r in metrics
    ]

    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            if rows:
                # A re-scrape supersedes the sections it covers. Without this,
                # rows from an earlier broken parse keep their old keys and sit
                # alongside the corrected ones indefinitely.
                sections = sorted({r["section"] for r in rows})
                con.execute(
                    "DELETE FROM fundamental_metrics "
                    "WHERE symbol = ? AND section IN "
                    f"({','.join('?' * len(sections))})",
                    [sym, *sections],
                )
                df = pd.DataFrame(rows)
                con.register("fund_df", df)
                con.execute(
                    """
                    INSERT OR REPLACE INTO fundamental_metrics
                    SELECT * FROM fund_df
                    """
                )
            if foreign:
                as_of_date = int(datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d"))
                con.execute(
                    """
                    INSERT OR REPLACE INTO foreign_flow_daily
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        sym,
                        as_of_date,
                        foreign.get("foreign_buy"),
                        foreign.get("foreign_sell"),
                        foreign.get("net_foreign"),
                        quote.get("last"),
                        "stockbit_overview_scrape",
                        now,
                    ],
                )
        finally:
            con.close()

    FUND_JSON_DIR.mkdir(parents=True, exist_ok=True)
    side = {
        "symbol": sym,
        "as_of": as_of,
        "metrics_n": len(metrics),
        "foreign": foreign,
        "quote": quote,
        "source_md": str(path),
        "domain": "fundamental",
    }
    (FUND_JSON_DIR / f"{sym}.json").write_text(
        json.dumps(side, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(metrics)


def transform_fundamentals(*, limit: int = 0) -> str:
    files = sorted(FUND_MD_DIR.glob("*.md")) if FUND_MD_DIR.is_dir() else []
    if limit and limit > 0:
        files = files[-int(limit) :]
    total_m = 0
    ok = 0
    for path in files:
        try:
            n = ingest_fundamental_file(path)
            if n:
                ok += 1
                total_m += n
        except Exception:  # noqa: BLE001
            logger.exception("fund transform %s", path)
    return (
        f"transform fundamentals: files_ok={ok}/{len(files)} metrics={total_m}\n"
        f"db={db_path()} table=fundamental_metrics + foreign_flow_daily"
    )


def get_fundamentals(symbol: str, *, limit: int = 40) -> str:
    sym = (symbol or "").strip().upper()
    limit = max(1, min(int(limit or 40), 200))
    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            df = con.execute(
                """
                SELECT section, metric, period, value_raw, value_num, as_of
                FROM fundamental_metrics
                WHERE symbol = ?
                ORDER BY section, metric, period DESC
                LIMIT ?
                """,
                [sym, limit],
            ).df()
            fx = con.execute(
                """
                SELECT as_of_date, foreign_buy, foreign_sell, net_foreign, last_price
                FROM foreign_flow_daily
                WHERE symbol = ?
                ORDER BY as_of_date DESC
                LIMIT 3
                """,
                [sym],
            ).df()
        finally:
            con.close()

    lines = [
        f"FUNDAMENTALS (keystats/profile → metrics) {sym}",
        f"db={db_path()}",
        "",
    ]
    if fx is not None and not fx.empty:
        lines.append("### Foreign flow snapshots (dari scrape overview)")
        for _, r in fx.iterrows():
            lines.append(
                f"- {int(r['as_of_date'])}: "
                f"FBuy={r.get('foreign_buy')} FSell={r.get('foreign_sell')} "
                f"net={r.get('net_foreign')} last={r.get('last_price')}"
            )
        lines.append("")
    if df is None or df.empty:
        lines.append(
            "(belum ada metrics — jalankan stockbit_scrape / "
            "farm_stockbit_fundamentals lalu run_data_transform scope=fundamentals)"
        )
        return "\n".join(lines)
    cur = None
    for _, r in df.iterrows():
        sec = r["section"]
        if sec != cur:
            lines.append(f"### {sec}")
            cur = sec
        label = f"{r['metric']} [{r['period']}]" if r["period"] else r["metric"]
        lines.append(f"- {label}: {r['value_raw']}")
    return "\n".join(lines)


def foreign_flow_from_transforms(
    start_date: int,
    end_date: int,
    *,
    side: str = "buy",
    limit: int = 10,
) -> str:
    """Rank Net foreign from transformed Scrape snapshots."""
    side_norm = (side or "buy").strip().lower()
    order = "ASC" if side_norm == "sell" else "DESC"
    having = ""
    if side_norm == "buy":
        having = "HAVING net_foreign > 0"
    elif side_norm == "sell":
        having = "HAVING net_foreign < 0"

    with _LOCK:
        con = connect()
        try:
            init_schema(con)
            n = con.execute(
                "SELECT COUNT(*) FROM foreign_flow_daily "
                "WHERE as_of_date BETWEEN ? AND ?",
                [start_date, end_date],
            ).fetchone()[0]
            if not n:
                return (
                    "GAP_DATA: belum ada foreign_flow_daily dari transform.\n"
                    "Chartbit OHLCV TIDAK punya foreign buy/sell "
                    "(panel daily_stock berisi NULL).\n"
                    "Cara isi: stockbit_scrape(symbol, sections='overview') "
                    "lalu run_data_transform(scope='fundamentals') "
                    "— parser ambil F Buy / F Sell dari halaman."
                )
            df = con.execute(
                f"""
                SELECT symbol,
                       SUM(foreign_buy) AS total_foreign_buy,
                       SUM(foreign_sell) AS total_foreign_sell,
                       SUM(net_foreign) AS net_foreign,
                       COUNT(*) AS snapshots
                FROM foreign_flow_daily
                WHERE as_of_date BETWEEN ? AND ?
                GROUP BY symbol
                {having}
                ORDER BY net_foreign {order}
                LIMIT {int(limit)}
                """,
                [start_date, end_date],
            ).df()
        finally:
            con.close()

    lines = [
        f"foreign_flow_daily period={start_date}-{end_date} side={side_norm} "
        f"source=stockbit_overview_transform",
        f"rows={len(df)}",
        "",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"- {r['symbol']}: net={r['net_foreign']} "
            f"(buy={r['total_foreign_buy']} sell={r['total_foreign_sell']} "
            f"n={int(r['snapshots'])})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def run_data_transform(
    scope: str = "all",
    *,
    symbols: str | None = None,
    limit: int = 0,
) -> str:
    """scrape/farm artifacts → transformed stores."""
    scope_n = (scope or "all").strip().lower()
    sym_list = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else None
    )
    parts: list[str] = [
        f"run_data_transform scope={scope_n}",
        "pipeline: scrape/farm → ingest raw → TRANSFORM → tools query",
        "",
    ]
    if scope_n in {"all", "technical", "technicals", "charts", "pasar"}:
        parts.append(
            recompute_chart_indicators(symbols=sym_list, limit=limit or 0)
        )
        parts.append("")
    if scope_n in {"all", "fundamental", "fundamentals", "keystats"}:
        parts.append(transform_fundamentals(limit=limit or 0))
        parts.append("")
    if scope_n in {"all", "news", "sentiment"}:
        # Transform latest market news sample into sentiment store
        parts.append(search_news_sentiment("IHSG OR IDX", limit=8))
        parts.append("")
    if scope_n in {"foreign", "flow"}:
        parts.append(
            foreign_flow_from_transforms(
                20260101,
                int(datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d")),
                side="net",
                limit=15,
            )
        )
    return "\n".join(parts).strip()
