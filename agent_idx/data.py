from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import duckdb

# Columns present in daily_stock_summary parquet files.
SCHEMA_COLUMNS: list[tuple[str, str]] = [
    ("symbol", "Symbol code, e.g. BBCA"),
    ("name", "Company name"),
    ("prev_close", "Previous close price"),
    ("open_price", "Open price"),
    ("high", "Day high"),
    ("low", "Day low"),
    ("close", "Close price"),
    ("change", "Price change"),
    ("volume", "Trade volume (shares)"),
    ("value", "Trade value (IDR)"),
    ("frequency", "Number of trades"),
    ("index_individual", "Individual index"),
    ("weight_for_index", "Index weight"),
    ("foreign_buy", "Foreign buy volume"),
    ("foreign_sell", "Foreign sell volume"),
    ("offer", "Best offer price"),
    ("offer_volume", "Offer volume"),
    ("bid", "Best bid price"),
    ("bid_volume", "Bid volume"),
    ("listed_shares", "Listed shares"),
    ("tradable_shares", "Tradable shares"),
    ("non_regular_volume", "Non-regular volume"),
    ("non_regular_value", "Non-regular value"),
    ("non_regular_frequency", "Non-regular frequency"),
    ("date", "Trading date as integer YYYYMMDD (e.g. 20220103)"),
    ("upload_file", "Source upload filename"),
]

_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|COPY|"
    r"EXPORT|IMPORT|PRAGMA|CALL|EXECUTE|INSTALL|LOAD|SET|RESET|VACUUM|"
    r"TRUNCATE|REPLACE|MERGE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def _to_yyyymmdd(value: str | int) -> int:
    text = str(value).strip().replace("-", "").replace("/", "")
    if not re.fullmatch(r"\d{8}", text):
        raise ValueError(f"Invalid date '{value}'. Use YYYYMMDD or YYYY-MM-DD.")
    return int(text)


def _market_source() -> str:
    """Which daily_stock_summary_*.parquet set to load.

    - ``idx`` / ``drive`` / ``bei`` — Drive Ringkasan Saham day files + tvkit gap
      (preferred: has foreign_buy/sell).
    - ``stockbit`` — Stockbit/tvkit rebuild (`*_stockbit_*`); falls back to others.
    - ``yahoo`` / ``yf`` — Yahoo bootstrap only.
    - ``all`` — every matching file (may duplicate symbol+date; prefer idx).
    """
    raw = (os.getenv("MARKET_SOURCE") or "idx").strip().lower()
    if raw in {"yahoo", "yf", "yfinance"}:
        return "yahoo"
    if raw in {"all", "both", "*"}:
        return "all"
    if raw in {"idx", "drive", "bei", "ringkasan"}:
        return "idx"
    if raw in {"stockbit", "chartbit", "tvkit"}:
        return "stockbit"
    return "idx"


def _is_drive_day_file(name: str) -> bool:
    n = name.lower()
    if "_stockbit_" in n or "_yf_" in n or "_tvkit_" in n:
        return False
    return bool(re.match(r"daily_stock_summary_\d{8}\.parquet$", n))


def _is_tvkit_gap_file(name: str) -> bool:
    return "_tvkit_" in name.lower()


class StockDataStore:
    """Read-only DuckDB access over daily_stock_summary parquet files."""

    def __init__(self, parquet_dir: Path, max_rows: int = 100) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.max_rows = max_rows
        self._con = duckdb.connect(database=":memory:")
        self._register_view()

    def _selected_files(self) -> list[Path]:
        if not self.parquet_dir.is_dir():
            return []
        all_files = sorted(self.parquet_dir.glob("daily_stock_summary_*.parquet"))
        source = _market_source()
        drive = [p for p in all_files if _is_drive_day_file(p.name)]
        tvkit_gap = [p for p in all_files if _is_tvkit_gap_file(p.name)]
        stockbit = [p for p in all_files if "_stockbit_" in p.name.lower()]
        yahoo = [
            p
            for p in all_files
            if "_stockbit_" not in p.name.lower()
            and not _is_drive_day_file(p.name)
            and not _is_tvkit_gap_file(p.name)
        ]
        if source == "idx":
            picked = drive + tvkit_gap
            if picked:
                return picked
            # Fallback chain before first Drive sync.
            return stockbit or yahoo or all_files
        if source == "stockbit":
            if stockbit:
                return stockbit
            if drive or tvkit_gap:
                return drive + tvkit_gap
            return yahoo or all_files
        if source == "yahoo":
            return yahoo or all_files
        return all_files

    def _register_view(self) -> None:
        if not self.parquet_dir.is_dir():
            raise FileNotFoundError(f"Parquet directory not found: {self.parquet_dir}")

        files = self._selected_files()
        if not files:
            raise FileNotFoundError(
                f"No daily_stock_summary_*.parquet files in {self.parquet_dir} "
                f"(MARKET_SOURCE={_market_source()})"
            )

        listed = ", ".join("'" + p.as_posix().replace("'", "''") + "'" for p in files)
        src = _market_source()
        # Normalize date; prefer Drive (Ringkasan) over tvkit gap on duplicate keys.
        self._con.execute(
            f"""
            CREATE OR REPLACE VIEW daily_stock AS
            SELECT * EXCLUDE (rn, _prio) FROM (
                SELECT
                    * EXCLUDE (date),
                    CAST(replace(CAST(date AS VARCHAR), '-', '') AS INTEGER) AS date,
                    CASE
                        WHEN lower(CAST(upload_file AS VARCHAR)) LIKE '%ringkasan%' THEN 0
                        WHEN lower(CAST(upload_file AS VARCHAR)) LIKE '%tvkit%' THEN 2
                        WHEN lower(CAST(upload_file AS VARCHAR)) LIKE '%stockbit%' THEN 1
                        ELSE 3
                    END AS _prio,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            UPPER(CAST(symbol AS VARCHAR)),
                            CAST(replace(CAST(date AS VARCHAR), '-', '') AS INTEGER)
                        ORDER BY
                            CASE
                                WHEN lower(CAST(upload_file AS VARCHAR)) LIKE '%ringkasan%' THEN 0
                                WHEN lower(CAST(upload_file AS VARCHAR)) LIKE '%stockbit%' THEN 1
                                WHEN lower(CAST(upload_file AS VARCHAR)) LIKE '%tvkit%' THEN 2
                                ELSE 3
                            END
                    ) AS rn
                FROM read_parquet([{listed}], union_by_name=true)
            ) WHERE rn = 1
            """
        )
        self._active_files = files
        if src == "idx" and any(_is_drive_day_file(p.name) for p in files):
            self._active_source = "idx_drive+tvkit_gap"
        elif src == "stockbit" and any("_stockbit_" in p.name.lower() for p in files):
            self._active_source = "stockbit"
        else:
            self._active_source = src


    def close(self) -> None:
        self._con.close()

    def describe_schema(self) -> str:
        files = getattr(self, "_active_files", [])
        src = getattr(self, "_active_source", _market_source())
        lines = [
            f"Table: daily_stock (MARKET_SOURCE={src})",
            f"files={[p.name for p in files]}",
            "date is INTEGER YYYYMMDD - filter with e.g. date BETWEEN 20220101 AND 20221231",
            "Preferred source: IDX Drive Ringkasan Saham + tvkit gap (MARKET_SOURCE=idx).",
            "Yahoo/yfinance ignored when idx/stockbit panels exist.",
            "",
            "Columns:",
        ]
        for name, desc in SCHEMA_COLUMNS:
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def list_date_range(self) -> str:
        row = self._con.execute(
            "SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(DISTINCT date) AS trading_days, "
            "COUNT(DISTINCT symbol) AS symbols "
            "FROM daily_stock"
        ).fetchone()
        assert row is not None
        src = getattr(self, "_active_source", _market_source())
        files = getattr(self, "_active_files", [])
        names = ",".join(p.name for p in files[:3])
        if len(files) > 3:
            names += f",+{len(files)-3}"
        return (
            f"source={src}, min_date={row[0]}, max_date={row[1]}, "
            f"trading_days={row[2]}, symbols={row[3]}, "
            f"files={names}, parquet_dir={self.parquet_dir}"
        )

    def get_stock_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        columns: str | None = None,
    ) -> str:
        symbol = symbol.strip().upper()
        start = _to_yyyymmdd(start_date)
        end = _to_yyyymmdd(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")

        allowed = {c for c, _ in SCHEMA_COLUMNS}
        if columns:
            cols = [c.strip() for c in columns.split(",") if c.strip()]
            bad = [c for c in cols if c not in allowed]
            if bad:
                raise ValueError(f"Unknown columns: {bad}")
            select_cols = ", ".join(cols)
        else:
            select_cols = "date, symbol, name, open_price, high, low, close, change, volume, value, frequency, foreign_buy, foreign_sell"

        sql = f"""
            SELECT {select_cols}
            FROM daily_stock
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
            LIMIT {self.max_rows}
        """
        return self._format_df(self._con.execute(sql, [symbol, start, end]).fetchdf())

    def summarize_period(
        self,
        start_date: str,
        end_date: str,
        symbol: str | None = None,
    ) -> str:
        start = _to_yyyymmdd(start_date)
        end = _to_yyyymmdd(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")

        if symbol:
            symbol = symbol.strip().upper()
            sql = """
                SELECT
                    symbol,
                    ANY_VALUE(name) AS name,
                    MIN(date) AS first_date,
                    MAX(date) AS last_date,
                    COUNT(*) AS trading_days,
                    MIN(low) AS period_low,
                    MAX(high) AS period_high,
                    ARG_MIN(close, date) AS first_close,
                    ARG_MAX(close, date) AS last_close,
                    SUM(volume) AS total_volume,
                    SUM(value) AS total_value,
                    AVG(volume) AS avg_volume,
                    AVG(value) AS avg_value,
                    SUM(foreign_buy) AS total_foreign_buy,
                    SUM(foreign_sell) AS total_foreign_sell
                FROM daily_stock
                WHERE symbol = ? AND date BETWEEN ? AND ?
                GROUP BY symbol
            """
            df = self._con.execute(sql, [symbol, start, end]).fetchdf()
        else:
            sql = """
                SELECT
                    COUNT(DISTINCT symbol) AS symbols,
                    COUNT(DISTINCT date) AS trading_days,
                    MIN(date) AS first_date,
                    MAX(date) AS last_date,
                    SUM(volume) AS total_volume,
                    SUM(value) AS total_value,
                    AVG(volume) AS avg_volume,
                    AVG(value) AS avg_value
                FROM daily_stock
                WHERE date BETWEEN ? AND ?
            """
            df = self._con.execute(sql, [start, end]).fetchdf()

        return self._format_df(df)

    def rank_foreign_flow(
        self,
        start_date: str,
        end_date: str,
        side: str = "sell",
        limit: int = 10,
    ) -> str:
        """Rank symbols by net foreign flow (buy - sell) over a period."""
        start = _to_yyyymmdd(start_date)
        end = _to_yyyymmdd(end_date)
        if start > end:
            raise ValueError("start_date must be <= end_date")

        side_norm = (side or "sell").strip().lower()
        if side_norm not in {"sell", "buy", "net"}:
            raise ValueError("side must be 'sell', 'buy', or 'net'")

        limit = max(1, min(int(limit), self.max_rows))
        order = "ASC" if side_norm == "sell" else "DESC"
        having = ""
        if side_norm == "sell":
            having = "HAVING net_foreign < 0"
        elif side_norm == "buy":
            having = "HAVING net_foreign > 0"

        coverage = self._con.execute(
            "SELECT MIN(date), MAX(date), COUNT(DISTINCT date) "
            "FROM daily_stock WHERE date BETWEEN ? AND ?",
            [start, end],
        ).fetchone()
        assert coverage is not None
        actual_min, actual_max, days = coverage
        if days == 0 or actual_min is None:
            return (
                f"(no rows) Tidak ada data transaksi untuk "
                f"{start}–{end}. Panggil list_date_range untuk cakupan tersedia."
            )

        sql = f"""
            SELECT
                symbol,
                ANY_VALUE(name) AS name,
                SUM(foreign_buy) AS total_foreign_buy,
                SUM(foreign_sell) AS total_foreign_sell,
                SUM(foreign_buy) - SUM(foreign_sell) AS net_foreign
            FROM daily_stock
            WHERE date BETWEEN ? AND ?
              AND foreign_buy IS NOT NULL
              AND foreign_sell IS NOT NULL
            GROUP BY symbol
            {having}
            ORDER BY net_foreign {order}
            LIMIT {limit}
        """
        header = (
            f"period_requested={start}-{end}, "
            f"period_actual={actual_min}-{actual_max}, "
            f"trading_days={days}, side={side_norm}, "
            f"net_foreign = foreign_buy - foreign_sell "
            f"(negatif = net sell)"
        )
        body = self._format_df(self._con.execute(sql, [start, end]).fetchdf())
        # Chartbit carries no foreign fields; detect whether any real coverage exists.
        covered = self._con.execute(
            """
            SELECT COUNT(*) FROM daily_stock
            WHERE date BETWEEN ? AND ?
              AND foreign_buy IS NOT NULL
              AND foreign_sell IS NOT NULL
            """,
            [start, end],
        ).fetchone()[0]
        if not covered:
            from agent_idx.transforms import foreign_flow_from_transforms

            alt = foreign_flow_from_transforms(
                start, end, side=side_norm, limit=limit
            )
            return (
                f"{header}\n"
                f"(no rows) Panel daily_stock tidak punya foreign flow nyata "
                f"(Chartbit OHLCV → foreign_buy/sell = NULL).\n\n"
                f"Fallback transform store:\n{alt}"
            )
        return f"{header}\n{body}"

    def run_sql(self, sql: str) -> str:
        cleaned = sql.strip().rstrip(";")
        if not cleaned:
            raise ValueError("Empty SQL")

        # Allow only a single statement.
        if ";" in cleaned:
            raise ValueError("Only a single SQL statement is allowed")

        if _FORBIDDEN_SQL.search(cleaned):
            raise ValueError("Only read-only SELECT / WITH queries are allowed")

        upper = cleaned.lstrip().upper()
        if not (upper.startswith("SELECT") or upper.startswith("WITH")):
            raise ValueError("Only SELECT / WITH queries are allowed")

        # Force a row cap if the model forgot LIMIT.
        wrapped = f"SELECT * FROM ({cleaned}) AS q LIMIT {self.max_rows}"
        df = self._con.execute(wrapped).fetchdf()
        return self._format_df(df)

    def _format_df(self, df: Any) -> str:
        if df is None or len(df) == 0:
            return "(no rows)"
        # Keep LLM context small.
        text = df.to_string(index=False)
        if len(text) > 12_000:
            text = text[:12_000] + "\n... (truncated)"
        return f"rows={len(df)}\n{text}"


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "describe_schema",
            "description": "Describe the daily_stock table columns and how dates are stored.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_date_range",
            "description": "Return min/max trading dates and number of trading days available in the parquet data.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_history",
            "description": (
                "Get daily OHLCV-style history for one symbol between dates. "
                "Dates: YYYYMMDD or YYYY-MM-DD. Result is capped by MAX_SQL_ROWS."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol, e.g. BBCA"},
                    "start_date": {"type": "string", "description": "Start date YYYYMMDD or YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "End date YYYYMMDD or YYYY-MM-DD"},
                    "columns": {
                        "type": "string",
                        "description": "Optional comma-separated columns to return",
                    },
                },
                "required": ["symbol", "start_date", "end_date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_period",
            "description": (
                "Aggregate stats for a date range. Optionally filter to one symbol. "
                "Useful for full-year questions like 'BBCA throughout 2022'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "symbol": {"type": "string", "description": "Optional symbol filter"},
                },
                "required": ["start_date", "end_date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_foreign_flow",
            "description": (
                "Rank symbols by net foreign flow over a period. "
                "net_foreign = foreign_buy - foreign_sell. "
                "PENTING: panel Stockbit Chartbit menyimpan foreign=NULL. "
                "Jika kosong/GAP, tool akan fallback ke foreign_flow_daily "
                "hasil transform scrape overview (F Buy/F Sell)."
                "side='sell' = largest net foreign sell (most negative net). "
                "side='buy' = largest net foreign buy. "
                "Use for questions like 'siapa net foreign sell 2026'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "side": {
                        "type": "string",
                        "enum": ["sell", "buy", "net"],
                        "description": "sell | buy | net (all, sorted by net desc)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Top N symbols (default 10)",
                    },
                },
                "required": ["start_date", "end_date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run a read-only SQL SELECT against the daily_stock view. "
                "date is INTEGER YYYYMMDD. Always filter by date/symbol when possible. "
                "Results are auto-limited."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SELECT or WITH query only",
                    }
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
]


def dispatch_tool(store: StockDataStore, name: str, arguments: dict[str, Any]) -> str:
    try:
        if name == "describe_schema":
            return store.describe_schema()
        if name == "list_date_range":
            return store.list_date_range()
        if name == "get_stock_history":
            return store.get_stock_history(
                symbol=arguments["symbol"],
                start_date=arguments["start_date"],
                end_date=arguments["end_date"],
                columns=arguments.get("columns"),
            )
        if name == "summarize_period":
            return store.summarize_period(
                start_date=arguments["start_date"],
                end_date=arguments["end_date"],
                symbol=arguments.get("symbol"),
            )
        if name == "rank_foreign_flow":
            return store.rank_foreign_flow(
                start_date=arguments["start_date"],
                end_date=arguments["end_date"],
                side=arguments.get("side", "sell"),
                limit=int(arguments.get("limit", 10)),
            )
        if name == "run_sql":
            return store.run_sql(arguments["sql"])
        return f"Unknown tool: {name}"
    except Exception as exc:  # noqa: BLE001 — surface tool errors to the model
        return f"ERROR: {exc}"
