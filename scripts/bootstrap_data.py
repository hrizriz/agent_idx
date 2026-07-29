"""Bootstrap local daily_stock_summary parquet data from Yahoo Finance.

The production pipeline (Airflow) that produced the original parquet files
lives on another machine, so this script rebuilds an equivalent local dataset
for the agent: daily OHLCV for liquid IDX tickers, written in the exact
daily_stock_summary schema that StockDataStore expects.

Columns Yahoo Finance cannot provide use NULL. In particular, missing foreign
flow is never represented as measured zero.

Usage:
    py -3 scripts/bootstrap_data.py [--start 2022-01-01] [--end today]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.backtest_data import fetch_yfinance_symbol  # noqa: E402

DEFAULT_SYMBOLS_FILE = ROOT / "data" / "symbols_list.txt"

# Optional company-name lookup (Yahoo does not always return ID names).
# Unknown symbols fall back to the ticker code as name.
NAME_LOOKUP: dict[str, str] = {
    "BBCA": "Bank Central Asia Tbk.",
    "BBRI": "Bank Rakyat Indonesia (Persero) Tbk.",
    "BMRI": "Bank Mandiri (Persero) Tbk.",
    "BBNI": "Bank Negara Indonesia (Persero) Tbk.",
    "BRIS": "Bank Syariah Indonesia Tbk.",
    "ARTO": "Bank Jago Tbk.",
    "TLKM": "Telkom Indonesia (Persero) Tbk.",
    "ISAT": "Indosat Tbk.",
    "EXCL": "XL Axiata Tbk.",
    "TOWR": "Sarana Menara Nusantara Tbk.",
    "MTEL": "Dayamitra Telekomunikasi Tbk.",
    "ASII": "Astra International Tbk.",
    "UNTR": "United Tractors Tbk.",
    "UNVR": "Unilever Indonesia Tbk.",
    "ICBP": "Indofood CBP Sukses Makmur Tbk.",
    "INDF": "Indofood Sukses Makmur Tbk.",
    "MYOR": "Mayora Indah Tbk.",
    "CPIN": "Charoen Pokphand Indonesia Tbk.",
    "JPFA": "Japfa Comfeed Indonesia Tbk.",
    "KLBF": "Kalbe Farma Tbk.",
    "SIDO": "Industri Jamu dan Farmasi Sido Muncul Tbk.",
    "MIKA": "Mitra Keluarga Karyasehat Tbk.",
    "GOTO": "GoTo Gojek Tokopedia Tbk.",
    "BUKA": "Bukalapak.com Tbk.",
    "EMTK": "Elang Mahkota Teknologi Tbk.",
    "AMRT": "Sumber Alfaria Trijaya Tbk.",
    "MAPI": "Mitra Adiperkasa Tbk.",
    "ACES": "Aspirasi Hidup Indonesia Tbk.",
    "ADRO": "Alamtri Resources Indonesia Tbk.",
    "PTBA": "Bukit Asam Tbk.",
    "ITMG": "Indo Tambangraya Megah Tbk.",
    "HRUM": "Harum Energy Tbk.",
    "ANTM": "Aneka Tambang Tbk.",
    "INCO": "Vale Indonesia Tbk.",
    "MDKA": "Merdeka Copper Gold Tbk.",
    "TINS": "Timah Tbk.",
    "MEDC": "Medco Energi Internasional Tbk.",
    "PGAS": "Perusahaan Gas Negara Tbk.",
    "AKRA": "AKR Corporindo Tbk.",
    "ESSA": "ESSA Industries Indonesia Tbk.",
    "TPIA": "Chandra Asri Pacific Tbk.",
    "BRPT": "Barito Pacific Tbk.",
    "SMGR": "Semen Indonesia (Persero) Tbk.",
    "INTP": "Indocement Tunggal Prakarsa Tbk.",
    "JSMR": "Jasa Marga (Persero) Tbk.",
    "PGEO": "Pertamina Geothermal Energy Tbk.",
    "BREN": "Barito Renewables Energy Tbk.",
    "AMMN": "Amman Mineral Internasional Tbk.",
    "CTRA": "Ciputra Development Tbk.",
    "BSDE": "Bumi Serpong Damai Tbk.",
}


def load_symbols(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Symbols file not found: {path}\n"
            "Create data/symbols_list.txt (one ticker per line)."
        )
    symbols: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Allow "BBCA" or "BBCA,Bank Central Asia"
        sym = line.split(",", 1)[0].split("\t", 1)[0].strip().upper()
        sym = sym.replace(".JK", "").replace(".JK", "")
        if not sym or not sym.isalnum() or sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)
    if not symbols:
        raise ValueError(f"No tickers found in {path}")
    return symbols


# Full daily_stock_summary schema (see agent_idx/data.py SCHEMA_COLUMNS).
OUTPUT_COLUMNS = [
    "symbol", "name", "prev_close", "open_price", "high", "low", "close",
    "change", "volume", "value", "frequency", "index_individual",
    "weight_for_index", "foreign_buy", "foreign_sell", "offer",
    "offer_volume", "bid", "bid_volume", "listed_shares", "tradable_shares",
    "non_regular_volume", "non_regular_value", "non_regular_frequency",
    "date", "upload_file",
]


def to_schema(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["symbol"] = df["symbol"]
    out["name"] = name
    out["prev_close"] = df["prev_close"]
    out["open_price"] = df["open_price"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["close"] = df["close"]
    out["change"] = df["close"] - df["prev_close"]
    out["volume"] = df["volume"].astype("float64")
    out["value"] = df["value"]
    out["frequency"] = 0.0
    out["index_individual"] = None
    out["weight_for_index"] = None
    out["foreign_buy"] = float("nan")
    out["foreign_sell"] = float("nan")
    out["offer"] = None
    out["offer_volume"] = None
    out["bid"] = None
    out["bid_volume"] = None
    out["listed_shares"] = None
    out["tradable_shares"] = None
    out["non_regular_volume"] = 0.0
    out["non_regular_value"] = 0.0
    out["non_regular_frequency"] = 0.0
    out["date"] = df["date"].astype("int64")
    out["upload_file"] = "yfinance_bootstrap"
    return out[OUTPUT_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument(
        "--out-dir", default=str(ROOT / "data" / "parquet"), help="Output parquet directory"
    )
    parser.add_argument(
        "--symbols-file",
        default=str(DEFAULT_SYMBOLS_FILE),
        help="Path to symbols_list.txt (one ticker per line)",
    )
    args = parser.parse_args()

    start = int(args.start.replace("-", ""))
    end = int(args.end.replace("-", ""))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = load_symbols(Path(args.symbols_file))
    print(f"Loaded {len(symbols)} symbols from {args.symbols_file}")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for i, sym in enumerate(symbols, start=1):
        name = NAME_LOOKUP.get(sym, sym)
        try:
            raw = fetch_yfinance_symbol(sym, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(symbols)}] {sym}: FAILED ({exc})")
            failed.append(sym)
            continue
        if raw.empty:
            print(f"[{i}/{len(symbols)}] {sym}: no data")
            failed.append(sym)
            continue
        frames.append(to_schema(raw, name))
        print(f"[{i}/{len(symbols)}] {sym}: {len(raw)} rows")
        time.sleep(0.3)  # be polite to Yahoo

    if not frames:
        print("ERROR: no data fetched at all - check internet connection.")
        return 1

    panel = pd.concat(frames, ignore_index=True)
    panel.sort_values(["date", "symbol"], inplace=True)

    out_file = out_dir / f"daily_stock_summary_yf_{start}_{end}.parquet"
    con = duckdb.connect(":memory:")
    con.register("panel", panel)
    con.execute(
        f"COPY (SELECT * FROM panel) TO '{out_file.as_posix()}' (FORMAT PARQUET)"
    )
    con.close()

    print()
    print(f"Wrote {len(panel):,} rows for {panel['symbol'].nunique()} symbols")
    print(f"Dates: {panel['date'].min()} - {panel['date'].max()}")
    print(f"Output: {out_file}")
    if failed:
        print(f"Failed/skipped: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
