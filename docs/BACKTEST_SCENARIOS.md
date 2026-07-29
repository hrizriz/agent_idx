# Backtest skenario trading — agent_idx

Modul: `agent_idx/backtest.py` + `agent_idx/backtest_data.py`

## Tujuan

Menguji setup trading sederhana pada data transaksi harian IDX (parquet) dari **2022-01-01** sampai data terakhir.

**Cap filter default:** market cap >= IDR 1 Triliun pada **hari sinyal (entry only)**. Panel menyimpan seluruh bar kronologis per emiten agar exit +N memakai N hari bursa, termasuk bar yang tidak eligible entry.

**Target win rate:** band **75–85%** (bukan hanya minimum). Skenario agregat di seluruh large cap jarang masuk band ini — hasil realistis ada di **per emiten + grid search**.

## Cara menjalankan

### CLI

```bash
py -3 main.py backtest
py -3 main.py backtest --min-win-rate 0.75 --max-win-rate 0.85 --min-trades 15
py -3 main.py backtest --cap-filter all --no-supplement
# legacy alias:
py -3 main.py backtest --cap-filter none
```

### Telegram bot

```
/ask cari setup trading win rate 75-85% market cap 1 triliun
/ask backtest skenario large cap sejak 2022
```

### Tool agent

`run_backtest_scan(start_date, min_win_rate, max_win_rate, min_trades, min_market_cap, supplement_external, cap_filter)`

(`universe` tetap diterima sebagai alias legacy di API Python.)

Output:

- `exports/backtest/scan_YYYYMMDD_HHMMSS.json`
- `exports/backtest/scan_YYYYMMDD_HHMMSS.md`
- Section **## SETUP TRADING** — semua hit dalam band win rate (aggregate, grid, per emiten), tanpa truncasi

## Metodologi

1. **Load panel** — DuckDB baca parquet, filter tanggal. Cap filter menandai `cap_eligible` (tidak membuang bar).
2. **Supplement yfinance** — emiten dengan coverage absolut < 85% dari jumlah trading-date panel diisi OHLCV dari Yahoo (`.JK`). Tidak ada batas diam 150 emiten. Foreign flow tidak tersedia di yfinance → biarkan **NaN** (bukan 0).
3. **16 skenario bawaan** + **grid search** agregat ekstra: drop, foreign streak, momentum, volume saja (breakout / MA-cross / oversold tetap di skenario bawaan, tidak di grid ekstra).
4. **Per emiten** — skenario bawaan + grid exhaustive (termasuk keluarga volume) dengan **min_trades sama** seperti agregat (default 15).
5. **Entry** — beli **close** hari sinyal, hanya jika `cap_eligible`.
6. **Exit** — jual **close** +N hari bursa pada deret lengkap.
7. **Win** — return forward > 0%.
8. **Filter lulus** — win rate dalam band [min, max] dan trades >= threshold.
9. **Foreign** — `net_foreign` hanya jika `foreign_buy` dan `foreign_sell` keduanya ada; sinyal foreign mensyaratkan `notna`.

## Skenario bawaan (16)

| ID | Deskripsi | Hold |
|---|---|---:|
| `foreign_2d_positive` | Net foreign positif 2 hari berturut | 5d |
| `foreign_3d_streak` | Net foreign positif 3 hari berturut | 5d |
| `foreign_reversal` | 3 hari net foreign total negatif, hari ini positif & > MA20 | 5d |
| `drop_3pct_bounce` | Change <= -3% | 3d |
| `drop_5pct_bounce` | Change <= -5% | 5d |
| `breakout_20d` | Close > high 20 hari sebelumnya | 10d |
| `volume_spike_up` | Volume > 2× MA20 & close > prev_close | 5d |
| `gap_down_open` | Open <= 98% prev_close | 5d |
| `net_foreign_top_day` | Net foreign >= quantile 95 rolling 20d | 5d |
| `rsi_oversold_proxy` | Return 5 hari <= -8% | 5d |
| `ma_cross_5_20` | MA5 cross above MA20 | 7d |
| `momentum_5d` | Return 5 hari >= +5% | 5d |
| `pullback_ma20` | Close <= MA20, change <= -2%, MA20 > MA50 | 5d |
| `foreign_intensity` | Net foreign > 2× abs(MA20) & close naik | 5d |
| `breakout_10d` | Close > high 10 hari sebelumnya | 7d |
| `volume_spike_3x` | Volume > 3× MA20 & close naik | 5d |

Grid tambahan (agregat & per emiten): `grid_drop_*`, `grid_foreign_*`, `grid_momentum_*`, `grid_vol_*`.

## Interpretasi (penting)

- **75–85% win rate agregat large cap** sangat jarang — normal untuk pasar efisien.
- Win rate tinggi **per emiten** dengan sample kecil sering **overfitting** — wajib validasi out-of-sample.
- Skenario agregat ~40–45% win rate lebih realistis untuk rule sederhana.
- **Bukan saran investasi.** Research tool only.

## Perluas skenario

Edit `SCENARIOS` di `backtest.py` atau tambah rule di `_signal_mask` / `grid_search`.
