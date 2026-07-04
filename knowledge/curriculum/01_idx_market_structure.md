# Struktur Pasar IDX (Bursa Efek Indonesia)

## Instrumen utama
- **Saham**: kepemilikan sebagian perusahaan terbuka (Tbk).
- **Indeks**: IHSG (semua), LQ45 (likuid), IDX30, sektor (IDXSECTOR).
- **Sesi**: reguler (pre-opening, continuous, pre-closing), plus sesi negosiasi.

## Pelaku pasar
- **Retail**: individu domestik.
- **Institusi domestik**: reksa dana, asuransi, dana pensiun.
- **Asing (foreign)**: tercermin di `foreign_buy` / `foreign_sell` data harian.
- **Market maker / broker**: menyediakan likuiditas.

## Data transaksi harian yang relevan
- OHLCV: open, high, low, close, volume.
- **Value**: nilai transaksi (Rp) — lebih penting dari volume lot untuk likuiditas.
- **Frequency**: jumlah transaksi — indikasi partisipasi.
- **Foreign buy/sell**: aliran asing (bukan ownership penuh, tapi flow harian).
- Net foreign = foreign_buy − foreign_sell (negatif = net sell).

## Batasan data transaksi
- Tidak berisi laporan keuangan (laba, ekuitas, utang).
- Tidak otomatis menjelaskan *mengapa* harga bergerak (butuh berita/fundamental).
- Corporate action (stock split, dividen, right issue) bisa membuat harga tampak “loncat”.

## Kerangka analisis IDX
1. **Fundamental**: kesehatan bisnis & valuasi.
2. **Teknikal/charting**: harga & volume.
3. **Flow**: asing, likuiditas, rotasi sektor.
4. **Makro**: BI-Rate, USDIDR, komoditas, risk-on/off global.
