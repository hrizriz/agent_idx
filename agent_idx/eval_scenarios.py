"""200 QA scenarios for agent routing / context backtests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    id: str
    category: str
    question: str
    history: list[dict[str, Any]] = field(default_factory=list)
    # Deterministic expectations (None = don't assert)
    drop_history: bool | None = None
    needs_tools: bool | None = None
    intent_in: list[str] = field(default_factory=list)
    intent_not: list[str] = field(default_factory=list)
    # Soft answer checks used only with --llm
    answer_must: list[str] = field(default_factory=list)  # regex
    answer_must_not: list[str] = field(default_factory=list)  # regex
    notes: str = ""


DSSA_HISTORY: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": (
            "Analisis teknik Elliott Wave timeframe 1H untuk DSSA "
            "dari major bottom Mei 2026 Rp432 sampai close 24 Juli 2026 Rp870."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "DSSA menyelesaikan Wave (C) korektif. Target 830 lalu rebound ke 985. "
            "Proyeksi 13-20 candle 1H untuk target 1."
        ),
    },
]

BBCA_HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "Ringkas BBCA sepanjang 2024"},
    {
        "role": "assistant",
        "content": "BBCA menguat sepanjang 2024 dengan volume solid dan net foreign positif di beberapa bulan.",
    },
]


def _s(
    sid: str,
    cat: str,
    q: str,
    *,
    history: list[dict[str, Any]] | None = None,
    drop_history: bool | None = None,
    needs_tools: bool | None = None,
    intent_in: list[str] | None = None,
    intent_not: list[str] | None = None,
    answer_must: list[str] | None = None,
    answer_must_not: list[str] | None = None,
    notes: str = "",
) -> Scenario:
    return Scenario(
        id=sid,
        category=cat,
        question=q,
        history=list(history or []),
        drop_history=drop_history,
        needs_tools=needs_tools,
        intent_in=list(intent_in or []),
        intent_not=list(intent_not or []),
        answer_must=list(answer_must or []),
        answer_must_not=list(answer_must_not or []),
        notes=notes,
    )


def build_scenarios() -> list[Scenario]:
    out: list[Scenario] = []

    # ---- 1) Meta bot / cron / capability (20) ----
    meta_qs = [
        ("M01", "Kamu bisa buat cronjob gak?"),
        ("M02", "Bisa tidak membuat cron job harian?"),
        ("M03", "Apakah ada scheduler / jadwal otomatis?"),
        ("M04", "Cara atur cron market digest?"),
        ("M05", "Siapa kamu?"),
        ("M06", "Fitur apa saja yang kamu punya?"),
        ("M07", "Kamu bisa apa aja?"),
        ("M08", "Bagaimana cara pakai bot ini?"),
        ("M09", "Status bot sekarang gimana?"),
        ("M10", "Perlu /reset dulu gak?"),
        ("M11", "Jangan buat PDF kalau ga diminta"),
        ("M12", "Kamu sanggupkah bikin jadwal farming otomatis?"),
        ("M13", "Ada cron untuk berita nggak?"),
        ("M14", "Bisa set CRON_NEWS lewat chat?"),
        ("M15", "Who are you and what can you do?"),
        ("M16", "Clear history dong"),
        ("M17", "Apa bedanya /ask dan /council?"),
        ("M18", "Kamu bisa login Stockbit sendiri?"),
        ("M19", "Butuh restart bot gak biar cron jalan?"),
        ("M20", "Jelaskan kemampuan agent_idx singkat"),
    ]
    for sid, q in meta_qs:
        out.append(
            _s(
                sid,
                "meta_bot",
                q,
                history=DSSA_HISTORY,
                drop_history=True,
                needs_tools=False,
                intent_in=["bot_meta", "learning_meta", "general"],
                answer_must_not=[r"\bDSSA\b", r"Elliott", r"Rp870", r"Wave\s*\(C\)"],
                notes="Jangan nyambung ke analisis saham lama",
            )
        )

    # ---- 2) Topic switch after DSSA (20) ----
    switches = [
        ("T01", "Analisis BBCA YTD", ["market_analysis", "composite_report"], [r"\bDSSA\b"]),
        ("T02", "Berita terbaru BMRI", ["news", "market_analysis"], [r"\bDSSA\b"]),
        ("T03", "Foreign flow GOTO minggu ini", ["market_analysis"], [r"\bDSSA\b"]),
        ("T04", "Scrape Stockbit Reports 7 hari", ["stockbit_scrape"], [r"Elliott"]),
        ("T05", "Farming chart 1H semua emiten", ["general", "stockbit_chart", "market_analysis"], [r"\bDSSA\b"]),
        ("T06", "Buka chart BBRI di Stockbit", ["stockbit_chart"], [r"\bDSSA\b"]),
        ("T07", "Apa itu PE ratio?", ["knowledge"], [r"\bDSSA\b"]),
        ("T08", "Siapa bilang tentang MSCI di group?", ["chat_search"], [r"\bDSSA\b"]),
        ("T09", "Jalankan backtest scan winrate 75-85", ["backtest_scan"], [r"\bDSSA\b"]),
        ("T10", "Ringkas ADRO 2025", ["market_analysis", "composite_report"], [r"\bDSSA\b"]),
        ("T11", "Bandingkan BBCA vs BMRI return 2024", ["market_analysis"], [r"Elliott"]),
        ("T12", "Cari berita IHSG hari ini", ["news"], [r"\bDSSA\b"]),
        ("T13", "List folder data/charts", ["general", "bot_meta"], [r"Wave"]),
        ("T14", "Kamu sudah belajar apa saja?", ["learning_meta"], [r"\bDSSA\b"]),
        ("T15", "Export CSV history TLKM 2024", ["market_analysis", "general"], [r"\bDSSA\b"]),
        ("T16", "Login Stockbit dong", ["general", "bot_meta"], [r"Elliott"]),
        ("T17", "Proyek WTE proxy BNBR TPIA", ["general", "composite_report", "news", "market_analysis"], [r"\bDSSA\b"]),
        ("T18", "RSI ASII daily gimana?", ["market_analysis", "general"], [r"\bDSSA\b"]),
        ("T19", "Net foreign top 10 kemarin", ["market_analysis"], [r"\bDSSA\b"]),
        ("T20", "Buat PDF laporan BBCA", ["market_analysis", "composite_report", "general"], [r"\bDSSA\b"]),
    ]
    for sid, q, intents, forbid in switches:
        out.append(
            _s(
                sid,
                "topic_switch",
                q,
                history=DSSA_HISTORY,
                drop_history=True,
                intent_in=intents,
                answer_must_not=forbid,
            )
        )

    # ---- 3) Explicit continue should KEEP history (15) ----
    continues = [
        ("C01", "lanjut yang tadi"),
        ("C02", "Hitung proyeksi candle 1H dari analisa tadi"),
        ("C03", "Perjelas target wave C itu"),
        ("C04", "Revisi proyeksi DSSA barusan"),
        ("C05", "Berapa hari bursa untuk target tadi?"),
        ("C06", "Jelasin lagi invalidasinya"),
        ("C07", "Lanjutkan analisis sebelumnya"),
        ("C08", "Yang barusan, tambah risiko"),
        ("C09", "Soal itu, entry di mana?"),
        ("C10", "Masih tentang DSSA, hitung extension 161.8"),
        ("C11", "Dari perhitungan Elliott wave tadi, bantu hitung future proyeksi"),
        ("C12", "Update angka close-nya dong"),
        ("C13", "Itu time cycle-nya berapa candle?"),
        ("C14", "Di atas kamu bilang 985, dari mana?"),
        ("C15", "Sebelumnya target 830 — masih valid?"),
    ]
    for sid, q in continues:
        out.append(
            _s(
                sid,
                "continue",
                q,
                history=DSSA_HISTORY,
                drop_history=False,
                notes="Harus tetap pakai konteks DSSA",
            )
        )

    # ---- 4) Market / symbol analysis (30) ----
    symbols = [
        "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP",
        "INDF", "ADRO", "PTBA", "ITMG", "ANTM", "MDKA", "BRPT", "GOTO",
        "BUKA", "EMTK", "EXCL", "ISAT", "PGAS", "ELSA", "BNBR", "TPIA",
        "DSSA", "BRIS", "ARTO", "BBYB", "ACES", "MAPI",
    ]
    for i, t in enumerate(symbols, start=1):
        out.append(
            _s(
                f"K{i:02d}",
                "market",
                f"Ringkas pergerakan {t} YTD dan risiko utamanya",
                intent_in=["market_analysis", "composite_report"],
                needs_tools=True,
                answer_must_not=[r"NEED_STOCKBIT_CREDENTIALS"],
            )
        )

    # ---- 5) Stockbit / reports / chart (20) ----
    stockbit = [
        ("S01", "Scrape Stockbit Reports 10 hari", ["stockbit_scrape"], True),
        ("S02", "Ambil Stockbit Reports Januari sampai Juli 2026", ["stockbit_scrape"], True),
        ("S03", "Baca scrape Stockbit tanggal 2026-07-24", ["stockbit_read_date", "stockbit_scrape"], None),
        ("S04", "Buka chart DSSA di Stockbit", ["stockbit_chart"], True),
        ("S05", "Lihat chartbit BBCA", ["stockbit_chart"], True),
        ("S06", "Open https://stockbit.com/symbol/BBRI/chartbit", ["stockbit_chart"], True),
        ("S07", "Tarik data Stockbit untuk ADRO", ["general", "stockbit_scrape", "market_analysis"], True),
        ("S08", "Scrape overview keystats BMRI di Stockbit", ["general", "stockbit_scrape", "market_analysis"], True),
        ("S09", "Cari di Stockbit Reports tentang MSCI", ["general", "stockbit_scrape", "news", "market_analysis"], None),
        ("S10", "Status login Stockbit", ["bot_meta", "general"], False),
        ("S11", "Farming chart timeframe 1H semua emiten", ["general", "stockbit_chart", "market_analysis"], True),
        ("S12", "Farm stockbit charts 1D limit 20", ["general"], True),
        ("S13", "Scrape chart 1H BBCA BBRI BMRI", ["general", "stockbit_chart"], True),
        ("S14", "Jangan scrape dulu, cuma status", ["bot_meta", "general"], False),
        ("S15", "Pindah ke halaman financials TLKM Stockbit", ["stockbit_chart", "general", "market_analysis"], True),
        ("S16", "Unduh riwayat Stockbit Reports minggu lalu", ["stockbit_scrape"], True),
        ("S17", "List file scrape Stockbit yang ada", ["general"], True),
        ("S18", "Baca laporan scrape tanggal 15 Juli 2026", ["stockbit_read_date", "general"], None),
        ("S19", "Chart ASII 1W dari Stockbit", ["stockbit_chart", "general"], True),
        ("S20", "Apakah sesi Stockbit masih aktif?", ["bot_meta", "general"], False),
    ]
    for sid, q, intents, tools in stockbit:
        out.append(
            _s(
                sid,
                "stockbit",
                q,
                intent_in=intents,
                needs_tools=tools,
                answer_must_not=[r"data intraday tidak ada", r"tidak bisa scraping chart"],
            )
        )

    # ---- 6) PDF preference / output format (15) ----
    pdfs = [
        ("P01", "Jangan buat pdf kalo ga diminta!", True, False, [r"belum bisa mengakses file PDF"]),
        ("P02", "JANGAN BUAT DALAM PDF KETIKA GA DIMINTA", True, False, [r"Kirim ulang PDF"]),
        ("P03", "Buatkan PDF laporan ringkas BBCA 2024", False, True, []),
        ("P04", "Kirim PDF analisa BMRI", False, True, []),
        ("P05", "Generate PDF proyeksi DSSA", False, True, []),
        ("P06", "Jawaban teks saja, tanpa dokumen", True, False, [r"FILE:.*\.pdf"]),
        ("P07", "Ringkas BBCA tanpa PDF", False, True, [r"belum bisa mengakses file PDF"]),
        ("P08", "Tolong jangan export pdf", True, False, [r"belum bisa mengakses"]),
        ("P09", "Buat laporan BBCA di chat saja", False, True, []),
        ("P10", "Export CSV saja jangan PDF", False, None, [r"belum bisa mengakses file PDF"]),
        ("P11", "Baca dokumen PDF ini", False, None, []),  # attachment path separate
        ("P12", "Preferensi: output default teks Telegram", True, False, []),
        ("P13", "Stop bikin PDF otomatis", True, False, [r"Sedang memproses.*PDF"]),
        ("P14", "Oke lanjut analisa, tetap tanpa pdf", False, None, []),
        ("P15", "Minta PDF sekarang untuk GOTO", False, True, []),
    ]
    for sid, q, drop, tools, forbid in pdfs:
        out.append(
            _s(
                sid,
                "pdf_pref",
                q,
                history=DSSA_HISTORY if drop else [],
                drop_history=drop if drop else None,
                needs_tools=tools,
                answer_must_not=forbid,
            )
        )

    # ---- 7) News / knowledge / chat (20) ----
    misc = [
        ("N01", "Berita terbaru soal IHSG", ["news"], True),
        ("N02", "Headline bank besar hari ini", ["news"], True),
        ("N03", "Sentimen berita batu bara", ["news"], True),
        ("N04", "Apa itu ROE dan cara bacanya?", ["knowledge"], True),
        ("N05", "Jelaskan foreign flow di BEI", ["knowledge", "market_analysis"], True),
        ("N06", "Kurikulum belajar apa yang kamu punya?", ["learning_meta", "knowledge"], False),
        ("N07", "Siapa bilang BBCA undervalued di group?", ["chat_search"], True),
        ("N08", "Ringkas diskusi group soal GOTO", ["chat_search"], True),
        ("N09", "Cari chat tentang rights issue", ["chat_search"], True),
        ("N10", "Teori support resistance singkat", ["knowledge"], True),
        ("N11", "Rasio DER itu apa?", ["knowledge"], True),
        ("N12", "News MSCI Indonesia rebalancing", ["news"], True),
        ("N13", "Apa saja topik knowledge base?", ["learning_meta"], False),
        ("N14", "Belajar fundamental bank dong", ["knowledge", "learning_meta"], True),
        ("N15", "Ada digest harian nggak?", ["learning_meta", "bot_meta", "general"], False),
        ("N16", "Cari berita DSSA akuisisi", ["news"], True),
        ("N17", "Diskusi telegram soal IPO kemarin", ["chat_search"], True),
        ("N18", "Jelaskan MACD untuk pemula", ["knowledge"], True),
        ("N19", "Apa beda SMA dan EMA?", ["knowledge"], True),
        ("N20", "Berita energi terbarukan IDX", ["news"], True),
    ]
    for sid, q, intents, tools in misc:
        out.append(
            _s(
                sid,
                "news_knowledge_chat",
                q,
                intent_in=intents,
                needs_tools=tools,
            )
        )

    # ---- 8) Formatting / anti-ngaco style prompts (15) ----
    style = [
        ("F01", "Jelaskan proxy BNBR TPIA OASA untuk WTE tanpa tabel markdown"),
        ("F02", "List 5 emiten energi + catalyst, format bullet"),
        ("F03", "Jawab singkat: apa itu take-or-pay PPA?"),
        ("F04", "Ringkas dalam 5 bullet saja tentang IHSG"),
        ("F05", "Jangan pakai HTML tag di jawaban"),
        ("F06", "Bandingkan PGAS vs ELSA singkat"),
        ("F07", "Susun watchlist 5 ticker bank"),
        ("F08", "Tolong jawaban netral tanpa rekomendasi beli/jual agresif"),
        ("F09", "Sebut batasan data kalau foreign flow kosong"),
        ("F10", "Jangan menyuruh aku cari data sendiri"),
        ("F11", "Kalau gap data, bilang jujur"),
        ("F12", "Output bahasa Indonesia baku"),
        ("F13", "Jangan upsell atau self-puji"),
        ("F14", "Hanya pakai angka dari tools"),
        ("F15", "Kalau tidak tahu, bilang tidak tahu"),
    ]
    for sid, q in style:
        out.append(
            _s(
                sid,
                "style",
                q,
                history=BBCA_HISTORY,
                answer_must_not=[
                    r"<b>\s*</b>",
                    r"\| :---",
                    r"tolong carikan data",
                    r"silakan Anda cari",
                ],
            )
        )

    # ---- 9) Greetings / short / edge (10) ----
    edges = [
        ("E01", "halo", False, False),
        ("E02", "pagi", False, False),
        ("E03", "ok", None, False),
        ("E04", "thanks", None, False),
        ("E05", "?", None, False),
        ("E06", "BBCA", True, True),
        ("E07", "2024", True, True),
        ("E08", "foreign", True, True),
        ("E09", "help", None, False),
        ("E10", "/ask ringkas BBCA", True, True),
    ]
    for sid, q, tools, _marketish in edges:
        out.append(
            _s(
                sid,
                "edge",
                q,
                needs_tools=tools,
            )
        )

    # ---- 10) Cross-symbol confusion traps (20) ----
    traps = [
        ("X01", "Analisis BMRI", DSSA_HISTORY, True, [r"\bDSSA\b"]),
        ("X02", "Goto gimana?", DSSA_HISTORY, True, [r"Elliott Wave"]),
        ("X03", "TLKM valuation", DSSA_HISTORY, True, [r"Rp432"]),
        ("X04", "Bandingkan UNVR dan ICBP", DSSA_HISTORY, True, [r"\bDSSA\b"]),
        ("X05", "Berita EXCL", DSSA_HISTORY, True, [r"Wave \(C\)"]),
        ("X06", "ASII foreign flow", BBCA_HISTORY, True, [r"\bBBCA\b"]),
        ("X07", "ADRO vs PTBA", BBCA_HISTORY, True, []),
        ("X08", "Cron weekly jalan jam berapa?", DSSA_HISTORY, True, [r"\bDSSA\b"]),
        ("X09", "Kamu bisa farming chart otomatis tiap malam?", DSSA_HISTORY, True, [r"Rp870"]),
        ("X10", "Reset history", DSSA_HISTORY, True, [r"proyeksi"]),
        ("X11", "Buat cron scrape reports", DSSA_HISTORY, True, [r"\bDSSA\b"]),
        ("X12", "Apa status cron_enabled?", DSSA_HISTORY, True, [r"Elliott"]),
        ("X13", "Jangan nyambung ke DSSA; bahas BBRI", DSSA_HISTORY, True, []),
        ("X14", "Ganti topik: properti CTRA", DSSA_HISTORY, True, [r"\bDSSA\b"]),
        ("X15", "Stop, pertanyaan baru: apa itu NPL?", DSSA_HISTORY, True, [r"\bDSSA\b"]),
        ("X16", "Farming 1H offset 724", DSSA_HISTORY, True, [r"Wave"]),
        ("X17", "List project dir scripts", DSSA_HISTORY, True, [r"\bDSSA\b"]),
        ("X18", "Baca file data/symbols_list.txt baris awal", DSSA_HISTORY, True, [r"Elliott"]),
        ("X19", "Apakah bot sudah restart?", DSSA_HISTORY, True, [r"\bDSSA\b"]),
        ("X20", "Jangan analisis saham; jawab soal cron saja", DSSA_HISTORY, True, [r"\bDSSA\b"]),
    ]
    for sid, q, hist, drop, forbid in traps:
        out.append(
            _s(
                sid,
                "trap",
                q,
                history=hist,
                drop_history=drop,
                answer_must_not=forbid,
            )
        )

    # ---- Anti-hallucination routes (Aug 2026 chat failures) ----
    anti_halu = [
        (
            "H01",
            "Emang berapa free float dssa?",
            ["general"],
            ["market_analysis"],
            "fundamentals not parquet",
        ),
        (
            "H02",
            "Berapa value 5% dssa itu ya dengan last close price",
            ["general"],
            ["market_analysis"],
            "pct ownership value",
        ),
        (
            "H03",
            "Kategori HSC itu gimana? Dari msci ada minimal free float?",
            ["knowledge", "general"],
            ["market_analysis"],
            "HSC/MSCI theory",
        ),
        (
            "H04",
            "Dalam 1 kalimat: SBN yield hari ini, USDIDR, IHSG",
            ["general", "news"],
            ["market_analysis"],
            "macro GAP path",
        ),
        (
            "H05",
            "Cek kredibilitas data PHK Kemnaker 2022-2026",
            ["general", "news"],
            ["market_analysis"],
            "PHK external",
        ),
        (
            "H06",
            "Apakah $ketr grup Sinarmas?",
            ["general"],
            ["chat_search", "market_analysis"],
            "ownership not chat",
        ),
        (
            "H07",
            "Progress akuisisi ketr oleh Sinarmas terakhir gimana?",
            ["general"],
            ["market_analysis"],
            "akuisisi reports",
        ),
        (
            "H08",
            "Cek teknikal & fundamental GOTO dong harga wajarnya diberapa?",
            ["general"],
            ["knowledge", "market_analysis"],
            "analyze_stock path",
        ),
        (
            "H09",
            "Biasanya RAJA ini avg dihargai berapa pbv?",
            ["general"],
            ["market_analysis"],
            "PBV fundamentals",
        ),
        (
            "H10",
            "Bukannya data terakhir 29 juli yaa? Kok masih baca 24 juli?",
            ["general", "bot_meta", "market_analysis"],
            [],
            "baca=verb not ticker BACA",
        ),
        (
            "H11",
            "Analisis coba gubernur BI yang baru? Proyeksi masa depan",
            ["general", "news", "knowledge"],
            ["market_analysis"],
            "BI governor not COBA ticker",
        ),
        (
            "H12",
            "Dssa ini kena hsc dari bei kan?",
            ["general"],
            ["market_analysis"],
            "HSC fact per symbol",
        ),
    ]
    for sid, q, intent_in, intent_not, notes in anti_halu:
        out.append(
            _s(
                sid,
                "anti_halu",
                q,
                intent_in=intent_in,
                intent_not=intent_not,
                notes=notes,
            )
        )

    # Ensure we have at least 200; pad with generated market variants if short.
    n = 1
    while len(out) < 200:
        t = symbols[(n - 1) % len(symbols)]
        out.append(
            _s(
                f"Z{n:03d}",
                "pad_market",
                f"Summarize {t} volume and close last 30 trading days",
                intent_in=["market_analysis", "composite_report", "general"],
                needs_tools=True,
            )
        )
        n += 1

    # Stable order, unique ids
    seen: set[str] = set()
    unique: list[Scenario] = []
    for s in out:
        if s.id in seen:
            continue
        seen.add(s.id)
        unique.append(s)
    return unique[:200]


def scenarios_by_category(scenarios: list[Scenario] | None = None) -> dict[str, int]:
    scenarios = scenarios or build_scenarios()
    counts: dict[str, int] = {}
    for s in scenarios:
        counts[s.category] = counts.get(s.category, 0) + 1
    return counts
