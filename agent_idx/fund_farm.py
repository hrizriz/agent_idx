"""Polite Stockbit fundamentals farm (profile / keystats / financials / overview).

Design goals vs Chartbit farm:
- Much slower pacing + jitter (human-like, lower bot fingerprint).
- Sequential only (shared Playwright thread already max_workers=1).
- Stage confirmation for bulk (>BULK_THRESHOLD symbols).
- Stop immediately on login/OTP/captcha-ish blank pages.
- Persist markdown + light JSON sidecar — no aggressive parallel scrape.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from agent_idx.chart_farm import (
    BULK_THRESHOLD,
    MAX_SYMBOLS_PER_JOB,
    resolve_symbols,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FUND_DIR = ROOT / "data" / "fundamentals"
EXPORT_DIR = ROOT / "exports" / "stockbit"

DEFAULT_SECTIONS = ("overview", "keystats", "financials", "profile")
ALLOWED_SECTIONS = frozenset(
    {
        "overview",
        "keystats",
        "financials",
        "company",
        "profile",
        "bit",
        "chart",
        "chartbit",
    }
)

# Polite pacing — intentionally slower than chart farm.
SECTION_SLEEP = (1.8, 3.5)  # between Sections of the same Symbol
SYMBOL_SLEEP = (4.0, 9.0)  # between Symbols
ProgressFn = Callable[[str], None]

_PENDING_JOBS: dict[int, dict] = {}
WIB = ZoneInfo("Asia/Jakarta")


def normalize_sections(raw: str | list[str] | None) -> list[str]:
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.replace(";", ",").split(",") if p.strip()]
    elif raw:
        parts = [str(p).strip().lower() for p in raw if str(p).strip()]
    else:
        parts = list(DEFAULT_SECTIONS)
    alias = {"company": "profile", "chart": "chartbit"}
    out: list[str] = []
    for p in parts:
        p = alias.get(p, p)
        if p in ALLOWED_SECTIONS and p not in out:
            out.append(p)
    return out or list(DEFAULT_SECTIONS)


def _sleep_range(lo_hi: tuple[float, float]) -> None:
    lo, hi = lo_hi
    time.sleep(random.uniform(lo, hi))


def sidecar_freshness(
    symbol: str,
    sections: list[str] | None = None,
    *,
    now: datetime | None = None,
) -> dict:
    """Describe the latest successful fundamentals Scrape from its Sidecar."""
    sym = (symbol or "").strip().upper()
    requested = set(normalize_sections(sections))
    path = FUND_DIR / f"{sym}.json"
    out = {
        "symbol": sym,
        "fresh": False,
        "age_days": None,
        "scraped_at": None,
        "sections_complete": False,
        "reason": "GAP_DATA: sidecar missing",
    }
    if not path.is_file():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        out["reason"] = "GAP_DATA: sidecar unreadable"
        return out

    scraped_raw = str(data.get("scraped_at") or "").strip()
    stored_sections = set(normalize_sections(data.get("sections") or []))
    out["scraped_at"] = scraped_raw or None
    out["sections_complete"] = requested.issubset(stored_sections)
    if not out["sections_complete"]:
        missing = sorted(requested - stored_sections)
        out["reason"] = f"GAP_DATA: missing sections {','.join(missing)}"
        return out
    try:
        scraped = datetime.fromisoformat(scraped_raw.replace("Z", "+00:00"))
        if scraped.tzinfo is None:
            scraped = scraped.replace(tzinfo=WIB)
        scraped = scraped.astimezone(WIB)
    except ValueError:
        out["reason"] = "GAP_DATA: invalid scraped_at"
        return out

    current = (now or datetime.now(WIB)).astimezone(WIB)
    age_days = max(0, (current.date() - scraped.date()).days)
    out["age_days"] = age_days
    out["fresh"] = True
    out["reason"] = "ok"
    return out


def filter_fresh_symbols(
    symbols: list[str],
    sections: list[str] | None = None,
    *,
    fresh_days: int = 7,
    now: datetime | None = None,
) -> tuple[list[str], list[str]]:
    """Return (to_scrape, skipped_fresh), using calendar days in WIB."""
    days = max(0, int(fresh_days or 0))
    picked = [(s or "").strip().upper() for s in symbols if (s or "").strip()]
    if days <= 0:
        return picked, []
    stale: list[str] = []
    fresh: list[str] = []
    for sym in picked:
        status = sidecar_freshness(sym, sections, now=now)
        age = status.get("age_days")
        if status.get("sections_complete") and age is not None and age < days:
            fresh.append(sym)
        else:
            stale.append(sym)
    return stale, fresh


def _extract_quote_hint(text: str) -> dict:
    """Best-effort last/change parse from overview text (not guaranteed)."""
    out: dict = {}
    m = re.search(
        r"\b([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*"
        r"([+-]\s*[0-9]+(?:\.[0-9]+)?)\s*"
        r"\(\s*([+-]?\s*[0-9]+(?:\.[0-9]+)?)\s*%\s*\)",
        text or "",
    )
    if m:
        try:
            out["last"] = float(m.group(1).replace(",", ""))
            out["change"] = float(m.group(2).replace(" ", "").replace(",", ""))
            out["change_pct"] = float(m.group(3).replace(" ", "").replace(",", ""))
        except ValueError:
            pass
    return out


def scrape_one(
    symbol: str,
    sections: list[str],
    *,
    headless: bool | None = None,
) -> dict:
    """Scrape one Symbol via StockbitBrowser; save markdown + Sidecar."""
    from agent_idx.browser_stockbit import NEED_CREDENTIALS, NEED_OTP, run_sync
    from agent_idx.tools import _stockbit_browser, _stockbit_headless

    sym = (symbol or "").strip().upper()
    secs = normalize_sections(sections)
    hl = _stockbit_headless() if headless is None else headless

    def _run() -> str:
        browser = _stockbit_browser(hl)
        browser.start()
        # Extra settle between sections: monkey-patch via sequential opens
        # by calling scrape_symbol (already waits 2s/section).
        return browser.scrape_symbol(symbol=sym, sections=",".join(secs))

    result_text = run_sync(_run)
    payload: dict = {
        "symbol": sym,
        "sections": secs,
        "scraped_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "ok": False,
        "error": None,
        "file": None,
        "quote": {},
        "preview": (result_text or "")[:500],
    }
    if NEED_CREDENTIALS in (result_text or ""):
        payload["error"] = NEED_CREDENTIALS
        return payload
    if NEED_OTP in (result_text or ""):
        payload["error"] = NEED_OTP
        return payload
    if (result_text or "").startswith("ERROR:"):
        payload["error"] = result_text.split("\n", 1)[0]
        return payload

    m = re.search(r"^FILE:\s*(.+)\s*$", result_text or "", re.M)
    path = Path(m.group(1).strip()) if m else None
    if path and path.is_file():
        body = path.read_text(encoding="utf-8", errors="replace")
        payload["file"] = str(path)
        payload["quote"] = _extract_quote_hint(body)
        payload["ok"] = True
        FUND_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from agent_idx.transforms import ingest_fundamental_file

            n_metrics = ingest_fundamental_file(path)
            payload["metrics_ingested"] = n_metrics
        except Exception as exc:  # noqa: BLE001
            logger.warning("fundamental transform skip: %s", exc)
            n_metrics = 0
        side = FUND_DIR / f"{sym}.json"
        existing: dict = {}
        if side.is_file():
            try:
                existing = json.loads(side.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                existing = {}
        existing.update(
            {
                "symbol": sym,
                "sections": secs,
                "scraped_at": payload["scraped_at"],
                "markdown": str(path),
                "quote": payload["quote"],
                "chars": len(body),
                "metrics_ingested": n_metrics,
                "domain": "fundamental",
            }
        )
        side.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        payload["json"] = str(side)
    else:
        payload["error"] = "NO_FILE_WRITTEN"
    return payload


def run_fund_farm(
    symbols: list[str],
    sections: list[str] | None = None,
    *,
    headless: bool | None = None,
    on_progress: ProgressFn | None = None,
    skip_fresh_days: int = 7,
) -> str:
    from agent_idx.browser_stockbit import NEED_CREDENTIALS, NEED_OTP

    secs = normalize_sections(sections)
    symbols, skipped_fresh = filter_fresh_symbols(
        symbols, secs, fresh_days=skip_fresh_days
    )
    if not symbols:
        return (
            f"Fund farm: 0 perlu di-scrape; SKIP_FRESH={len(skipped_fresh)} "
            f"(fresh < {max(0, int(skip_fresh_days or 0))} hari, sections lengkap)."
        )

    def progress(msg: str) -> None:
        logger.info("[fund_farm] %s", msg)
        if on_progress:
            try:
                on_progress(msg)
            except Exception:  # noqa: BLE001
                pass

    ok: list[str] = []
    failed: list[str] = []
    lines: list[str] = []
    stopped = ""

    total = len(symbols)
    progress(
        f"Mulai fund farm {total} simbol x {secs} "
        f"(pace ~{SYMBOL_SLEEP[0]:.0f}-{SYMBOL_SLEEP[1]:.0f}s/emiten)"
    )

    for i, sym in enumerate(symbols, start=1):
        try:
            # Extra pause between sections is already in scrape_symbol;
            # we add symbol-level jitter here for bulk politeness.
            if i > 1:
                _sleep_range(SYMBOL_SLEEP)
            result = scrape_one(sym, secs, headless=headless)
        except Exception as exc:  # noqa: BLE001
            logger.exception("fund farm failed %s", sym)
            failed.append(sym)
            lines.append(f"{sym}: GAGAL ({exc})")
            continue

        err = result.get("error")
        if err in {NEED_CREDENTIALS, NEED_OTP}:
            stopped = (
                f"{err}\nBerhenti di {sym}: login/OTP Stockbit diperlukan.\n"
                "Kirim /stockbit, selesaikan login, lalu lanjut offset batch."
            )
            failed.append(sym)
            lines.append(f"{sym}: GAGAL login ({err})")
            break
        if not result.get("ok"):
            failed.append(sym)
            lines.append(f"{sym}: GAGAL ({err or 'unknown'})")
            # Back off harder after a soft failure (possible soft block).
            _sleep_range((6.0, 12.0))
            continue

        ok.append(sym)
        q = result.get("quote") or {}
        qhint = ""
        if q.get("last") is not None:
            qhint = f" last={q['last']}"
        lines.append(f"{sym}: OK{qhint} file={result.get('file')}")

        if i % 5 == 0 or i == total:
            progress(f"{i}/{total} — OK={len(ok)} gagal={len(failed)}")

    summary = [
        f"Fund farm selesai: OK={len(ok)} gagal={len(failed)} dari {total}",
        f"SKIP_FRESH={len(skipped_fresh)} fresh_days={max(0, int(skip_fresh_days or 0))}",
        f"sections={','.join(secs)}",
        f"markdown=exports/stockbit/  json=data/fundamentals/",
        "pace=polite (jitter antar emiten; stop on login/OTP)",
    ]
    if stopped:
        summary.append(stopped)
    summary.extend(lines[:50])
    if len(lines) > 50:
        summary.append(f"... (+{len(lines) - 50} simbol lagi)")
    return "\n".join(summary)


# Rough wall-clock per section for one Symbol, including page load, the
# statement/period toggles and the deep scrolls. Financials dominates because it
# walks three statements across three periods.
_SECTION_SECONDS = {
    "financials": 45.0,
    "keystats": 20.0,
    "profile": 15.0,
    "overview": 8.0,
    "chartbit": 12.0,
    "bit": 10.0,
}
_SYMBOL_OVERHEAD = sum(SYMBOL_SLEEP) / 2


def estimate_minutes(symbols: list[str], sections: list[str]) -> int:
    per_symbol = _SYMBOL_OVERHEAD + sum(
        _SECTION_SECONDS.get(s, 15.0) + sum(SECTION_SLEEP) / 2 for s in sections
    )
    return max(1, round(len(symbols) * per_symbol / 60))


def stage_fund_job(
    chat_key: int,
    symbols: list[str],
    sections: list[str],
    *,
    skip_fresh_days: int = 7,
) -> str:
    picked, skipped_fresh = filter_fresh_symbols(
        symbols, sections, fresh_days=skip_fresh_days
    )
    if not picked:
        return (
            f"SKIP_FRESH: semua {len(skipped_fresh)} simbol masih fresh "
            f"(< {max(0, int(skip_fresh_days or 0))} hari) dan sections lengkap. "
            "Tidak ada Staged action dibuat. Gunakan force=true untuk scrape ulang."
        )
    est_min = estimate_minutes(picked, sections)
    job = {
        "kind": "fundamentals",
        "symbols": picked[:MAX_SYMBOLS_PER_JOB],
        "sections": sections,
        "skip_fresh_days": max(0, int(skip_fresh_days or 0)),
        "skipped_fresh": skipped_fresh,
        "est_min": est_min,
    }
    _PENDING_JOBS[chat_key] = job
    preview = ", ".join(picked[:15]) + ("..." if len(picked) > 15 else "")
    return (
        "STAGED (menunggu konfirmasi user)\n"
        f"job=farm_stockbit_fundamentals simbol={len(picked)} "
        f"sections={','.join(sections)}\n"
        f"skip_fresh={len(skipped_fresh)} (< {max(0, int(skip_fresh_days or 0))} hari)\n"
        f"estimasi≈{_fmt_duration(est_min)} (pace pelan, anti-spam Stockbit)\n"
        f"daftar: {preview}\n"
        "CATATAN: scrape BELUM jalan. Setujui (ya/tidak) di Telegram."
    )


def _fmt_duration(minutes: int) -> str:
    if minutes < 90:
        return f"{minutes} menit"
    return f"{minutes / 60:.1f} jam"


def has_pending_job(chat_key: int) -> bool:
    return chat_key in _PENDING_JOBS


def peek_pending_job(chat_key: int) -> dict | None:
    return _PENDING_JOBS.get(chat_key)


def describe_pending_job(chat_key: int) -> str:
    job = _PENDING_JOBS.get(chat_key)
    if not job:
        return "(tidak ada job tertunda)"
    symbols = job["symbols"]
    preview = ", ".join(symbols[:20]) + ("..." if len(symbols) > 20 else "")
    return (
        "Job farm fundamentals Stockbit menunggu konfirmasi:\n"
        f"• simbol: {len(symbols)} ({preview})\n"
        f"• sections: {', '.join(job['sections'])}\n"
        f"• skip fresh: {len(job.get('skipped_fresh') or [])} "
        f"(< {job.get('skip_fresh_days', 0)} hari)\n"
        f"• estimasi: ~{_fmt_duration(job['est_min'])} (pace pelan)\n"
        f"• output: exports/stockbit/*.md + data/fundamentals/{{SYM}}.json"
    )


def clear_pending_job(chat_key: int) -> bool:
    return _PENDING_JOBS.pop(chat_key, None) is not None


def run_pending_job(chat_key: int, *, on_progress: ProgressFn | None = None) -> str:
    job = _PENDING_JOBS.pop(chat_key, None)
    if not job:
        return "Tidak ada job fundamentals untuk dijalankan."
    return run_fund_farm(
        job["symbols"],
        job["sections"],
        on_progress=on_progress,
        skip_fresh_days=job.get("skip_fresh_days", 7),
    )
