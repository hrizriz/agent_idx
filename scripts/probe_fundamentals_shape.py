"""Scrape one ticker's fundamentals and report the shape of the parsed rows.

Feedback loop for designing the Postgres schema: shows how many rows each
section produces, how value_num is populated, and a sample of every distinct
row shape (scalar metric, financial statement cell, profile entry).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.browser_stockbit import (
    get_browser,
    run_sync,
    shutdown_browser,
    symbol_page_url,
)
from agent_idx.fund_farm import run_fund_farm
from agent_idx.transforms import connect, db_path, init_schema

PROFILE = ROOT / "data" / "stockbit_profile"


def diagnose(symbol: str, sections: list[str], headless: bool) -> None:
    """Print the auth signals before and after each section navigation."""

    def signals(browser) -> str:
        page = browser._page_or_raise()
        return (
            f"url={page.url[:70]} logged_in={browser.is_logged_in()} "
            f"needs_login={browser.needs_login()} needs_otp={browser.needs_otp()} "
            f"pwd_visible={browser._has_visible_password()} "
            f"otp_visible={browser._has_visible_otp_inputs()}"
        )

    def _run() -> None:
        browser = get_browser(headless=headless, user_data_dir=str(PROFILE))
        print("start:", browser.start())
        print("  after start   ", signals(browser))
        for name in sections:
            url = symbol_page_url(symbol, "profile" if name == "company" else name)
            result = browser.open(url=url)
            head = (result or "").splitlines()[0] if result else "(empty)"
            print(f"  open {name:11}", head[:60])
            print(f"  {'':16}", signals(browser))
            from agent_idx.browser_stockbit import (
                _OTP_STRONG_SELECTORS,
                _OTP_WEAK_SELECTORS,
                _visible_count,
            )

            probes = [*_OTP_STRONG_SELECTORS, *_OTP_WEAK_SELECTORS] + [
                'input[type="tel"]',
                'input[type="text"]',
                "input",
            ]
            for fi, frame in enumerate(browser._frames()):
                hits = {
                    sel: _visible_count(frame.locator(sel))
                    for sel in probes
                    if _visible_count(frame.locator(sel)) > 0
                }
                if hits:
                    print(f"  {'':16} frame[{fi}] playwright-visible: {hits}")

    run_sync(_run)


def report(symbol: str) -> None:
    con = connect()
    try:
        init_schema(con)
        total, with_num = con.execute(
            """
            SELECT COUNT(*), COUNT(value_num)
            FROM fundamental_metrics WHERE symbol = ?
            """,
            [symbol],
        ).fetchone()
        print(f"\n=== {symbol}: {total} rows, {with_num} with value_num ===")
        print(f"db={db_path()}\n")

        sections = con.execute(
            """
            SELECT section, COUNT(*) AS n, COUNT(value_num) AS n_num
            FROM fundamental_metrics WHERE symbol = ?
            GROUP BY section ORDER BY n DESC
            """,
            [symbol],
        ).fetchall()
        print(f"{'section':44} {'rows':>5} {'numeric':>8}")
        for sec, n, n_num in sections:
            print(f"{sec[:44]:44} {n:>5} {n_num:>8}")

        for sec, _, _ in sections:
            rows = con.execute(
                """
                SELECT metric, value_raw, value_num
                FROM fundamental_metrics
                WHERE symbol = ? AND section = ?
                ORDER BY metric LIMIT 4
                """,
                [symbol, sec],
            ).fetchall()
            print(f"\n--- {sec}")
            for metric, raw, num in rows:
                print(f"    {metric[:58]:58} | {str(raw)[:22]:22} | {num}")
    finally:
        con.close()


_FIN_DOM_JS = """
() => {
  const vis = el => {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const clean = s => (s || '').replace(/\\s+/g, ' ').trim();

  const tabs = [];
  const seen = new Set();
  for (const el of document.querySelectorAll(
        '[role=tab],button,[class*=tab i],[class*=Tab],li,a,span[class*=chip i]')) {
    if (!vis(el)) continue;
    const t = clean(el.innerText);
    if (!t || t.length > 34) continue;
    const key = t + '|' + el.tagName;
    if (seen.has(key)) continue;
    seen.add(key);
    tabs.push({text: t, tag: el.tagName, role: el.getAttribute('role') || '',
               cls: (el.className || '').toString().slice(0, 46)});
  }

  const tables = [];
  document.querySelectorAll('table').forEach((tb, i) => {
    const rows = Array.from(tb.querySelectorAll('tr')).slice(0, 3).map(tr =>
      Array.from(tr.querySelectorAll('th,td')).slice(0, 8).map(c => clean(c.innerText)));
    tables.push({idx: i, nrows: tb.querySelectorAll('tr').length,
                 ncols: (tb.querySelector('tr')?.children.length) || 0,
                 head: rows});
  });

  // Anything that looks like a period label (Q1 '26, 2026, Mar 26)
  const periodish = [];
  const re = /^(Q[1-4][ '\\-/]?\\d{2,4}|\\d{4}|[A-Z][a-z]{2}[ '\\-]?\\d{2,4}|TTM)$/;
  for (const el of document.querySelectorAll('div,span,th,td,p')) {
    if (el.children.length) continue;
    if (!vis(el)) continue;
    const t = clean(el.innerText);
    if (re.test(t)) periodish.push({text: t, tag: el.tagName,
                                    cls: (el.className || '').toString().slice(0, 46)});
    if (periodish.length > 24) break;
  }
  return {tabs, tables, periodish};
}
"""


def diagnose_financials(symbol: str, headless: bool) -> None:
    import json as _json

    def _run() -> None:
        browser = get_browser(headless=headless, user_data_dir=str(PROFILE))
        browser.start()
        print(browser.open(url=symbol_page_url(symbol, "financials")))
        page = browser._page_or_raise()
        page.wait_for_timeout(4000)
        browser._scroll_page_deep(4)
        data = page.evaluate(_FIN_DOM_JS)

        print(f"\n=== clickable candidates ({len(data['tabs'])}) ===")
        for t in data["tabs"]:
            print(f"  {t['tag']:6} role={t['role']:8} {t['text'][:34]:34} .{t['cls']}")

        print(f"\n=== tables ({len(data['tables'])}) ===")
        for tb in data["tables"]:
            print(f"  table[{tb['idx']}] rows={tb['nrows']} cols={tb['ncols']}")
            for row in tb["head"]:
                print(f"      {_json.dumps(row, ensure_ascii=False)[:150]}")

        print(f"\n=== period-looking leaf nodes ({len(data['periodish'])}) ===")
        for p in data["periodish"][:24]:
            print(f"  {p['tag']:5} {p['text'][:14]:14} .{p['cls']}")

        selects = page.evaluate(
            """() => [...document.querySelectorAll('select')].map((s, i) => ({
                 idx: i, id: s.id, name: s.name,
                 cls: (s.className || '').toString().slice(0, 40),
                 value: s.value,
                 options: [...s.options].map(o => o.label || o.text).slice(0, 12)
               }))"""
        )
        print(f"\n=== select elements ({len(selects)}) ===")
        for s in selects:
            print(f"  select[{s['idx']}] id={s['id']!r} name={s['name']!r} value={s['value']!r}")
            print(f"      options: {s['options']}")

        print("\n=== why the toggles don't click ===")
        for label in ("Income Statement", "Balance Sheet", "Cash Flow",
                      "Quarterly", "Annual", "Key Ratio"):
            bits = []
            for role in ("tab", "button", "link"):
                try:
                    bits.append(f"{role}={page.get_by_role(role, name=label, exact=True).count()}")
                except Exception as exc:  # noqa: BLE001
                    bits.append(f"{role}=err({type(exc).__name__})")
            try:
                loc = page.get_by_text(label, exact=True)
                n = loc.count()
                bits.append(f"text={n}")
                for i in range(min(n, 3)):
                    el = loc.nth(i)
                    info = el.evaluate(
                        "el => el.tagName + '.' + (el.className||'').toString().slice(0,26)"
                    )
                    try:
                        el.click(timeout=1500)
                        outcome = "CLICK_OK"
                    except Exception as exc:  # noqa: BLE001
                        outcome = f"CLICK_FAIL {type(exc).__name__}"
                    bits.append(f"[{i}]{info} vis={el.is_visible()} {outcome}")
            except Exception as exc:  # noqa: BLE001
                bits.append(f"text=err({type(exc).__name__})")
            print(f"  {label:18} " + " | ".join(bits))

    run_sync(_run)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="BBCA")
    ap.add_argument("--sections", default="financials,keystats,profile")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--diag-financials", action="store_true")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    sym = args.symbol.strip().upper()

    if args.diag_financials:
        try:
            diagnose_financials(sym, args.headless)
        finally:
            shutdown_browser()
        return 0

    if args.diagnose:
        secs = [s.strip() for s in args.sections.split(",") if s.strip()]
        try:
            diagnose(sym, secs, args.headless)
        finally:
            shutdown_browser()
        return 0

    if not args.report_only:
        secs = [s.strip() for s in args.sections.split(",") if s.strip()]
        print(f"scraping {sym} sections={secs} ...", flush=True)
        try:
            print(run_fund_farm([sym], secs, on_progress=lambda m: print("  ", m, flush=True)))
        finally:
            shutdown_browser()

    report(sym)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
