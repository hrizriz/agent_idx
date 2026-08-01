"""Probe Stockbit symbol page for accessible feature tabs/links."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.browser_stockbit import get_browser, run_sync, shutdown_browser

PROFILE = ROOT / "data" / "stockbit_profile"
OUT = ROOT / "exports" / "stockbit" / "_probe_features.json"


def main() -> int:
    def _run():
        browser = get_browser(headless=True, user_data_dir=str(PROFILE))
        print(browser.start())
        print("status:", browser.status().replace("\n", " | "))
        page = browser._page_or_raise()
        page.goto("https://stockbit.com/symbol/BBCA", wait_until="domcontentloaded")
        page.wait_for_timeout(4500)
        cur = page.url
        print("url:", cur)

        hrefs = page.eval_on_selector_all(
            'a[href*="/symbol/"]',
            """els => [...new Set(
                els.map(e => e.getAttribute('href')).filter(Boolean)
            )].sort()""",
        )

        labeled = page.eval_on_selector_all(
            "a[href]",
            """els => {
              const out = [];
              for (const e of els) {
                const h = e.getAttribute('href') || '';
                if (!(h.includes('stockbit.com') || h.startsWith('/'))) continue;
                const t = (e.innerText || '').trim().replace(/\\s+/g, ' ');
                if (t && t.length < 48) out.push({text: t, href: h});
              }
              // unique by href+text
              const seen = new Set();
              const uniq = [];
              for (const x of out) {
                const k = x.text + '|' + x.href;
                if (seen.has(k)) continue;
                seen.add(k);
                uniq.push(x);
              }
              return uniq.slice(0, 120);
            }""",
        )

        # Probe known candidate slugs
        candidates = [
            "overview",
            "chartbit",
            "keystats",
            "financials",
            "company",
            "bit",
            "orderbook",
            "order-book",
            "broker",
            "broker-summary",
            "bandarmology",
            "corp-action",
            "corporate-action",
            "holder",
            "holders",
            "news",
            "stream",
            "profile",
            "ratio",
            "dividend",
            "seasonality",
        ]
        probe = []
        for slug in candidates:
            url = (
                f"https://stockbit.com/symbol/BBCA/{slug}"
                if slug != "overview"
                else "https://stockbit.com/symbol/BBCA"
            )
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(1500)
                status = resp.status if resp else None
                final = page.url
                title = page.title()
                loginish = "login" in (final or "").lower()
                probe.append(
                    {
                        "slug": slug,
                        "requested": url,
                        "final_url": final,
                        "status": status,
                        "title": title,
                        "redirected_to_login": loginish,
                    }
                )
                print(f"probe {slug}: status={status} final={final}")
            except Exception as exc:  # noqa: BLE001
                probe.append({"slug": slug, "error": str(exc)})
                print(f"probe {slug}: ERROR {exc}")

        # Global feature URLs
        globals_ = [
            "https://stockbit.com/",
            "https://stockbit.com/StockbitReports?source=0",
            "https://stockbit.com/screener",
            "https://stockbit.com/calendar",
            "https://stockbit.com/watchlist",
            "https://stockbit.com/heatmap",
            "https://stockbit.com/market",
            "https://stockbit.com/stream",
            "https://stockbit.com/chartbit",
            "https://stockbit.com/orderbook",
            "https://stockbit.com/broker-summary",
            "https://stockbit.com/bandarmology",
        ]
        global_probe = []
        for url in globals_:
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(1200)
                global_probe.append(
                    {
                        "requested": url,
                        "final_url": page.url,
                        "status": resp.status if resp else None,
                        "title": page.title(),
                        "redirected_to_login": "login" in (page.url or "").lower(),
                    }
                )
                print(f"global {url} -> {page.url} ({resp.status if resp else None})")
            except Exception as exc:  # noqa: BLE001
                global_probe.append({"requested": url, "error": str(exc)})

        payload = {
            "symbol_page_url": cur,
            "symbol_hrefs": hrefs,
            "labeled_links": labeled,
            "slug_probe": probe,
            "global_probe": global_probe,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {OUT}")
        return 0

    try:
        return int(run_sync(_run))
    finally:
        try:
            shutdown_browser()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
