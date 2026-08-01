"""Deep probe Chartbit resolution + network for one symbol/TF."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.browser_stockbit import (
    _CHART_TF_LABELS,
    get_browser,
    run_sync,
    shutdown_browser,
    symbol_page_url,
)
from agent_idx.tools import _stockbit_headless, _stockbit_profile_dir

OUT = ROOT / "exports" / "stockbit" / "_chart_probe_5m.json"


def main() -> int:
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "BBCA").upper()
    tf = (sys.argv[2] if len(sys.argv) > 2 else "5M").upper()

    def _run():
        browser = get_browser(
            headless=_stockbit_headless(), user_data_dir=_stockbit_profile_dir()
        )
        print(browser.start(), flush=True)
        page = browser._page_or_raise()
        captured: list[dict] = []

        def on_resp(resp):
            try:
                url = resp.url
                ctype = (resp.headers.get("content-type") or "").lower()
                low = url.lower()
                if resp.status != 200:
                    return
                interesting = any(
                    k in low
                    for k in (
                        "history",
                        "candle",
                        "ohlc",
                        "bars",
                        "chart",
                        "udf",
                        "tradingview",
                        "exodus",
                        "price",
                        "quote",
                        "time",
                        "symbol",
                        "resolution",
                        "aggregate",
                        "intraday",
                    )
                )
                if "json" not in ctype and not interesting:
                    return
                item = {"url": url[:250], "ctype": ctype[:80], "status": resp.status}
                try:
                    data = resp.json()
                    item["type"] = type(data).__name__
                    if isinstance(data, dict):
                        item["keys"] = list(data.keys())[:30]
                        # UDF?
                        if isinstance(data.get("t"), list):
                            item["udf_n"] = len(data["t"])
                            item["udf_res_hint"] = data.get("s")
                    elif isinstance(data, list):
                        item["list_n"] = len(data)
                except Exception:
                    item["json"] = False
                captured.append(item)
            except Exception:
                return

        page.on("response", on_resp)
        target = symbol_page_url(symbol, "chart")
        page.goto(target, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)
        print("url", page.url, flush=True)
        print("loginish", "login" in page.url.lower(), flush=True)

        # Dump resolution-like buttons
        ui = page.evaluate(
            """() => {
              const texts = [];
              for (const el of document.querySelectorAll('button, [role="button"], div')) {
                const t = (el.innerText || '').trim().replace(/\\s+/g, ' ');
                if (!t || t.length > 20) continue;
                if (/^(1|3|5|15|30|45|60|120|240|1[HDWM]|[DHWM]|D|W|M)$/i.test(t)
                    || /min|jam|hour|day|week|menit/i.test(t)) {
                  texts.push(t);
                }
              }
              return [...new Set(texts)].slice(0, 80);
            }"""
        )
        print("ui_labels", ui, flush=True)

        # Try widget resolutions
        widget = page.evaluate(
            """() => {
              const out = {};
              try {
                const w = window.tvWidget || window.chartWidget || window.__chartWidget;
                out.hasWidget = !!w;
                if (w && typeof w.activeChart === 'function') {
                  const c = w.activeChart();
                  out.res = c.resolution && c.resolution();
                  try { c.setResolution('5'); out.set5 = true; } catch(e) { out.set5 = String(e); }
                }
              } catch(e) { out.err = String(e); }
              return out;
            }"""
        )
        print("widget", widget, flush=True)
        page.wait_for_timeout(3000)

        labels = _CHART_TF_LABELS.get(tf, [tf, "5"])
        print("try_labels", labels, flush=True)
        ok = browser._set_chart_resolution(labels)
        print("set_resolution", ok, flush=True)
        page.wait_for_timeout(4500)

        # Also click any visible "5"
        for sel in ('button:has-text("5")', 'div[role="button"]:has-text("5")', 'text="5"'):
            try:
                loc = page.locator(sel)
                n = loc.count()
                print(f"click_candidates {sel}={n}", flush=True)
                if n:
                    loc.first.click(timeout=2000, force=True)
                    page.wait_for_timeout(2500)
                    break
            except Exception as exc:
                print("click fail", sel, exc, flush=True)

        page.screenshot(
            path=str(ROOT / "exports" / "stockbit" / "_chart_probe_5m.png"),
            full_page=False,
        )

        # Now full scrape path
        page.remove_listener("response", on_resp)
        result = browser.scrape_chart_candles(symbol, [tf], settle_ms=4500)
        payload = {
            "page_url": page.url,
            "ui_labels": ui,
            "widget": widget,
            "captured_n": len(captured),
            "captured": captured[:40],
            "scrape_error": result.get("error"),
            "scrape_n": {
                k: (v or {}).get("n") for k, v in (result.get("timeframes") or {}).items()
            },
            "scrape_urls": (result.get("intercepted_urls") or [])[:20],
        }
        OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("saved", OUT, flush=True)
        print("scrape", payload["scrape_error"], payload["scrape_n"], flush=True)
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
