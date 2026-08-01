"""Quick probe: can we intercept OHLCV from Stockbit Chartbit?"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.browser_stockbit import get_browser, run_sync, shutdown_browser


def main() -> int:
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "BBCA").upper()
    out = ROOT / "data" / "charts" / "_meta"
    out.mkdir(parents=True, exist_ok=True)

    def _run():
        browser = get_browser(headless=False)
        print(browser.start())
        browser.open(url="https://stockbit.com/")
        print("Status awal:", browser.status().replace("\n", " | "))
        print("Login di Chrome jika diminta. Menunggu max 180s...")
        deadline = time.time() + 180
        while time.time() < deadline:
            if browser.is_logged_in() and not browser._has_visible_password():
                # Prefer real cookie/session over false-positive homepage.
                cookies = browser._context.cookies() if browser._context else []
                names = {c.get("name", "").lower() for c in cookies}
                if any("token" in n or "session" in n or "auth" in n for n in names):
                    print("Login OK (session cookie)")
                    break
            if browser.needs_otp():
                print("OTP — selesaikan di browser...")
            time.sleep(2)
        else:
            print("Timeout tunggu login — lanjut probe Chartbit tetap dicoba")

        result = browser.scrape_chart_candles(symbol, ["1D", "1H", "1W"], settle_ms=4000)
        path = out / f"probe_{symbol}.json"
        dump = {
            "symbol": result.get("symbol"),
            "error": result.get("error"),
            "page_url": browser._page.url if browser._page else None,
            "intercepted_urls": result.get("intercepted_urls") or [],
            "timeframes": {},
        }
        for tf, payload in (result.get("timeframes") or {}).items():
            bars = payload.get("bars") or []
            dump["timeframes"][tf] = {
                "n": len(bars),
                "source_urls": payload.get("source_urls") or [],
                "sample": bars[:3],
            }
        path.write_text(json.dumps(dump, indent=2), encoding="utf-8")
        print(json.dumps(dump, indent=2)[:4000])
        print(f"\nSaved: {path}")
        return 0 if any(v.get("n", 0) > 0 for v in dump["timeframes"].values()) else 2

    try:
        return int(run_sync(_run))
    finally:
        try:
            shutdown_browser()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
