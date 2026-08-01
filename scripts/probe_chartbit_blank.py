from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.browser_stockbit import get_browser, run_sync, shutdown_browser
from agent_idx.tools import _stockbit_headless, _stockbit_profile_dir


def main() -> int:
    def _run():
        b = get_browser(
            headless=_stockbit_headless(), user_data_dir=_stockbit_profile_dir()
        )
        b.start()
        page = b._page_or_raise()
        page.goto(
            "https://stockbit.com/symbol/BBCA/chartbit",
            wait_until="domcontentloaded",
            timeout=60000,
        )
        page.wait_for_timeout(12000)
        html = page.content()
        print("html_len", len(html))
        print("has_canvas", "canvas" in html.lower())
        print("has_login", "/login" in html.lower())
        print("has_chartbit", "chartbit" in html.lower())
        print("body_text_len", len((page.inner_text("body") or "").strip()))
        marker = 'id="__next"'
        i = html.find(marker)
        print("next_snip", (html[i : i + 400] if i >= 0 else html[:400]).replace("\n", " "))
        cookies = b._context.cookies() if b._context else []
        auth = [
            c["name"]
            for c in cookies
            if any(x in c["name"].lower() for x in ("token", "session", "auth", "access"))
        ]
        print("auth_cookies", auth[:30], "total", len(cookies))
        print("is_logged_in", b.is_logged_in(), "needs_login", b.needs_login())
        # try 1H scrape too
        r1 = b.scrape_chart_candles("BBCA", ["1H"], settle_ms=5000)
        print(
            "1H",
            r1.get("error"),
            {k: v.get("n") for k, v in (r1.get("timeframes") or {}).items()},
            "urls",
            len(r1.get("intercepted_urls") or []),
        )
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
