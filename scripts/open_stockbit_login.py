"""Open the persistent-profile Chromium so a human can finish login/OTP.

Holds the browser open, polling until the session reports logged in, then
closes cleanly so the profile lock is released for the bot or a farm script.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.browser_stockbit import get_browser, run_sync, shutdown_browser

PROFILE = ROOT / "data" / "stockbit_profile"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0)
    args = ap.parse_args()
    deadline = time.time() + args.minutes * 60

    def _open() -> str:
        browser = get_browser(headless=False, user_data_dir=str(PROFILE))
        browser.start()
        page = browser._page_or_raise()
        page.goto("https://stockbit.com/login", wait_until="domcontentloaded")
        page.wait_for_timeout(3000)
        return "logged_in" if browser.is_logged_in() else "needs_login"

    def _check() -> str:
        browser = get_browser(headless=False, user_data_dir=str(PROFILE))
        return "logged_in" if browser.is_logged_in() else "needs_login"

    print(f"opening chromium (profile={PROFILE})", flush=True)
    state = run_sync(_open)
    print(f"state={state}", flush=True)

    while state != "logged_in" and time.time() < deadline:
        time.sleep(10)
        try:
            state = run_sync(_check)
        except Exception as exc:  # noqa: BLE001
            print(f"check failed: {exc}", flush=True)
            break
        print(f"state={state} ({int(deadline - time.time())}s left)", flush=True)

    print(f"LOGIN_STATE={state}", flush=True)
    shutdown_browser()
    return 0 if state == "logged_in" else 1


if __name__ == "__main__":
    raise SystemExit(main())
