"""Explore Stockbit compose UI (no submit)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.browser_stockbit import get_browser, run_sync, shutdown_browser

PROFILE = ROOT / "data" / "stockbit_profile"
OUT_DIR = ROOT / "exports" / "stockbit"


def main() -> int:
    def _run():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        browser = get_browser(headless=True, user_data_dir=str(PROFILE))
        print(browser.start())
        page = browser._page_or_raise()

        for url in (
            "https://stockbit.com/stream",
            "https://stockbit.com/kharisamiruddin",
        ):
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(4000)
            name = "stream" if "/stream" in url else "profile"
            page.screenshot(path=str(OUT_DIR / f"_ui_{name}.png"), full_page=False)

            info = page.evaluate(
                """() => {
                  const buttons = [...document.querySelectorAll('button, [role="button"], a')]
                    .map(e => ({
                      tag: e.tagName,
                      text: (e.innerText||'').trim().replace(/\\s+/g,' ').slice(0,60),
                      aria: e.getAttribute('aria-label'),
                      href: e.getAttribute('href'),
                      title: e.getAttribute('title'),
                      class: (e.className||'').toString().slice(0,80),
                    }))
                    .filter(x => x.text || x.aria || x.title)
                    .slice(0, 150);

                  const inputs = [...document.querySelectorAll(
                    'input, textarea, [contenteditable], [role="textbox"], form'
                  )].map(e => ({
                    tag: e.tagName,
                    type: e.getAttribute('type'),
                    role: e.getAttribute('role'),
                    placeholder: e.getAttribute('placeholder'),
                    aria: e.getAttribute('aria-label'),
                    contenteditable: e.getAttribute('contenteditable'),
                    class: (e.className||'').toString().slice(0,100),
                    text: (e.innerText||'').trim().slice(0,80),
                  }));

                  // Look for create / plus icons near top
                  const svgs = [...document.querySelectorAll('svg')].slice(0, 40).map(s => {
                    const p = s.closest('button, a, [role="button"], div');
                    return {
                      parentTag: p ? p.tagName : null,
                      parentText: p ? (p.innerText||'').trim().slice(0,40) : null,
                      parentAria: p ? p.getAttribute('aria-label') : null,
                      parentClass: p ? (p.className||'').toString().slice(0,80) : null,
                    };
                  });

                  return {
                    title: document.title,
                    url: location.href,
                    bodyStart: (document.body.innerText||'').slice(0, 1500),
                    buttons,
                    inputs,
                    svgs,
                  };
                }"""
            )
            (OUT_DIR / f"_ui_{name}.json").write_text(
                json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"=== {name} ===")
            print("url:", info["url"])
            print("bodyStart:", info["bodyStart"][:400].replace("\n", " | "))
            interesting = [
                b
                for b in info["buttons"]
                if any(
                    k in ((b.get("text") or "") + " " + (b.get("aria") or "")).lower()
                    for k in (
                        "post",
                        "kirim",
                        "tulis",
                        "share",
                        "create",
                        "buat",
                        "compose",
                        "new",
                        "+",
                        "bit",
                    )
                )
            ]
            print("interesting buttons:", interesting[:20])
            print("inputs:", info["inputs"][:20])

        # Try clicking common FAB / create
        page.goto("https://stockbit.com/stream", wait_until="domcontentloaded")
        page.wait_for_timeout(3000)
        click_tries = [
            'button:has-text("Create")',
            'button:has-text("Buat")',
            'button:has-text("Post")',
            'button:has-text("Tulis")',
            '[aria-label*="create" i]',
            '[aria-label*="post" i]',
            '[aria-label*="compose" i]',
            '[aria-label*="tulis" i]',
            'text=/Start a post/i',
            'text=/Mulai/i',
        ]
        for sel in click_tries:
            try:
                loc = page.locator(sel)
                n = loc.count()
                if n == 0:
                    continue
                print(f"found {n} for {sel}")
                loc.first.click(timeout=2000)
                page.wait_for_timeout(1500)
                page.screenshot(path=str(OUT_DIR / "_ui_after_click.png"), full_page=False)
                after = page.evaluate(
                    """() => ({
                      textareas: document.querySelectorAll('textarea').length,
                      editables: document.querySelectorAll('[contenteditable="true"]').length,
                      dialogs: document.querySelectorAll('[role="dialog"], .modal, [class*="Modal"]').length,
                      body: (document.body.innerText||'').slice(0,800),
                    })"""
                )
                print("after click", sel, after)
                if after["textareas"] or after["editables"] or after["dialogs"]:
                    (OUT_DIR / "_ui_modal.json").write_text(
                        json.dumps(after, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    break
            except Exception as exc:  # noqa: BLE001
                print("click fail", sel, exc)
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
