"""Post 'test' via Stockbit stream compose box."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.browser_stockbit import get_browser, run_sync, shutdown_browser

PROFILE = ROOT / "data" / "stockbit_profile"
OUT_DIR = ROOT / "exports" / "stockbit"
TEXT = "test"


def main() -> int:
    def _run():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        browser = get_browser(headless=True, user_data_dir=str(PROFILE))
        print(browser.start())
        page = browser._page_or_raise()
        page.goto("https://stockbit.com/stream", wait_until="domcontentloaded")
        page.wait_for_timeout(4000)

        # Inspect compose form internals
        detail = page.evaluate(
            """() => {
              const forms = [...document.querySelectorAll('form')];
              const target = forms.find(f => (f.innerText||'').includes('Tulis ide'));
              if (!target) return {error: 'compose form not found', formCount: forms.length};
              const kids = [...target.querySelectorAll('*')].slice(0, 80).map(e => ({
                tag: e.tagName,
                type: e.getAttribute('type'),
                role: e.getAttribute('role'),
                contenteditable: e.getAttribute('contenteditable'),
                placeholder: e.getAttribute('placeholder'),
                class: (e.className||'').toString().slice(0, 100),
                text: (e.innerText||'').trim().slice(0, 60),
                html: e.outerHTML.slice(0, 200),
              }));
              return {kids, html: target.outerHTML.slice(0, 2500)};
            }"""
        )
        (OUT_DIR / "_compose_form.json").write_text(
            json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("form detail keys:", list(detail.keys()))
        if detail.get("error"):
            print(detail)
            return 2

        # Click placeholder text then type
        box = page.get_by_text("Tulis ide kamu disini...", exact=False).first
        box.click(timeout=5000)
        page.wait_for_timeout(500)

        # Try contenteditable / textbox inside compose
        filled = False
        for sel in (
            'form:has-text("Tulis ide") [contenteditable="true"]',
            'form:has-text("Tulis ide") [role="textbox"]',
            'form:has-text("Tulis ide") textarea',
            'form:has-text("Tulis ide") div[contenteditable]',
            '[contenteditable="true"]',
            '[role="textbox"]',
        ):
            loc = page.locator(sel)
            if loc.count() == 0:
                continue
            try:
                target = loc.first
                target.click(timeout=2000)
                try:
                    target.fill(TEXT)
                except Exception:
                    page.keyboard.press("Control+A")
                    page.keyboard.type(TEXT, delay=40)
                filled = True
                print("filled via", sel)
                break
            except Exception as exc:  # noqa: BLE001
                print("fill fail", sel, exc)

        if not filled:
            # Fallback: just type after clicking placeholder
            page.keyboard.type(TEXT, delay=40)
            filled = True
            print("filled via keyboard after placeholder click")

        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT_DIR / "_post_filled.png"), full_page=False)

        # Enable + click Post inside compose form
        post_btn = page.locator('form:has-text("Tulis ide") button:has-text("Post")').first
        if post_btn.count() == 0:
            post_btn = page.locator('button.ant-btn-primary:has-text("Post")').first

        # Wait until enabled
        enabled = False
        for _ in range(10):
            try:
                disabled = post_btn.get_attribute("disabled")
                aria = post_btn.get_attribute("aria-disabled")
                visible = post_btn.is_visible()
                print(f"post btn disabled={disabled} aria={aria} visible={visible}")
                if visible and disabled is None and aria != "true":
                    enabled = True
                    break
            except Exception as exc:  # noqa: BLE001
                print("btn check", exc)
            page.wait_for_timeout(400)

        if not enabled:
            # force click anyway
            print("Post still disabled; trying force click")
            try:
                post_btn.click(timeout=3000, force=True)
            except Exception as exc:  # noqa: BLE001
                print("force click failed", exc)
                page.keyboard.press("Control+Enter")
                print("tried Ctrl+Enter")
        else:
            post_btn.click(timeout=5000)
            print("clicked Post")

        page.wait_for_timeout(5000)
        page.screenshot(path=str(OUT_DIR / "_post_after.png"), full_page=False)

        # Verify on profile Ideas
        page.goto("https://stockbit.com/kharisamiruddin", wait_until="domcontentloaded")
        page.wait_for_timeout(4000)
        page.screenshot(path=str(OUT_DIR / "_post_profile.png"), full_page=False)
        check = page.evaluate(
            """() => {
              const text = document.body.innerText || '';
              // Look for a short standalone 'test' near top ideas
              const lines = text.split('\\n').map(s => s.trim()).filter(Boolean);
              const hit = lines.find(l => l.toLowerCase() === 'test');
              return {
                hasExactTestLine: !!hit,
                firstLines: lines.slice(0, 40),
                containsTest: /\\btest\\b/i.test(text),
              };
            }"""
        )
        (OUT_DIR / "_post_verify.json").write_text(
            json.dumps(check, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("verify:", check)
        return 0 if check.get("hasExactTestLine") or check.get("containsTest") else 3

    try:
        return int(run_sync(_run))
    finally:
        try:
            shutdown_browser()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
