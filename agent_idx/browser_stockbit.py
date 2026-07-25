from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Playwright sync API must stay on one thread (login vs agent worker threads).
_BROWSER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stockbit-pw")

ALLOWED_HOST_SUFFIXES = (
    "stockbit.com",
)
LOGIN_URL = "https://stockbit.com/login"
HOME_URL = "https://stockbit.com/"
REPORTS_URL = "https://stockbit.com/StockbitReports?source=0"

SYMBOL_PAGE_SLUGS = {
    "overview": "",
    "chart": "chartbit",
    "chartbit": "chartbit",
    "keystats": "keystats",
    "financials": "financials",
    "company": "company",
    "bit": "bit",
}


def symbol_page_url(symbol: str, page: str = "overview") -> str:
    sym = (symbol or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{3,5}", sym):
        raise ValueError(f"symbol tidak valid: {symbol!r}")
    slug = SYMBOL_PAGE_SLUGS.get((page or "overview").lower(), (page or "").strip("/"))
    if slug:
        return f"https://stockbit.com/symbol/{sym}/{slug}"
    return f"https://stockbit.com/symbol/{sym}"

NEED_CREDENTIALS = "NEED_STOCKBIT_CREDENTIALS"
NEED_OTP = "NEED_STOCKBIT_OTP"

_OTP_INPUT_SELECTORS = [
    'input[autocomplete="one-time-code"]',
    'input[inputmode="numeric"]',
    'input[name*="otp" i]',
    'input[name*="code" i]',
    'input[id*="otp" i]',
    'input[id*="code" i]',
    'input[placeholder*="kode" i]',
    'input[placeholder*="otp" i]',
    'input[placeholder*="verifikasi" i]',
    'input[maxlength="1"]',
]


def _host_allowed(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    return any(host == s or host.endswith("." + s) for s in ALLOWED_HOST_SUFFIXES)


def _visible_count(locator) -> int:
    try:
        total = locator.count()
    except Exception:  # noqa: BLE001
        return 0
    visible = 0
    for i in range(min(total, 20)):
        try:
            if locator.nth(i).is_visible():
                visible += 1
        except Exception:  # noqa: BLE001
            continue
    return visible


class StockbitBrowser:
    """Ephemeral (guest-like) Chromium session limited to Stockbit domains."""

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    @property
    def ready(self) -> bool:
        return self._page is not None

    def status(self) -> str:
        """Human-readable Stockbit session status for bot/agent."""
        lines = [
            f"browser_ready={self.ready}",
            f"headless={self.headless}",
            "mode=ephemeral/guest-like (session hilang saat bot restart)",
            "allowlist=stockbit.com only",
        ]
        if not self.ready:
            lines.append("login=not_started")
            lines.append(
                "hint: jalankan /stockbit untuk login interaktif, "
                "atau /ask buka stockbit ..."
            )
            return "\n".join(lines)

        page = self._page_or_raise()
        url = page.url
        lines.append(f"url={url}")
        try:
            logged_in = self.is_logged_in()
            needs_login = self.needs_login()
            needs_otp = self.needs_otp()
        except Exception as exc:  # noqa: BLE001
            lines.append(f"login=error ({exc})")
            return "\n".join(lines)

        if logged_in:
            state = "logged_in"
        elif needs_otp:
            state = "needs_otp"
        elif needs_login:
            state = "needs_login"
        else:
            state = "unknown"
        lines.append(f"login={state}")
        lines.append(f"needs_login={needs_login}")
        lines.append(f"needs_otp={needs_otp}")
        if state == "logged_in":
            lines.append("hint: siap scrape — /ask scrape Stockbit BBCA")
        elif state == "needs_otp":
            lines.append("hint: kirim kode OTP di chat, atau /stockbit ulang")
        elif state == "needs_login":
            lines.append("hint: /stockbit untuk login interaktif")
        else:
            lines.append("hint: /stockbit untuk memastikan sesi")
        return "\n".join(lines)

    def start(self) -> str:
        if self._page is not None:
            return "OK: browser already running (ephemeral session)"
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright belum terpasang. Jalankan: "
                "py -3 -m pip install playwright && py -3 -m playwright install chromium"
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._context = self._browser.new_context(
            viewport={"width": 1400, "height": 900},
            locale="id-ID",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(30_000)
        self._page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        return "OK: ephemeral Chrome started (guest-like, no persistent profile)"

    def close(self) -> str:
        for obj in (self._context, self._browser):
            try:
                if obj is not None:
                    obj.close()
            except Exception:  # noqa: BLE001
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:  # noqa: BLE001
                pass
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        return "OK: browser closed"

    def _page_or_raise(self):
        if self._page is None:
            self.start()
        assert self._page is not None
        return self._page

    def _frames(self):
        page = self._page_or_raise()
        return page.frames

    def _has_visible_password(self) -> bool:
        for frame in self._frames():
            if _visible_count(frame.locator('input[type="password"]')) > 0:
                return True
        return False

    def _otp_digit_boxes(self):
        """Return (frame, locator) for multi-box OTP if present."""
        for frame in self._frames():
            loc = frame.locator('input[maxlength="1"]')
            if _visible_count(loc) >= 4:
                return frame, loc
        return None, None

    def _otp_single_input(self):
        for frame in self._frames():
            for sel in _OTP_INPUT_SELECTORS:
                if sel == 'input[maxlength="1"]':
                    continue
                loc = frame.locator(sel)
                if _visible_count(loc) > 0:
                    return frame, loc.first
        # Fallback: any visible text/tel input when password is gone.
        if not self._has_visible_password():
            for frame in self._frames():
                for sel in ('input[type="tel"]', 'input[type="text"]', "input"):
                    loc = frame.locator(sel)
                    n = _visible_count(loc)
                    if n == 1:
                        return frame, loc.first
        return None, None

    def _has_visible_otp_inputs(self) -> bool:
        _, boxes = self._otp_digit_boxes()
        if boxes is not None:
            return True
        _, single = self._otp_single_input()
        return single is not None

    def open(
        self,
        url: str | None = None,
        symbol: str | None = None,
        page: str | None = None,
    ) -> str:
        tab = self._page_or_raise()
        subpage = page
        if symbol:
            try:
                target = symbol_page_url(symbol, subpage or "overview")
            except ValueError as exc:
                return f"ERROR: {exc}"
        else:
            target = (url or HOME_URL).strip()
        if not target.startswith("http"):
            target = "https://" + target
        if not _host_allowed(target):
            return f"ERROR: domain tidak diizinkan: {target}. Hanya stockbit.com."

        tab.goto(target, wait_until="domcontentloaded")
        tab.wait_for_timeout(1500)
        final = tab.url
        if not _host_allowed(final):
            tab.goto(HOME_URL, wait_until="domcontentloaded")
            return f"ERROR: navigasi keluar allowlist diblokir ({final})"

        if self.needs_login():
            return (
                f"{NEED_CREDENTIALS}\n"
                f"url={final}\n"
                "Halaman login Stockbit terdeteksi. Minta kredensial interaktif ke user."
            )
        if self.needs_otp():
            return (
                f"{NEED_OTP}\n"
                f"url={final}\n"
                "Halaman kode verifikasi terdeteksi."
            )
        return f"OK: opened {final}\n\n{self.snapshot(max_chars=4000)}"

    def needs_login(self) -> bool:
        if self.is_logged_in():
            return False
        if self.needs_otp():
            return False
        page = self._page_or_raise()
        url = (page.url or "").lower()
        if "login" in url or "signin" in url:
            return True
        return self._has_visible_password()

    def needs_otp(self) -> bool:
        """True only when OTP inputs are actually visible."""
        if self.is_logged_in():
            return False
        if self._has_visible_password():
            return False
        return self._has_visible_otp_inputs()

    def is_logged_in(self) -> bool:
        """Positive login check — avoid false OTP failures."""
        page = self._page_or_raise()
        url = (page.url or "").lower()
        if not _host_allowed(url):
            return False
        # Still on auth gates.
        if any(
            x in url
            for x in ("/login", "/signin", "/otp", "/verify", "/2fa", "/mfa", "/challenge")
        ):
            if self._has_visible_password() or self._has_visible_otp_inputs():
                return False
        if self._has_visible_password() or self._has_visible_otp_inputs():
            return False

        # Logged-in UI signals.
        for frame in self._frames():
            for sel in (
                'a[href*="logout" i]',
                'button:has-text("Keluar")',
                'a:has-text("Keluar")',
                '[data-testid*="avatar" i]',
                'a[href*="/stream"]',
                'a[href*="/portfolio"]',
                'a[href*="/watchlist"]',
            ):
                if _visible_count(frame.locator(sel)) > 0:
                    return True

        # Cookie heuristic.
        try:
            cookies = self._context.cookies() if self._context else []
            names = {c.get("name", "").lower() for c in cookies}
            if any("token" in n or "session" in n or "auth" in n for n in names):
                if "login" not in url and "signin" not in url:
                    return True
        except Exception:  # noqa: BLE001
            pass

        # Default: if not on login and no auth inputs, treat as logged in.
        if "login" not in url and "signin" not in url:
            return True
        return False

    def login(self, username: str, password: str) -> str:
        page = self._page_or_raise()
        username = (username or "").strip()
        password = password or ""
        if not username or not password:
            return "ERROR: username/password kosong"

        if self.needs_otp():
            return (
                f"{NEED_OTP}\n"
                "Halaman kode verifikasi sudah terbuka. Kirim kode OTP dari chat."
            )

        if not self.needs_login():
            page.goto(LOGIN_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)

        if self.is_logged_in() and not self.needs_login() and not self.needs_otp():
            return (
                f"OK: sudah login\nurl={page.url}\n\n"
                f"{self.snapshot(max_chars=3000)}"
            )

        if not _host_allowed(page.url):
            return "ERROR: bukan halaman Stockbit"

        # Scope to login form / auth modal — never the navbar search box
        # (placeholder contains "username" and was matching incorrectly).
        self._dismiss_blocking_modals()
        form_roots = self._login_form_roots()
        user_selectors = [
            'input[name="username"]',
            'input[name="email"]',
            'input[type="email"]',
            'input[autocomplete="username"]',
            'input[autocomplete="email"]',
            'input[placeholder*="email" i]',
            'input[placeholder*="Username" i]',
            'input[placeholder*="Email" i]',
            'input[type="text"]:not([role="combobox"])',
        ]
        pass_selectors = [
            'input[name="password"]',
            'input[type="password"]',
            'input[autocomplete="current-password"]',
        ]
        submit_selectors = [
            'button[type="submit"]',
            'button:has-text("Masuk")',
            'button:has-text("Login")',
            'button:has-text("Sign in")',
            'button:has-text("Lanjut")',
            'button:has-text("Continue")',
        ]

        filled_user = self._fill_first(form_roots, user_selectors, username)
        if not filled_user:
            return (
                "ERROR: field username login tidak ditemukan "
                "(bukan search bar). UI Stockbit mungkin berubah."
            )

        filled_pass = self._fill_first(form_roots, pass_selectors, password)
        if not filled_pass:
            return "ERROR: field password tidak ditemukan"

        clicked = self._click_first(form_roots, submit_selectors)
        if not clicked:
            page.keyboard.press("Enter")

        page.wait_for_timeout(5000)
        if self.needs_otp():
            return (
                f"{NEED_OTP}\n"
                "Password diterima. Stockbit meminta kode verifikasi (OTP/2FA).\n"
                "Kirim kode dari email/SMS/authenticator di chat."
            )
        if self.is_logged_in():
            return f"OK: login berhasil\nurl={page.url}\n\n{self.snapshot(max_chars=3000)}"
        if self.needs_login():
            return (
                "ERROR: login gagal (password salah atau captcha). "
                "Coba lagi; jika captcha, set STOCKBIT_HEADLESS=false."
            )
        return f"OK: login berhasil\nurl={page.url}\n\n{self.snapshot(max_chars=3000)}"

    def submit_otp(self, code: str) -> str:
        page = self._page_or_raise()
        code = re.sub(r"\s+", "", code or "")
        if not code:
            return "ERROR: kode verifikasi kosong"
        if not _host_allowed(page.url):
            return "ERROR: bukan halaman Stockbit"

        before_url = page.url
        filled = self._fill_otp(code)
        if not filled:
            debug = self._debug_dump("otp_no_field")
            return (
                "ERROR: field kode verifikasi tidak ditemukan.\n"
                f"url={before_url}\n{debug}"
            )

        # Wait for navigation / UI change.
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:  # noqa: BLE001
            page.wait_for_timeout(4000)

        # Sometimes OTP auto-submits; give it a moment then go home.
        page.wait_for_timeout(2000)
        if self.is_logged_in():
            return f"OK: login berhasil\nurl={page.url}\n\n{self.snapshot(max_chars=3000)}"

        # Try home — session cookie may already be set.
        try:
            page.goto(HOME_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
        except Exception:  # noqa: BLE001
            pass

        if self.is_logged_in():
            return f"OK: login berhasil\nurl={page.url}\n\n{self.snapshot(max_chars=3000)}"

        if self.needs_otp():
            debug = self._debug_dump("otp_still_required")
            return (
                "ERROR: masih di halaman verifikasi setelah mengirim kode.\n"
                "Kemungkinan: kode sudah expired, atau UI butuh klik manual.\n"
                "Coba kode baru, atau set STOCKBIT_HEADLESS=false lalu /stockbit ulang.\n"
                f"url={page.url}\n{debug}"
            )
        if self.needs_login():
            return "ERROR: kembali ke halaman login setelah OTP"
        return f"OK: login berhasil\nurl={page.url}\n\n{self.snapshot(max_chars=3000)}"

    def _fill_otp(self, code: str) -> bool:
        page = self._page_or_raise()
        digits = list(code)

        frame, boxes = self._otp_digit_boxes()
        if boxes is not None:
            n = _visible_count(boxes)
            # Clear then type digit-by-digit (React-friendly).
            for i in range(min(n, len(digits))):
                box = boxes.nth(i)
                box.click()
                box.fill("")
                box.type(digits[i], delay=80)
            if len(digits) >= n:
                # Auto-submit often happens; also try Enter / verify button.
                page.keyboard.press("Enter")
                self._click_verify_buttons()
            return True

        frame, single = self._otp_single_input()
        if single is None:
            return False
        single.click()
        single.fill("")
        single.type(code, delay=60)
        page.keyboard.press("Enter")
        self._click_verify_buttons()
        return True

    def _login_form_roots(self) -> list:
        """Prefer login form / auth dialog over whole page (avoids navbar search)."""
        page = self._page_or_raise()
        roots = []
        for frame in self._frames():
            for sel in (
                'form:has(input[type="password"])',
                'form:has(input[name="password"])',
                '.ant-modal-wrap:not([style*="display: none"]) form',
                '.ant-modal-content:has(input[type="password"])',
                '[role="dialog"]:has(input[type="password"])',
                'div:has(> input[type="password"])',
            ):
                loc = frame.locator(sel)
                if _visible_count(loc) > 0:
                    roots.append(loc.first)
        if not roots:
            # Fallback: any frame, but callers must use strict selectors.
            roots.extend(self._frames())
        return roots

    def _fill_first(self, roots: list, selectors: list[str], value: str) -> bool:
        for root in roots:
            for sel in selectors:
                try:
                    loc = root.locator(sel)
                except Exception:  # noqa: BLE001
                    continue
                if _visible_count(loc) == 0:
                    continue
                target = loc.first
                try:
                    # Skip navbar/search widgets.
                    input_type = (target.get_attribute("type") or "").lower()
                    role = (target.get_attribute("role") or "").lower()
                    placeholder = (target.get_attribute("placeholder") or "").lower()
                    if input_type == "search" or role == "combobox":
                        continue
                    if "search for brand" in placeholder or "search" in placeholder:
                        continue
                    target.click(timeout=5000)
                    target.fill("")
                    target.fill(value)
                    return True
                except Exception as exc:  # noqa: BLE001
                    logger.info("fill skip %s: %s", sel, exc)
                    continue
        return False

    def _click_first(self, roots: list, selectors: list[str]) -> bool:
        for root in roots:
            for sel in selectors:
                try:
                    loc = root.locator(sel)
                except Exception:  # noqa: BLE001
                    continue
                if _visible_count(loc) == 0:
                    continue
                try:
                    loc.first.click(timeout=5000)
                    return True
                except Exception as exc:  # noqa: BLE001
                    logger.info("click skip %s: %s", sel, exc)
                    try:
                        loc.first.click(timeout=3000, force=True)
                        return True
                    except Exception:  # noqa: BLE001
                        continue
        return False

    def _dismiss_blocking_modals(self) -> None:
        """Close cookie/promo modals that intercept pointer events."""
        page = self._page_or_raise()
        for frame in self._frames():
            for sel in (
                'button:has-text("Tutup")',
                'button:has-text("Close")',
                'button:has-text("Mengerti")',
                'button:has-text("OK")',
                'button:has-text("Nanti")',
                'button.ant-modal-close',
                '[aria-label="Close"]',
                ".ant-modal-close",
            ):
                loc = frame.locator(sel)
                if _visible_count(loc) > 0:
                    try:
                        loc.first.click(timeout=1500)
                        page.wait_for_timeout(300)
                    except Exception:  # noqa: BLE001
                        continue
        # Escape key often closes ant modals.
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(200)
        except Exception:  # noqa: BLE001
            pass

    def _click_verify_buttons(self) -> None:
        for frame in self._frames():
            for sel in (
                'button[type="submit"]',
                'button:has-text("Verifikasi")',
                'button:has-text("Verify")',
                'button:has-text("Konfirmasi")',
                'button:has-text("Confirm")',
                'button:has-text("Lanjut")',
                'button:has-text("Continue")',
                'button:has-text("Submit")',
                'button:has-text("Masuk")',
            ):
                loc = frame.locator(sel)
                if _visible_count(loc) > 0:
                    try:
                        loc.first.click(timeout=2000)
                        return
                    except Exception:  # noqa: BLE001
                        continue

    def _debug_dump(self, tag: str) -> str:
        page = self._page_or_raise()
        out_dir = Path(__file__).resolve().parent.parent / "exports" / "stockbit_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        shot = out_dir / f"{tag}.png"
        html = out_dir / f"{tag}.html"
        try:
            page.screenshot(path=str(shot), full_page=True)
        except Exception:  # noqa: BLE001
            shot = None
        try:
            html.write_text(page.content(), encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            html = None
        bits = [f"debug_url={page.url}"]
        if shot:
            bits.append(f"screenshot={shot}")
        if html:
            bits.append(f"html={html}")
        return " | ".join(bits)

    def snapshot(self, max_chars: int = 8000) -> str:
        page = self._page_or_raise()
        if not _host_allowed(page.url):
            return "ERROR: halaman di luar stockbit.com"
        text = self._extract_main_text()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return f"url={page.url}\n\n{text}"

    def scrape_symbol(self, symbol: str, sections: str | None = None) -> str:
        """Scrape Stockbit pages for a ticker and save a markdown report."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        symbol = (symbol or "").strip().upper()
        if not re.fullmatch(r"[A-Z]{3,5}", symbol):
            return f"ERROR: symbol tidak valid: {symbol!r}"

        if self.needs_login():
            return (
                f"{NEED_CREDENTIALS}\n"
                "Login diperlukan sebelum scrape Stockbit."
            )
        if self.needs_otp():
            return f"{NEED_OTP}\nHalaman OTP masih terbuka."

        section_map = {
            "overview": symbol_page_url(symbol, "overview"),
            "chart": symbol_page_url(symbol, "chart"),
            "chartbit": symbol_page_url(symbol, "chart"),
            "keystats": symbol_page_url(symbol, "keystats"),
            "financials": symbol_page_url(symbol, "financials"),
            "company": symbol_page_url(symbol, "company"),
            "bit": symbol_page_url(symbol, "bit"),
        }
        if sections:
            wanted = [s.strip().lower() for s in sections.split(",") if s.strip()]
        else:
            wanted = ["overview", "keystats", "financials", "company"]

        blocks: list[str] = [
            f"# Stockbit scrape — {symbol}",
            "",
            f"Scraped: {datetime.now(ZoneInfo('Asia/Jakarta')).isoformat()}",
            "",
        ]
        ok_sections = 0
        for name in wanted:
            url = section_map.get(name)
            if not url:
                blocks.append(f"## {name}\nSKIP: section tidak dikenal\n")
                continue
            result = self.open(url=url)
            if NEED_CREDENTIALS in result or NEED_OTP in result:
                return result
            if result.startswith("ERROR:"):
                blocks.append(f"## {name}\n{result}\n")
                continue
            page = self._page_or_raise()
            page.wait_for_timeout(2000)
            # Dismiss popups that cover content.
            self._dismiss_blocking_modals()
            text = self._extract_main_text()
            # Drop huge nav chrome noise.
            text = _clean_scrape_text(text)
            if len(text) > 12000:
                text = text[:12000] + "\n... (truncated)"
            blocks.append(f"## {name}")
            blocks.append(f"url: {page.url}")
            blocks.append("")
            blocks.append(text or "(halaman kosong / tidak terbaca)")
            blocks.append("")
            ok_sections += 1

        if ok_sections == 0:
            return f"ERROR: gagal scrape semua section untuk {symbol}"

        report = "\n".join(blocks)
        out_dir = Path(__file__).resolve().parent.parent / "exports" / "stockbit"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{symbol}_{stamp}.md"
        path.write_text(report, encoding="utf-8")
        preview = report if len(report) <= 10000 else report[:10000] + "\n... (truncated)"
        return (
            f"OK: scrape {symbol} ({ok_sections} section)\n"
            f"FILE: {path}\n\n{preview}"
        )

    def scrape_reports_stream(
        self,
        days: int | None = 10,
        url: str | None = None,
        max_scrolls: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
        """Scrape stream posts from Stockbit Reports (or similar profile URL)."""
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        if self.needs_login():
            return (
                f"{NEED_CREDENTIALS}\n"
                "Login diperlukan sebelum scrape Stockbit Reports."
            )
        if self.needs_otp():
            return f"{NEED_OTP}\nHalaman OTP masih terbuka."

        tz = ZoneInfo("Asia/Jakarta")
        now = datetime.now(tz)
        range_label = ""

        if date_from or date_to:
            start = _parse_iso_date(date_from, tz) if date_from else None
            end = _parse_iso_date(date_to, tz) if date_to else now
            if start is None and date_from:
                return f"ERROR: date_from tidak valid: {date_from!r} (pakai YYYY-MM-DD)"
            if end is None and date_to:
                return f"ERROR: date_to tidak valid: {date_to!r} (pakai YYYY-MM-DD)"
            if start is None:
                start = end - timedelta(days=90)
            if end is None:
                end = now
            if start > end:
                return "ERROR: date_from harus <= date_to"
            span_days = (end.date() - start.date()).days + 1
            if span_days > 366:
                return "ERROR: rentang maksimal 366 hari per scrape"
            max_scrolls = max_scrolls or min(120, max(40, span_days // 2))
            range_label = f"{start.date().isoformat()} .. {end.date().isoformat()}"
        else:
            days = 10 if days is None else int(days)
            if days < 1 or days > 90:
                return "ERROR: days harus antara 1 dan 90"
            start = now - timedelta(days=days)
            end = now
            max_scrolls = max_scrolls or 30
            range_label = f"{days} hari terakhir"

        target = (url or REPORTS_URL).strip()
        page = self._page_or_raise()
        api_posts: list[dict[str, str]] = []

        def _on_response(response) -> None:
            try:
                if response.status != 200:
                    return
                req_url = response.url.lower()
                if not any(
                    k in req_url
                    for k in ("stream", "timeline", "activity", "feed", "post", "user")
                ):
                    return
                ctype = (response.headers.get("content-type") or "").lower()
                if "json" not in ctype:
                    return
                data = response.json()
                _walk_json_for_posts(data, api_posts)
            except Exception:  # noqa: BLE001
                return

        page.on("response", _on_response)
        open_result = self.open(url=target)
        if NEED_CREDENTIALS in open_result or NEED_OTP in open_result:
            page.remove_listener("response", _on_response)
            return open_result
        if open_result.startswith("ERROR:"):
            page.remove_listener("response", _on_response)
            return open_result

        self._dismiss_blocking_modals()
        page.wait_for_timeout(2000)

        dom_posts: list[dict[str, str]] = []
        stale_rounds = 0
        start_naive = start.replace(tzinfo=None)
        logger.info(
            "Stockbit Reports scrape start range=%s scrolls=%s url=%s",
            range_label,
            max_scrolls,
            target,
        )
        for scroll_i in range(max_scrolls):
            batch = page.evaluate(_EXTRACT_STREAM_POSTS_JS)
            if isinstance(batch, list):
                dom_posts.extend(item for item in batch if isinstance(item, dict))
            before = len(dom_posts)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1800)
            after_scroll = page.evaluate(_EXTRACT_STREAM_POSTS_JS)
            if isinstance(after_scroll, list):
                dom_posts.extend(item for item in after_scroll if isinstance(item, dict))
            if len(dom_posts) == before:
                stale_rounds += 1
            else:
                stale_rounds = 0

            if scroll_i % 5 == 0 or scroll_i == max_scrolls - 1:
                logger.info(
                    "Stockbit Reports scroll %s/%s raw=%s stale=%s",
                    scroll_i + 1,
                    max_scrolls,
                    len(dom_posts),
                    stale_rounds,
                )

            merged_now = _merge_stream_posts(api_posts + dom_posts)
            oldest: datetime | None = None
            for item in merged_now:
                when = _parse_stockbit_when(
                    item.get("date_raw") or item.get("datetime") or "", now
                )
                if when is None:
                    continue
                if oldest is None or when < oldest:
                    oldest = when
            if oldest is not None and oldest < start_naive:
                break
            if stale_rounds >= 3:
                break

        page.remove_listener("response", _on_response)

        merged = _merge_stream_posts(api_posts + dom_posts)
        filtered: list[tuple[datetime, dict[str, str]]] = []
        end_naive = end.replace(tzinfo=None)
        for item in merged:
            when = _parse_stockbit_when(item.get("date_raw") or item.get("datetime") or "", now)
            if when is None:
                continue
            if when.tzinfo is None:
                when_cmp = when
            else:
                when_cmp = when.astimezone(tz).replace(tzinfo=None)
            if start_naive <= when_cmp <= end_naive:
                filtered.append((when_cmp, item))

        filtered.sort(key=lambda x: x[0], reverse=True)
        logger.info(
            "Stockbit Reports scrape done posts=%s candidates=%s",
            len(filtered),
            len(merged),
        )

        if not filtered:
            debug = self._debug_dump("reports_empty")
            return (
                f"ERROR: tidak ada post Stockbit Reports untuk rentang {range_label}.\n"
                f"url={page.url}\n"
                f"raw_candidates={len(merged)}\n"
                f"scrolls={max_scrolls}\n"
                f"{debug}\n"
                "Pastikan sudah login (/stockbit). Coba STOCKBIT_HEADLESS=false jika perlu."
            )

        lines = [
            "# Stockbit Reports — stream scrape",
            "",
            f"Source: {page.url}",
            f"Window: {range_label}",
            f"Scraped: {now.isoformat()}",
            f"Posts: {len(filtered)}",
            "",
        ]
        for when, item in filtered:
            text = (item.get("text") or "").strip()
            text = re.sub(r"\n{3,}", "\n\n", text)
            if len(text) > 5000:
                text = text[:5000] + "\n... (truncated)"
            href = (item.get("href") or "").strip()
            lines.append(f"## {when.strftime('%Y-%m-%d %H:%M')} WIB")
            if href:
                lines.append(f"link: {href}")
            lines.append("")
            lines.append(text or "(kosong)")
            lines.append("")

        report = "\n".join(lines)
        out_dir = Path(__file__).resolve().parent.parent / "exports" / "stockbit"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y%m%d_%H%M%S")
        if date_from or date_to:
            tag = f"{start.date().isoformat()}_{end.date().isoformat()}"
            path = out_dir / f"StockbitReports_{tag}_{stamp}.md"
        else:
            path = out_dir / f"StockbitReports_{int(days or 10)}d_{stamp}.md"
        path.write_text(report, encoding="utf-8")

        preview = report if len(report) <= 10000 else report[:10000] + "\n... (truncated)"
        return (
            f"OK: scrape Stockbit Reports ({len(filtered)} post, {range_label})\n"
            f"FILE: {path}\n\n{preview}"
        )

    def _extract_main_text(self) -> str:
        page = self._page_or_raise()
        for sel in (
            "main",
            "[role='main']",
            ".ant-layout-content",
            "#root",
            "body",
        ):
            try:
                loc = page.locator(sel).first
                if loc.count() > 0:
                    text = loc.inner_text(timeout=5000)
                    if text and len(text.strip()) > 40:
                        return re.sub(r"\n{3,}", "\n\n", text).strip()
            except Exception:  # noqa: BLE001
                continue
        try:
            return re.sub(r"\n{3,}", "\n\n", page.inner_text("body")).strip()
        except Exception as exc:  # noqa: BLE001
            return f"(gagal extract: {exc})"


_EXTRACT_STREAM_POSTS_JS = """
() => {
  const results = [];
  const seen = new Set();
  const add = (dateRaw, text, href) => {
    const t = (text || '').trim();
    if (t.length < 25) return;
    const key = (href || '') + '|' + t.slice(0, 160);
    if (seen.has(key)) return;
    seen.add(key);
    results.push({
      date_raw: (dateRaw || '').trim(),
      text: t.slice(0, 6000),
      href: href || '',
    });
  };

  document.querySelectorAll('time[datetime]').forEach((t) => {
    const dt = t.getAttribute('datetime') || t.innerText || '';
    const root = t.closest(
      'article, li, [class*="stream"], [class*="Stream"], [class*="post"], '
      + '[class*="Post"], [class*="activity"], [class*="Activity"], [data-testid]'
    ) || t.parentElement?.parentElement;
    if (!root) return;
    const link = root.querySelector(
      'a[href*="/post/"], a[href*="/stream/"], a[href*="StockbitReports"]'
    );
    add(dt, root.innerText, link ? link.href : '');
  });

  document.querySelectorAll(
    'a[href*="/post/"], a[href*="/stream/"], a[href*="/status/"]'
  ).forEach((a) => {
    const root = a.closest(
      'article, li, [class*="stream"], [class*="Stream"], [class*="post"], [class*="Post"]'
    ) || a.parentElement?.parentElement;
    if (!root) return;
    const timeEl = root.querySelector('time');
    const dateRaw = timeEl
      ? (timeEl.getAttribute('datetime') || timeEl.innerText || '')
      : '';
    add(dateRaw, root.innerText, a.href);
  });

  return results;
}
"""


def _walk_json_for_posts(obj, out: list[dict[str, str]], depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(obj, dict):
        text = None
        for key in ("content", "body", "text", "message", "description", "title"):
            val = obj.get(key)
            if isinstance(val, str) and len(val.strip()) > 25:
                text = val.strip()
                break
        ts = None
        for key in (
            "created_at",
            "createdAt",
            "posted_at",
            "postedAt",
            "date",
            "timestamp",
            "time",
            "published_at",
            "publishedAt",
        ):
            val = obj.get(key)
            if val is not None and str(val).strip():
                ts = str(val).strip()
                break
        if text and ts:
            href = ""
            for key in ("url", "link", "permalink", "post_url", "postUrl"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    href = val.strip()
                    break
            out.append({"date_raw": ts, "text": text, "href": href})
        for val in obj.values():
            _walk_json_for_posts(val, out, depth + 1)
    elif isinstance(obj, list):
        for val in obj:
            _walk_json_for_posts(val, out, depth + 1)


def _merge_stream_posts(items: list[dict]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if len(text) < 25:
            continue
        href = str(raw.get("href") or "").strip()
        key = href or text[:160]
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "date_raw": str(raw.get("date_raw") or raw.get("datetime") or "").strip(),
                "text": text,
                "href": href,
            }
        )
    return merged


def _parse_stockbit_when(raw: str, now) -> "datetime | None":
    from datetime import datetime, timedelta

    if not raw:
        return None
    raw = raw.strip()
    try:
        if "T" in raw or re.match(r"\d{4}-\d{2}-\d{2}", raw):
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                return dt.astimezone(now.tzinfo).replace(tzinfo=None)
            return dt
    except ValueError:
        pass

    if raw.isdigit() and len(raw) >= 10:
        try:
            ts = int(raw[:10])
            return datetime.fromtimestamp(ts, tz=now.tzinfo).replace(tzinfo=None)
        except (ValueError, OSError):
            pass

    low = raw.lower()
    if "baru saja" in low or "just now" in low:
        return now.replace(tzinfo=None)
    if "kemarin" in low or "yesterday" in low:
        return (now - timedelta(days=1)).replace(tzinfo=None)

    rel = re.search(
        r"(\d+)\s*(detik|menit|men|jam|hr?|hari|day|minggu|week|bulan|month|tahun|year)",
        low,
    )
    if rel:
        n = int(rel.group(1))
        unit = rel.group(2)
        if unit.startswith("detik"):
            delta = timedelta(seconds=n)
        elif unit.startswith("men"):
            delta = timedelta(minutes=n)
        elif unit.startswith("jam"):
            delta = timedelta(hours=n)
        elif unit.startswith("h") or unit.startswith("day"):
            delta = timedelta(days=n)
        elif unit.startswith("minggu") or unit.startswith("week"):
            delta = timedelta(weeks=n)
        elif unit.startswith("bulan") or unit.startswith("month"):
            delta = timedelta(days=n * 30)
        else:
            delta = timedelta(days=n * 365)
        return (now - delta).replace(tzinfo=None)

    for fmt in ("%d %b %Y", "%d %B %Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw[:24].strip(), fmt)
        except ValueError:
            continue
    return None


def _parse_iso_date(raw: str, tz) -> "datetime | None":
    from datetime import datetime

    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(raw[:10], fmt)
            return dt.replace(tzinfo=tz)
        except ValueError:
            continue
    return None


def _clean_scrape_text(text: str) -> str:
    lines = []
    skip_prefixes = (
        "cookie",
        "download app",
        "install app",
    )
    for line in text.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        low = s.lower()
        if any(low.startswith(p) for p in skip_prefixes):
            continue
        lines.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


_BROWSER: StockbitBrowser | None = None


def run_sync(fn, *args, **kwargs):
    """Run Playwright work on the dedicated browser thread."""
    return _BROWSER_EXECUTOR.submit(lambda: fn(*args, **kwargs)).result(timeout=900)


def get_browser(headless: bool = True) -> StockbitBrowser:
    global _BROWSER
    if _BROWSER is None:
        _BROWSER = StockbitBrowser(headless=headless)
    return _BROWSER


def shutdown_browser() -> None:
    import threading

    def _close() -> None:
        global _BROWSER
        if _BROWSER is not None:
            _BROWSER.close()
            _BROWSER = None

    if _BROWSER is None:
        return
    if threading.current_thread().name.startswith("stockbit-pw"):
        _close()
    else:
        run_sync(_close)
