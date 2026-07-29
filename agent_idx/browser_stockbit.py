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
    "stockbit.io",
)
LOGIN_URL = "https://stockbit.com/login"
HOME_URL = "https://stockbit.com/"
REPORTS_URL = "https://stockbit.com/StockbitReports?source=0"
# Official @Stockbit profile — often posts market notes + mid-day foreign flow.
OFFICIAL_URL = "https://stockbit.com/Stockbit"
STREAM_SOURCES = {
    "reports": REPORTS_URL,
    "stockbitreports": REPORTS_URL,
    "official": OFFICIAL_URL,
    "stockbit": OFFICIAL_URL,
}


def stream_source_slug(url: str) -> str:
    """Map profile URL → filename/chroma-friendly slug."""
    low = (url or "").lower()
    if "stockbitreports" in low:
        return "StockbitReports"
    if re.search(r"stockbit\.com/stockbit(?:/|\?|$)", low):
        return "Stockbit"
    m = re.search(r"stockbit\.com/([^/?#]+)", low)
    if m:
        return re.sub(r"[^A-Za-z0-9_-]+", "", m.group(1))[:40] or "StockbitStream"
    return "StockbitStream"


def resolve_stream_url(url: str | None = None, source: str | None = None) -> str:
    """Resolve tool args → absolute Stockbit stream/profile URL."""
    if url and str(url).strip():
        raw = str(url).strip()
        if not raw.startswith("http"):
            raw = "https://" + raw
        return raw
    key = (source or "reports").strip().lower()
    return STREAM_SOURCES.get(key, REPORTS_URL)


SYMBOL_PAGE_SLUGS = {
    "overview": "",
    "chart": "chartbit",
    "chartbit": "chartbit",
    "keystats": "keystats",
    "financials": "financials",
    # Stockbit UI uses /profile (not /company) for company background.
    "company": "profile",
    "profile": "profile",
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

# Fields that only ever appear on a real OTP screen.
_OTP_STRONG_SELECTORS = [
    'input[autocomplete="one-time-code"]',
    'input[name*="otp" i]',
    'input[id*="otp" i]',
    'input[placeholder*="kode" i]',
    'input[placeholder*="otp" i]',
    'input[placeholder*="verifikasi" i]',
]
# Fields an ordinary Stockbit page also matches, so they count only once the
# page is known to be an auth screen.
_OTP_WEAK_SELECTORS = [
    'input[inputmode="numeric"]',
    'input[name*="code" i]',
    'input[id*="code" i]',
]
_OTP_INPUT_SELECTORS = [
    *_OTP_STRONG_SELECTORS,
    *_OTP_WEAK_SELECTORS,
    'input[maxlength="1"]',
]

_AUTH_URL_MARKERS = ("/login", "/signin", "/otp", "/verify", "/2fa", "/mfa", "/challenge")
_AUTH_TEXT_SELECTORS = (
    "text=/one[- ]time (password|code)/i",
    "text=/verification code/i",
    "text=/kode (otp|verifikasi)/i",
    "text=/masukkan kode/i",
)


def _host_allowed(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    return any(host == s or host.endswith("." + s) for s in ALLOWED_HOST_SUFFIXES)


def _is_login_url(url: str) -> bool:
    low = (url or "").lower()
    return any(x in low for x in ("/login", "/signin", "/register", "/otp", "/verify"))


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
    """Single Stockbit Chromium session, normally backed by a persistent profile."""

    def __init__(self, headless: bool = True, user_data_dir: str | None = None) -> None:
        self.headless = headless
        self.user_data_dir = user_data_dir
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    @property
    def ready(self) -> bool:
        if self._page is None:
            return False
        try:
            # Touch a cheap attribute; closed pages raise.
            _ = self._page.url
            return True
        except Exception:  # noqa: BLE001
            return False

    def _mark_dead(self) -> None:
        """Clear handles after TargetClosed / external browser kill."""
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def status(self) -> str:
        """Human-readable Stockbit session status for bot/agent."""
        mode = (
            f"persistent ({self.user_data_dir})"
            if self.user_data_dir
            else "ephemeral/guest-like (hilang saat bot restart)"
        )
        lines = [
            f"browser_ready={self.ready}",
            f"headless={self.headless}",
            f"mode={mode}",
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
            lines.append(
                "hint: login=unknown — buka /stream via /stockbit; "
                "jika Chartbit kosong coba STOCKBIT_HEADLESS=false"
            )
        return "\n".join(lines)

    def start(self) -> str:
        if self.ready:
            mode = "persistent" if self.user_data_dir else "ephemeral"
            return f"OK: browser already running ({mode} session)"
        # Stale handles after an external close cannot be cleaned up safely:
        # Playwright's sync objects may already be invalid. Drop only our
        # references; the next explicit start reuses this singleton.
        if self._page is not None or self._context is not None or self._playwright is not None:
            self._mark_dead()
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright belum terpasang. Jalankan: "
                "py -3 -m pip install playwright && py -3 -m playwright install chromium"
            ) from exc

        self._playwright = sync_playwright().start()
        ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
        if self.user_data_dir:
            # Persistent profile: Stockbit login survives across runs.
            Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)
            self._context = self._playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
                viewport={"width": 1400, "height": 900},
                locale="id-ID",
                user_agent=ua,
            )
            self._browser = None
            pages = self._context.pages
            self._page = pages[0] if pages else self._context.new_page()
        else:
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            self._context = self._browser.new_context(
                viewport={"width": 1400, "height": 900},
                locale="id-ID",
                user_agent=ua,
            )
            self._page = self._context.new_page()
        self._page.set_default_timeout(30_000)
        self._page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        if self.user_data_dir:
            return f"OK: Chrome started with persistent profile ({self.user_data_dir})"
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
        if not self.ready:
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

    def _in_auth_context(self) -> bool:
        """True when the page is an auth screen — URL or its own wording says so."""
        page = self._page_or_raise()
        url = (page.url or "").lower()
        if any(x in url for x in _AUTH_URL_MARKERS):
            return True
        for frame in self._frames():
            for sel in _AUTH_TEXT_SELECTORS:
                if _visible_count(frame.locator(sel)) > 0:
                    return True
        return False

    def _otp_single_input(self):
        for frame in self._frames():
            for sel in _OTP_STRONG_SELECTORS:
                loc = frame.locator(sel)
                if _visible_count(loc) > 0:
                    return frame, loc.first
        # Weak signals and the bare-input fallback match ordinary Stockbit pages
        # too (a keystats page renders exactly one visible input), so they only
        # count on a page that identifies itself as an auth screen.
        if not self._in_auth_context():
            return None, None
        for frame in self._frames():
            for sel in _OTP_WEAK_SELECTORS:
                loc = frame.locator(sel)
                if _visible_count(loc) > 0:
                    return frame, loc.first
        if not self._has_visible_password():
            for frame in self._frames():
                for sel in ('input[type="tel"]', 'input[type="text"]', "input"):
                    loc = frame.locator(sel)
                    if _visible_count(loc) == 1:
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

    def create_post(self, text: str) -> str:
        """Publish a short idea post on the logged-in Stockbit stream/profile."""
        body = (text or "").strip()
        if not body:
            return "ERROR: teks post kosong"
        if len(body) > 2000:
            return "ERROR: teks post terlalu panjang (max 2000 karakter)"

        page = self._page_or_raise()
        if self.needs_login():
            return (
                f"{NEED_CREDENTIALS}\n"
                "Login Stockbit diperlukan sebelum membuat post."
            )
        if self.needs_otp():
            return f"{NEED_OTP}\nHalaman kode verifikasi terdeteksi."

        page.goto("https://stockbit.com/stream", wait_until="domcontentloaded")
        page.wait_for_timeout(3500)
        if self.needs_login():
            return (
                f"{NEED_CREDENTIALS}\n"
                "Login Stockbit diperlukan sebelum membuat post."
            )

        # Focus compose ("Tulis ide kamu disini...")
        try:
            page.get_by_text("Tulis ide kamu disini...", exact=False).first.click(
                timeout=8000
            )
        except Exception as exc:  # noqa: BLE001
            return (
                "ERROR: kotak compose Stockbit tidak ditemukan "
                f"(Tulis ide kamu disini...). Detail: {exc}"
            )
        page.wait_for_timeout(500)

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
                    target.fill(body)
                except Exception:  # noqa: BLE001
                    page.keyboard.press("Control+A")
                    page.keyboard.type(body, delay=25)
                filled = True
                break
            except Exception:  # noqa: BLE001
                continue
        if not filled:
            page.keyboard.type(body, delay=25)

        page.wait_for_timeout(600)
        post_btn = page.locator(
            'form:has-text("Tulis ide") button:has-text("Post")'
        ).first
        if post_btn.count() == 0:
            post_btn = page.locator(
                'button.ant-btn-primary:has-text("Post")'
            ).first
        if post_btn.count() == 0:
            return "ERROR: tombol Post tidak ditemukan"

        try:
            disabled = post_btn.get_attribute("disabled")
            aria = post_btn.get_attribute("aria-disabled")
            if disabled is not None or aria == "true":
                post_btn.click(timeout=4000, force=True)
            else:
                post_btn.click(timeout=5000)
        except Exception:  # noqa: BLE001
            try:
                page.keyboard.press("Control+Enter")
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: gagal klik Post: {exc}"

        page.wait_for_timeout(4000)
        # Soft verify: text appears on stream / own profile
        on_stream = page.evaluate(
            """(expected) => {
              const t = (document.body.innerText || '');
              return t.includes(expected);
            }""",
            body,
        )
        profile_href = page.evaluate(
            """() => {
              const links = [...document.querySelectorAll('a[href]')];
              for (const a of links) {
                const label = (a.innerText || '').trim().toLowerCase();
                const href = a.getAttribute('href') || '';
                if (label === 'profile' && href.startsWith('/') && !href.includes('symbol'))
                  return href;
              }
              return null;
            }"""
        )
        profile_ok = False
        profile_url = ""
        if profile_href:
            profile_url = (
                f"https://stockbit.com{profile_href}"
                if profile_href.startswith("/")
                else profile_href
            )
            try:
                page.goto(profile_url, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)
                profile_ok = page.evaluate(
                    """(expected) => (document.body.innerText || '').includes(expected)""",
                    body,
                )
            except Exception:  # noqa: BLE001
                profile_ok = False

        preview = body if len(body) <= 120 else body[:117] + "..."
        if on_stream or profile_ok:
            where = profile_url or "https://stockbit.com/stream"
            return (
                f"OK: post Stockbit terkirim\n"
                f"teks={preview!r}\n"
                f"verify_stream={bool(on_stream)} verify_profile={bool(profile_ok)}\n"
                f"url={where}"
            )
        return (
            "WARN: tombol Post diklik tetapi teks belum terverifikasi di feed. "
            f"Cek manual di Stockbit. teks={preview!r}"
        )

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
        """Positive login check — avoid false OTP failures / crisp-cookie traps."""
        page = self._page_or_raise()
        url = (page.url or "").lower()
        if not _host_allowed(url):
            return False
        # Still on auth gates.
        if any(x in url for x in _AUTH_URL_MARKERS):
            if self._has_visible_password() or self._has_visible_otp_inputs():
                return False
        if self._has_visible_password() or self._has_visible_otp_inputs():
            return False

        # Strong UI signals (compose / logout).
        for frame in self._frames():
            for sel in (
                'text=/Tulis ide kamu disini/i',
                'textarea[placeholder*="Tulis ide" i]',
                'div[contenteditable="true"][data-placeholder*="Tulis" i]',
                'a[href*="logout" i]',
                'button:has-text("Keluar")',
                'a:has-text("Keluar")',
            ):
                if _visible_count(frame.locator(sel)) > 0:
                    return True

        # localStorage / sessionStorage JWT-style keys (Stockbit often stores auth here).
        try:
            keys = page.evaluate(
                """() => {
                  const out = [];
                  for (const store of [window.localStorage, window.sessionStorage]) {
                    try {
                      for (let i = 0; i < store.length; i++) {
                        const k = store.key(i) || '';
                        if (/access|refresh|auth|token|jwt|bearer|sb[_-]?token/i.test(k)) {
                          const v = store.getItem(k) || '';
                          if (v.length > 20) out.push(k);
                        }
                      }
                    } catch (e) {}
                  }
                  return out;
                }"""
            )
            if keys:
                return True
        except Exception:  # noqa: BLE001
            pass

        # Cookie heuristic: ignore analytics / crisp / youtube rollouts.
        try:
            cookies = self._context.cookies() if self._context else []
            skip = ("crisp", "rollout", "sentry", "ttcsid", "wzrk", "_ttp", "grecaptcha")
            for c in cookies:
                name = (c.get("name") or "").lower()
                if any(s in name for s in skip):
                    continue
                host = (c.get("domain") or "").lower().lstrip(".")
                if "stockbit" not in host:
                    continue
                val = c.get("value") or ""
                if len(val) < 16:
                    continue
                if any(k in name for k in ("token", "session", "auth", "access", "jwt", "refresh")):
                    return True
        except Exception:  # noqa: BLE001
            pass

        return False

    def _await_logged_in(self, timeout_sec: float = 25.0) -> bool:
        """Poll until compose/auth is visible; navigate to /stream to verify."""
        import time

        page = self._page_or_raise()
        deadline = time.monotonic() + timeout_sec
        stream = "https://stockbit.com/stream"
        tried_stream = False

        while time.monotonic() < deadline:
            if self.needs_otp():
                return False
            if self.is_logged_in():
                return True
            url = (page.url or "").lower()
            if ("login" in url or "signin" in url) and not tried_stream:
                try:
                    page.goto(stream, wait_until="domcontentloaded")
                    page.wait_for_timeout(2000)
                    tried_stream = True
                    continue
                except Exception:  # noqa: BLE001
                    pass
            page.wait_for_timeout(1000)

        if not tried_stream:
            try:
                page.goto(stream, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)
            except Exception:  # noqa: BLE001
                pass
        return self.is_logged_in() and not self.needs_otp() and not self.needs_login()

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

        if self._await_logged_in(timeout_sec=5):
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

        page.wait_for_timeout(4000)
        if self.needs_otp():
            return (
                f"{NEED_OTP}\n"
                "Password diterima. Stockbit meminta kode verifikasi (OTP/2FA).\n"
                "Kirim kode dari email/SMS/authenticator di chat."
            )
        if self._await_logged_in(timeout_sec=20):
            return f"OK: login berhasil\nurl={page.url}\n\n{self.snapshot(max_chars=3000)}"
        if self.needs_otp():
            return (
                f"{NEED_OTP}\n"
                "Stockbit meminta kode verifikasi (OTP/2FA).\n"
                "Kirim kode dari email/SMS/authenticator di chat."
            )
        if self.needs_login():
            return (
                "ERROR: login gagal (password salah atau captcha). "
                "Coba lagi; jika captcha, set STOCKBIT_HEADLESS=false."
            )
        debug = self._debug_dump("login_unverified")
        return (
            "ERROR: login belum terverifikasi (compose/stream auth tidak muncul).\n"
            "Kemungkinan: OTP tertunda, captcha, atau headless diblokir.\n"
            "Set STOCKBIT_HEADLESS=false di .env, restart bot, lalu /stockbit ulang.\n"
            f"url={page.url}\n{debug}"
        )

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

        page.wait_for_timeout(2000)
        if self._await_logged_in(timeout_sec=20):
            return f"OK: login berhasil\nurl={page.url}\n\n{self.snapshot(max_chars=3000)}"

        # Try home/stream — session cookie may already be set.
        try:
            page.goto("https://stockbit.com/stream", wait_until="domcontentloaded")
            page.wait_for_timeout(2500)
        except Exception:  # noqa: BLE001
            pass

        if self._await_logged_in(timeout_sec=10):
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
        debug = self._debug_dump("otp_unverified")
        return (
            "ERROR: OTP dikirim tetapi sesi belum terverifikasi.\n"
            "Set STOCKBIT_HEADLESS=false, restart bot, /stockbit ulang.\n"
            f"url={page.url}\n{debug}"
        )
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
        """Scrape Stockbit Sections for one Symbol and save a markdown report."""
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
            "chartbit": symbol_page_url(symbol, "chart"),
            "keystats": symbol_page_url(symbol, "keystats"),
            "financials": symbol_page_url(symbol, "financials"),
            "profile": symbol_page_url(symbol, "profile"),
            "bit": symbol_page_url(symbol, "bit"),
        }
        if sections:
            wanted = [s.strip().lower() for s in sections.split(",") if s.strip()]
        else:
            wanted = ["overview", "keystats", "financials", "profile"]
        # Normalize aliases so we don't scrape the same page twice.
        alias = {"company": "profile", "chart": "chartbit"}
        normalized: list[str] = []
        for name in wanted:
            name = alias.get(name, name)
            if name not in normalized:
                normalized.append(name)
        wanted = normalized

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
            if name == "keystats":
                text = self._scrape_keystats_deep()
                max_chars = 120000
            elif name == "profile":
                text = self._scrape_profile_deep()
                max_chars = 150000
            elif name == "financials":
                text = self._scrape_financials_deep()
                max_chars = 400000
            else:
                text = _clean_scrape_text(self._extract_main_text())
                max_chars = 40000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"
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
        source: str | None = None,
    ) -> str:
        """Scrape stream posts from a Stockbit profile URL.

        Defaults to @StockbitReports. Also supports official @Stockbit
        (https://stockbit.com/Stockbit) which often posts mid-day foreign flow.
        Pass source='official'|'reports' or an absolute url=.
        """
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        if self.needs_login():
            return (
                f"{NEED_CREDENTIALS}\n"
                "Login diperlukan sebelum scrape Stockbit stream."
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

        target = resolve_stream_url(url, source)
        slug = stream_source_slug(target)
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
            "Stockbit stream scrape start source=%s range=%s scrolls=%s url=%s",
            slug,
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
                    "Stockbit stream (%s) scroll %s/%s raw=%s stale=%s",
                    slug,
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
            "Stockbit stream scrape done source=%s posts=%s candidates=%s",
            slug,
            len(filtered),
            len(merged),
        )

        if not filtered:
            debug = self._debug_dump("reports_empty")
            return (
                f"ERROR: tidak ada post {slug} untuk rentang {range_label}.\n"
                f"url={page.url}\n"
                f"raw_candidates={len(merged)}\n"
                f"scrolls={max_scrolls}\n"
                f"{debug}\n"
                "Pastikan sudah login (/stockbit). Coba STOCKBIT_HEADLESS=false jika perlu."
            )

        lines = [
            f"# Stockbit stream scrape — {slug}",
            "",
            f"Source: {page.url}",
            f"Account: {slug}",
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
            path = out_dir / f"{slug}_{tag}_{stamp}.md"
        else:
            path = out_dir / f"{slug}_{int(days or 10)}d_{stamp}.md"
        path.write_text(report, encoding="utf-8")

        preview = report if len(report) <= 10000 else report[:10000] + "\n... (truncated)"
        return (
            f"OK: scrape {slug} ({len(filtered)} post, {range_label})\n"
            f"FILE: {path}\n\n{preview}"
        )

    def scrape_chart_candles(
        self,
        symbol: str,
        timeframes: list[str] | None = None,
        *,
        settle_ms: int = 2500,
        wait_login_sec: int = 0,
    ) -> dict:
        """Farm OHLCV from Chartbit by intercepting network JSON + switching TF.

        Returns a dict with keys: symbol, timeframes, intercepted_urls, error.
        Each timeframe value: {bars: [{ts, open, high, low, close, volume}, ...],
        source_urls: [...], n: int}.
        """
        symbol = (symbol or "").strip().upper()
        if not re.fullmatch(r"[A-Z]{3,5}", symbol):
            return {"symbol": symbol, "timeframes": {}, "intercepted_urls": [], "error": "invalid symbol"}

        tfs = [t.strip().upper() for t in (timeframes or ["1H", "1D", "1W"]) if t.strip()]
        page = self._page_or_raise()

        # Soft gate only: hard abort if OTP/password form is actually visible.
        if self._has_visible_password() and (
            "login" in (page.url or "").lower() or "signin" in (page.url or "").lower()
        ):
            return {
                "symbol": symbol,
                "timeframes": {},
                "intercepted_urls": [],
                "error": NEED_CREDENTIALS,
            }
        if self.needs_otp():
            return {
                "symbol": symbol,
                "timeframes": {},
                "intercepted_urls": [],
                "error": NEED_OTP,
            }

        captured: list[tuple[str, object]] = []
        urls: list[str] = []
        unauthorized: list[str] = []

        def _on_response(response) -> None:
            try:
                req_url = response.url
                low = req_url.lower()
                host = ""
                try:
                    host = (urlparse(req_url).hostname or "").lower()
                except ValueError:
                    return
                if not any(
                    host == s or host.endswith("." + s)
                    for s in ("stockbit.com", "stockbit.io")
                ):
                    return
                if "sentry" in host:
                    return
                if response.status in {401, 403}:
                    unauthorized.append(req_url)
                    return
                if response.status != 200:
                    return
                ctype = (response.headers.get("content-type") or "").lower()
                looks_chart = any(
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
                        "market",
                        "symbol",
                        "timeframe",
                        "resolution",
                        "intraday",
                        "aggregate",
                    )
                )
                if "json" not in ctype and not looks_chart:
                    return
                data = response.json()
                captured.append((req_url, data))
                urls.append(req_url)
            except Exception:  # noqa: BLE001
                return

        page.on("response", _on_response)
        result_tfs: dict[str, dict] = {}
        blank_chart = False
        try:
            target = symbol_page_url(symbol, "chart")
            # Navigate directly — avoid open()'s early NEED_CREDENTIALS abort so we
            # can still intercept chart APIs behind soft login walls / false positives.
            page.goto(target, wait_until="domcontentloaded")
            page.wait_for_timeout(max(settle_ms, 3500))
            final = page.url
            if not _host_allowed(final):
                page.goto(HOME_URL, wait_until="domcontentloaded")
                return {
                    "symbol": symbol,
                    "timeframes": {},
                    "intercepted_urls": list(urls),
                    "error": f"ERROR: navigasi keluar allowlist ({final})",
                }

            if wait_login_sec > 0 and _is_login_url(final):
                logger.info(
                    "Chartbit redirected to login — waiting %ss for manual login",
                    wait_login_sec,
                )
                import time as _time

                deadline = _time.monotonic() + wait_login_sec
                while _time.monotonic() < deadline:
                    _time.sleep(2)
                    cur = page.url
                    if not _is_login_url(cur):
                        break
                # Retry chart page once the login gate is cleared.
                if not _is_login_url(page.url):
                    page.goto(target, wait_until="domcontentloaded")
                    page.wait_for_timeout(settle_ms)
                final = page.url

            hard_login = _is_login_url(final) and self._has_visible_password()
            if hard_login:
                return {
                    "symbol": symbol,
                    "timeframes": {},
                    "intercepted_urls": list(urls),
                    "error": NEED_CREDENTIALS,
                }

            self._dismiss_blocking_modals()
            page.wait_for_timeout(settle_ms)

            # Blank Chartbit shell (no canvas / no text) usually means expired auth.
            try:
                body_len = len((page.inner_text("body") or "").strip())
                canvas_n = page.locator("canvas").count()
            except Exception:  # noqa: BLE001
                body_len, canvas_n = 0, 0
            if body_len < 40 and canvas_n == 0:
                page.wait_for_timeout(4000)
                try:
                    body_len = len((page.inner_text("body") or "").strip())
                    canvas_n = page.locator("canvas").count()
                except Exception:  # noqa: BLE001
                    body_len, canvas_n = 0, 0
            blank_chart = body_len < 40 and canvas_n == 0

            for tf in tfs:
                labels = _CHART_TF_LABELS.get(tf, [tf])
                before = len(captured)
                self._set_chart_resolution(labels)
                page.wait_for_timeout(settle_ms)
                batch = captured[before:]
                bars, source_urls = _pick_best_bars(batch if batch else captured, symbol)
                if not bars:
                    bars, source_urls = _pick_best_bars(captured, symbol)
                result_tfs[tf] = {
                    "bars": bars,
                    "source_urls": source_urls,
                    "n": len(bars),
                }
                logger.info(
                    "Chart farm %s %s bars=%s urls=%s",
                    symbol,
                    tf,
                    len(bars),
                    len(source_urls),
                )
        finally:
            try:
                page.remove_listener("response", _on_response)
            except Exception:  # noqa: BLE001
                pass

        any_bars = any((v.get("n") or 0) > 0 for v in result_tfs.values())
        err = None
        if not any_bars:
            logged = False
            try:
                logged = self.is_logged_in()
            except Exception:  # noqa: BLE001
                logged = False
            if (unauthorized or blank_chart) and not logged:
                err = NEED_CREDENTIALS
            elif self._has_visible_password():
                err = NEED_CREDENTIALS
            elif self.needs_otp():
                err = NEED_OTP
            elif blank_chart and logged:
                err = "CHARTBIT_BLANK_WHILE_LOGGED_IN"
            else:
                err = "NO_OHLCV_INTERCEPTED"

        return {
            "symbol": symbol,
            "timeframes": result_tfs,
            "intercepted_urls": list(dict.fromkeys(urls)),
            "unauthorized_urls": list(dict.fromkeys(unauthorized))[:10],
            "error": err,
        }

    def _set_chart_resolution(self, labels: list[str]) -> bool:
        """Try TradingView widget API, then click resolution UI labels."""
        page = self._page_or_raise()
        for label in labels:
            try:
                ok = page.evaluate(
                    """(res) => {
                        try {
                            const w = window.tvWidget || window.chartWidget || window.__chartWidget;
                            if (w && typeof w.activeChart === 'function') {
                                w.activeChart().setResolution(String(res));
                                return true;
                            }
                            if (w && w.chart && typeof w.chart === 'function') {
                                w.chart().setResolution(String(res));
                                return true;
                            }
                        } catch (e) {}
                        return false;
                    }""",
                    label,
                )
                if ok:
                    return True
            except Exception:  # noqa: BLE001
                pass

        for label in labels:
            selectors = [
                f'button:has-text("{label}")',
                f'div[role="button"]:has-text("{label}")',
                f'[data-value="{label}"]',
                f'[data-name="resolution"] :text-is("{label}")',
                f'text="{label}"',
            ]
            for sel in selectors:
                try:
                    loc = page.locator(sel).first
                    if loc.count() == 0:
                        continue
                    if not loc.is_visible():
                        continue
                    loc.click(timeout=2000)
                    return True
                except Exception:  # noqa: BLE001
                    continue
        return False

    def _scroll_page_deep(self, rounds: int = 10) -> None:
        page = self._page_or_raise()
        for _ in range(max(1, rounds)):
            page.evaluate("window.scrollBy(0, Math.max(600, window.innerHeight * 0.9))")
            page.wait_for_timeout(350)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(400)

    def _click_exact_label(self, label: str) -> bool:
        """Click a visible Net Income / EPS / Revenue (or similar) tab/button."""
        page = self._page_or_raise()
        # Prefer role=tab / button exact text.
        for role in ("tab", "button"):
            try:
                loc = page.get_by_role(role, name=label, exact=True)
                if loc.count() > 0:
                    target = loc.first
                    if target.is_visible():
                        target.click(timeout=2500)
                        return True
            except Exception:  # noqa: BLE001
                pass
        try:
            loc = page.get_by_text(label, exact=True)
            n = min(loc.count(), 8)
            for i in range(n):
                el = loc.nth(i)
                try:
                    if el.is_visible():
                        el.click(timeout=2500)
                        return True
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            return False
        return False

    def _click_any_label(self, labels: tuple[str, ...]) -> str:
        """Click the first label that exists; return the clicked label ('' if none)."""
        for label in labels:
            if self._click_exact_label(label):
                return label
        return ""

    def _select_option_by_label(self, labels: tuple[str, ...]) -> str:
        """Choose an option in a native <select>; return the chosen label.

        The Financials statement and period switches are `<select>` dropdowns,
        so their options can never be clicked — they must be selected.
        """
        page = self._page_or_raise()
        try:
            selects = page.locator("select")
            count = min(selects.count(), 12)
        except Exception:  # noqa: BLE001
            return ""
        for i in range(count):
            sel = selects.nth(i)
            try:
                options = sel.evaluate("s => [...s.options].map(o => o.label || o.text)")
            except Exception:  # noqa: BLE001
                continue
            available = {str(o).strip() for o in options or ()}
            for label in labels:
                if label not in available:
                    continue
                try:
                    sel.select_option(label=label)
                    return label
                except Exception:  # noqa: BLE001
                    break
        return ""

    def _switch_to(self, labels: tuple[str, ...]) -> str:
        """Activate a control by label, whether it is a dropdown or a tab."""
        return self._select_option_by_label(labels) or self._click_any_label(labels)

    def _scrape_financials_deep(self) -> str:
        """Capture Financials: statements x period toggles + Key Ratio sub-tabs."""
        page = self._page_or_raise()
        self._scroll_page_deep(6)

        lines: list[str] = ["### Financials structured", ""]

        header = page.evaluate(_EXTRACT_FIN_HEADER_JS)
        if isinstance(header, dict) and header:
            unit = str(header.get("unit") or "").strip()
            basis = str(header.get("basis") or "").strip()
            if unit or basis:
                lines.append("### Financials meta")
                lines.append("")
                if unit:
                    lines.append(f"Unit: {unit}")
                if basis:
                    lines.append(f"Basis: {basis}")
                lines.append("")

        statements = (
            ("Income Statement", ("Income Statement", "Laba Rugi")),
            ("Balance Sheet", ("Balance Sheet", "Neraca")),
            ("Cash Flow", ("Cash Flow", "Cash Flow Statement", "Arus Kas")),
        )
        periods = (
            ("Quarterly", ("Quarterly", "Kuartalan")),
            ("Annual", ("Annual", "Yearly", "Tahunan")),
            ("TTM", ("TTM",)),
        )

        for stmt_name, stmt_labels in statements:
            stmt_clicked = self._switch_to(stmt_labels)
            page.wait_for_timeout(1600 if stmt_clicked else 500)
            self._scroll_page_deep(4)

            for period_name, period_labels in periods:
                period_clicked = self._switch_to(period_labels)
                # Skip periods the UI doesn't offer to avoid duplicate dumps.
                if not period_clicked and period_name != "Quarterly":
                    continue
                page.wait_for_timeout(1600 if period_clicked else 600)
                self._scroll_page_deep(4)
                tables = page.evaluate(_EXTRACT_FINANCIAL_TABLES_JS)
                lines.append(f"### Statement — {stmt_name} ({period_name})")
                lines.append(
                    f"clicked_statement={bool(stmt_clicked)} "
                    f"clicked_period={bool(period_clicked)}"
                )
                lines.append("")
                lines.append(
                    tables.strip()
                    if isinstance(tables, str) and tables.strip()
                    else "(no table)"
                )
                lines.append("")

        # Key Ratio sub-tabs (Financial Health / Efficiency / etc.)
        ratio_tabs = (
            "Financial Health",
            "Efficiency",
            "Profitability",
            "Growth",
            "Valuation",
            "Per Share",
            "Key Ratio",
        )
        for tab in ratio_tabs:
            clicked = self._click_exact_label(tab)
            if not clicked:
                continue
            page.wait_for_timeout(1400)
            self._scroll_page_deep(4)
            tables = page.evaluate(_EXTRACT_FINANCIAL_TABLES_JS)
            lines.append(f"### Key Ratio — {tab}")
            lines.append("")
            lines.append(
                tables.strip()
                if isinstance(tables, str) and tables.strip()
                else "(no table)"
            )
            lines.append("")

        raw = _clean_scrape_text(self._extract_main_text())
        raw = _trim_profile_noise(raw)
        lines.append("### Raw page text")
        lines.append("")
        lines.append(raw[:80000] if raw else "(empty)")
        return "\n".join(lines)

    def _scrape_keystats_deep(self) -> str:
        """Capture Key Stats cards + Net Income/EPS/Revenue series tables."""
        page = self._page_or_raise()
        self._scroll_page_deep(12)
        pairs = page.evaluate(_EXTRACT_KEYSTATS_PAIRS_JS)
        if not isinstance(pairs, list):
            pairs = []

        lines: list[str] = [
            "### Metrics (structured)",
            "",
        ]
        for item in pairs:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            value = str(item.get("value") or "").strip()
            group = str(item.get("group") or "").strip()
            if not label or not value:
                continue
            prefix = f"[{group}] " if group else ""
            lines.append(f"{prefix}{label}: {value}")

        # Clickable series: Net Income / EPS / Revenue
        for tab in ("Net Income", "EPS", "Revenue"):
            clicked = self._click_exact_label(tab)
            page.wait_for_timeout(1400 if clicked else 400)
            table_md = page.evaluate(_EXTRACT_VISIBLE_TABLES_JS)
            lines.append("")
            lines.append(f"### Series — {tab}")
            lines.append(f"clicked={clicked}")
            lines.append("")
            lines.append(table_md.strip() if isinstance(table_md, str) and table_md.strip() else "(no table)")

        # Dividend history + other leftover tables already in DOM
        extra_tables = page.evaluate(_EXTRACT_VISIBLE_TABLES_JS)
        if isinstance(extra_tables, str) and extra_tables.strip():
            lines.append("")
            lines.append("### Tables (visible)")
            lines.append("")
            lines.append(extra_tables.strip())

        raw = _clean_scrape_text(self._extract_main_text())
        lines.append("")
        lines.append("### Raw page text")
        lines.append("")
        # Keep raw as fallback for anything the card extractor missed.
        lines.append(raw[:50000] if raw else "(empty)")
        return "\n".join(lines)

    def _expand_see_more(self) -> None:
        """Click common Stockbit 'see more' / expand controls on profile."""
        page = self._page_or_raise()
        labels = (
            "See more",
            "See More",
            "Show more",
            "Selengkapnya",
            "Lihat selengkapnya",
            "Load more",
            "Tampilkan lebih banyak",
        )
        for label in labels:
            try:
                loc = page.get_by_text(label, exact=False)
                n = min(loc.count(), 6)
                for i in range(n):
                    el = loc.nth(i)
                    try:
                        if el.is_visible():
                            el.click(timeout=1500)
                            page.wait_for_timeout(400)
                    except Exception:  # noqa: BLE001
                        continue
            except Exception:  # noqa: BLE001
                continue

    def _scrape_profile_deep(self) -> str:
        """Capture full Stockbit /profile: background, shareholders, BOD, subsidiaries."""
        page = self._page_or_raise()
        self._scroll_page_deep(14)
        self._expand_see_more()
        self._scroll_page_deep(6)

        # Optional: click common profile sub-tabs / filters if present.
        for tab in (
            "Shareholder > 1%",
            "Shareholders",
            "Shareholder Composition",
            "Holding Composition",
            "Board of Directors",
            "Board of Commissioners",
            "Subsidiary Companies",
            "Company History",
            "Number of Shareholders",
            "Ultimate Beneficiary Owner",
            "All",
            "Local",
            "Foreign",
        ):
            try:
                loc = page.get_by_text(tab, exact=True)
                if loc.count() > 0 and loc.first.is_visible():
                    # Don't force-click every header; only role=tab / button-like.
                    pass
            except Exception:  # noqa: BLE001
                pass

        structured = page.evaluate(_EXTRACT_PROFILE_JS)
        if not isinstance(structured, dict):
            structured = {}

        lines: list[str] = ["### Profile structured", ""]

        bg = str(structured.get("background") or "").strip()
        if bg:
            lines.append("### Company Background")
            lines.append("")
            lines.append(bg)
            lines.append("")

        info = structured.get("info") or []
        if isinstance(info, list) and info:
            lines.append("### Company Info")
            lines.append("")
            for item in info:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").strip()
                value = str(item.get("value") or "").strip()
                if label and value:
                    lines.append(f"{label}: {value}")
            lines.append("")

        for key, title in (
            ("shareholders_gt1", "Shareholder > 1%"),
            ("shareholder_composition", "Shareholder Composition"),
            ("holding_composition", "Holding Composition"),
            ("shareholders_simple", "Shareholders"),
            ("directors_ownership", "Directors and Commissioners Ownership"),
            ("number_of_shareholders", "Number of Shareholders"),
            ("subsidiaries", "Subsidiary Companies"),
            ("other_tables", "Other Tables"),
        ):
            md = structured.get(key)
            if isinstance(md, str) and md.strip():
                lines.append(f"### {title}")
                lines.append("")
                lines.append(md.strip())
                lines.append("")

        ubo = structured.get("ubo") or []
        if isinstance(ubo, list) and ubo:
            lines.append("### Ultimate Beneficiary Owner")
            lines.append("")
            for name in ubo:
                n = str(name or "").strip()
                if n:
                    lines.append(f"- {n}")
            lines.append("")

        for key, title in (
            ("board_directors", "Board of Directors"),
            ("board_commissioners", "Board of Commissioners"),
        ):
            people = structured.get(key) or []
            if isinstance(people, list) and people:
                lines.append(f"### {title}")
                lines.append("")
                for person in people:
                    if isinstance(person, dict):
                        role = str(person.get("role") or "").strip()
                        name = str(person.get("name") or "").strip()
                        if role and name:
                            lines.append(f"{role}: {name}")
                        elif name:
                            lines.append(f"- {name}")
                    elif str(person).strip():
                        lines.append(f"- {str(person).strip()}")
                lines.append("")

        addr = structured.get("address") or {}
        if isinstance(addr, dict) and any(addr.values()):
            lines.append("### Address")
            lines.append("")
            for k in ("street", "phone", "fax", "npwp", "email", "website"):
                v = str(addr.get(k) or "").strip()
                if v:
                    lines.append(f"{k.capitalize()}: {v}")
            lines.append("")

        raw = _clean_scrape_text(self._extract_main_text())
        raw = _trim_profile_noise(raw)
        lines.append("### Raw page text")
        lines.append("")
        lines.append(raw[:70000] if raw else "(empty)")
        return "\n".join(lines)

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


_EXTRACT_KEYSTATS_PAIRS_JS = """
() => {
  const pairs = [];
  const seen = new Set();
  const skipLabels = new Set([
    'stream','profile','watchlist','portfolio','orderbook','chartbit','e-ipo',
    'screener','broker analysis','academy','insider activity','valuation',
    'financials','fundachart','broker activity','seasonality','sector',
    'calendar','ktur','earnings','glossary','ebook','faqs','start trading',
    'analysis','corp. action','comparison','key stats','today'
  ]);
  const add = (label, value, group) => {
    label = String(label || '').replace(/\\s+/g, ' ').trim();
    value = String(value || '').replace(/\\s+/g, ' ').trim();
    group = String(group || '').replace(/\\s+/g, ' ').trim();
    if (!label || !value) return;
    if (label.length < 2 || label.length > 90) return;
    if (value.length > 80) return;
    const low = label.toLowerCase();
    if (skipLabels.has(low)) return;
    if (/^(net income|eps|revenue|period|q[1-4]|annualised|ttm)$/i.test(label)) return;
    // values are usually numeric / percent / dash / date-ish
    if (!/[0-9%]/.test(value) && value !== '-') return;
    if (!/[A-Za-z(]/.test(label)) return;
    const key = (group + '|' + label + '|' + value).toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    pairs.push({label, value, group});
  };

  // Card-like nodes with 2 text children (label + value)
  const nodes = Array.from(document.querySelectorAll('div, li, section, article'));
  for (const n of nodes) {
    if (!n.children || n.children.length < 2 || n.children.length > 3) continue;
    const kids = Array.from(n.children).map(c => (c.innerText || '').trim()).filter(Boolean);
    if (kids.length === 2 && !kids[0].includes('\\n') && !kids[1].includes('\\n')) {
      add(kids[0], kids[1], '');
    }
  }

  // Alternate-line text inside main content (Stockbit often renders label\\nvalue)
  const root = document.querySelector('main') || document.querySelector('[role=main]') || document.body;
  const lines = (root.innerText || '').split('\\n').map(s => s.trim()).filter(s => s.length > 0);
  const sectionHeaders = new Set([
    'Current Valuation','Per Share','Solvency','Management Effectiveness',
    'Profitability','Growth','Dividend','Market Rank','Income Statement',
    'Balance Sheet','Cash Flow Statement','Price Performance','Dividend History'
  ]);
  let group = '';
  for (let i = 0; i < lines.length - 1; i++) {
    const a = lines[i];
    const b = lines[i + 1];
    if (sectionHeaders.has(a)) { group = a; continue; }
    if (sectionHeaders.has(b)) continue;
    if (/^[A-Za-z(].{1,80}$/.test(a) && /^(?:[-−]|\\(?[0-9].*|\\d{1,2}\\s+[A-Za-z]{3}\\s+\\d{2})$/.test(b)) {
      // Avoid treating table headers / years as labels
      if (/^(Period|Q[1-4]|Annualised|TTM|Div\\b)/i.test(a)) continue;
      if (/^20\\d{2}$/.test(a)) continue;
      add(a, b, group);
      i++; // consume value line
    }
  }
  return pairs;
}
"""

_EXTRACT_VISIBLE_TABLES_JS = """
() => {
  const tables = Array.from(document.querySelectorAll('table'));
  const chunks = [];
  for (const table of tables) {
    const rect = table.getBoundingClientRect();
    if (rect.width < 40 || rect.height < 20) continue;
    const rows = Array.from(table.querySelectorAll('tr')).map(tr =>
      Array.from(tr.querySelectorAll('th,td')).map(c => (c.innerText || '').replace(/\\s+/g, ' ').trim())
    ).filter(r => r.some(Boolean));
    if (rows.length < 2) continue;
    const width = Math.max(...rows.map(r => r.length));
    if (width < 2) continue;
    const md = rows.map(r => {
      const cells = r.slice(0, width);
      while (cells.length < width) cells.push('');
      return '| ' + cells.join(' | ') + ' |';
    });
    if (md.length) {
      const sep = '| ' + Array(width).fill('---').join(' | ') + ' |';
      chunks.push([md[0], sep, ...md.slice(1)].join('\\n'));
    }
  }
  // Dedup identical tables
  return Array.from(new Set(chunks)).join('\\n\\n');
}
"""

_EXTRACT_FIN_HEADER_JS = """
() => {
  const root = document.querySelector('main') || document.body;
  const text = (root.innerText || '').split('\\n').map(s => s.trim()).filter(Boolean);
  const out = {unit: '', basis: ''};
  for (const line of text.slice(0, 120)) {
    if (!out.unit && /^Dalam\\s+.+|^In\\s+(Millions|Billions|Thousands)/i.test(line)) {
      out.unit = line.split('\\t')[0];
    }
    if (!out.basis && /^(As Reported|Standardized|Disajikan)/i.test(line)) {
      out.basis = line;
    }
  }
  return out;
}
"""

_EXTRACT_FINANCIAL_TABLES_JS = """
() => {
  const chunks = [];
  const clean = c => (c || '').replace(/\\s+/g, ' ').trim();
  const variants = [];

  const collect = (rows) => {
    rows = rows.filter(r => r.some(c => c && c.length));
    if (rows.length < 2) return;
    const width = Math.max(...rows.map(r => r.length));
    if (width < 2) return;
    variants.push(rows.map(r => {
      const cells = r.slice(0, width);
      while (cells.length < width) cells.push('');
      return cells;
    }));
  };

  // 1) Real HTML tables
  for (const table of document.querySelectorAll('table')) {
    const rect = table.getBoundingClientRect();
    if (rect.width < 40) continue;
    collect(Array.from(table.querySelectorAll('tr')).map(tr =>
      Array.from(tr.querySelectorAll('th,td')).map(c => clean(c.innerText))
    ));
  }

  // 2) ARIA grids / virtualized financial tables
  for (const grid of document.querySelectorAll('[role=table], [role=grid], [role=treegrid]')) {
    const rowEls = grid.querySelectorAll('[role=row]');
    if (!rowEls.length) continue;
    collect(Array.from(rowEls).map(r =>
      Array.from(r.querySelectorAll('[role=cell], [role=columnheader], [role=rowheader], [role=gridcell]'))
        .map(c => clean(c.innerText))
    ));
  }

  // Stockbit renders each financial table twice: once with unrounded values and
  // no period header, once with formatted values and the period header. Group
  // the twins by their row labels so one emitted table carries both.
  // A header cell is a period label ("Q2 2026", "12M 2025", "2026", "TTM") —
  // not merely a non-empty cell, or a row of figures would pass as a header.
  const periodRe = /^(?:(?:Q[1-4]|\\d{1,2}M)\\s+)?(?:19|20)\\d{2}$|^(?:TTM|LTM)$/;
  const isHeader = row => row.slice(1).filter(c => periodRe.test(c)).length >= 2;
  const rawScore = rows => {
    let n = 0;
    for (const r of rows.slice(0, 10))
      for (const c of r.slice(1, 10))
        if (/^-?\\d{7,}(\\.\\d+)?$/.test(c)) n++;
    return n;
  };

  const groups = new Map();
  const headerByWidth = new Map();
  for (const rows of variants) {
    const header = isHeader(rows[0]) ? rows[0] : null;
    const body = header ? rows.slice(1) : rows;
    if (!body.length) continue;
    if (header && !headerByWidth.has(header.length)) {
      headerByWidth.set(header.length, header);
    }
    const key = body.map(r => r[0]).join('|');
    if (!groups.has(key)) groups.set(key, {header: null, bodies: []});
    const g = groups.get(key);
    if (header && !g.header) g.header = header;
    g.bodies.push(body);
  }

  for (const g of groups.values()) {
    const body = g.bodies.slice().sort((a, b) => rawScore(b) - rawScore(a))[0];
    // Every table in a given period mode shares the same columns, so a table
    // whose own twin was missed can still borrow a header of equal width.
    const header = g.header || headerByWidth.get(body[0].length) || null;
    const rows = header && header.length === body[0].length
      ? [header, ...body]
      : body;
    const width = rows[0].length;
    const md = rows.map(r => '| ' + r.map(c => c.replace(/\\|/g, '/')).join(' | ') + ' |');
    const sep = '| ' + Array(width).fill('---').join(' | ') + ' |';
    chunks.push([md[0], sep, ...md.slice(1)].join('\\n'));
  }

  // 3) Fallback: tab-separated innerText lines (Stockbit financial grid)
  const mdFromRows = (rows) => {
    rows = rows.filter(r => r.some(c => c && c.length));
    if (rows.length < 2) return '';
    const width = Math.max(...rows.map(r => r.length));
    if (width < 2) return '';
    const md = rows.map(r => {
      const cells = r.slice(0, width);
      while (cells.length < width) cells.push('');
      return '| ' + cells.map(c => c.replace(/\\|/g, '/')).join(' | ') + ' |';
    });
    const sep = '| ' + Array(width).fill('---').join(' | ') + ' |';
    return [md[0], sep, ...md.slice(1)].join('\\n');
  };

  if (!chunks.length) {
    const root = document.querySelector('main') || document.body;
    const lines = (root.innerText || '').split('\\n');
    let buf = [];
    const flush = () => {
      if (buf.length >= 2) {
        const md = mdFromRows(buf.map(l => l.split('\\t').map(c => c.trim())));
        if (md) chunks.push(md);
      }
      buf = [];
    };
    for (const line of lines) {
      if ((line.match(/\\t/g) || []).length >= 2) buf.push(line);
      else flush();
    }
    flush();
  }

  return Array.from(new Set(chunks)).join('\\n\\n');
}
"""

_EXTRACT_PROFILE_JS = """
() => {
  const root = document.querySelector('main') || document.querySelector('[role=main]') || document.body;
  const fullText = root.innerText || '';
  const lines = fullText.split('\\n').map(s => s.trim()).filter(Boolean);

  const headerIdx = (name) => {
    const low = String(name || '').toLowerCase();
    for (let i = 0; i < lines.length; i++) {
      const cur = lines[i].toLowerCase();
      if (cur === low || cur.startsWith(low)) return i;
    }
    return -1;
  };

  const sliceUntil = (startName, stopNames) => {
    const start = headerIdx(startName);
    if (start < 0) return [];
    const stops = stopNames.map(n => n.toLowerCase());
    const out = [];
    for (let i = start + 1; i < lines.length; i++) {
      const low = lines[i].toLowerCase();
      if (stops.some(s => low === s || low.startsWith(s))) break;
      out.push(lines[i]);
    }
    return out;
  };

  const tableToMd = (table) => {
    const rows = Array.from(table.querySelectorAll('tr')).map(tr =>
      Array.from(tr.querySelectorAll('th,td')).map(c => (c.innerText || '').replace(/\\s+/g, ' ').trim())
    ).filter(r => r.some(Boolean));
    if (rows.length < 2) return '';
    const width = Math.max(...rows.map(r => r.length));
    if (width < 2) return '';
    const md = rows.map(r => {
      const cells = r.slice(0, width);
      while (cells.length < width) cells.push('');
      return '| ' + cells.join(' | ') + ' |';
    });
    const sep = '| ' + Array(width).fill('---').join(' | ') + ' |';
    return [md[0], sep, ...md.slice(1)].join('\\n');
  };

  const nearestHeader = (el) => {
    let cur = el;
    for (let depth = 0; depth < 8 && cur; depth++) {
      let prev = cur.previousElementSibling;
      while (prev) {
        const t = (prev.innerText || '').trim().split('\\n')[0];
        if (t && t.length < 80) return t;
        prev = prev.previousElementSibling;
      }
      cur = cur.parentElement;
    }
    return '';
  };

  // --- Company Background ---
  let background = '';
  const bgLines = sliceUntil('Company Background', [
    'Shareholder > 1%', 'Shareholders', 'Shareholder Composition', 'Holding Composition',
    'Directors and Commissioners', 'Company History', 'Board of Directors',
    'Board of Commissioners', 'Address', 'Subsidiary Companies', 'Number of Shareholders',
    'Ultimate Beneficiary Owner', 'Stream', 'Key Stats'
  ]);
  // Keep long prose paragraphs, skip index tag soup if very dense
  background = bgLines
    .filter(l => l.length > 40 && !/^(Minyak|Syariah|DAYTRADE|Papan|IDX|IHSG|JII|LQ45)/i.test(l.replace(/\\s/g,'')))
    .join('\\n\\n');
  if (!background && bgLines.length) {
    background = bgLines.filter(l => l.length > 20).slice(0, 3).join('\\n\\n');
  }

  // --- Classify HTML tables by nearest header ---
  const buckets = {
    shareholders_gt1: [],
    shareholder_composition: [],
    holding_composition: [],
    shareholders_simple: [],
    directors_ownership: [],
    number_of_shareholders: [],
    subsidiaries: [],
    other_tables: [],
  };
  for (const table of root.querySelectorAll('table')) {
    const md = tableToMd(table);
    if (!md) continue;
    const hdr = (nearestHeader(table) || '').toLowerCase();
    let key = 'other_tables';
    if (hdr.includes('shareholder > 1') || hdr.includes('shareholder >1')) key = 'shareholders_gt1';
    else if (hdr.includes('shareholder composition')) key = 'shareholder_composition';
    else if (hdr.includes('holding composition')) key = 'holding_composition';
    else if (hdr.includes('directors and commissioners')) key = 'directors_ownership';
    else if (hdr.includes('number of shareholders')) key = 'number_of_shareholders';
    else if (hdr.includes('subsidiary')) key = 'subsidiaries';
    else if (hdr === 'shareholders' || hdr.startsWith('shareholders')) key = 'shareholders_simple';
    buckets[key].push(md);
  }

  // Fallback: rebuild Shareholder > 1% / composition from text blocks if tables empty
  const textTable = (startName, stopNames, minCols) => {
    const block = sliceUntil(startName, stopNames);
    if (block.length < 4) return '';
    // Heuristic: consecutive numeric-heavy lines after header row names
    return block.slice(0, 80).map(l => l).join('\\n');
  };

  // --- Company History as label:value ---
  const info = [];
  const hist = sliceUntil('Company History', [
    'Number of Shareholders', 'Board of Directors', 'Board of Commissioners',
    'Address', 'Subsidiary Companies', 'Ultimate Beneficiary Owner', 'Shareholders'
  ]);
  const histLabels = [
    'Listing Date', 'IPO Price', 'IPO Amount', 'IPO Shares', 'Free Float',
    'Underwriters', 'Administration Bureau', 'Listing Board'
  ];
  for (let i = 0; i < hist.length; i++) {
    const line = hist[i];
    for (const lab of histLabels) {
      if (line.toLowerCase() === lab.toLowerCase() && i + 1 < hist.length) {
        info.push({label: lab, value: hist[i + 1]});
      }
      // same-line "Listing Date\\t30 Jul 1990"
      const m = line.match(new RegExp('^' + lab + '\\s+(.+)$', 'i'));
      if (m) info.push({label: lab, value: m[1]});
    }
  }

  // --- UBO ---
  const uboBlock = sliceUntil('Ultimate Beneficiary Owner', [
    'Company History', 'Number of Shareholders', 'Board of Directors',
    'Board of Commissioners', 'Address', 'Subsidiary Companies'
  ]);
  const ubo = [];
  const seenUbo = new Set();
  for (const l of uboBlock) {
    if (/^(ultimate|nirwan|anthony|controller|director|commissioner|:)/i.test(l) || l.length < 3) {
      // keep person-like ALL CAPS / Title Case names
    }
    if (/^[A-Z][A-Z\\s.'-]{5,80}$/.test(l) || /^[A-Z][a-z]+(?:\\s+[A-Z][a-zA-Z.]+){1,5}$/.test(l)) {
      const key = l.toUpperCase();
      if (!seenUbo.has(key) && !/OWNER|CONTROLLER|DIRECTOR|COMMISSIONER/.test(key)) {
        seenUbo.add(key);
        ubo.push(l);
      }
    }
  }

  // --- Board lists ---
  const parseBoard = (startName, stopNames) => {
    const block = sliceUntil(startName, stopNames);
    const roles = [
      'President Director', 'Vice President Director', 'Director',
      'President Independent Commissioner', 'Independent Commissioner', 'Commissioner',
      'President Commissioner'
    ];
    const people = [];
    let role = '';
    for (const l of block) {
      const hit = roles.find(r => l.toLowerCase() === r.toLowerCase());
      if (hit) { role = hit; continue; }
      if (/^(commissioner|director)\\s*:/i.test(l)) continue;
      if (l.length < 3 || l.length > 90) continue;
      if (/^[A-Z0-9]/.test(l) && !/^(Updated|Company|Business|Location|Phone|Fax|Email|Website)/i.test(l)) {
        people.push({role: role || 'Member', name: l});
      }
    }
    return people;
  };

  const board_directors = parseBoard('Board of Directors', [
    'Board of Commissioners', 'Address', 'Subsidiary Companies', 'Number of Shareholders'
  ]);
  const board_commissioners = parseBoard('Board of Commissioners', [
    'Address', 'Subsidiary Companies', 'Board of Directors', 'All Watchlist'
  ]);

  // --- Address block ---
  const addrLines = sliceUntil('Address', [
    'Subsidiary Companies', 'Board of Directors', 'Board of Commissioners',
    'All Watchlist', 'Portfolio'
  ]);
  const address = {street: '', phone: '', fax: '', npwp: '', email: '', website: ''};
  for (const l of addrLines) {
    if (/^Phone:/i.test(l)) address.phone = l.replace(/^Phone:\\s*/i, '');
    else if (/^Fax:/i.test(l)) address.fax = l.replace(/^Fax:\\s*/i, '');
    else if (/^NPWP:/i.test(l)) address.npwp = l.replace(/^NPWP:\\s*/i, '');
    else if (/^Email:/i.test(l)) address.email = l.replace(/^Email:\\s*/i, '');
    else if (/^Website:/i.test(l)) address.website = l.replace(/^Website:\\s*/i, '');
    else if (!address.street && l.length > 15 && !/^(Phone|Fax|NPWP|Email|Website)/i.test(l)) {
      address.street = l;
    }
  }

  // Text fallback for shareholder composition when no HTML table
  if (!buckets.shareholder_composition.length) {
    const block = sliceUntil('Shareholder Composition', [
      'Holding Composition', 'Shareholders', 'Directors and Commissioners',
      'Ultimate Beneficiary Owner', 'Company History'
    ]);
    // lines like: Private Equity / 183.58B / 49.44%
    const rows = [['Category', 'Shares', 'Percentage']];
    for (let i = 0; i < block.length - 2; i++) {
      const a = block[i], b = block[i+1], c = block[i+2];
      if (/^[A-Za-z \\/.-]{3,40}$/.test(a) && /[0-9]/.test(b) && /%$/.test(c)) {
        rows.push([a, b, c]);
        i += 2;
      }
    }
    if (rows.length > 3) {
      const md = rows.map(r => '| ' + r.join(' | ') + ' |');
      const sep = '| --- | --- | --- |';
      buckets.shareholder_composition.push([md[0], sep, ...md.slice(1)].join('\\n'));
    }
  }

  if (!buckets.shareholders_simple.length) {
    const block = sliceUntil('Shareholders', [
      'Directors and Commissioners', 'Ultimate Beneficiary Owner', 'Company History',
      'Board of Directors', 'Shares'
    ]);
    // pattern: NAME / 170.00 B / 45.78%
    const rows = [['Name', 'Shares', 'Percentage']];
    for (let i = 0; i < block.length - 2; i++) {
      const a = block[i], b = block[i+1], c = block[i+2];
      if (a.length > 3 && /[0-9].*[KMBT]?$/i.test(b.replace(/\\s/g,'')) && /%$/.test(c)) {
        rows.push([a.replace(/controller$/i, ' (controller)'), b, c]);
        i += 2;
      }
    }
    if (rows.length > 1) {
      const md = rows.map(r => '| ' + r.join(' | ') + ' |');
      buckets.shareholders_simple.push([md[0], '| --- | --- | --- |', ...md.slice(1)].join('\\n'));
    }
  }

  if (!buckets.number_of_shareholders.length) {
    const block = sliceUntil('Number of Shareholders', [
      'Board of Directors', 'Board of Commissioners', 'Address', 'Subsidiary Companies'
    ]);
    const rows = [['Date', 'Count', 'Change']];
    for (let i = 0; i < block.length - 1; i++) {
      const a = block[i], b = block[i+1], c = block[i+2] || '';
      if (/^\\d{1,2}\\s+[A-Za-z]{3}\\s+\\d{4}$/.test(a) && /^[0-9.,]+$/.test(b.replace(/\\s/g,''))) {
        rows.push([a, b, /^[()]/.test(c) || /^[+-]/.test(c) ? c : '']);
        i += (/^[()]/.test(c) || /^[+-]/.test(c)) ? 2 : 1;
      }
    }
    if (rows.length > 1) {
      const md = rows.map(r => '| ' + r.join(' | ') + ' |');
      buckets.number_of_shareholders.push([md[0], '| --- | --- | --- |', ...md.slice(1)].join('\\n'));
    }
  }

  const joinUnique = (arr) => Array.from(new Set(arr.filter(Boolean))).join('\\n\\n');

  return {
    background,
    info,
    shareholders_gt1: joinUnique(buckets.shareholders_gt1),
    shareholder_composition: joinUnique(buckets.shareholder_composition),
    holding_composition: joinUnique(buckets.holding_composition),
    shareholders_simple: joinUnique(buckets.shareholders_simple),
    directors_ownership: joinUnique(buckets.directors_ownership),
    number_of_shareholders: joinUnique(buckets.number_of_shareholders),
    subsidiaries: joinUnique(buckets.subsidiaries),
    other_tables: joinUnique(buckets.other_tables),
    ubo,
    board_directors,
    board_commissioners,
    address,
  };
}
"""


def _trim_profile_noise(text: str) -> str:
    """Drop watchlist / orderbook chrome that often trails profile pages."""
    out = text or ""
    patterns = (
        r"\nAll Watchlist\b",
        r"\nPortfolio\nSymbol\b",
        r"\nBid\nOffer\b",
        r"\nIndeks Harga Saham Gabungan\b",
    )
    cut_at = None
    for pat in patterns:
        m = re.search(pat, out)
        if m and m.start() > 400:
            cut_at = m.start() if cut_at is None else min(cut_at, m.start())
    if cut_at is not None:
        out = out[:cut_at].rstrip()
    return out


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


_CHART_TF_LABELS: dict[str, list[str]] = {
    "1M": ["1", "1m", "1M", "1 min", "1 Minute"],
    "5M": ["5", "5m", "5M", "5 min", "5 Minute", "5 Menit"],
    "15M": ["15", "15m", "15M", "15 min", "15 Minute"],
    "30M": ["30", "30m", "30M", "30 min", "30 Minute"],
    "1H": ["60", "1H", "1h", "60m", "1 Hour", "1 Jam"],
    "4H": ["240", "4H", "4h", "4 Hour", "4 Jam"],
    "1D": ["1D", "D", "1d", "Daily", "1 Day"],
    "1W": ["1W", "W", "1w", "Weekly", "1 Week"],
}


def _as_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_ts(value) -> int | None:
    """Normalize timestamp to unix seconds."""
    try:
        if value is None:
            return None
        n = int(float(value))
        # ms → s
        if n > 10_000_000_000:
            n //= 1000
        return n
    except (TypeError, ValueError):
        return None


def _bars_from_udf(obj: dict) -> list[dict]:
    """TradingView UDF history shape: {t,o,h,l,c,v}."""
    t = obj.get("t") or obj.get("time") or obj.get("timestamp")
    o = obj.get("o") or obj.get("open")
    h = obj.get("h") or obj.get("high")
    low = obj.get("l") or obj.get("low")
    c = obj.get("c") or obj.get("close")
    v = obj.get("v") or obj.get("volume")
    if not isinstance(t, list) or not t:
        return []
    if not all(isinstance(x, list) for x in (o, h, low, c) if x is not None):
        return []
    n = len(t)
    bars: list[dict] = []
    for i in range(n):
        ts = _as_ts(t[i])
        if ts is None:
            continue
        bars.append(
            {
                "ts": ts,
                "open": _as_float(o[i]) if o else None,
                "high": _as_float(h[i]) if h else None,
                "low": _as_float(low[i]) if low else None,
                "close": _as_float(c[i]) if c else None,
                "volume": _as_float(v[i]) if isinstance(v, list) and i < len(v) else 0.0,
            }
        )
    return [b for b in bars if b.get("close") is not None]


def _bars_from_list(items: list) -> list[dict]:
    bars: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Common key aliases
        ts = (
            item.get("ts")
            or item.get("time")
            or item.get("timestamp")
            or item.get("datetime")
            or item.get("date")
            or item.get("t")
        )
        # Nested candle
        if ts is None and isinstance(item.get("candle"), dict):
            item = item["candle"]
            ts = item.get("time") or item.get("timestamp") or item.get("t")
        ts_n = _as_ts(ts)
        if ts_n is None and isinstance(ts, str) and len(ts) >= 10:
            # YYYY-MM-DD or ISO
            try:
                from datetime import datetime

                ts_n = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
            except ValueError:
                try:
                    from datetime import datetime

                    ts_n = int(datetime.strptime(ts[:10], "%Y-%m-%d").timestamp())
                except ValueError:
                    ts_n = None
        if ts_n is None:
            continue
        o = _as_float(item.get("open") or item.get("o") or item.get("Open"))
        h = _as_float(item.get("high") or item.get("h") or item.get("High"))
        low = _as_float(item.get("low") or item.get("l") or item.get("Low"))
        c = _as_float(item.get("close") or item.get("c") or item.get("Close"))
        v = _as_float(item.get("volume") or item.get("v") or item.get("Volume") or 0)
        if c is None:
            continue
        bars.append(
            {
                "ts": ts_n,
                "open": o if o is not None else c,
                "high": h if h is not None else c,
                "low": low if low is not None else c,
                "close": c,
                "volume": v if v is not None else 0.0,
            }
        )
    return bars


def _walk_json_for_bars(node, out: list[list[dict]], depth: int = 0) -> None:
    if depth > 8 or node is None:
        return
    if isinstance(node, dict):
        udf = _bars_from_udf(node)
        if len(udf) >= 5:
            out.append(udf)
        for key in (
            "bars",
            "candles",
            "ohlc",
            "ohlcv",
            "data",
            "result",
            "items",
            "history",
            "prices",
            "series",
        ):
            if key in node:
                _walk_json_for_bars(node[key], out, depth + 1)
        # Avoid exploding over every key — only recurse a few more promising ones.
        for key, val in list(node.items())[:40]:
            if key in {"bars", "candles", "ohlc", "ohlcv", "data", "result", "items", "history"}:
                continue
            if isinstance(val, (dict, list)):
                _walk_json_for_bars(val, out, depth + 1)
    elif isinstance(node, list):
        if node and isinstance(node[0], dict):
            bars = _bars_from_list(node)
            if len(bars) >= 5:
                out.append(bars)
        for item in node[:30]:
            if isinstance(item, (dict, list)):
                _walk_json_for_bars(item, out, depth + 1)


def _pick_best_bars(
    captured: list[tuple[str, object]], symbol: str
) -> tuple[list[dict], list[str]]:
    """Choose the richest OHLCV series from intercepted payloads."""
    candidates: list[tuple[int, list[dict], str]] = []
    for url, data in captured:
        found: list[list[dict]] = []
        _walk_json_for_bars(data, found)
        for bars in found:
            # Prefer payloads that mention the symbol when URL contains it.
            score = len(bars)
            low = url.lower()
            if symbol.lower() in low:
                score += 50
            if any(k in low for k in ("history", "candle", "ohlc", "bars", "udf")):
                score += 20
            candidates.append((score, bars, url))
    if not candidates:
        return [], []
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    # Deduplicate by ts keep last
    by_ts: dict[int, dict] = {}
    for bar in best[1]:
        by_ts[int(bar["ts"])] = bar
    bars = [by_ts[k] for k in sorted(by_ts)]
    urls = list(dict.fromkeys(c[2] for c in candidates[:5]))
    return bars, urls


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
_BROWSER_THREAD_ID: int | None = None


def _is_browser_closed_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "targetclosed" in name
        or "target closed" in msg
        or "browser_closed" in msg
        or "has been closed" in msg
        or "browser has been closed" in msg
        or "context or browser has been closed" in msg
    )


def run_sync(fn, *args, **kwargs):
    """Run Playwright work on the dedicated browser thread.

    Playwright sync API is thread-affine: start + all calls must stay on one
    thread. If we're already on that thread, run inline (avoids deadlock on the
    single-worker executor). A closed browser is marked unavailable and must be
    started explicitly; it is never closed/replaced as automatic cleanup.
    """
    import threading

    def _call():
        global _BROWSER_THREAD_ID
        _BROWSER_THREAD_ID = threading.get_ident()
        return fn(*args, **kwargs)

    def _mark_browser_unavailable() -> None:
        global _BROWSER_THREAD_ID
        if _BROWSER is not None:
            _BROWSER._mark_dead()
        _BROWSER_THREAD_ID = None

    current = threading.current_thread()
    if current.name.startswith("stockbit-pw"):
        try:
            return _call()
        except Exception as exc:  # noqa: BLE001
            if _is_browser_closed_error(exc):
                logger.warning("Stockbit browser closed externally: %s", exc)
                _mark_browser_unavailable()
                raise RuntimeError("BROWSER_CLOSED") from exc
            raise

    def _submit(target):
        return _BROWSER_EXECUTOR.submit(target).result(timeout=900)

    try:
        return _submit(_call)
    except Exception as exc:  # noqa: BLE001
        if _is_browser_closed_error(exc):
            logger.warning("Stockbit browser closed externally: %s", exc)

            def _mark() -> None:
                _mark_browser_unavailable()

            _submit(_mark)
            raise RuntimeError("BROWSER_CLOSED") from exc
        raise


def get_browser(headless: bool = True, user_data_dir: str | None = None) -> StockbitBrowser:
    global _BROWSER
    if _BROWSER is None:
        _BROWSER = StockbitBrowser(headless=headless, user_data_dir=user_data_dir)
        return _BROWSER
    if not _BROWSER.ready:
        _BROWSER.headless = headless
        if user_data_dir is not None:
            _BROWSER.user_data_dir = user_data_dir
        return _BROWSER
    if user_data_dir and _BROWSER.user_data_dir != user_data_dir:
        logger.warning(
            "Stockbit browser already running with profile=%r (requested=%r) — keeping current",
            _BROWSER.user_data_dir,
            user_data_dir,
        )
    return _BROWSER


def shutdown_browser() -> None:
    import threading

    def _close() -> None:
        global _BROWSER, _BROWSER_THREAD_ID
        if _BROWSER is not None:
            _BROWSER.close()
            _BROWSER = None
        _BROWSER_THREAD_ID = None

    if _BROWSER is None:
        return
    if threading.current_thread().name.startswith("stockbit-pw"):
        _close()
    else:
        run_sync(_close)
