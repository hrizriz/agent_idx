from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.error import URLError, HTTPError
from urllib.parse import quote_plus, urlparse, parse_qs
from urllib.request import Request, urlopen


def search_news(query: str, limit: int = 5) -> str:
    """Search recent news via Google News RSS (title + link + source + date)."""
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")

    limit = max(1, min(int(limit), 15))
    # Prefer Indonesian finance context.
    rss_query = f"{q} saham OR IDX OR BEI"
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(rss_query)}&hl=id&gl=ID&ceid=ID:id"
    )

    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; agent-idx/0.1; +local)",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        },
    )
    try:
        with urlopen(req, timeout=20) as resp:
            payload = resp.read()
    except (URLError, HTTPError, TimeoutError) as exc:
        raise RuntimeError(f"Gagal mengambil berita: {exc}") from exc

    root = ET.fromstring(payload)
    items = root.findall("./channel/item")
    if not items:
        return f"(no rows) Tidak ada berita untuk query: {q}"

    lines = [f"query={q}, results={min(limit, len(items))}"]
    for i, item in enumerate(items[:limit], start=1):
        title = _text(item.findtext("title"))
        link = _resolve_link(item.findtext("link") or "")
        source = _text(item.findtext("source"))
        pub = item.findtext("pubDate") or ""
        when = pub
        try:
            when = parsedate_to_datetime(pub).strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError, IndexError):
            pass
        lines.append(f"{i}. {title}")
        lines.append(f"   sumber: {source or '-'} | waktu: {when or '-'}")
        lines.append(f"   link: {link}")
    return "\n".join(lines)


def _text(value: str | None) -> str:
    if not value:
        return ""
    return unescape(re.sub(r"\s+", " ", value)).strip()


def _resolve_link(link: str) -> str:
    """Google News RSS often wraps the real URL in a redirect query param."""
    link = (link or "").strip()
    if not link:
        return ""
    try:
        parsed = urlparse(link)
        qs = parse_qs(parsed.query)
        if "url" in qs and qs["url"]:
            return qs["url"][0]
    except ValueError:
        pass
    return link
