"""Normalize LLM/markdown output into Telegram-safe HTML."""
from __future__ import annotations

import html
import re


_ALLOWED_TAGS = frozenset({"b", "strong", "i", "em", "u", "s", "code", "pre", "a"})
_EMPTY_FMT = re.compile(
    r"<(b|strong|i|em|u|s)>\s*</\1>",
    re.I,
)
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")


def to_telegram_html(text: str) -> str:
    """Normalize model output into Telegram-safe HTML."""
    text = (text or "").strip()
    if not text:
        return text

    # Decode common HTML entities the model may already have escaped.
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )

    # Drop empty formatting tags early (e.g. <b></b>, <b> </b>).
    text = _EMPTY_FMT.sub("", text)

    # Convert markdown tables to plain readable lines before other transforms.
    text = _convert_tables(text)

    # Horizontal rules
    text = re.sub(r"(?m)^\s*([-*_])\1{2,}\s*$", "----------", text)

    # If the model mixed HTML tags in, strip them and rebuild from markdown-ish text.
    # Trusting model HTML often yields unbalanced <b>… which Telegram rejects,
    # causing a plain-text fallback that shows raw tags.
    text = _strip_html_to_text(text)

    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            lines.append("")
            continue

        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
        if heading:
            lines.append(f"<b>{_inline_md(heading.group(2))}</b>")
            continue

        bullet = re.match(r"^\s*([-*•])\s+(.*)$", line)
        if bullet:
            lines.append(f"• {_inline_md(bullet.group(2))}")
            continue

        numbered = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
        if numbered:
            lines.append(
                f"<b>{numbered.group(1)}.</b> {_inline_md(numbered.group(2))}"
            )
            continue

        if line.strip() == "----------":
            lines.append("----------")
            continue

        lines.append(_inline_md(line))

    out: list[str] = []
    blank = 0
    for line in lines:
        # Clean leftover empty tags after inline conversion.
        line = _EMPTY_FMT.sub("", line).rstrip()
        if not line:
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line)

    result = "\n".join(out).strip()
    return _balance_tags(result)


def to_plain_text(text: str) -> str:
    """Strip markdown/HTML markers for clean terminal output."""
    text = (text or "").strip()
    if not text:
        return text

    text = _strip_html_to_text(text)
    text = _convert_tables(text)

    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            lines.append("")
            continue

        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
        if heading:
            lines.append(_strip_inline_md(heading.group(2)).upper())
            continue

        bullet = re.match(r"^(\s*)([-*•])\s+(.*)$", line)
        if bullet:
            lines.append(f"{bullet.group(1)}- {_strip_inline_md(bullet.group(3))}")
            continue

        if re.fullmatch(r"\s*([-*_])\1{2,}\s*", line) or line.strip() == "----------":
            lines.append("-" * 40)
            continue

        lines.append(_strip_inline_md(line))

    out: list[str] = []
    blank = 0
    for line in lines:
        if line == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line)
    return "\n".join(out).strip()


def _strip_inline_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def _inline_md(text: str) -> str:
    # Escape first, then restore intentional formatting markers we convert.
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Drop accidental empty tags after conversion.
    text = _EMPTY_FMT.sub("", text)
    return text


def _strip_html_to_text(text: str) -> str:
    """Convert a small set of HTML tags back to markdown-ish plain text."""
    # Bold / italic → markdown so _inline_md can re-apply cleanly.
    text = re.sub(r"</?(?:strong|b)\s*>", "**", text, flags=re.I)
    text = re.sub(r"</?(?:em|i)\s*>", "*", text, flags=re.I)
    text = re.sub(r"</?u\s*>", "", text, flags=re.I)
    text = re.sub(r"</?s\s*>", "", text, flags=re.I)
    text = re.sub(r"</?code\s*>", "`", text, flags=re.I)
    text = re.sub(r"</?pre\s*>", "", text, flags=re.I)
    text = re.sub(r"<a\s+[^>]*>", "", text, flags=re.I)
    text = re.sub(r"</a\s*>", "", text, flags=re.I)
    # Drop any other tags (including broken leftovers).
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    # Collapse accidental **** from nested bold.
    text = text.replace("****", "**")
    text = re.sub(r"\*{4,}", "**", text)
    # Fix unpaired markers per line (global odd-count can hide orphans).
    fixed_lines = []
    for line in text.splitlines():
        line = _pair_markers(line, "**")
        line = _pair_markers(line, "*")
        line = _pair_markers(line, "`")
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def _pair_markers(text: str, marker: str) -> str:
    """Keep only well-paired markers; drop orphans.

    For '**' / '`': toggle open/close. For lone '*', ignore those that are part of '**'.
    """
    if marker == "*":
        chars: list[str] = []
        i = 0
        open_i: int | None = None
        while i < len(text):
            if text.startswith("**", i):
                chars.append("**")
                i += 2
                continue
            if text[i] == "*":
                if open_i is None:
                    open_i = len(chars)
                    chars.append("*")
                else:
                    chars.append("*")
                    open_i = None
                i += 1
                continue
            chars.append(text[i])
            i += 1
        if open_i is not None:
            chars.pop(open_i)
        return "".join(chars)

    parts = text.split(marker)
    # n markers => n+1 parts. Even marker count => odd parts count?
    # "a**b**c" -> ['a','b','c'] = 3 parts = 2 markers (paired).
    # "a**b" -> ['a','b'] = 2 parts = 1 marker (orphan).
    if len(parts) <= 1:
        return text
    if (len(parts) - 1) % 2 == 0:
        return marker.join(parts)
    # Odd markers: drop the last one by merging the last two parts without marker.
    return marker.join(parts[:-2] + [parts[-2] + parts[-1]])


def _convert_tables(text: str) -> str:
    """Turn markdown pipe-tables into readable key: value lines."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect a table: header row + separator row.
        if (
            i + 1 < len(lines)
            and _TABLE_ROW.match(line)
            and _TABLE_SEP.match(lines[i + 1])
        ):
            headers = _split_row(line)
            i += 2  # skip header + separator
            # Blank line before table block (if previous content exists).
            if out and out[-1] != "":
                out.append("")
            while i < len(lines) and _TABLE_ROW.match(lines[i]):
                cells = _split_row(lines[i])
                while len(cells) < len(headers):
                    cells.append("")
                cells = cells[: len(headers)]

                # Build "Ticker — Name (Category): catalyst" when possible.
                by_header = {
                    h.strip().lower(): c.strip()
                    for h, c in zip(headers, cells)
                    if c.strip()
                }
                ticker = (
                    by_header.get("ticker")
                    or by_header.get("kode")
                    or by_header.get("symbol")
                    or ""
                )
                name = (
                    by_header.get("nama emiten / proxy")
                    or by_header.get("nama")
                    or by_header.get("emiten")
                    or by_header.get("proxy")
                    or ""
                )
                kategori = by_header.get("kategori") or by_header.get("category") or ""
                catalyst = (
                    by_header.get("catalyst & exposure")
                    or by_header.get("catalyst")
                    or by_header.get("exposure")
                    or by_header.get("catatan")
                    or ""
                )

                if ticker or name:
                    lead = ticker or name
                    bits: list[str] = []
                    if name and ticker:
                        bits.append(name)
                    if kategori:
                        bits.append(kategori)
                    if catalyst:
                        bits.append(catalyst)
                    if bits:
                        out.append(f"• **{lead}** — " + " | ".join(bits))
                    else:
                        out.append(f"• **{lead}**")
                else:
                    parts = [
                        f"{h.strip()}: {c.strip()}"
                        for h, c in zip(headers, cells)
                        if c.strip()
                    ]
                    if parts:
                        out.append("• " + "; ".join(parts))
                i += 1
            if out and out[-1] != "":
                out.append("")
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _split_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [c.strip() for c in inner.split("|")]


def _balance_tags(text: str) -> str:
    """Remove empty tags and drop unmatched open/close formatting tags."""
    text = _EMPTY_FMT.sub("", text)

    # Stack-based balance for b/i/u/s/code (no nesting of same tag assumed).
    tag_re = re.compile(r"</?(b|i|u|s|code)>", re.I)
    stack: list[str] = []
    pieces: list[str] = []
    last = 0
    for m in tag_re.finditer(text):
        pieces.append(text[last : m.start()])
        raw = m.group(0)
        name = m.group(1).lower()
        closing = raw.startswith("</")
        if not closing:
            stack.append(name)
            pieces.append(f"<{name}>")
        else:
            if stack and stack[-1] == name:
                stack.pop()
                pieces.append(f"</{name}>")
            # else: drop orphan closing tag
        last = m.end()
    pieces.append(text[last:])

    # Close any still-open tags.
    while stack:
        pieces.append(f"</{stack.pop()}>")

    result = "".join(pieces)
    result = _EMPTY_FMT.sub("", result)
    # Collapse spaces left by removed empty tags.
    result = re.sub(r"[ \t]+\n", "\n", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
