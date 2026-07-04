from __future__ import annotations

import html
import re


def to_telegram_html(text: str) -> str:
    """Normalize model output into Telegram-safe HTML."""
    text = text.strip()
    if not text:
        return text

    # Already looks like intentional HTML from the model.
    if re.search(r"<(b|i|u|code|pre|a)\b", text, re.I):
        return _sanitize_html(text)

    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            lines.append("")
            continue

        heading = re.match(r"^\s{0,3}(#{1,4})\s+(.*)$", line)
        if heading:
            title = _inline_md(heading.group(2))
            lines.append(f"<b>{title}</b>")
            continue

        bullet = re.match(r"^\s*([-*•])\s+(.*)$", line)
        if bullet:
            lines.append(f"• {_inline_md(bullet.group(2))}")
            continue

        numbered = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
        if numbered:
            lines.append(f"<b>{numbered.group(1)}.</b> {_inline_md(numbered.group(2))}")
            continue

        # Horizontal rules / separators
        if re.fullmatch(r"\s*([-*_])\1{2,}\s*", line):
            lines.append("----------")
            continue

        lines.append(_inline_md(line))

    # Collapse excessive blank lines
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


def _inline_md(text: str) -> str:
    # Escape first, then restore intentional formatting markers we convert.
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def _sanitize_html(text: str) -> str:
    """Escape everything except a small allowlist of Telegram HTML tags."""
    # Protect allowed tags, escape the rest.
    tokens: list[str] = []

    def protect(match: re.Match[str]) -> str:
        tokens.append(match.group(0))
        return f"\x00TAG{len(tokens) - 1}\x00"

    allowed = re.compile(
        r"</?(?:b|strong|i|em|u|s|code|pre|a)(?:\s+href=\"[^\"]*\")?\s*>",
        re.I,
    )
    protected = allowed.sub(protect, text)
    protected = html.escape(protected)
    for i, token in enumerate(tokens):
        # Normalize strong/em to b/i
        token = re.sub(r"</?strong>", lambda m: "</b>" if m.group(0).startswith("</") else "<b>", token, flags=re.I)
        token = re.sub(r"</?em>", lambda m: "</i>" if m.group(0).startswith("</") else "<i>", token, flags=re.I)
        protected = protected.replace(f"\x00TAG{i}\x00", token)
    return protected
