from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass
class KnowledgeHit:
    path: Path
    score: float
    snippet: str


class KnowledgeBase:
    """Simple keyword search over markdown knowledge files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("curriculum", "daily", "news", "lessons", "notes", "docs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def list_topics(self) -> str:
        lines = [f"knowledge_root={self.root}"]
        for sub in ("curriculum", "daily", "news", "lessons", "notes", "docs"):
            folder = self.root / sub
            files = sorted(folder.glob("*.md"))
            lines.append(f"\n## {sub} ({len(files)} files)")
            for f in files[-20:]:
                lines.append(f"- {f.name}")
        return "\n".join(lines)

    def stats(self) -> str:
        counts: dict[str, int] = {}
        for sub in ("curriculum", "daily", "news", "lessons", "notes", "docs"):
            counts[sub] = len(list((self.root / sub).glob("*.md")))
        total = sum(counts.values())
        return (
            f"knowledge_total={total} "
            f"(curriculum={counts['curriculum']}, daily={counts['daily']}, "
            f"news={counts['news']}, lessons={counts['lessons']}, "
            f"notes={counts['notes']}, docs={counts['docs']})"
        )

    def save_note(self, title: str, body: str) -> str:
        title = (title or "catatan").strip()
        body = (body or "").strip()
        if not body:
            raise ValueError("body catatan tidak boleh kosong")
        stamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", title).strip("_")[:40] or "note"
        path = self.root / "notes" / f"{stamp}_{slug}.md"
        path.write_text(
            f"# {title}\n\n"
            f"Saved: {datetime.now(ZoneInfo('Asia/Jakarta')).isoformat()}\n\n"
            f"{body}\n",
            encoding="utf-8",
        )
        return f"OK: catatan disimpan\nFILE: {path}"

    def search(self, query: str, limit: int = 5) -> str:
        q = (query or "").strip()
        if not q:
            raise ValueError("query is required")

        tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", q) if len(t) > 2]
        if not tokens:
            tokens = [q.lower()]

        hits: list[KnowledgeHit] = []
        for path in self.root.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lower = text.lower()
            score = 0.0
            for tok in tokens:
                score += lower.count(tok)
            # Prefer curriculum for theory questions.
            if "curriculum" in path.parts:
                score *= 1.2
            if score <= 0:
                continue
            snippet = _best_snippet(text, tokens)
            hits.append(KnowledgeHit(path=path, score=score, snippet=snippet))

        hits.sort(key=lambda h: h.score, reverse=True)
        if not hits:
            return f"(no rows) Tidak ada knowledge untuk: {q}"

        lines = [f"query={q}, hits={min(limit, len(hits))}"]
        for i, hit in enumerate(hits[:limit], start=1):
            rel = hit.path.relative_to(self.root)
            lines.append(f"\n### {i}. {rel} (score={hit.score:.1f})")
            lines.append(hit.snippet.strip())
        return "\n".join(lines)


def _best_snippet(text: str, tokens: list[str], radius: int = 280) -> str:
    lower = text.lower()
    pos = -1
    for tok in tokens:
        pos = lower.find(tok)
        if pos >= 0:
            break
    if pos < 0:
        snippet = text[: radius * 2]
    else:
        start = max(0, pos - radius)
        end = min(len(text), pos + radius)
        snippet = text[start:end]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) > 700:
        snippet = snippet[:700] + "..."
    return snippet


_META_LEARNING = re.compile(
    r"\b("
    r"sejauh\s+mana|"
    r"sudah\s+belajar|"
    r"belajar\s+(?:apa|sejauh|mana|dimana|sampai)|"
    r"kemampuan\s+(?:kamu|mu|bot|agent)|"
    r"progress\s+belajar|"
    r"apa\s+yang\s+(?:sudah\s+)?(?:kamu\s+)?(?:pelajari|dipelajari)|"
    r"how\s+far.*learn|"
    r"what\s+have\s+you\s+learned|"
    r"kamu\s+(?:sudah\s+)?belajar|"
    r"topik\s+knowledge|knowledge\s+base|list\s+knowledge|inventaris\s+knowledge"
    r")\b",
    re.I,
)


def is_meta_learning_query(text: str) -> bool:
    """User asks about the bot's own learning progress — not market theory."""
    raw = (text or "").strip()
    if not re.search(r"\b(belajar|learn(?:ing|ed)?|knowledge\s+base)\b", raw, re.I):
        return False
    return bool(_META_LEARNING.search(raw))
