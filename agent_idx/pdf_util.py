from __future__ import annotations

from pathlib import Path


def extract_pdf_text(path: Path, max_chars: int = 20_000) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    text = "\n".join(chunks).strip()
    if not text:
        return (
            "(PDF tidak berisi teks yang bisa diekstrak. "
            "Kemungkinan file hasil scan/gambar.)"
        )
    if len(text) > max_chars:
        return text[:max_chars] + "\n... (truncated)"
    return text
