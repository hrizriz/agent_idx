"""Image helpers for Telegram photo / PNG / JPEG analysis."""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from openai import OpenAI

from agent_idx.config import Settings

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_IMAGE_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS


def mime_for_path(path: Path) -> str:
    return _IMAGE_MIMES.get(path.suffix.lower(), "image/jpeg")


def compress_image_for_llm(path: Path, max_side: int = 1280, quality: int = 85) -> tuple[bytes, str]:
    """Return (bytes, mime) suitable for vision API; keep PNG if small enough."""
    raw = path.read_bytes()
    if len(raw) <= 1_200_000 and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return raw, mime_for_path(path)

    try:
        from PIL import Image
    except ImportError:
        return raw, mime_for_path(path)

    img = Image.open(io.BytesIO(raw))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), "image/jpeg"


def describe_image(
    path: Path,
    settings: Settings,
    *,
    user_hint: str = "",
) -> str:
    """Ask the LLM (vision) to describe a chart/screenshot for IDX analysis."""
    data, mime = compress_image_for_llm(path)
    b64 = base64.b64encode(data).decode("ascii")
    hint = (user_hint or "").strip()
    prompt = (
        "Kamu asisten analis saham IDX. Deskripsikan gambar ini secara faktual dan terstruktur.\n"
        "Fokus: Symbol (jika terlihat), Timeframe, arah harga, level support/resistance, "
        "indikator (MA/RSI/MACD/volume) jika ada, anotasi user, dan risiko visual.\n"
        "Jangan mengarang angka yang tidak terbaca. Bahasa Indonesia, ringkas.\n"
    )
    if hint:
        prompt += f"\nPertanyaan / konteks user: {hint}\n"

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        tool_choice="none",
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Model tidak mengembalikan deskripsi gambar")
    if "Image input not supported" in text or "not supported in this API" in text:
        raise RuntimeError(
            "LLM backend belum menerima image. Restart gemini-web2api dengan: "
            "py -3 -m gemini_web2api (package multimodal)."
        )
    logger.info("Described image %s chars=%s", path.name, len(text))
    return text
