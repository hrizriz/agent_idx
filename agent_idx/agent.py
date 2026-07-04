from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent_idx.chat_log import ChatLog
from agent_idx.config import Settings
from agent_idx.data import StockDataStore
from agent_idx.knowledge import KnowledgeBase
from agent_idx.tools import TOOL_DEFINITIONS, dispatch_tool, extract_files

logger = logging.getLogger(__name__)

_TOOL_HINT = re.compile(
    r"\b("
    r"analisis|rekomendasi|pantau|watchlist|foreign|volume|return|"
    r"berita|news|laporan|report|export|csv|data|sql|ticker|emiten|"
    r"banding|bandingkan|ranking|top|net\s*sell|net\s*buy|ohlc|harga|"
    r"periode|tahun|ytd|bbca|bmri|bbri|bbni|goto|idx|saham|"
    r"buat\s+pdf|kirim\s+pdf|generate\s+pdf|"
    r"fundamental|valuasi|rasio|charting|teknikal|nim|npl|roe|roa|der|pe\b|pb\b|"
    r"belajar|kurikulum|knowledge|digest|ajarin|ingat\s+bahwa|catat|"
    r"chat|group|grup|siapa\s+bilang|diskusi"
    r")\b|"
    r"\b20\d{2}\b|"
    r"\b[A-Z]{3,5}\b",
    re.I,
)

SYSTEM_PROMPT = """\
Kamu adalah analis & tutor pasar saham IDX.
Kamu asisten 2 arah: data transaksi, berita, PDF user, laporan, dan knowledge base pembelajaran.

KEMAMPUAN TOOLS
- Data pasar: list_date_range, describe_schema, get_stock_history, summarize_period,
  rank_foreign_flow, run_sql (SELECT only)
- Knowledge belajar: search_knowledge, list_knowledge_topics
  (fundamental, rasio, charting, foreign flow, playbook, daily digest, news digest, weekly lesson)
- Chat group Telegram yang di-collect (bisa banyak group): list_collected_chats,
  search_chat_history, recent_chat_history, chat_log_stats
  (pesan sejak bot aktif di group itu; bukan history sebelum bot masuk.
   Interaksi user hanya di group utama; group lain di-collect diam-diam.)
- Berita live: search_news
- File: create_pdf_report, export_query_csv (terkirim otomatis ke Telegram)
- Blok "--- ISI PDF ... ---" = dokumen user. Jawab HANYA dari situ kecuali diminta bandingkan data.

PEMBELAJARAN
- Untuk pertanyaan teori/fundamental/charting/cara analisis: WAJIB search_knowledge dulu.
- Jika user menyuruh belajar/update knowledge/digest: panggil run_learning_job
  (market | news | weekly | all).
- Jika user mengajari ("ingat bahwa", "catat", "ajarin"): panggil save_learning_note.
- Gabungkan: kerangka fundamental (knowledge) + angka pasar (tools data) + berita bila relevan.
- Data transaksi BUKAN laporan keuangan. Jangan mengarang NIM/ROE/NPL/laba tanpa sumber PDF/knowledge/berita.

ATURAN DATA
- WAJIB tools untuk fakta angka/berita/knowledge. Jangan mengarang.
- JANGAN bilang kendala teknis kecuali tool ERROR.
- date = INTEGER YYYYMMDD. net_foreign = foreign_buy - foreign_sell.

GAYA
- Bahasa Indonesia baku, profesional, netral.
- Analisis biasa ~350 kata; rekomendasi/belajar komprehensif ~700 kata.
- Markdown: ## ### - **teks**
- DILARANG kalimat self-puji ("analisis saya terbukti akurat", "fantastis", dll.).
- DILARANG menutup dengan pertanyaan upsell/engagement bait
  (contoh: "Apakah Anda ingin saya buatkan PDF/CSV/cari berita?").
- Akhiri di ## Catatan. Jika user ingin PDF/CSV/berita, mereka akan meminta sendiri.
- Jangan perintah beli/jual absolut.

STRUKTUR ANALISIS LENGKAP (jika diminta rekomendasi/fundamental)
1) Ringkasan
2) Kerangka fundamental (dari knowledge / PDF) — lewati jika tidak relevan
3) Data pasar & flow (tools)
4) Charting ringkas (jika relevan)
5) Risiko & invalidasi
6) Catatan / sumber
"""


@dataclass
class AgentResult:
    text: str
    files: list[Path] = field(default_factory=list)


class AnalystAgent:
    def __init__(
        self,
        settings: Settings,
        store: StockDataStore,
        knowledge: KnowledgeBase | None = None,
        chat_log: ChatLog | None = None,
        chat_id: int | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.knowledge = knowledge or KnowledgeBase(settings.knowledge_dir)
        self.chat_log = chat_log
        self.chat_id = chat_id if chat_id is not None else settings.telegram_chat_id
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self._coverage = store.list_date_range()
        self.export_dir = settings.export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _system_prompt(self) -> str:
        chat_stats = ""
        if self.chat_log is not None:
            chat_stats = f"\nCHAT LOG: {self.chat_log.stats()}"
            if self.chat_id is not None:
                chat_stats += f"\nINTERACTION CHAT_ID: {self.chat_id}"
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"CAKUPAN DATA TERSEDIA:\n{self._coverage}\n"
            f"FOLDER EXPORT: {self.export_dir}\n"
            f"KNOWLEDGE DIR: {self.knowledge.root}"
            f"{chat_stats}"
        )

    def ask(
        self, user_message: str, history: list[dict[str, Any]] | None = None
    ) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        files: list[Path] = []
        used_tool = False
        needs_tools = self._needs_tools(user_message, history)

        for step in range(self.settings.max_agent_steps):
            logger.info("Agent step %s/%s", step + 1, self.settings.max_agent_steps)
            tool_choice: Any = "required" if (step == 0 and needs_tools) else "auto"
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice=tool_choice,
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            assistant_payload: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if tool_calls:
                assistant_payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_payload)

            if not tool_calls:
                content = (message.content or "").strip()
                if needs_tools and not used_tool:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Panggil tools yang relevan dulu "
                                "(data / search_news / create_pdf_report / export_query_csv)."
                            ),
                        }
                    )
                    needs_tools = True
                    continue
                return AgentResult(
                    text=content or "(model tidak mengembalikan jawaban)",
                    files=files,
                )

            used_tool = True
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    result = f"ERROR: invalid JSON arguments: {tc.function.arguments!r}"
                else:
                    logger.info("Tool %s args=%s", name, args)
                    result = dispatch_tool(
                        self.store,
                        name,
                        args,
                        export_dir=self.export_dir,
                        knowledge=self.knowledge,
                        chat_log=self.chat_log,
                        chat_id=self.chat_id,
                    )
                    files.extend(extract_files(result))

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        return AgentResult(
            text=(
                "Saya sudah mencapai batas langkah analisis. "
                "Lanjutkan dengan perintah lebih spesifik, atau /reset lalu mulai lagi."
            ),
            files=files,
        )

    @staticmethod
    def _needs_tools(
        user_message: str, history: list[dict[str, Any]] | None
    ) -> bool:
        text = user_message.strip()
        if not text:
            return False
        # PDF content already injected — answer from context, no forced tools.
        if "--- ISI PDF" in text:
            return False
        # Greetings / short chat.
        if re.fullmatch(r"(halo|hai|hi|hello|pagi|siang|sore|malam)[.! ]*", text, re.I):
            return False
        if history and len(text) < 12 and not _TOOL_HINT.search(text):
            return False
        if _TOOL_HINT.search(text):
            return True
        if history and re.search(
            r"\b(itu|tadi|lanjut|buatkan|kirim|export|csv|berita|buat\s+pdf)\b",
            text,
            re.I,
        ):
            return True
        return len(text) > 40
