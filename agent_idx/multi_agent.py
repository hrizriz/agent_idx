"""Multi-agent council: Researcher → Analyst → Critic (with revision loops).

Three specialized roles share one scratchpad and correct each other before
the final answer reaches the user. Designed for IDX / fintech analysis:
evidence first, thesis second, challenge third.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from agent_idx.agent import AgentResult
from agent_idx.chat_log import ChatLog
from agent_idx.config import Settings
from agent_idx.data import StockDataStore
from agent_idx.knowledge import KnowledgeBase
from agent_idx.stockbit_store import StockbitReportsStore
from agent_idx.tools import TOOL_DEFINITIONS, dispatch_tool, extract_files

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]

# Researcher may only gather evidence (no PDF export / scrape / browser side-effects).
RESEARCHER_TOOL_NAMES = frozenset(
    {
        "describe_schema",
        "list_date_range",
        "get_stock_history",
        "summarize_period",
        "rank_foreign_flow",
        "run_sql",
        "search_knowledge",
        "list_knowledge_topics",
        "search_news",
        "search_stockbit_reports",
        "read_stockbit_scrape_date",
        "list_stockbit_scrape_files",
        "run_backtest_scan",
        "stockbit_status",
        "stockbit_open",
        "stockbit_read",
        "stockbit_scrape",
        "query_chart_ohlcv",
        "chart_features",
        "chart_db_stats",
        "get_stockbit_ohlcv",
        "get_technicals",
        "get_fundamentals",
        "analyze_stock",
        "search_news_sentiment",
        "run_data_transform",
        "rebuild_market_from_stockbit",
        "list_project_dir",
        "read_project_file",
    }
)

RESEARCHER_PROMPT = """\
Kamu adalah RESEARCHER di council analis IDX (fintech).

TUGAS
- Kumpulkan BUKTI saja lewat tools. Jangan buat rekomendasi beli/jual.
- Output akhir wajib berupa Evidence Pack berbahasa Indonesia.

FORMAT EVIDENCE PACK
## Ringkasan permintaan
## Data pasar (angka + periode + symbol)
## Konteks knowledge / teori (jika relevan)
## Berita / katalis (jika relevan)
## Gap data (apa yang TIDAK tersedia — mis. foreign flow = NULL dari bootstrap)
## Sumber tools yang dipakai

ATURAN
- WAJIB panggil tools untuk angka. Jangan mengarang harga/volume.
- Sumber pasar UTAMA = Stockbit Chartbit (daily_stock dari MARKET_SOURCE=stockbit /
  daily_stock_summary_stockbit_*.parquet). JANGAN mengandalkan Yahoo/yfinance.
- Cek list_date_range dulu; sebutkan max_date + source= di Evidence Pack.
- Harga terkini = close TERAKHIR + date saja. Jangan tulis rentang multi-hari
  sebagai "harga sekarang".
- Foreign flow: jika tool bilang GAP_DATA / NULL, JANGAN mengarang net buy.
  Arahkan scrape overview + run_data_transform / get_fundamentals.
- Untuk analisis emiten: utamakan analyze_stock(symbol, horizon).
  Sebut freshness, konflik, dan GAP_DATA/STALE_DATA dari outputnya.
  Drill-down MA/RSI lewat get_technicals bila perlu angka bar detail.
- Sentimen berita: search_news_sentiment (bukan mengarang positif/negatif).
- Jika data Stockbit tertinggal: get_stockbit_ohlcv(symbol, timeframes='1D', refresh=true)
  atau rebuild_market_from_stockbit / chart_features.
- Untuk multi-TF chart: get_stockbit_ohlcv(symbol, timeframes='5M,30M,1H,1D,1W').
- date = INTEGER YYYYMMDD. net_foreign = foreign_buy - foreign_sell.
- Jika kolom foreign/bid/offer kosong, sebutkan eksplisit di Gap data
  (Chartbit sync sering tidak punya foreign flow).
- Untuk data live Stockbit halaman: stockbit_status → stockbit_open/scrape (butuh sesi login).
- Bahasa Indonesia, ringkas, markdown.
"""

ANALYST_PROMPT = """\
Kamu adalah ANALYST di council analis IDX (fintech).

TUGAS
- Baca Evidence Pack (+ kritik putaran sebelumnya jika ada).
- Susun draft tesis investasi / analisis. JANGAN memanggil tools.
- Hanya gunakan fakta yang ada di evidence. Jika gap data, akui batasan.

FORMAT DRAFT
## Ringkasan eksekutif (2–3 kalimat)
## Pandangan (menarik / netral / hati-hati) + horizon
## Dasar dari data pasar
## Fundamental / kerangka (hanya jika evidence punya sumber)
## Risiko & invalidasi
## Batasan data

ATURAN
- Jangan mengarang angka baru.
- Jangan menyuruh user mencari data.
- Bahasa Indonesia baku, profesional, netral.
"""

CRITIC_PROMPT = """\
Kamu adalah CRITIC di council analis IDX (fintech).

TUGAS
- Serang draft Analyst: bias, overclaim, angka tanpa sumber, gap diabaikan, horizon kabur.
- Putuskan APPROVE atau REVISE.

OUTPUT WAJIB JSON MURNI (tanpa markdown fence):
{
  "verdict": "APPROVE" atau "REVISE",
  "issues": ["poin masalah 1", "..."],
  "requests_to_researcher": ["data tambahan yang dibutuhkan, atau []"],
  "requests_to_analyst": ["perbaikan thesis, atau []"],
  "lesson": "satu pelajaran singkat untuk knowledge base, atau string kosong",
  "summary": "1–2 kalimat alasan verdict"
}

ATURAN
- REVISE jika ada klaim tanpa evidence, foreign flow diklaim kuat padahal gap, atau rekomendasi tanpa invalidasi.
- APPROVE jika draft jujur soal batasan data dan kesimpulan proporsional.
- Jangan ambil data sendiri. Jangan tulis selain JSON.
"""


@dataclass
class RoundTrace:
    round_no: int
    evidence: str = ""
    thesis: str = ""
    critique_raw: str = ""
    verdict: str = ""
    issues: list[str] = field(default_factory=list)


@dataclass
class CouncilResult(AgentResult):
    rounds: list[RoundTrace] = field(default_factory=list)
    trace_text: str = ""


def _filter_tools(names: frozenset[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in TOOL_DEFINITIONS:
        fn = tool.get("function") or {}
        if fn.get("name") in names:
            out.append(tool)
    return out


def _parse_critique(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {
            "verdict": "REVISE",
            "issues": ["Critic tidak mengembalikan output"],
            "requests_to_researcher": [],
            "requests_to_analyst": ["Perjelas draft"],
            "lesson": "",
            "summary": "empty critic response",
        }

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        upper = raw.upper()
        verdict = "APPROVE" if "APPROVE" in upper and "REVISE" not in upper else "REVISE"
        return {
            "verdict": verdict,
            "issues": ["Critic output bukan JSON valid"],
            "requests_to_researcher": [],
            "requests_to_analyst": [],
            "lesson": "",
            "summary": raw[:500],
        }

    verdict = str(data.get("verdict", "REVISE")).strip().upper()
    if verdict not in {"APPROVE", "REVISE"}:
        verdict = "REVISE"
    return {
        "verdict": verdict,
        "issues": [str(x) for x in (data.get("issues") or [])],
        "requests_to_researcher": [str(x) for x in (data.get("requests_to_researcher") or [])],
        "requests_to_analyst": [str(x) for x in (data.get("requests_to_analyst") or [])],
        "lesson": str(data.get("lesson") or "").strip(),
        "summary": str(data.get("summary") or "").strip(),
    }


class MultiAgentCouncil:
    """Researcher / Analyst / Critic council with revision loops."""

    def __init__(
        self,
        settings: Settings,
        store: StockDataStore,
        knowledge: KnowledgeBase | None = None,
        chat_log: ChatLog | None = None,
        stockbit_store: StockbitReportsStore | None = None,
        chat_id: int | None = None,
        max_rounds: int = 3,
    ) -> None:
        self.settings = settings
        self.store = store
        self.knowledge = knowledge or KnowledgeBase(settings.knowledge_dir)
        self.chat_log = chat_log
        self.stockbit_store = stockbit_store
        self.chat_id = chat_id if chat_id is not None else settings.telegram_chat_id
        self.max_rounds = max(1, min(int(max_rounds), 5))
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self._coverage = store.list_date_range()
        self.export_dir = settings.export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._researcher_tools = _filter_tools(RESEARCHER_TOOL_NAMES)

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in RESEARCHER_TOOL_NAMES:
            return f"ERROR: tool '{name}' tidak diizinkan untuk Researcher"
        return dispatch_tool(
            self.store,
            name,
            arguments,
            export_dir=self.export_dir,
            knowledge=self.knowledge,
            chat_log=self.chat_log,
            chat_id=self.chat_id,
            stockbit_store=self.stockbit_store,
        )

    def _tool_loop(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict[str, Any]] | None,
        force_tools_first: bool,
        files: list[Path],
        safety_cap: int = 24,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        used_tool = False
        step = 0
        while True:
            step += 1
            force_answer = step >= safety_cap or tools is None
            if force_answer or tools is None:
                tool_choice: Any = "none"
            elif step == 1 and force_tools_first:
                tool_choice = "required"
            else:
                tool_choice = "auto"

            kwargs: dict[str, Any] = {
                "model": self.settings.llm_model,
                "messages": messages,
                "tool_choice": tool_choice,
            }
            if tools is not None and not force_answer:
                kwargs["tools"] = tools

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            tool_calls = [] if force_answer or tools is None else (message.tool_calls or [])

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
                if tools is not None and force_tools_first and not used_tool and not force_answer:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Panggil tools data/knowledge/news yang relevan dulu, "
                                "baru tulis Evidence Pack."
                            ),
                        }
                    )
                    continue
                return content or "(kosong)"

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
                    logger.info("Council tool %s args=%s", name, args)
                    result = self._dispatch(name, args)
                    files.extend(extract_files(result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

    def _researcher(
        self,
        question: str,
        *,
        critique: dict[str, Any] | None,
        prior_evidence: str,
        files: list[Path],
    ) -> str:
        parts = [
            f"CAKUPAN DATA: {self._coverage}",
            f"Pertanyaan user:\n{question}",
        ]
        if critique and critique.get("requests_to_researcher"):
            parts.append(
                "Permintaan tambahan dari Critic:\n- "
                + "\n- ".join(critique["requests_to_researcher"])
            )
            if prior_evidence:
                parts.append(f"Evidence sebelumnya (perbarui/lengkapi):\n{prior_evidence}")
        system = (
            f"{RESEARCHER_PROMPT}\n\n"
            f"FOLDER EXPORT: {self.export_dir}\n"
            f"KNOWLEDGE: {self.knowledge.root}"
        )
        return self._tool_loop(
            system=system,
            user="\n\n".join(parts),
            tools=self._researcher_tools,
            force_tools_first=True,
            files=files,
        )

    def _analyst(
        self,
        question: str,
        evidence: str,
        *,
        critique: dict[str, Any] | None,
        prior_thesis: str,
    ) -> str:
        parts = [
            f"Pertanyaan user:\n{question}",
            f"Evidence Pack:\n{evidence}",
        ]
        if critique:
            if critique.get("issues"):
                parts.append("Isu dari Critic:\n- " + "\n- ".join(critique["issues"]))
            if critique.get("requests_to_analyst"):
                parts.append(
                    "Perbaikan yang diminta Critic:\n- "
                    + "\n- ".join(critique["requests_to_analyst"])
                )
            if prior_thesis:
                parts.append(f"Draft sebelumnya:\n{prior_thesis}")
        return self._tool_loop(
            system=ANALYST_PROMPT,
            user="\n\n".join(parts),
            tools=None,
            force_tools_first=False,
            files=[],
            safety_cap=1,
        )

    def _critic(self, question: str, evidence: str, thesis: str) -> dict[str, Any]:
        user = (
            f"Pertanyaan user:\n{question}\n\n"
            f"Evidence Pack:\n{evidence}\n\n"
            f"Draft Analyst:\n{thesis}\n\n"
            "Kembalikan JSON verdict saja."
        )
        raw = self._tool_loop(
            system=CRITIC_PROMPT,
            user=user,
            tools=None,
            force_tools_first=False,
            files=[],
            safety_cap=1,
        )
        parsed = _parse_critique(raw)
        parsed["_raw"] = raw
        return parsed

    def _save_lesson(self, question: str, lesson: str, files: list[Path]) -> None:
        if not lesson:
            return
        title = f"Council lesson: {question[:60]}"
        result = dispatch_tool(
            self.store,
            "save_learning_note",
            {"title": title, "body": lesson},
            export_dir=self.export_dir,
            knowledge=self.knowledge,
            chat_log=self.chat_log,
            chat_id=self.chat_id,
            stockbit_store=self.stockbit_store,
        )
        files.extend(extract_files(result))
        logger.info("Saved council lesson: %s", lesson[:120])

    def ask(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,  # noqa: ARG002 — API parity
        *,
        on_progress: ProgressFn | None = None,
        include_trace: bool = True,
    ) -> CouncilResult:
        def progress(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        files: list[Path] = []
        rounds: list[RoundTrace] = []
        evidence = ""
        thesis = ""
        critique: dict[str, Any] | None = None

        for round_no in range(1, self.max_rounds + 1):
            progress(f"[council] Putaran {round_no}/{self.max_rounds} — Researcher…")
            evidence = self._researcher(
                user_message,
                critique=critique,
                prior_evidence=evidence,
                files=files,
            )

            progress(f"[council] Putaran {round_no}/{self.max_rounds} — Analyst…")
            thesis = self._analyst(
                user_message,
                evidence,
                critique=critique,
                prior_thesis=thesis,
            )

            progress(f"[council] Putaran {round_no}/{self.max_rounds} — Critic…")
            critique = self._critic(user_message, evidence, thesis)
            verdict = critique["verdict"]
            trace = RoundTrace(
                round_no=round_no,
                evidence=evidence,
                thesis=thesis,
                critique_raw=str(critique.get("_raw", "")),
                verdict=verdict,
                issues=list(critique.get("issues") or []),
            )
            rounds.append(trace)
            progress(
                f"[council] Critic: {verdict}"
                + (f" — {critique.get('summary')}" if critique.get("summary") else "")
            )

            if verdict == "APPROVE":
                self._save_lesson(user_message, critique.get("lesson", ""), files)
                break

            # Last round: force finalize even if still REVISE.
            if round_no == self.max_rounds:
                progress("[council] Max rounds tercapai — finalize dengan catatan critic.")
                if critique.get("issues"):
                    thesis = (
                        f"{thesis.rstrip()}\n\n"
                        "## Catatan Critic (belum fully resolved)\n"
                        + "\n".join(f"- {i}" for i in critique["issues"])
                    )
                self._save_lesson(user_message, critique.get("lesson", ""), files)

        trace_lines = [
            f"Council selesai dalam {len(rounds)} putaran "
            f"(max={self.max_rounds})."
        ]
        for r in rounds:
            trace_lines.append(f"- Putaran {r.round_no}: Critic={r.verdict}")
            for issue in r.issues[:5]:
                trace_lines.append(f"  · {issue}")

        final = thesis.strip()
        if include_trace:
            final = f"{final}\n\n---\n### Jejak council\n" + "\n".join(trace_lines)

        return CouncilResult(
            text=final or "(council tidak menghasilkan jawaban)",
            files=files,
            rounds=rounds,
            trace_text="\n".join(trace_lines),
        )
