"""Structured tool pipelines — system (and LLM) define which tools run, in order.

Each intent maps to a named pipeline of ToolSteps. Routers in agent.py call
`run_pipeline`; the LLM may also call the same tools individually when allowed.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_idx.decision import Intent

logger = logging.getLogger(__name__)

ArgsFn = Callable[["PipelineContext"], dict[str, Any]]


@dataclass
class PipelineContext:
    """Runtime inputs for building tool arguments."""

    user_message: str
    intent: str = ""
    symbol: str | None = None
    symbols: list[str] = field(default_factory=list)
    timeframe: str = "1H"
    timeframes: list[str] = field(default_factory=list)
    days: int | None = None
    date_from: str | None = None
    date_to: str | None = None
    url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def primary_symbol(self) -> str | None:
        if self.symbol:
            return self.symbol.upper()
        if self.symbols:
            return self.symbols[0].upper()
        return None


@dataclass
class ToolStep:
    """One backend tool invocation in a pipeline."""

    tool: str
    args: dict[str, Any] | ArgsFn
    required: bool = True
    label: str = ""
    # If tool output contains these, stop pipeline and surface to caller.
    stop_substrings: tuple[str, ...] = (
        "NEED_STOCKBIT_CREDENTIALS",
        "NEED_STOCKBIT_OTP",
        "STAGED (menunggu konfirmasi",
    )


@dataclass
class PipelineDef:
    id: str
    description: str
    steps: list[ToolStep]
    # Tools the LLM may call when this pipeline's intent is active.
    allowed_tools: list[str] = field(default_factory=list)


@dataclass
class StepResult:
    tool: str
    label: str
    ok: bool
    output: str
    files: list[Path] = field(default_factory=list)


@dataclass
class PipelineResult:
    pipeline_id: str
    run_id: str
    steps: list[StepResult] = field(default_factory=list)
    stopped_early: bool = False
    need_stockbit_login: bool = False
    status: str = "ok"

    def combined_text(self, *, max_per_step: int = 6000) -> str:
        parts: list[str] = [
            f"pipeline={self.pipeline_id} run_id={self.run_id} status={self.status}"
        ]
        for i, s in enumerate(self.steps, 1):
            head = s.label or s.tool
            body = s.output if len(s.output) <= max_per_step else s.output[:max_per_step] + "…"
            parts.append(f"--- step {i}: {head} ({s.tool}) ok={s.ok} ---\n{body}")
        return "\n\n".join(parts)

    def all_files(self) -> list[Path]:
        out: list[Path] = []
        for s in self.steps:
            out.extend(s.files)
        return out


def _farm_args(ctx: PipelineContext) -> dict[str, Any]:
    sym = ctx.primary_symbol()
    tfs = ctx.timeframes or ([ctx.timeframe] if ctx.timeframe else ["1H"])
    return {
        "symbols": sym or "",
        "timeframes": ",".join(tfs),
    }


def _features_args(ctx: PipelineContext) -> dict[str, Any]:
    sym = ctx.primary_symbol() or ""
    tfs = ctx.timeframes or [ctx.timeframe, "1H", "1D"]
    # Dedupe preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tfs:
        u = t.upper()
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    if ctx.timeframe.upper() not in seen:
        ordered.insert(0, ctx.timeframe.upper())
    return {"symbol": sym, "timeframes": ",".join(ordered), "lookback": 40}


def _ohlcv_args(ctx: PipelineContext) -> dict[str, Any]:
    return {
        "symbol": ctx.primary_symbol() or "",
        "timeframe": ctx.timeframe or "1H",
        "n": 30,
    }


def _sync_args(ctx: PipelineContext) -> dict[str, Any]:
    sym = ctx.primary_symbol()
    tfs = ctx.timeframes or ([ctx.timeframe] if ctx.timeframe else [])
    return {
        "symbols": sym or "",
        "timeframes": ",".join(tfs) if tfs else "",
    }


def _reports_args(ctx: PipelineContext) -> dict[str, Any]:
    args: dict[str, Any] = {
        "url": ctx.url or "https://stockbit.com/StockbitReports?source=0",
    }
    if ctx.date_from:
        args["date_from"] = ctx.date_from
    if ctx.date_to:
        args["date_to"] = ctx.date_to
    if ctx.days is not None:
        args["days"] = ctx.days
    return args


# --------------------------------------------------------------------------- #
# Pipeline catalog — system-defined tool graphs
# --------------------------------------------------------------------------- #

PIPELINES: dict[str, PipelineDef] = {
    "chart_analyze": PipelineDef(
        id="chart_analyze",
        description=(
            "Ambil OHLCV Stockbit (5M/30M/1H/1D/1W) via get_stockbit_ohlcv "
            "(farm Chartbit bila perlu + sync DuckDB + features). "
            "Lalu LLM ringkas analisa dari hasil tools (bukan mengarang)."
        ),
        steps=[
            ToolStep(
                "get_stockbit_ohlcv",
                lambda ctx: {
                    "symbol": ctx.primary_symbol() or "",
                    "timeframes": ",".join(
                        ctx.timeframes
                        or ([ctx.timeframe] if ctx.timeframe else ["1H", "1D"])
                    ),
                    "n": 40,
                    "refresh": True,
                },
                label="get Stockbit OHLCV",
            ),
        ],
        allowed_tools=[
            "get_stockbit_ohlcv",
            "farm_stockbit_charts",
            "sync_chart_db",
            "query_chart_ohlcv",
            "chart_features",
            "chart_db_stats",
            "stockbit_status",
            "stockbit_open",
        ],
    ),
    "chart_query": PipelineDef(
        id="chart_query",
        description="Query DuckDB/Stockbit OHLCV saja (tanpa farm wajib).",
        steps=[
            ToolStep(
                "get_stockbit_ohlcv",
                lambda ctx: {
                    "symbol": ctx.primary_symbol() or "",
                    "timeframes": ",".join(
                        ctx.timeframes
                        or ([ctx.timeframe] if ctx.timeframe else ["1D"])
                    ),
                    "n": 40,
                    "refresh": False,
                },
                label="get Stockbit OHLCV",
            ),
        ],
        allowed_tools=[
            "get_stockbit_ohlcv",
            "query_chart_ohlcv",
            "chart_features",
            "chart_db_stats",
            "sync_chart_db",
            "farm_stockbit_charts",
        ],
    ),
    "reports_scrape": PipelineDef(
        id="reports_scrape",
        description=(
            "Stage Stream scrape; after confirmation write markdown + ingest Chroma."
        ),
        steps=[
            ToolStep("stockbit_scrape_reports", _reports_args, label="scrape reports"),
        ],
        allowed_tools=[
            "stockbit_scrape_reports",
            "stockbit_status",
            "search_stockbit_reports",
            "read_stockbit_scrape",
            "stockbit_scrape_stats",
        ],
    ),
    "document_ingest": PipelineDef(
        id="document_ingest",
        description=(
            "Stage simpan dokumen user ke DuckDB + knowledge/docs. "
            "TIDAK menulis DB sampai user konfirmasi ya/tidak di Telegram."
        ),
        steps=[
            ToolStep(
                "stage_document_ingest",
                lambda ctx: {
                    "title": ctx.extra.get("title") or "dokumen_user",
                    "body": ctx.extra.get("body") or "",
                    "category": ctx.extra.get("category") or "user_docs",
                    "source_name": ctx.extra.get("source_name") or "",
                },
                label="stage document",
            ),
        ],
        allowed_tools=[
            "stage_document_ingest",
            "search_documents",
            "doc_store_stats",
            "search_knowledge",
            "list_knowledge_topics",
        ],
    ),
}

# Intent → default pipeline id
INTENT_PIPELINE: dict[str, str] = {
    Intent.STOCKBIT_CHART.value: "chart_analyze",
    Intent.STOCKBIT_SCRAPE.value: "reports_scrape",
    Intent.DOCUMENT_INGEST.value: "document_ingest",
}


def get_pipeline(pipeline_id: str) -> PipelineDef | None:
    return PIPELINES.get(pipeline_id)


def pipeline_for_intent(intent: str | Intent) -> PipelineDef | None:
    key = intent.value if isinstance(intent, Intent) else str(intent)
    pid = INTENT_PIPELINE.get(key)
    return PIPELINES.get(pid) if pid else None


def describe_pipelines() -> str:
    lines = ["Structured tool pipelines (system-defined):"]
    for p in PIPELINES.values():
        tools = " -> ".join(s.tool for s in p.steps)
        lines.append(f"- {p.id}: {tools}")
        lines.append(f"  {p.description}")
    return "\n".join(lines)


def run_pipeline(
    pipeline_id: str,
    ctx: PipelineContext,
    *,
    dispatch: Callable[[str, dict[str, Any]], str],
    extract_files: Callable[[str], list[Path]] | None = None,
) -> PipelineResult:
    """Execute a named pipeline. `dispatch(tool_name, args) -> str`."""
    from agent_idx import chart_db

    pdef = PIPELINES.get(pipeline_id)
    run_id = uuid.uuid4().hex[:12]
    if not pdef:
        return PipelineResult(
            pipeline_id=pipeline_id,
            run_id=run_id,
            status="error",
            steps=[
                StepResult(
                    tool="?",
                    label="missing pipeline",
                    ok=False,
                    output=f"ERROR: pipeline tidak dikenal: {pipeline_id}",
                )
            ],
        )

    ctx.intent = ctx.intent or pipeline_id
    result = PipelineResult(pipeline_id=pipeline_id, run_id=run_id)
    tool_names: list[str] = []

    logger.info(
        "Pipeline start id=%s run=%s symbol=%s tf=%s",
        pipeline_id,
        run_id,
        ctx.primary_symbol(),
        ctx.timeframe,
    )

    for step in pdef.steps:
        tool_names.append(step.tool)
        if callable(step.args):
            args = step.args(ctx)
        else:
            args = dict(step.args)

        logger.info("Pipeline step tool=%s args=%s", step.tool, args)
        try:
            output = dispatch(step.tool, args)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline tool %s failed", step.tool)
            output = f"ERROR: {exc}"

        files: list[Path] = []
        if extract_files:
            try:
                files = extract_files(output)
            except Exception:  # noqa: BLE001
                files = []

        ok = not str(output).startswith("ERROR:") and "GAGAL" not in str(output)[:80]
        # Farm "0 bars" is failure for required farm step
        if step.tool == "farm_stockbit_charts" and "0 bars" in output:
            ok = False

        step_res = StepResult(
            tool=step.tool,
            label=step.label or step.tool,
            ok=ok,
            output=output or "",
            files=files,
        )
        result.steps.append(step_res)

        stop = any(s in output for s in step.stop_substrings)
        if stop:
            result.stopped_early = True
            result.status = "stopped"
            if "NEED_STOCKBIT" in output:
                result.need_stockbit_login = True
            break
        if not ok and step.required:
            result.stopped_early = True
            result.status = "failed"
            break

    if result.status == "ok" and result.steps and not all(s.ok for s in result.steps):
        result.status = "partial"

    try:
        chart_db.log_pipeline_run(
            run_id,
            intent=ctx.intent,
            pipeline=pipeline_id,
            tools=tool_names,
            status=result.status,
            detail=result.combined_text(max_per_step=800)[:3500],
        )
    except Exception:  # noqa: BLE001
        logger.debug("pipeline_runs log skipped", exc_info=True)

    logger.info(
        "Pipeline done id=%s run=%s status=%s steps=%s",
        pipeline_id,
        run_id,
        result.status,
        len(result.steps),
    )
    return result
