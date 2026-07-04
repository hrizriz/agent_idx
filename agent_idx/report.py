from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from fpdf import FPDF


def _pick_font() -> tuple[str, str | None, str | None]:
    """Return (family, regular_path, bold_path). Prefer Windows Arial."""
    candidates = [
        (
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
        ),
        (
            Path(r"C:\Windows\Fonts\segoeui.ttf"),
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.is_file():
            return "ReportFont", str(regular), str(bold) if bold.is_file() else None
    return "Helvetica", None, None


class _ReportPDF(FPDF):
    def footer(self) -> None:  # noqa: N802 — fpdf API
        self.set_y(-15)
        self.set_font(self._font_family, size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Halaman {self.page_no()}/{{nb}}", align="C")


def create_pdf_report(
    export_dir: Path,
    title: str,
    body: str,
    subtitle: str | None = None,
    filename_prefix: str = "laporan_idx",
) -> str:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    title = (title or "Laporan IDX").strip()
    body = (body or "").strip()
    if not body:
        raise ValueError("body laporan tidak boleh kosong")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", filename_prefix).strip("_") or "laporan_idx"
    path = export_dir / f"{safe_prefix}_{stamp}.pdf"

    family, regular, bold = _pick_font()
    pdf = _ReportPDF()
    pdf._font_family = family  # used in footer
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    if regular:
        pdf.add_font(family, "", regular)
        if bold:
            pdf.add_font(family, "B", bold)
        else:
            pdf.add_font(family, "B", regular)
    # else Helvetica built-in

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    def write_line(text: str, *, bold: bool = False, size: int = 11, h: float = 6) -> None:
        style = "B" if bold else ""
        pdf.set_font(family, style, size)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_width, h, text)

    write_line(title, bold=True, size=16, h=9)
    pdf.ln(2)

    meta = datetime.now().strftime("%Y-%m-%d %H:%M")
    if subtitle:
        meta = f"{subtitle.strip()} | {meta}"
    pdf.set_text_color(90, 90, 90)
    write_line(meta, size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for raw in body.splitlines():
        line = raw.rstrip()
        if not line:
            pdf.ln(3)
            continue

        heading = re.match(r"^\s{0,3}(#{1,4})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            size = 14 if level <= 2 else 12
            pdf.ln(2)
            write_line(text, bold=True, size=size, h=7)
            continue

        bullet = re.match(r"^\s*([-*•])\s+(.*)$", line)
        if bullet:
            write_line(f"- {bullet.group(2)}")
            continue

        # Strip simple markdown bold markers for PDF body.
        plain = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        write_line(plain)

    pdf.output(str(path))
    return f"OK: PDF dibuat\nFILE: {path}"


def export_csv(
    export_dir: Path,
    filename_prefix: str,
    csv_text: str,
) -> str:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", filename_prefix).strip("_") or "data_idx"
    path = export_dir / f"{safe_prefix}_{stamp}.csv"
    path.write_text(csv_text.strip() + "\n", encoding="utf-8")
    return f"OK: CSV dibuat\nFILE: {path}"


def export_query_csv(
    export_dir: Path,
    store,
    sql: str,
    filename_prefix: str = "query_idx",
) -> str:
    """Run read-only SQL and save full result as CSV (not row-capped for display)."""
    cleaned = sql.strip().rstrip(";")
    if not cleaned:
        raise ValueError("Empty SQL")
    if ";" in cleaned:
        raise ValueError("Only a single SQL statement is allowed")
    upper = cleaned.lstrip().upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("Only SELECT / WITH queries are allowed")

    # Reuse store safety by calling run_sql validation path via private checks.
    from agent_idx.data import _FORBIDDEN_SQL

    if _FORBIDDEN_SQL.search(cleaned):
        raise ValueError("Only read-only SELECT / WITH queries are allowed")

    df = store._con.execute(cleaned).fetchdf()
    if df is None or len(df) == 0:
        return "(no rows) Query tidak mengembalikan data; CSV tidak dibuat."

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", filename_prefix).strip("_") or "query_idx"
    path = export_dir / f"{safe_prefix}_{stamp}.csv"
    df.to_csv(path, index=False)
    return f"OK: CSV dibuat ({len(df)} baris)\nFILE: {path}"
