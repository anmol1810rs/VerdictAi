"""
backend/export/pdf_exporter.py — Story 3.1

Generates a white, 2-page PDF evaluation report using ReportLab Platypus.

Page 1: Header → Verdict → Score Breakdown Table → Cost Breakdown Table
Page 2: Per-Prompt Breakdown (top 5 by variance) → GT Alignment → Footer

Fits within 2 pages for a 10-prompt, 2-model run.
"""
from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Colours ────────────────────────────────────────────────────────────────

_GREEN       = colors.HexColor("#2e7d32")
_AMBER       = colors.HexColor("#e65100")
_LIGHT_GREEN = colors.HexColor("#e8f5e9")
_ROW_ALT     = colors.HexColor("#f5f5f5")
_HEADER_BG   = colors.HexColor("#eeeeee")
_BORDER      = colors.HexColor("#cccccc")
_DARK        = colors.HexColor("#1a1a1a")
_MED_GRAY    = colors.HexColor("#555555")
_LIGHT_GRAY  = colors.HexColor("#888888")

# ── Page geometry ──────────────────────────────────────────────────────────

_MARGIN  = 36   # 0.5 inch
_W, _H   = LETTER                       # 612 × 792 pt
_TW      = _W - 2 * _MARGIN            # 540 pt usable width

# ── Paragraph styles ───────────────────────────────────────────────────────

def _styles() -> dict:
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=_DARK,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=9,
            textColor=_MED_GRAY,
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=_DARK,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            textColor=_DARK,
            spaceAfter=2,
            leading=13,
        ),
        "body_green": ParagraphStyle(
            "body_green",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=_GREEN,
            spaceAfter=2,
        ),
        "body_amber": ParagraphStyle(
            "body_amber",
            fontName="Helvetica",
            fontSize=9,
            textColor=_AMBER,
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica",
            fontSize=8,
            textColor=_LIGHT_GRAY,
            spaceAfter=2,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=8,
            textColor=_MED_GRAY,
            alignment=TA_CENTER,
        ),
        "prompt_header": ParagraphStyle(
            "prompt_header",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=_DARK,
            spaceAfter=2,
        ),
    }


# ── Table style helpers ────────────────────────────────────────────────────

def _base_table_style(n_data_rows: int, winner_row: int | None = None) -> list:
    """Standard alternating-row table commands."""
    cmds = [
        # Header row
        ("BACKGROUND",   (0, 0), (-1, 0), _HEADER_BG),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 9),
        ("TEXTCOLOR",    (0, 0), (-1, 0), _DARK),
        # Body rows
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("TEXTCOLOR",    (0, 1), (-1, -1), _DARK),
        # Grid + padding
        ("GRID",         (0, 0), (-1, -1), 0.5, _BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(1, n_data_rows + 1):
        if (i % 2) == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), _ROW_ALT))
    if winner_row is not None:
        cmds.append(("BACKGROUND", (0, winner_row), (-1, winner_row), _LIGHT_GREEN))
        cmds.append(("TEXTCOLOR",  (0, winner_row), (-1, winner_row), _GREEN))
        cmds.append(("FONTNAME",   (0, winner_row), (-1, winner_row), "Helvetica-Bold"))
    return cmds


def _small_table_style() -> list:
    """Compact 8pt table for per-prompt data."""
    return [
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",    (0, 0), (-1, -1), _DARK),
        ("BACKGROUND",   (0, 0), (-1, 0), _HEADER_BG),
        ("GRID",         (0, 0), (-1, -1), 0.4, _BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]


# ── Helpers ────────────────────────────────────────────────────────────────

def _trunc(text: str | None, n: int = 40) -> str:
    """Truncate to n chars with ellipsis."""
    if not text:
        return ""
    s = str(text).strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def _fmt_score(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_cost(v) -> str:
    if v is None:
        return "—"
    try:
        return f"${float(v):.6f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_int(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "—"


# ── Main entry point ────────────────────────────────────────────────────────


def generate_pdf_bytes(run_id: str, db) -> bytes:
    """
    Query the DB for all run data and return a PDF as raw bytes.
    Raises ValueError if the run is not found or not complete.
    """
    from backend.db.models import EvalRun, ModelResult, Prompt, Verdict

    # ── Fetch data ──────────────────────────────────────────────────────────
    run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise ValueError(f"Run '{run_id}' not found.")

    verdict_row = db.query(Verdict).filter(Verdict.eval_run_id == run_id).first()
    results     = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
    prompts     = db.query(Prompt).filter(Prompt.eval_run_id == run_id).all()

    prompt_by_id = {p.id: p for p in prompts}
    models       = sorted({r.model_name for r in results})
    score_bd     = (verdict_row.score_breakdown or {}) if verdict_row else {}
    cost_comp    = (verdict_row.cost_comparison or {}) if verdict_row else {}
    hall_warns   = (verdict_row.hallucination_warnings or []) if verdict_row else []
    winning_model = verdict_row.winning_model if verdict_row else (models[0] if models else "—")
    summary_text  = (verdict_row.summary or "") if verdict_row else ""

    # ── Aggregate cost & token data from ModelResult ───────────────────────
    cost_by_model: dict    = defaultdict(float)
    tin_by_model: dict     = defaultdict(int)
    tout_by_model: dict    = defaultdict(int)
    dim_sum: dict          = defaultdict(lambda: defaultdict(float))
    dim_cnt: dict          = defaultdict(lambda: defaultdict(int))
    DIMS = ["accuracy", "hallucination", "instruction_following", "conciseness"]

    for r in results:
        m = r.model_name
        cost_by_model[m] += r.cost_usd or 0.0
        tin_by_model[m]  += r.tokens_in or 0
        tout_by_model[m] += r.tokens_out or 0
        for d in DIMS:
            v = (r.dimension_scores or {}).get(d)
            if v is not None:
                dim_sum[m][d] += float(v)
                dim_cnt[m][d] += 1

    # ── Per-prompt data sorted by variance ────────────────────────────────
    prompts_map: dict = defaultdict(list)
    for r in results:
        prompts_map[int(r.prompt_index)].append(r)

    def _variance(idx):
        rows = prompts_map[idx]
        vs = [r.variance_score for r in rows if r.variance_score is not None]
        return vs[0] if vs else 0.0

    sorted_indices = sorted(prompts_map.keys(), key=_variance, reverse=True)
    total_prompts  = len(sorted_indices)
    top_indices    = sorted_indices[:5]      # at most 5 prompts on page 2

    # ── GT data ────────────────────────────────────────────────────────────
    gt_sum: dict = defaultdict(float)
    gt_cnt: dict = defaultdict(int)
    gt_reasoning_by_model: dict = defaultdict(list)
    for r in results:
        if r.ground_truth_score is not None:
            gt_sum[r.model_name] += float(r.ground_truth_score)
            gt_cnt[r.model_name] += 1
            if r.ground_truth_reasoning:
                gt_reasoning_by_model[r.model_name].append(r.ground_truth_reasoning)
    has_gt = bool(gt_cnt)

    # ── Pricing last updated ───────────────────────────────────────────────
    pricing_last_updated = "—"
    try:
        import os, yaml
        pricing_path = os.path.join(os.path.dirname(__file__), "..", "..", "pricing.yaml")
        with open(pricing_path) as f:
            p_cfg = yaml.safe_load(f)
        pricing_last_updated = p_cfg.get("meta", {}).get("last_updated", "—")
    except Exception:
        pass

    # ── Build PDF ──────────────────────────────────────────────────────────
    buf   = BytesIO()
    S     = _styles()
    story = []

    # ╔══════════════════════════════════════════════════════════════════════╗
    # PAGE 1
    # ╚══════════════════════════════════════════════════════════════════════╝

    # ── Header ─────────────────────────────────────────────────────────────
    run_label = run.custom_label or f"Run {run_id[:8]}"
    run_date  = run.created_at.strftime("%d %b %Y") if run.created_at else "—"
    modality  = run.modality or "text"
    models_str = ", ".join(models) if models else "—"

    story.append(Paragraph("⚖️ VerdictAI — Evaluation Report", S["title"]))
    story.append(Paragraph(
        f"<b>{run_label}</b> &nbsp;|&nbsp; {run_date} &nbsp;|&nbsp; "
        f"Modality: {modality} &nbsp;|&nbsp; Models: {models_str}",
        S["subtitle"],
    ))
    story.append(HRFlowable(width=_TW, thickness=1, color=_BORDER, spaceAfter=6))

    # ── Verdict ─────────────────────────────────────────────────────────────
    story.append(Paragraph("VERDICT", S["section"]))
    story.append(Paragraph(f"Winner: {winning_model}", S["body_green"]))

    # Overall score from score_breakdown
    winner_score = None
    if winning_model in score_bd:
        winner_score = score_bd[winning_model].get("final_score")
    if winner_score is not None:
        story.append(Paragraph(f"Overall score: {_fmt_score(winner_score)}/10", S["body"]))

    # Verdict summary (can be multi-line)
    for line in summary_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("⚠️"):
            story.append(Paragraph(stripped, S["body_amber"]))
        elif stripped:
            story.append(Paragraph(stripped, S["body"]))

    for warning in hall_warns:
        story.append(Paragraph(f"⚠  {warning}", S["body_amber"]))

    story.append(Spacer(1, 6))

    # ── Score Breakdown Table ───────────────────────────────────────────────
    story.append(Paragraph("SCORE BREAKDOWN", S["section"]))

    DIM_SHORT = {
        "accuracy": "Acc",
        "hallucination": "Hall",
        "instruction_following": "Inst",
        "conciseness": "Con",
    }
    score_headers = ["Model", "Acc", "Hall", "Inst", "Con", "Cost Eff", "Total"]
    score_col_w   = [120, 60, 60, 60, 60, 70, 60]   # = 490 → centered in 540

    score_data = [score_headers]
    winner_row_idx = None
    for i, m in enumerate(models, start=1):
        dims_data = score_bd.get(m, {}).get("dimensions", {})
        ce_score  = score_bd.get(m, {}).get("cost_efficiency_score")
        total     = score_bd.get(m, {}).get("final_score")
        row = [
            m,
            _fmt_score(dims_data.get("accuracy")),
            _fmt_score(dims_data.get("hallucination")),
            _fmt_score(dims_data.get("instruction_following")),
            _fmt_score(dims_data.get("conciseness")),
            _fmt_score(ce_score),
            _fmt_score(total),
        ]
        score_data.append(row)
        if m == winning_model:
            winner_row_idx = i

    t = Table(score_data, colWidths=score_col_w, hAlign="LEFT")
    t.setStyle(TableStyle(_base_table_style(len(models), winner_row=winner_row_idx)))
    story.append(t)
    story.append(Spacer(1, 6))

    # ── Cost Breakdown Table ────────────────────────────────────────────────
    story.append(Paragraph("COST BREAKDOWN", S["section"]))

    cost_headers = ["Model", "Total $", "Tokens In", "Tokens Out", "$/1K tokens", "$/Quality Pt"]
    cost_col_w   = [120, 70, 75, 80, 85, 90]   # = 520

    cost_data = [cost_headers]
    for m in models:
        total_c = cost_by_model[m]
        tin     = tin_by_model[m]
        tout    = tout_by_model[m]
        total_t = tin + tout
        cost_1k = (total_c / total_t * 1000) if total_t > 0 else 0.0
        # quality from score_breakdown final_score
        q_score = score_bd.get(m, {}).get("final_score") or 1.0
        cpp     = total_c / q_score if q_score > 0 else 0.0
        cost_data.append([
            m,
            _fmt_cost(total_c),
            _fmt_int(tin),
            _fmt_int(tout),
            _fmt_cost(cost_1k),
            _fmt_cost(cpp),
        ])

    ct = Table(cost_data, colWidths=cost_col_w, hAlign="LEFT")
    ct.setStyle(TableStyle(_base_table_style(len(models))))
    story.append(ct)
    story.append(Paragraph(f"Prices last updated: {pricing_last_updated}", S["caption"]))

    callout = cost_comp.get("callout", "")
    if callout:
        story.append(Paragraph(f"💡  {callout}", S["body"]))

    # ╔══════════════════════════════════════════════════════════════════════╗
    # PAGE 2
    # ╚══════════════════════════════════════════════════════════════════════╝
    story.append(PageBreak())

    # ── Per-Prompt Breakdown ────────────────────────────────────────────────
    story.append(Paragraph("PER-PROMPT BREAKDOWN", S["section"]))
    if total_prompts > 5:
        story.append(Paragraph(
            f"Showing top 5 of {total_prompts} prompts. "
            "Full data available in JSON export.",
            S["caption"],
        ))
    story.append(Spacer(1, 4))

    pp_col_w  = [110, 100, 100, 110, 110]   # Model | Acc | Hall | Inst | Con
    pp_header = ["Model", "Acc", "Hall", "Inst", "Con"]

    for idx in top_indices:
        prompt_rows = prompts_map[idx]
        if not prompt_rows:
            continue
        prompt_rec = prompt_by_id.get(prompt_rows[0].prompt_id)
        prompt_text = (prompt_rec.prompt_text if prompt_rec else f"Prompt {idx}") or f"Prompt {idx}"
        variance   = _variance(idx)
        variance_badge = " ⚡" if variance and variance >= 0.5 else ""
        header_text = f"Prompt {idx}{variance_badge}: {_trunc(prompt_text, 80)}"
        story.append(Paragraph(header_text, S["prompt_header"]))

        # Build score + reasoning table for this prompt
        # Rows: header, then per-model: [score row, reasoning row]
        pp_data = [pp_header]
        pp_styles_cmds = _small_table_style()

        for model_idx, model in enumerate(models):
            # Get this model's result for this prompt
            m_results = [r for r in prompt_rows if r.model_name == model]
            if not m_results:
                continue
            mr = m_results[0]
            ds = mr.dimension_scores or {}
            dr = mr.dimension_reasoning or {}

            score_row = [
                model,
                _fmt_score(ds.get("accuracy")),
                _fmt_score(ds.get("hallucination")),
                _fmt_score(ds.get("instruction_following")),
                _fmt_score(ds.get("conciseness")),
            ]
            # Reasoning row: spans as one merged cell per dim
            reasoning_row = [
                "  ↳",
                _trunc(dr.get("accuracy"), 30),
                _trunc(dr.get("hallucination"), 30),
                _trunc(dr.get("instruction_following"), 30),
                _trunc(dr.get("conciseness"), 30),
            ]
            base_row = len(pp_data)
            pp_data.append(score_row)
            pp_data.append(reasoning_row)

            # Style the reasoning row (smaller, gray)
            reasoning_row_idx = base_row + 1
            pp_styles_cmds.append(
                ("FONTSIZE", (0, reasoning_row_idx), (-1, reasoning_row_idx), 7)
            )
            pp_styles_cmds.append(
                ("TEXTCOLOR", (0, reasoning_row_idx), (-1, reasoning_row_idx), _LIGHT_GRAY)
            )
            pp_styles_cmds.append(
                ("TOPPADDING", (0, reasoning_row_idx), (-1, reasoning_row_idx), 1)
            )
            pp_styles_cmds.append(
                ("BOTTOMPADDING", (0, reasoning_row_idx), (-1, reasoning_row_idx), 2)
            )
            # Alternate shading for model blocks
            if model_idx % 2 == 1:
                pp_styles_cmds.append(
                    ("BACKGROUND", (0, base_row), (-1, base_row + 1), _ROW_ALT)
                )

        if len(pp_data) > 1:
            pt_tbl = Table(pp_data, colWidths=pp_col_w, hAlign="LEFT")
            pt_tbl.setStyle(TableStyle(pp_styles_cmds))
            story.append(pt_tbl)
        story.append(Spacer(1, 4))

    # ── GT Alignment ───────────────────────────────────────────────────────
    if has_gt:
        story.append(Spacer(1, 6))
        story.append(Paragraph("GT ALIGNMENT", S["section"]))

        gt_headers = ["Model", "Avg GT Score", "Sample Reasoning"]
        gt_col_w   = [130, 100, 300]
        gt_data    = [gt_headers]

        best_gt_model = max(gt_cnt, key=lambda m: gt_sum[m] / gt_cnt[m]) if gt_cnt else None
        for m in models:
            if gt_cnt[m] == 0:
                continue
            avg_gt = gt_sum[m] / gt_cnt[m]
            sample_r = gt_reasoning_by_model[m][0] if gt_reasoning_by_model[m] else "—"
            gt_data.append([m, f"{avg_gt:.1f}/10", _trunc(sample_r, 60)])

        if len(gt_data) > 1:
            gt_table = Table(gt_data, colWidths=gt_col_w, hAlign="LEFT")
            cmds = _base_table_style(len(gt_data) - 1, winner_row=None)
            # Highlight best GT model
            for i, row in enumerate(gt_data[1:], start=1):
                if row[0] == best_gt_model:
                    cmds.append(("BACKGROUND", (0, i), (-1, i), _LIGHT_GREEN))
                    cmds.append(("TEXTCOLOR",  (0, i), (-1, i), _GREEN))
                    cmds.append(("FONTNAME",   (0, i), (-1, i), "Helvetica-Bold"))
            gt_table.setStyle(TableStyle(cmds))
            story.append(gt_table)

    # ── Footer ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width=_TW, thickness=0.5, color=_BORDER))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "github.com/anmol1810rs/verdictai  |  "
        "Generated by VerdictAI — open source LLM evaluation engine",
        S["footer"],
    ))

    # ── Compile ─────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
    )
    doc.build(story)
    return buf.getvalue()
