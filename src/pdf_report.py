# src/pdf_report.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def _fmt_dt(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        return ts_iso


def _safe(v, default: str = "—") -> str:
    if v is None:
        return default
    try:
        if isinstance(v, float) and pd.isna(v):
            return default
    except Exception:
        pass
    s = str(v).strip()
    return default if s == "" else s


def _fmt_float(v: Any, ndigits: int = 4, default: str = "—") -> str:
    try:
        if v is None:
            return default
        if isinstance(v, float) and pd.isna(v):
            return default
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return default


def _clip(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def _wrap_text_limited(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    line_height: float,
    font_name: str,
    font_size: float,
    max_lines: int,
) -> float:
    """
    Wrap text within max_width and max_lines.
    If truncated, last line ends with "…".
    Returns new y after drawing.
    """
    txt = (text or "").strip()
    if not txt:
        return y

    words = txt.split()
    c.setFont(font_name, font_size)

    lines: List[str] = []
    line = ""
    idx = 0
    while idx < len(words):
        w = words[idx]
        test = (line + " " + w).strip()
        if c.stringWidth(test, font_name, font_size) <= max_width:
            line = test
            idx += 1
        else:
            if line:
                lines.append(line)
                line = ""
                if len(lines) >= max_lines:
                    break
            else:
                # single word too long -> clip
                clipped = w
                while clipped and c.stringWidth(clipped + "…", font_name, font_size) > max_width:
                    clipped = clipped[:-1]
                lines.append((clipped + "…") if clipped else "…")
                idx += 1
                if len(lines) >= max_lines:
                    break

    if len(lines) < max_lines and line:
        lines.append(line)

    if idx < len(words) and lines:
        last = lines[-1]
        while last and c.stringWidth(last + "…", font_name, font_size) > max_width:
            last = last[:-1].rstrip()
        lines[-1] = (last + "…") if last else "…"

    for ln in lines:
        c.drawString(x, y, ln)
        y -= line_height
    return y


def _draw_image_fit(
    c: canvas.Canvas,
    img_path: Path,
    x: float,
    y: float,
    w: float,
    h: float,
) -> bool:
    """
    Draw an image to fit inside (x, y, w, h) preserving aspect ratio,
    centered both horizontally and vertically.
    Returns True if image drawn.
    """
    try:
        if not img_path.exists():
            return False
        reader = ImageReader(str(img_path))
        iw, ih = reader.getSize()
        if iw <= 0 or ih <= 0:
            return False

        scale = min(w / iw, h / ih)
        dw = iw * scale
        dh = ih * scale
        dx = x + (w - dw) / 2
        dy = y + (h - dh) / 2

        c.drawImage(
            reader,
            dx,
            dy,
            width=dw,
            height=dh,
            preserveAspectRatio=True,
            mask="auto",
        )
        return True
    except Exception:
        return False


def default_report_path(report_id: str) -> Path:
    return REPORTS_DIR / f"{report_id}.pdf"


# -----------------------------
# Main PDF generator (1 page)
# -----------------------------
def generate_clinical_pdf(
    *,
    out_path: Path,
    patient: Dict[str, Any],
    doctor: Dict[str, Any],
    encounter: Dict[str, Any],
    risk_factors: List[str],
    uploaded_image_path: Optional[Path] = None,
    gradcam_overlay_path: Optional[Path] = None,
    clinician_judgement: str = "",
) -> Path:
    c = canvas.Canvas(str(out_path), pagesize=A4)
    W, H = A4

    # Palette
    BLUE = colors.HexColor("#0B3B5B")
    INK = colors.HexColor("#0F172A")
    MUTED = colors.HexColor("#475569")
    BORDER = colors.HexColor("#D0D7DE")
    CARD_BG = colors.HexColor("#F8FAFC")
    SOFT = colors.HexColor("#EEF2F7")

    FONT_H = "Helvetica-Bold"
    FONT_SB = "Helvetica-Bold"
    FONT_B = "Helvetica"

    margin = 13 * mm
    x0 = margin
    x1 = W - margin
    y = H - margin

    # ---------------- Header ----------------
    header_h = 18 * mm
    c.setFillColor(BLUE)
    c.rect(0, H - header_h, W, header_h, stroke=0, fill=1)

    c.setFillColor(colors.white)
    c.setFont(FONT_H, 14)
    c.drawString(x0, H - 11.5 * mm, "Retina-AI — Diabetic Retinopathy Screening Report")
    c.setFont(FONT_B, 8.5)
    c.drawString(x0, H - 15.8 * mm, "AI-assisted screening • For clinical decision support only")

    # ---------------- Meta row ----------------
    y = H - header_h - 6 * mm

    report_id = _safe(encounter.get("report_id"))
    ts = _fmt_dt(_safe(encounter.get("timestamp_utc"), ""))
    model_version = _safe(encounter.get("model_version"))

    c.setFillColor(INK)
    c.setFont(FONT_B, 9)
    c.drawString(x0, y, f"Report ID: {report_id}")
    c.drawRightString(x1, y, f"Generated: {ts}")
    y -= 4.8 * mm
    c.drawString(x0, y, f"Model: {model_version}")
    y -= 6.0 * mm

    # ---------------- helpers ----------------
    def section(title: str) -> None:
        nonlocal y
        c.setFillColor(INK)
        c.setFont(FONT_H, 10.5)
        c.drawString(x0, y, title)
        y -= 3.2 * mm
        c.setStrokeColor(BORDER)
        c.line(x0, y, x1, y)
        y -= 5.0 * mm

    def card(x: float, y_top: float, w: float, h: float, title: str) -> None:
        c.setFillColor(CARD_BG)
        c.setStrokeColor(BORDER)
        c.roundRect(x, y_top - h, w, h, 6, stroke=1, fill=1)
        c.setFillColor(INK)
        c.setFont(FONT_SB, 9.5)
        c.drawString(x + 3.2 * mm, y_top - 4.4 * mm, title)

    def kv_rows(
        x: float,
        y_top: float,
        w: float,
        rows: List[tuple[str, str]],
        *,
        start_offset: float,
        step: float,
    ) -> None:
        """
        Cleanly aligned key/value rows inside a card.
        """
        key_x = x + 3.2 * mm
        val_x = x + w - 3.2 * mm
        yy = y_top - start_offset

        for k, v in rows:
            c.setFont(FONT_B, 8.7)
            c.setFillColor(MUTED)
            c.drawString(key_x, yy, k)

            c.setFont(FONT_SB, 8.7)
            c.setFillColor(INK)
            c.drawRightString(val_x, yy, _clip(v, 36))
            yy -= step

    # ---------------- Patient & Clinician ----------------
    section("Patient & Clinician Details")

    gap = 6 * mm
    col_w = (x1 - x0 - gap) / 2
    # slightly taller to avoid last-row cutoff
    card_h = 34 * mm

    left_x = x0
    right_x = x0 + col_w + gap
    y_top = y

    card(left_x, y_top, col_w, card_h, "Patient")
    card(right_x, y_top, col_w, card_h, "Clinician")

    p_id = _safe(patient.get("patient_id"))
    p_name = _safe(patient.get("name"))
    p_age = _safe(patient.get("age"))
    p_sex = _safe(patient.get("sex"))
    p_dur = _safe(patient.get("diabetes_years"))
    p_htn = "Yes" if bool(patient.get("hypertension")) else "No"
    p_a1c = _fmt_float(patient.get("last_hba1c"), 1)

    kv_rows(
        left_x,
        y_top,
        col_w,
        [
            ("Patient ID", p_id),
            ("Name", p_name),
            ("Age / Sex", f"{p_age} / {p_sex}"),
            ("DM Duration", f"{p_dur} yrs"),
            ("Hypertension", p_htn),
            ("HbA1c (last)", p_a1c),
        ],
        start_offset=10.0 * mm,
        step=4.45 * mm,
    )

    d_id = _safe(doctor.get("doctor_id"))
    d_name = _safe(doctor.get("name"))
    d_qual = _safe(doctor.get("qualification"))
    d_fac = _safe(doctor.get("hospital"))

    kv_rows(
        right_x,
        y_top,
        col_w,
        [
            ("Doctor ID", d_id),
            ("Name", d_name),
            ("Qualification", d_qual),
            ("Facility", d_fac),
        ],
        start_offset=10.0 * mm,
        step=4.75 * mm,
    )

    y = y_top - card_h - 7 * mm

    # ---------------- Screening summary ----------------
    section("Screening Summary")

    pred = _safe(encounter.get("prediction"))
    p_dr = float(encounter.get("p_dr", 0.0) or 0.0)
    conf = float(encounter.get("confidence", 0.0) or 0.0)
    risk_level = _safe(encounter.get("risk_level"))
    followup = _safe(encounter.get("followup_timeline"))
    reco = _safe(encounter.get("recommendation"))

    # Summary bar
    bar_h = 12 * mm
    c.setFillColor(SOFT)
    c.setStrokeColor(BORDER)
    c.roundRect(x0, y - bar_h, x1 - x0, bar_h, 6, stroke=1, fill=1)

    badge_color = {
        "Low": colors.HexColor("#1F7A1F"),
        "Mild": colors.HexColor("#B57F00"),
        "Moderate": colors.HexColor("#C05621"),
        "High": colors.HexColor("#B91C1C"),
    }.get(risk_level, BLUE)

    bx = x0 + 3.2 * mm
    by = y - 3.0 * mm
    badge_w = 36 * mm
    badge_h = 8 * mm

    c.setFillColor(badge_color)
    c.roundRect(bx, by - badge_h, badge_w, badge_h, 4, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont(FONT_H, 9)
    c.drawCentredString(bx + badge_w / 2, by - 5.8 * mm, f"RISK: {risk_level.upper()}")

    c.setFillColor(INK)
    c.setFont(FONT_SB, 9)
    c.drawString(bx + badge_w + 4.5 * mm, y - 7.4 * mm, f"Prediction: {pred}")

    c.setFont(FONT_B, 9)
    c.drawString(bx + badge_w + 44 * mm, y - 7.4 * mm, f"p(DR): {_fmt_float(p_dr, 4)}")

    c.drawRightString(
        x1 - 3.2 * mm,
        y - 7.4 * mm,
        f"Confidence: {conf*100:.2f}%  •  Follow-up: {followup}",
    )

    y -= (bar_h + 5.5 * mm)

    # Recommendation box (more height + clean wrap)
    rec_h = 24 * mm
    c.setFillColor(colors.white)
    c.setStrokeColor(BORDER)
    c.roundRect(x0, y - rec_h, x1 - x0, rec_h, 6, stroke=1, fill=1)

    c.setFillColor(INK)
    c.setFont(FONT_SB, 9.5)
    c.drawString(x0 + 3.2 * mm, y - 4.2 * mm, "Recommendation")

    c.setFont(FONT_B, 9)
    _wrap_text_limited(
        c,
        reco,
        x0 + 3.2 * mm,
        y - 9.2 * mm,
        max_width=(x1 - x0) - 6.4 * mm,
        line_height=4.2 * mm,
        font_name=FONT_B,
        font_size=9,
        max_lines=4,
    )

    y -= (rec_h + 6.0 * mm)

    # ---------------- Risk factors + Image quality (2 cards) ----------------
    left_w = (x1 - x0 - gap) / 2
    right_w = left_w
    block_h = 30 * mm  # a bit taller to avoid overlap
    y_top = y

    # Risk factors card
    c.setFillColor(colors.white)
    c.setStrokeColor(BORDER)
    c.roundRect(x0, y_top - block_h, left_w, block_h, 6, stroke=1, fill=1)
    c.setFillColor(INK)
    c.setFont(FONT_SB, 9.5)
    c.drawString(x0 + 3.2 * mm, y_top - 4.2 * mm, "Risk factors")

    rf = [str(r).strip() for r in (risk_factors or []) if str(r).strip()]
    rf = rf[:6]

    c.setFont(FONT_B, 8.8)
    c.setFillColor(INK)

    rf_x = x0 + 3.2 * mm
    rf_y = y_top - 10.0 * mm
    step = 4.5 * mm

    for i, item in enumerate(rf):
        if i >= 6:
            break
        yy = rf_y - step * i
        c.drawString(rf_x, yy, f"• {_clip(item, 52)}")

    if not rf:
        c.setFillColor(MUTED)
        c.drawString(rf_x, rf_y, "• Not available")
        c.setFillColor(INK)

    # Image quality card (FIXED: no overlap, clean stacked rows)
    qx = x0 + left_w + gap
    c.setFillColor(colors.white)
    c.setStrokeColor(BORDER)
    c.roundRect(qx, y_top - block_h, right_w, block_h, 6, stroke=1, fill=1)

    c.setFillColor(INK)
    c.setFont(FONT_SB, 9.5)
    c.drawString(qx + 3.2 * mm, y_top - 4.2 * mm, "Image quality & gates")

    q_enabled = bool(encounter.get("quality_enabled"))
    q_flagged = bool(encounter.get("quality_flagged"))
    q_reason = _safe(encounter.get("quality_reason"))

    bm = encounter.get("brightness_mean")
    cs = encounter.get("contrast_std")
    sp = encounter.get("sharpness_proxy")

    # Top line
    c.setFillColor(MUTED)
    c.setFont(FONT_B, 8.6)
    c.drawString(qx + 3.2 * mm, y_top - 9.8 * mm, f"Quality gate: {'On' if q_enabled else 'Off'}  •  Flagged: {'Yes' if q_flagged else 'No'}")

    # KPI lines
    base_y = y_top - 15.0 * mm
    c.setFillColor(INK)
    c.setFont(FONT_SB, 8.7)
    c.drawString(qx + 3.2 * mm, base_y, f"Brightness: {_fmt_float(bm, 4)}")
    c.drawString(qx + 3.2 * mm, base_y - 4.6 * mm, f"Contrast: {_fmt_float(cs, 4)}")
    c.drawString(qx + 3.2 * mm, base_y - 9.2 * mm, f"Sharpness: {_fmt_float(sp, 4)}")

    # Reason (bounded)
    c.setFillColor(MUTED)
    c.setFont(FONT_B, 8.6)
    c.drawString(qx + 3.2 * mm, y_top - 28.5 * mm, f"Reason: {_clip(q_reason, 60)}")

    y = y_top - block_h - 6.0 * mm

    # ---------------- Explainability images ----------------
    section("Explainability (Grad-CAM)")

    tile_h = 42 * mm  # bigger so fundus is clearly visible
    tile_w = (x1 - x0 - gap) / 2
    y_top = y

    def draw_tile(x: float, title: str, img_path: Optional[Path]) -> None:
        c.setFillColor(colors.white)
        c.setStrokeColor(BORDER)
        c.roundRect(x, y_top - tile_h, tile_w, tile_h, 6, stroke=1, fill=1)

        c.setFillColor(INK)
        c.setFont(FONT_SB, 9)
        c.drawString(x + 3.2 * mm, y_top - 4.2 * mm, title)

        # Inner image box (larger + centered fit)
        pad = 3.2 * mm
        img_box_x = x + pad
        img_box_y = (y_top - tile_h) + pad
        img_box_w = tile_w - 2 * pad
        img_box_h = tile_h - (pad * 2) - 6.0 * mm  # reserve header title space

        # shift box up to sit under title neatly
        img_box_y += 2.0 * mm

        c.setStrokeColor(colors.HexColor("#E5E7EB"))
        c.rect(img_box_x, img_box_y, img_box_w, img_box_h, stroke=1, fill=0)

        ok = False
        if img_path is not None:
            ok = _draw_image_fit(c, img_path, img_box_x, img_box_y, img_box_w, img_box_h)

        if not ok:
            c.setFillColor(MUTED)
            c.setFont(FONT_B, 8)
            c.drawString(img_box_x + 2 * mm, img_box_y + img_box_h - 6 * mm, "Image not available")
            c.setFillColor(INK)

    draw_tile(x0, "Model input (224×224)", uploaded_image_path)
    draw_tile(x0 + tile_w + gap, "Grad-CAM overlay", gradcam_overlay_path)

    y = y_top - tile_h - 6.0 * mm

    # ---------------- Clinician Judgement ----------------
    section("Clinician Judgement")

    footer_line_y = 12.8 * mm
    available = y - (footer_line_y + 6.0 * mm)
    box_h = max(18 * mm, min(30 * mm, available))

    c.setFillColor(colors.white)
    c.setStrokeColor(BORDER)
    c.roundRect(x0, y - box_h, x1 - x0, box_h, 6, stroke=1, fill=1)

    inner_x = x0 + 3.2 * mm
    inner_w = (x1 - x0) - 6.4 * mm

    if (clinician_judgement or "").strip():
        c.setFillColor(INK)
        c.setFont(FONT_B, 9)
        _wrap_text_limited(
            c,
            clinician_judgement.strip(),
            inner_x,
            y - 6.2 * mm,
            max_width=inner_w,
            line_height=4.2 * mm,
            font_name=FONT_B,
            font_size=9,
            max_lines=5,
        )
    else:
        c.setStrokeColor(colors.HexColor("#E5E7EB"))
        line_gap = 6.0 * mm
        yy = y - 9.0 * mm
        while yy > (y - box_h + 5.5 * mm):
            c.line(inner_x, yy, x1 - 3.2 * mm, yy)
            yy -= line_gap

    # ---------------- Footer disclaimer ----------------
    c.setStrokeColor(BORDER)
    c.line(x0, footer_line_y, x1, footer_line_y)

    disclaimer = (
        "Disclaimer: This report is generated by an AI model for decision support. "
        "Final diagnosis and management remain the responsibility of a qualified clinician."
    )

    c.setFillColor(MUTED)
    c.setFont(FONT_B, 7.5)
    _wrap_text_limited(
        c,
        disclaimer,
        x0,
        9.6 * mm,
        max_width=x1 - x0,
        line_height=3.1 * mm,
        font_name=FONT_B,
        font_size=7.5,
        max_lines=2,
    )

    # Single page only
    c.showPage()
    c.save()
    return out_path
