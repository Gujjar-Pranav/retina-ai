# src/ui_screening.py
from __future__ import annotations
import io

import base64
import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torch

from src.clinical_records import Encounter, append_encounter
from src.pdf_report import generate_clinical_pdf
from src.screening_core import (
    IMG_SIZE,
    compute_quality_metrics,
    predict_2class,
    risk_stratification,
    build_recommendation,
    derive_risk_factors,
    pil_to_tensor,
    GradCAM,
    find_last_conv_layer,
    overlay_cam_on_image,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _html(s: str) -> str:
    out = dedent(s).strip()
    out = re.sub(r"<!--.*?-->", "", out, flags=re.DOTALL)
    return out


def _safe_pdf_name(patient_id: str, report_id: str, ts_iso: str) -> str:
    try:
        d = datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).astimezone().strftime("%Y%m%d")
    except Exception:
        d = datetime.now().strftime("%Y%m%d")
    pid = str(patient_id).strip().replace(" ", "_")
    return f"{pid}_{d}_{report_id}.pdf"


def _pdf_preview(path: Path, max_pages: int = 2, zoom: float = 1.6) -> None:
    """
    Chrome-safe preview: render PDF pages as images using PyMuPDF.
    Shows first `max_pages` pages.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        st.info("Preview needs PyMuPDF. Install with: pip install pymupdf")
        return

    try:
        doc = fitz.open(str(path))
        pages = min(len(doc), max_pages)
        for i in range(pages):
            page = doc.load_page(i)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_bytes = pix.tobytes("png")
            st.image(img_bytes, use_container_width=True)

        if len(doc) > max_pages:
            st.caption(f"Showing first {max_pages} pages. Download to view full report.")
        doc.close()
    except Exception as e:
        st.warning(f"Preview failed: {e}")


def _find_pdf_for_row(reports_dir: Path, row: dict) -> Optional[Path]:
    """
    Robust lookup for history PDFs:
    1) If pdf_filename exists and file exists -> use it
    2) Else fallback: search reports_dir for *{report_id}*.pdf
    """
    pdf_name = str(row.get("pdf_filename", "") or "").strip()
    if pdf_name:
        p = reports_dir / pdf_name
        if p.exists():
            return p

    rid = str(row.get("report_id", "") or "").strip()
    if rid:
        matches = sorted(reports_dir.glob(f"*{rid}*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]

    return None


def render_screening(
    *,
    ROOT: Path,
    model: torch.nn.Module,
    backend: str,
    device: torch.device,
) -> None:
    data_dir = ROOT / "data"
    patients_xlsx = data_dir / "patients.xlsx"
    doctors_xlsx = data_dir / "doctors.xlsx"
    history_xlsx = data_dir / "patient_history.xlsx"
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load registry ----------
    p_df = pd.read_excel(patients_xlsx, engine="openpyxl") if patients_xlsx.exists() else pd.DataFrame()
    d_df = pd.read_excel(doctors_xlsx, engine="openpyxl") if doctors_xlsx.exists() else pd.DataFrame()

    if p_df.empty or d_df.empty:
        st.warning("Please create at least one patient and one clinician in **Registry**.")
        return

    # ---------- Order + history card ----------
    with st.container(border=True):
        st.markdown("### Order details")
        st.caption("Select patient & clinician. History & PDF access are available below.")

        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            patient_choice = st.selectbox(
                "Patient",
                options=p_df["patient_id"].astype(str).tolist(),
                format_func=lambda x: f"{x} — {p_df.loc[p_df.patient_id.astype(str) == x, 'name'].values[0]}",
            )

        with c2:
            doctor_choice = st.selectbox(
                "Clinician",
                options=d_df["doctor_id"].astype(str).tolist(),
                format_func=lambda x: f"{x} — {d_df.loc[d_df.doctor_id.astype(str) == x, 'name'].values[0]}",
            )

        patient_row = p_df[p_df["patient_id"].astype(str) == patient_choice].iloc[0].to_dict()
        doctor_row = d_df[d_df["doctor_id"].astype(str) == doctor_choice].iloc[0].to_dict()

        # Load history
        hx = pd.DataFrame()
        if history_xlsx.exists():
            h_df = pd.read_excel(history_xlsx, engine="openpyxl")
            if not h_df.empty and "patient_id" in h_df.columns:
                hx = h_df[h_df["patient_id"].astype(str) == str(patient_choice)].copy()

        if not hx.empty:
            if "timestamp_utc" in hx.columns:
                hx["timestamp_utc"] = hx["timestamp_utc"].astype(str)
                hx = hx.sort_values("timestamp_utc", ascending=False)

            if "doctor_id" in hx.columns and "doctor_id" in d_df.columns:
                d_map = d_df.copy()
                d_map["doctor_id"] = d_map["doctor_id"].astype(str)
                d_map = d_map[["doctor_id", "name"]].rename(columns={"name": "clinician_name"})
                hx["doctor_id"] = hx["doctor_id"].astype(str)
                hx = hx.merge(d_map, on="doctor_id", how="left")

            if "recommendation" in hx.columns:
                hx["recommendation_short"] = (
                    hx["recommendation"]
                    .astype(str)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                    .str.slice(0, 90)
                    + "…"
                )
            else:
                hx["recommendation_short"] = ""

        with st.expander("Recent history (last 5)", expanded=True):
            if hx.empty:
                st.info("No previous encounters.")
            else:
                show_cols = []
                for c in [
                    "timestamp_utc",
                    "clinician_name",
                    "risk_level",
                    "prediction",
                    "recommendation_short",
                    "report_id",
                    "pdf_filename",
                ]:
                    if c in hx.columns:
                        show_cols.append(c)

                st.dataframe(
                    hx.head(5)[show_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=220,
                )

        with st.expander("PDFs for recent history", expanded=False):
            if hx.empty:
                st.info("No history yet.")
            else:
                rows = hx.head(5).to_dict(orient="records")
                any_pdf = False

                for row in rows:
                    pdf_path = _find_pdf_for_row(reports_dir, row)
                    if not pdf_path:
                        continue

                    any_pdf = True
                    pdf_name = pdf_path.name

                    cols = st.columns([3, 1, 1], gap="medium")
                    with cols[0]:
                        st.caption(pdf_name)

                    with cols[1]:
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download",
                                data=f,
                                file_name=pdf_name,
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"hx_dl_{pdf_name}",
                            )

                    with cols[2]:
                        do_preview = st.toggle("Preview", value=False, key=f"hx_prev_{pdf_name}")

                    if do_preview:
                        _pdf_preview(pdf_path, max_pages=2)
                        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

                if not any_pdf:
                    st.info("No PDFs found for the last 5 encounters. Generate PDFs to enable downloads here.")

    st.markdown('<div class="spacer-12"></div>', unsafe_allow_html=True)

    # ---------- Upload ----------
    st.markdown("### Upload fundus image")
    uploaded = st.file_uploader("PNG/JPG/JPEG", type=["png", "jpg", "jpeg"])
    st.caption("Tip: centered macula, sharp focus, even illumination.")
    st.markdown('<div class="spacer-12"></div>', unsafe_allow_html=True)

    if uploaded is None:
        st.info("Upload an image to run screening.")
        return

    img = Image.open(uploaded).convert("RGB")

    # ---------- Sidebar controls ----------
    conf_gate = st.session_state.get("conf_gate", True)
    conf_th = float(st.session_state.get("conf_th", 0.90))

    q_gate = st.session_state.get("q_gate", True)
    bmin = float(st.session_state.get("bmin", 0.08))
    bmax = float(st.session_state.get("bmax", 0.30))
    cmin = float(st.session_state.get("cmin", 0.10))
    smin = float(st.session_state.get("smin", 0.00))

    show_cam = st.session_state.get("show_cam", True)
    facility_name = st.session_state.get("facility_name", "Retina-AI Clinic")

    # ---------- Stable report_id for this case ----------
    upload_sig = f"{getattr(uploaded, 'name', '')}|{getattr(uploaded, 'size', '')}"
    case_key = f"{patient_choice}|{doctor_choice}|{upload_sig}"
    if st.session_state.get("_case_key") != case_key:
        st.session_state["_case_key"] = case_key
        st.session_state["_report_id"] = f"RPT-{uuid.uuid4().hex[:10].upper()}"
        st.session_state["_case_ts_iso"] = utc_now_iso()
        st.session_state["_last_pdf_path"] = None

    report_id = str(st.session_state["_report_id"])
    case_ts_iso = str(st.session_state["_case_ts_iso"])

    # ---------- Save model input for PDF ----------
    tmp_dir = reports_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img_224 = img.resize((IMG_SIZE, IMG_SIZE))
    input_224_path = tmp_dir / "input_224.jpg"
    img_224.save(input_224_path, quality=92)

    overlay_path: Optional[Path] = None
    overlay_pil: Optional[Image.Image] = None

    # ---------- Quality gate ----------
    q = compute_quality_metrics(img)
    quality_flagged = False
    quality_reason = "OK"
    if q_gate:
        reasons = []
        if q["brightness_mean"] < bmin or q["brightness_mean"] > bmax:
            reasons.append("Brightness out-of-range")
        if q["contrast_std"] < cmin:
            reasons.append("Low contrast")
        if q["sharpness_proxy"] < smin:
            reasons.append("Blurry / out-of-focus")
        if reasons:
            quality_flagged = True
            quality_reason = "; ".join(reasons)

    # ---------- Prediction ----------
    out = predict_2class(model, device, img)
    p_dr = float(out["p_dr"])
    pred = str(out["pred"])
    conf = float(out["confidence"])

    confidence_flagged = bool(conf_gate and conf < conf_th)
    risk_level, followup = risk_stratification(p_dr)
    recommendation = build_recommendation(pred, risk_level, followup, quality_flagged, confidence_flagged)
    risk_factors = derive_risk_factors(patient_row)

    screening_status = "OK"
    if quality_flagged:
        screening_status = "RETAKE"
    elif confidence_flagged:
        screening_status = "REVIEW"

    # ---------- Grad-CAM ----------
    if show_cam:
        try:
            from torch._C import _InferenceMode
            with _InferenceMode(False):
                with torch.enable_grad():
                    target_layer = find_last_conv_layer(model)
                    cam = GradCAM(model, target_layer)
                    x = pil_to_tensor(img).to(device).clone().detach().requires_grad_(True)
                    cam01 = cam.generate(x, target_index=1)
                    cam.close()

            overlay_pil = overlay_cam_on_image(img_224, cam01, alpha=0.45)
            overlay_path = tmp_dir / "gradcam_overlay.jpg"
            overlay_pil.save(overlay_path, quality=92)
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")
            overlay_path = None
            overlay_pil = None

    # ---------- Result header ----------
    with st.container(border=True):
        st.markdown("### Screening result")
        r1, r2, r3, r4, r5, r6 = st.columns(6, gap="small")
        r1.metric("Status", screening_status)
        r2.metric("Prediction", pred)
        r3.metric("p(DR)", f"{p_dr:.4f}")
        r4.metric("Risk", risk_level)
        r5.metric("Follow-up", followup)
        r6.metric("Confidence", f"{conf*100:.2f}%")

        st.caption(
            f"Quality gate: {'FLAGGED' if quality_flagged else 'OK'} • "
            f"Confidence gate: {'FLAGGED' if confidence_flagged else 'OK'}"
        )

        if quality_flagged:
            st.warning(f"Image quality flagged: {quality_reason}")
        elif confidence_flagged:
            st.warning("Confidence flagged: route to clinician review.")
        else:
            st.success("Screening passed gates.")

    st.markdown('<div class="spacer-12"></div>', unsafe_allow_html=True)

    # ---------- Imaging block ----------
    with st.container(border=True):
        st.markdown("### Imaging & explainability")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("**Uploaded (model input)**")
            st.image(img_224, width=320)
        with c2:
            st.markdown("**Grad-CAM overlay**")
            if overlay_pil is not None:
                st.image(overlay_pil, width=320)
            else:
                st.info("Grad-CAM not available (disabled or failed).")

    st.markdown('<div class="spacer-12"></div>', unsafe_allow_html=True)

    # ---------- Clinical summary + clinician judgement ----------
    with st.container(border=True):
        st.markdown("### Clinical summary")
        st.markdown("**Recommendation**")
        st.markdown(
            f"<div style='font-size:16px; line-height:1.7; font-weight:650;'>{recommendation}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            st.markdown("**Image quality snapshot**")
            st.write(
                {
                    "brightness_mean": round(q["brightness_mean"], 4),
                    "contrast_std": round(q["contrast_std"], 4),
                    "sharpness_proxy": round(q["sharpness_proxy"], 4),
                    "quality_reason": quality_reason,
                }
            )

        with c2:
            st.markdown("**Risk factors**")
            for rf in risk_factors[:8]:
                st.markdown(f"- {rf}")

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        st.markdown("### Clinician judgement")
        st.caption("Optional. Will appear in the PDF even if left blank (for printout handwriting).")
        clinician_judgement = st.text_area(
            "Clinician judgement",
            placeholder="Write your judgement / notes here...",
            height=140,
            label_visibility="collapsed",
            key=f"judgement_{case_key}",
        )

    st.markdown('<div class="spacer-12"></div>', unsafe_allow_html=True)

    # ---------- Save & Export ----------
    st.markdown("### Save & Export")

    pdf_name = _safe_pdf_name(patient_choice, report_id, case_ts_iso)
    out_pdf = reports_dir / pdf_name

    encounter_obj = Encounter(
        report_id=report_id,
        timestamp_utc=case_ts_iso,
        patient_id=str(patient_row.get("patient_id")),
        doctor_id=str(doctor_row.get("doctor_id")),
        prediction=pred,
        p_dr=float(p_dr),
        confidence=float(conf),
        risk_level=str(risk_level),
        recommendation=str(recommendation),
        followup_timeline=str(followup),
        quality_enabled=bool(q_gate),
        quality_flagged=bool(quality_flagged),
        quality_reason=str(quality_reason),
        brightness_mean=float(q["brightness_mean"]),
        contrast_std=float(q["contrast_std"]),
        sharpness_proxy=float(q["sharpness_proxy"]),
        model_version=str(backend),
        image_filename=str(getattr(uploaded, "name", "uploaded_image")),
        pdf_filename=str(pdf_name),
        clinician_judgement=str(clinician_judgement or ""),
    )

    b1, b2, b3 = st.columns([1, 1, 1], gap="large")

    with b1:
        if st.button("💾 Save", type="primary", use_container_width=True):
            append_encounter(encounter_obj)  # upsert by report_id (from clinical_records.py)
            st.success("Saved ✅")
            st.rerun()

    with b2:
        if st.button("🧾 Generate PDF", use_container_width=True):
            # ✅ Critical: Save (upsert) encounter first so history gets pdf_filename
            append_encounter(encounter_obj)

            generate_clinical_pdf(
                out_path=out_pdf,
                patient=patient_row,
                doctor=doctor_row,
                encounter=asdict(encounter_obj),
                risk_factors=risk_factors,
                uploaded_image_path=input_224_path,
                gradcam_overlay_path=overlay_path,
                clinician_judgement=clinician_judgement or "",
            )

            st.session_state["_last_pdf_path"] = str(out_pdf)
            st.success(f"PDF generated ✅ ({pdf_name})")
            st.rerun()

    with b3:
        last_pdf_path = st.session_state.get("_last_pdf_path", None)
        if last_pdf_path and Path(last_pdf_path).exists():
            with open(last_pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download",
                    data=f,
                    file_name=Path(last_pdf_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.button("⬇️ Download", disabled=True, use_container_width=True)

    with st.expander("JSON output (integration-friendly)", expanded=False):
        payload = {
            "screening_status": screening_status,
            "report_id": report_id,
            "timestamp_utc": case_ts_iso,
            "patient_id": encounter_obj.patient_id,
            "doctor_id": encounter_obj.doctor_id,
            "prediction": pred,
            "risk_level": risk_level,
            "followup_timeline": followup,
            "confidence": conf,
            "p_dr": p_dr,
            "clinician_judgement": clinician_judgement or "",
            "pdf_filename": pdf_name,
            "gates": {
                "quality_enabled": q_gate,
                "quality_flagged": quality_flagged,
                "quality_reason": quality_reason,
                "confidence_enabled": conf_gate,
                "confidence_threshold": conf_th,
                "confidence_flagged": confidence_flagged,
            },
            "quality_metrics": q,
            "recommendation": recommendation,
            "risk_factors": risk_factors,
            "model_version": backend,
            "facility_name": facility_name,
        }
        st.code(json.dumps(payload, indent=2), language="json")
