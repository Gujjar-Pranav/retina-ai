# src/ui_registry.py
from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st


# ---------------------------- UI helpers (match Screening) ----------------------------

def _inject_global_ui_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1120px; }

          h1 { font-size: 34px !important; font-weight: 900 !important; margin-bottom: 0.25rem !important; }
          h2 { font-size: 24px !important; font-weight: 900 !important; margin-top: 1.25rem !important; }
          h3 { font-size: 18px !important; font-weight: 900 !important; margin-top: 1.00rem !important; }
          h4 { font-size: 16px !important; font-weight: 900 !important; margin-top: 0.75rem !important; }

          p, li, div, span, label { font-size: 14px; }
          small { font-size: 12px; color: #64748b; }

          .spacer-6  { height: 6px; }
          .spacer-10 { height: 10px; }
          .spacer-12 { height: 12px; }
          .spacer-16 { height: 16px; }
          .spacer-20 { height: 20px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _html(s: str) -> str:
    out = dedent(s).strip()
    out = re.sub(r"<!--.*?-->", "", out, flags=re.DOTALL)
    return out


def _load_xlsx(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_excel(path, engine="openpyxl")
        if isinstance(df, pd.DataFrame):
            return df
    return pd.DataFrame()


def _save_xlsx(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, engine="openpyxl")


def _upsert_row(df: pd.DataFrame, key_col: str, row: dict) -> pd.DataFrame:
    """Insert row if key not present; else update existing."""
    df = df.copy()
    key = str(row.get(key_col, "")).strip()
    if key == "":
        return df

    if df.empty:
        return pd.DataFrame([row])

    if key_col not in df.columns:
        df[key_col] = ""

    mask = df[key_col].astype(str) == key
    if mask.any():
        idx = df.index[mask][0]
        for k, v in row.items():
            if k not in df.columns:
                df[k] = ""
            df.at[idx, k] = v
    else:
        for k in row.keys():
            if k not in df.columns:
                df[k] = ""
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    return df


def _yn_to_bool(v: str):
    """Convert 'Yes'/'No'/'' to boolean/None."""
    s = (v or "").strip().lower()
    if s == "yes":
        return True
    if s == "no":
        return False
    return None


def _bool_to_yn(v):
    """Convert stored boolean/'Yes'/'No' to 'Yes'/'No'/'' for UI."""
    if isinstance(v, bool):
        return "Yes" if v else "No"
    s = str(v or "").strip()
    if s in {"Yes", "No"}:
        return s
    return ""


# ---------------------------- Main Registry UI ----------------------------

def render_registry(*, ROOT: Path) -> None:
    _inject_global_ui_css()

    data_dir = ROOT / "data"
    patients_xlsx = data_dir / "patients.xlsx"
    doctors_xlsx = data_dir / "doctors.xlsx"

    p_df = _load_xlsx(patients_xlsx)
    d_df = _load_xlsx(doctors_xlsx)

    with st.container(border=True):
        st.markdown(
            _html(
                """
                <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;">
                  <div>
                    <div style="font-weight:900; font-size:16px; color:#0f172a;">Registry</div>
                    <div style="color:#64748b; font-size:13px; margin-top:3px; line-height:1.5;">
                      Patient & clinician records. Add new entries or update details.
                    </div>
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='spacer-16'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    # ===================== PATIENT LOOKUP =====================
    with left:
        with st.container(border=True):
            st.markdown("### Patient lookup")
            st.markdown(
                "<div style='color:#64748b; font-size:13px; margin-top:-6px;'>"
                "Enter a Patient ID. If not found, you can create a new patient."
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='spacer-10'></div>", unsafe_allow_html=True)

            patient_id = st.text_input("Patient ID", placeholder="e.g., P-1001", label_visibility="visible").strip()

            patient_row = {}
            if patient_id and not p_df.empty and "patient_id" in p_df.columns:
                hit = p_df[p_df["patient_id"].astype(str) == patient_id]
                if not hit.empty:
                    patient_row = hit.iloc[0].to_dict()

            if not patient_id:
                st.info("Enter a Patient ID to view or add details.")
            else:
                is_new = (patient_row == {})
                if is_new:
                    st.warning("Patient not found. You can create a new patient record.")
                    patient_row = {"patient_id": patient_id}

                # ✅ Read HbA1c from either column
                hba1c_existing = patient_row.get("last_hba1c", patient_row.get("hba1c", ""))

                # ✅ Read hypertension for UI regardless of storage type
                htn_existing = _bool_to_yn(patient_row.get("hypertension", ""))

                with st.form("patient_form", clear_on_submit=False):
                    name = st.text_input("Full name", value=str(patient_row.get("name", "")))
                    age = st.text_input("Age", value=str(patient_row.get("age", "")))
                    sex = st.selectbox(
                        "Sex",
                        options=["", "Male", "Female", "Other"],
                        index=["", "Male", "Female", "Other"].index(
                            str(patient_row.get("sex", "")) if str(patient_row.get("sex", "")) in ["Male", "Female", "Other"] else ""
                        ),
                    )
                    diabetes_years = st.text_input(
                        "Diabetes duration (years)",
                        value=str(patient_row.get("diabetes_years", patient_row.get("diabetes_duration_years", ""))),
                    )

                    # ✅ Keep as text input but save to last_hba1c too
                    hba1c = st.text_input("HbA1c (%)", value=str(hba1c_existing))

                    hypertension = st.selectbox(
                        "Hypertension",
                        options=["", "Yes", "No"],
                        index=["", "Yes", "No"].index(htn_existing if htn_existing in ["Yes", "No"] else ""),
                    )

                    st.markdown("<div class='spacer-10'></div>", unsafe_allow_html=True)
                    submitted = st.form_submit_button("💾 Save patient", use_container_width=True)

                if submitted:
                    htn_bool = _yn_to_bool(hypertension)

                    row_out = {
                        "patient_id": patient_id,
                        "name": name.strip(),
                        "age": age.strip(),
                        "sex": sex,
                        "diabetes_years": diabetes_years.strip(),

                        # ✅ store both (compat + new)
                        "hba1c": hba1c.strip(),
                        "last_hba1c": hba1c.strip(),

                        # ✅ CRITICAL FIX: store as bool, not "Yes"/"No"
                        "hypertension": htn_bool,
                    }

                    p_df = _upsert_row(p_df, "patient_id", row_out)
                    _save_xlsx(p_df, patients_xlsx)
                    st.success("Patient saved ✅")

    # ===================== CLINICIAN LOOKUP =====================
    with right:
        with st.container(border=True):
            st.markdown("### Clinician lookup")
            st.markdown(
                "<div style='color:#64748b; font-size:13px; margin-top:-6px;'>"
                "Enter a Clinician/Doctor ID. If not found, you can create a new clinician."
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='spacer-10'></div>", unsafe_allow_html=True)

            doctor_id = st.text_input("Clinician/Doctor ID", placeholder="e.g., D-9001", label_visibility="visible").strip()

            clinician_row = {}
            if doctor_id and not d_df.empty and "doctor_id" in d_df.columns:
                hit = d_df[d_df["doctor_id"].astype(str) == doctor_id]
                if not hit.empty:
                    clinician_row = hit.iloc[0].to_dict()

            if not doctor_id:
                st.info("Enter a Clinician/Doctor ID to view or add details.")
            else:
                is_new = (clinician_row == {})
                if is_new:
                    st.warning("Clinician not found. You can create a new clinician record.")
                    clinician_row = {"doctor_id": doctor_id}

                with st.form("clinician_form", clear_on_submit=False):
                    name = st.text_input("Full name", value=str(clinician_row.get("name", "")))
                    specialty = st.text_input("Specialty", value=str(clinician_row.get("specialty", "")))
                    phone = st.text_input("Phone", value=str(clinician_row.get("phone", "")))
                    email = st.text_input("Email", value=str(clinician_row.get("email", "")))

                    st.markdown("<div class='spacer-10'></div>", unsafe_allow_html=True)
                    submitted = st.form_submit_button("💾 Save clinician", use_container_width=True)

                if submitted:
                    row_out = {
                        "doctor_id": doctor_id,
                        "name": name.strip(),
                        "specialty": specialty.strip(),
                        "phone": phone.strip(),
                        "email": email.strip(),
                    }
                    d_df = _upsert_row(d_df, "doctor_id", row_out)
                    _save_xlsx(d_df, doctors_xlsx)
                    st.success("Clinician saved ✅")

    # ✅ Intentionally removed "Encounter history" from Registry as requested.
