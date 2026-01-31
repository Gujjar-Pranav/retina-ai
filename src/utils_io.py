# src/utils_io.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS_XLSX = DATA_DIR / "patients.xlsx"
DOCTORS_XLSX = DATA_DIR / "doctors.xlsx"
HISTORY_XLSX = DATA_DIR / "patient_history.xlsx"


# -----------------------------
# Internal helper
# -----------------------------
def _read_xlsx(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        return pd.DataFrame()


# -----------------------------
# Public API (used by UI)
# -----------------------------
def get_patients_df() -> pd.DataFrame:
    df = _read_xlsx(PATIENTS_XLSX)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "name",
                "age",
                "sex",
                "diabetes_years",
                "hypertension",
                "last_hba1c",
            ]
        )

    want = [
        "patient_id",
        "name",
        "age",
        "sex",
        "diabetes_years",
        "hypertension",
        "last_hba1c",
    ]
    for c in want:
        if c not in df.columns:
            df[c] = None
    return df[want]


def get_doctors_df() -> pd.DataFrame:
    df = _read_xlsx(DOCTORS_XLSX)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "doctor_id",
                "name",
                "qualification",
                "hospital",
            ]
        )

    want = ["doctor_id", "name", "qualification", "hospital"]
    for c in want:
        if c not in df.columns:
            df[c] = None
    return df[want]


def get_history_df() -> pd.DataFrame:
    df = _read_xlsx(HISTORY_XLSX)
    return df if not df.empty else pd.DataFrame()
