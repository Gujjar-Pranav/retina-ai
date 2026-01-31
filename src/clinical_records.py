# src/clinical_records.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_XLSX = DATA_DIR / "patient_history.xlsx"


@dataclass
class Encounter:
    report_id: str
    timestamp_utc: str
    patient_id: str
    doctor_id: str

    prediction: str
    p_dr: float
    confidence: float
    risk_level: str
    recommendation: str
    followup_timeline: str

    quality_enabled: bool
    quality_flagged: bool
    quality_reason: str
    brightness_mean: float
    contrast_std: float
    sharpness_proxy: float

    model_version: str
    image_filename: str

    # ✅ NEW (must be persisted)
    pdf_filename: str = ""
    clinician_judgement: str = ""


def append_encounter(encounter: Encounter, path: Optional[Path] = None) -> None:
    """
    Append (or update) a screening encounter to Excel history.

    - Adds any missing columns automatically.
    - If the same report_id already exists, updates that row instead of duplicating.
    """
    out_path = path or HISTORY_XLSX
    row: Dict[str, Any] = asdict(encounter)

    if out_path.exists():
        df = pd.read_excel(out_path, engine="openpyxl")
    else:
        df = pd.DataFrame()

    # Ensure all columns exist
    for k in row.keys():
        if k not in df.columns:
            df[k] = None

    # Upsert by report_id (avoid duplicates, and allow later updates like pdf_filename)
    if not df.empty and "report_id" in df.columns:
        mask = df["report_id"].astype(str) == str(row["report_id"])
        if mask.any():
            idx = df.index[mask][0]
            for k, v in row.items():
                df.at[idx, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False, engine="openpyxl")
