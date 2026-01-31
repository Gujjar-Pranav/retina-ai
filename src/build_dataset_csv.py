from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

APTOS_RAW = ROOT / "data/raw/aptos"
MESS_RAW  = ROOT / "data/raw/messidor"

APTOS_CACHE = ROOT / "data/cache_224/aptos"
MESS_CACHE  = ROOT / "data/cache_224/messidor"

OUT_DIR = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Option: keep only gradable messidor images (recommended) ---
ONLY_GRADABLE_MESSIDOR = True

def build_aptos_df():
    """
    APTOS Binary folder structure:
      data/raw/aptos/.../APTOS 2019 (Original) (Binary)/DR/*.png
      data/raw/aptos/.../APTOS 2019 (Original) (Binary)/No DR/*.png

    Cached filenames:
      DR_<stem>.jpg  and  NoDR_<stem>.jpg
    """
    dr_files = (
        list(APTOS_RAW.rglob("DR/*.png")) +
        list(APTOS_RAW.rglob("DR/*.jpg")) +
        list(APTOS_RAW.rglob("DR/*.jpeg"))
    )
    nodr_files = (
        list(APTOS_RAW.rglob("No DR/*.png")) +
        list(APTOS_RAW.rglob("No DR/*.jpg")) +
        list(APTOS_RAW.rglob("No DR/*.jpeg"))
    )

    rows = []

    # DR = 1
    for p in dr_files:
        cache_path = APTOS_CACHE / f"DR_{p.stem}.jpg"
        if cache_path.exists():
            rows.append({"path": str(cache_path), "label": 1, "source": "aptos"})

    # No DR = 0
    for p in nodr_files:
        cache_path = APTOS_CACHE / f"NoDR_{p.stem}.jpg"
        if cache_path.exists():
            rows.append({"path": str(cache_path), "label": 0, "source": "aptos"})

    return pd.DataFrame(rows)

def build_messidor_df():
    """
    Messidor labels file:
      data/raw/messidor/messidor_data.csv

    Supports BOTH formats:

    Format A (google-brain):
      columns: image_id, adjudicated_dr_grade, adjudicated_dme, adjudicated_gradable
      map: adjudicated_dr_grade == 0 -> No DR (0), else DR (1)
      filters:
        - drop NaN grades
        - if ONLY_GRADABLE_MESSIDOR=True, keep adjudicated_gradable == 1

    Format B (kaggle-style):
      columns: id_code, diagnosis
      map: diagnosis == 0 -> No DR (0), else DR (1)

    Cached filenames:
      <stem>.jpg
    """
    label_csv = MESS_RAW / "messidor_data.csv"
    if not label_csv.exists():
        raise FileNotFoundError(f"Missing messidor_data.csv at: {label_csv}")

    labels = pd.read_csv(label_csv)
    cols = set(labels.columns.tolist())

    rows = []

    # --- Format A: google-brain ---
    if {"image_id", "adjudicated_dr_grade"}.issubset(cols):
        # drop NaN grades
        labels = labels.dropna(subset=["adjudicated_dr_grade"])

        # keep only gradable if flag enabled and column exists
        if ONLY_GRADABLE_MESSIDOR and "adjudicated_gradable" in labels.columns:
            labels = labels[labels["adjudicated_gradable"] == 1]

        for _, r in labels.iterrows():
            fname = str(r["image_id"])
            grade = int(float(r["adjudicated_dr_grade"]))  # safe after dropna
            stem = Path(fname).stem

            cache_path = MESS_CACHE / f"{stem}.jpg"
            if cache_path.exists():
                y = 0 if grade == 0 else 1
                rows.append({"path": str(cache_path), "label": y, "source": "messidor"})

    # --- Format B: kaggle-style ---
    elif {"id_code", "diagnosis"}.issubset(cols):
        labels = labels.dropna(subset=["diagnosis"])
        for _, r in labels.iterrows():
            fname = str(r["id_code"])
            diag = int(r["diagnosis"])
            stem = Path(fname).stem

            cache_path = MESS_CACHE / f"{stem}.jpg"
            if cache_path.exists():
                y = 0 if diag == 0 else 1
                rows.append({"path": str(cache_path), "label": y, "source": "messidor"})

    else:
        raise ValueError(f"Unexpected columns in messidor_data.csv: {labels.columns.tolist()}")

    return pd.DataFrame(rows)

def main():
    aptos = build_aptos_df()
    mess  = build_messidor_df()

    print("APTOS:", aptos.shape, aptos["label"].value_counts().to_dict())
    print("MESSIDOR:", mess.shape, mess["label"].value_counts().to_dict())
    print("ONLY_GRADABLE_MESSIDOR:", ONLY_GRADABLE_MESSIDOR)

    full = pd.concat([aptos, mess], ignore_index=True)
    print("TOTAL:", full.shape)

    train_df, val_df = train_test_split(
        full,
        test_size=0.2,
        random_state=42,
        stratify=full["label"]
    )

    full_path  = OUT_DIR / "dataset.csv"
    train_path = OUT_DIR / "train.csv"
    val_path   = OUT_DIR / "val.csv"

    full.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("\n✅ Saved:")
    print(" ", full_path)
    print(" ", train_path)
    print(" ", val_path)
    print("\nSplit sizes:", {"train": len(train_df), "val": len(val_df)})
    print("Train balance:", train_df["label"].value_counts().to_dict())
    print("Val balance:", val_df["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()
