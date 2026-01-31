from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/dataset.csv"
OUT_DIR = ROOT / "data/processed"

def main():
    df = pd.read_csv(DATASET)

    mess = df[df["source"] == "messidor"].copy()
    other = df[df["source"] != "messidor"].copy()

    # External test = 20% of messidor (stratified)
    mess_train, mess_test = train_test_split(
        mess,
        test_size=0.20,
        random_state=42,
        stratify=mess["label"]
    )

    # Train/Val pool = (all aptos) + (80% messidor)
    pool = pd.concat([other, mess_train], ignore_index=True)

    # Split train/val from pool
    train_df, val_df = train_test_split(
        pool,
        test_size=0.20,
        random_state=42,
        stratify=pool["label"]
    )

    train_path = OUT_DIR / "train.csv"
    val_path   = OUT_DIR / "val.csv"
    test_path  = OUT_DIR / "test_external.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    mess_test.to_csv(test_path, index=False)

    print("✅ Saved:")
    print(" ", train_path, len(train_df), train_df["label"].value_counts().to_dict())
    print(" ", val_path, len(val_df), val_df["label"].value_counts().to_dict())
    print(" ", test_path, len(mess_test), mess_test["label"].value_counts().to_dict())
    print("\nNOTE: test_external.csv is MESSIDOR-only and never used for training.")

if __name__ == "__main__":
    main()
