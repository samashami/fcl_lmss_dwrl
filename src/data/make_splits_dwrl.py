import os
from pathlib import Path
from collections import Counter
import pandas as pd

CLASSES = ["PET", "PP", "PE", "TETRA", "PS", "PVC", "Other"]

ROOT = Path("src/data/DWRL_clean")          # input images
OUT  = Path("src/data/splits")             # output CSVs
SEED = 42
TEST_FRAC = 0.10
VAL_FRAC  = 0.20   # fraction of (train+val) that becomes val

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def list_images(class_dir: Path):
    files = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in CLASSES:
        class_dir = ROOT / c
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for p in list_images(class_dir):
            rows.append({"path": str(p.resolve()), "label": c})

    df = pd.DataFrame(rows)
    print("Total images:", len(df))
    print("Per-class counts:\n", df["label"].value_counts())

    # We'll do per-class shuffles using random from Python for stability
    import random
    random.seed(SEED)

    test_idx = []
    trainval_idx = []

    for c in CLASSES:
        idx = df.index[df["label"] == c].tolist()
        random.shuffle(idx)
        n_test = int(round(TEST_FRAC * len(idx)))
        test_idx += idx[:n_test]
        trainval_idx += idx[n_test:]

    df_test = df.loc[test_idx].reset_index(drop=True)
    df_trainval = df.loc[trainval_idx].reset_index(drop=True)

    # Now split train/val stratified inside trainval
    train_idx = []
    val_idx = []

    for c in CLASSES:
        idx = df_trainval.index[df_trainval["label"] == c].tolist()
        random.shuffle(idx)
        n_val = int(round(VAL_FRAC * len(idx)))
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]

    df_train = df_trainval.loc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.loc[val_idx].reset_index(drop=True)

    # Save
    df_train.to_csv(OUT / "train.csv", index=False)
    df_val.to_csv(OUT / "val.csv", index=False)
    df_test.to_csv(OUT / "test.csv", index=False)

    print("\nSaved:")
    print(" train:", len(df_train), Counter(df_train["label"]))
    print(" val  :", len(df_val),   Counter(df_val["label"]))
    print(" test :", len(df_test),  Counter(df_test["label"]))

if __name__ == "__main__":
    main()