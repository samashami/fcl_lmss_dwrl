# src/data/augment_dwrl.py
from __future__ import annotations

from pathlib import Path
import random
import math

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms


# -----------------------------
# Config
# -----------------------------
SEED = 42
IMG_SIZE = 224

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "src" / "data" / "splits"
AUG_ROOT   = REPO_ROOT / "src" / "data" / "DWRL_aug"

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV   = SPLITS_DIR / "val.csv"

# We only augment minority classes (start with these 3)
AUG_CLASSES = ["TETRA", "PS", "PVC"]

# Target number of images per class AFTER augmentation, for train+val pool
TARGET_PER_CLASS = 2000

# Save format (JPG is smaller than PNG)
JPG_QUALITY = 90


# -----------------------------
# Augmentation transform (fast/moderate)
# -----------------------------
aug_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
])


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def _load_pool(train_csv: Path, val_csv: Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val")], ignore_index=True)

    # Basic checks
    assert "path" in df.columns and "label" in df.columns, "CSV must have columns: path,label"
    return df


def _ensure_dirs():
    AUG_ROOT.mkdir(parents=True, exist_ok=True)
    for c in AUG_CLASSES:
        (AUG_ROOT / c).mkdir(parents=True, exist_ok=True)


def _count_trainval(df: pd.DataFrame) -> dict[str, int]:
    # counts in train+val originals (because splits are from originals only)
    return df["label"].value_counts().to_dict()


def _plan_needed(counts: dict[str, int]) -> dict[str, int]:
    need = {}
    for c in AUG_CLASSES:
        n = int(counts.get(c, 0))
        need[c] = max(0, TARGET_PER_CLASS - n)
    return need


def _augment_from_list(paths: list[Path], class_name: str, n_to_generate: int, split_tag: str) -> list[dict]:
    """
    Generate n_to_generate augmented images by sampling source images (with replacement).
    Returns rows for an *aug csv* with columns: path,label
    """
    if n_to_generate <= 0:
        return []

    dst_dir = AUG_ROOT / class_name
    rows = []

    # sample sources with replacement so we can generate arbitrary N
    for j in tqdm(range(n_to_generate), desc=f"Gen {class_name} ({split_tag})"):
        src = random.choice(paths)
        img = Image.open(src).convert("RGB")
        out = aug_tf(img)

        # stable-ish naming: aug_<split>_<srcstem>_<j>.jpg
        out_name = f"aug_{split_tag}_{src.stem}_{j}.jpg"
        out_path = dst_dir / out_name

        out.save(out_path, format="JPEG", quality=JPG_QUALITY)
        rows.append({"path": str(out_path), "label": class_name})

    return rows


def main():
    _set_seed(SEED)
    _ensure_dirs()

    df = _load_pool(TRAIN_CSV, VAL_CSV)
    counts = _count_trainval(df)
    need = _plan_needed(counts)

    print("Train+Val original counts:", {c: counts.get(c, 0) for c in AUG_CLASSES})
    print("Target per class:", TARGET_PER_CLASS)
    print("Need to generate:", need)
    print("Saving under:", AUG_ROOT)

    train_aug_rows = []
    val_aug_rows = []

    # For each class, split the generation proportionally between train/val
    for c in AUG_CLASSES:
        n_need = need[c]
        if n_need <= 0:
            continue

        train_paths = [Path(p) for p in df[(df["label"] == c) & (df["split"] == "train")]["path"].tolist()]
        val_paths   = [Path(p) for p in df[(df["label"] == c) & (df["split"] == "val")]["path"].tolist()]

        # If val has 0, push all into train_aug
        n_train = len(train_paths)
        n_val = len(val_paths)
        if n_train + n_val == 0:
            print(f"[WARN] No sources found for class {c} in train/val. Skipping.")
            continue

        # Keep ratio similar to original split sizes
        frac_train = n_train / (n_train + n_val)
        gen_train = int(round(n_need * frac_train))
        gen_val = n_need - gen_train

        # Safety: if one side has no sources, shift generation
        if n_train == 0:
            gen_val = n_need
            gen_train = 0
        if n_val == 0:
            gen_train = n_need
            gen_val = 0

        print(f"{c}: sources train={n_train}, val={n_val} | gen_train={gen_train}, gen_val={gen_val}")

        train_aug_rows += _augment_from_list(train_paths, c, gen_train, split_tag="train")
        val_aug_rows   += _augment_from_list(val_paths,   c, gen_val,   split_tag="val")

    # Write aug csvs
    train_aug_df = pd.DataFrame(train_aug_rows)
    val_aug_df   = pd.DataFrame(val_aug_rows)

    train_aug_csv = SPLITS_DIR / "train_aug.csv"
    val_aug_csv   = SPLITS_DIR / "val_aug.csv"
    train_plus_csv = SPLITS_DIR / "train_plus_aug.csv"
    val_plus_csv   = SPLITS_DIR / "val_plus_aug.csv"

    train_aug_df.to_csv(train_aug_csv, index=False)
    val_aug_df.to_csv(val_aug_csv, index=False)

    # Merge originals + aug
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    train_plus = pd.concat([train_df, train_aug_df], ignore_index=True)
    val_plus   = pd.concat([val_df,   val_aug_df],   ignore_index=True)

    train_plus.to_csv(train_plus_csv, index=False)
    val_plus.to_csv(val_plus_csv, index=False)

    print("\nSaved:")
    print(" ", train_aug_csv, f"({len(train_aug_df)})")
    print(" ", val_aug_csv,   f"({len(val_aug_df)})")
    print(" ", train_plus_csv, f"({len(train_plus)})")
    print(" ", val_plus_csv,   f"({len(val_plus)})")
    print("\nDone.")


if __name__ == "__main__":
    main()