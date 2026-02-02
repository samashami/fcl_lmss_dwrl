# src/data/augment_dwrl.py
from __future__ import annotations

from pathlib import Path
import random
from typing import Dict

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

TRAIN_CSV = SPLITS_DIR / "train.csv"   # original-only split
VAL_CSV   = SPLITS_DIR / "val.csv"     # keep clean
TEST_CSV  = SPLITS_DIR / "test.csv"    # keep clean

AUG_ROOT = REPO_ROOT / "src" / "data" / "DWRL_aug"

OUT_TRAIN_AUG  = SPLITS_DIR / "train_aug.csv"
OUT_TRAIN_PLUS = SPLITS_DIR / "train_plus_aug.csv"
OUT_VAL_AUG    = SPLITS_DIR / "val_aug.csv"
OUT_VAL_PLUS   = SPLITS_DIR / "val_plus_aug.csv"


# -----------------------------
# Augmentations (class-specific)
# -----------------------------
ps_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
])

tetra_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
])

pvc_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.90, 1.0)),
])

AUG_TF: Dict[str, transforms.Compose] = {
    "PS": ps_aug,
    "TETRA": tetra_aug,
    "PVC": pvc_aug,
}


# -----------------------------
# Helpers
# -----------------------------
def clear_aug_dir(cls: str) -> None:
    """Delete previously generated aug_*.png for this class (rerunnable)."""
    out_dir = AUG_ROOT / cls
    if not out_dir.exists():
        return
    for p in out_dir.glob("aug_*.png"):
        p.unlink()


def generate_for_class(df_train: pd.DataFrame, cls: str, target_train: int) -> pd.DataFrame:
    """
    Generate augmented images for a single class so that:
      len(train_original_for_cls) + len(train_aug_for_cls) == target_train
    Uses ONLY train.csv as source. Saves files under DWRL_aug/<cls>/.
    Returns dataframe with columns: path,label for the generated samples.
    """
    src = df_train[df_train["label"] == cls].copy()
    n_src = len(src)
    if n_src == 0:
        print(f"[WARN] No train images for class {cls}")
        return pd.DataFrame(columns=["path", "label"])

    need = max(0, target_train - n_src)
    print(f"{cls}: train originals={n_src} target={target_train} need={need}")

    if need == 0:
        return pd.DataFrame(columns=["path", "label"])

    tf = AUG_TF[cls]
    out_dir = AUG_ROOT / cls
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    paths = src["path"].tolist()

    for i in tqdm(range(need), desc=f"Gen {cls} (train)"):
        p = Path(random.choice(paths))
        img = Image.open(p).convert("RGB")
        out = tf(img)

        # unique + stable filename
        out_path = out_dir / f"aug_{cls}_{p.stem}_{i:05d}.png"
        out.save(out_path)
        rows.append({"path": str(out_path), "label": cls})

    return pd.DataFrame(rows)


def generate_for_class_from_split(df_split: pd.DataFrame, split_name: str, cls: str, target_count: int) -> pd.DataFrame:
    """
    Same as generate_for_class, but works for any split (train or val).
    Generates enough augmented samples so:
      len(split_original_for_cls) + len(split_aug_for_cls) == target_count
    Saves under DWRL_aug/<cls>/ with filenames tagged by split_name.
    """
    src = df_split[df_split["label"] == cls].copy()
    n_src = len(src)
    if n_src == 0:
        print(f"[WARN] No {split_name} images for class {cls}")
        return pd.DataFrame(columns=["path", "label"])

    need = max(0, target_count - n_src)
    print(f"{cls}: {split_name} originals={n_src} target={target_count} need={need}")
    if need == 0:
        return pd.DataFrame(columns=["path", "label"])

    tf = AUG_TF[cls]
    out_dir = AUG_ROOT / cls
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    paths = src["path"].tolist()

    for i in tqdm(range(need), desc=f"Gen {cls} ({split_name})"):
        p = Path(random.choice(paths))
        img = Image.open(p).convert("RGB")
        out = tf(img)

        out_path = out_dir / f"aug_{split_name}_{cls}_{p.stem}_{i:05d}.png"
        out.save(out_path)
        rows.append({"path": str(out_path), "label": cls})

    return pd.DataFrame(rows)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- safety checks ----
    for f in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not f.exists():
            raise FileNotFoundError(f"Missing split file: {f}")

    df_train = pd.read_csv(TRAIN_CSV)
    if "path" not in df_train.columns or "label" not in df_train.columns:
        raise ValueError("train.csv must have columns: path,label")
    
    df_val = pd.read_csv(VAL_CSV)
    if "path" not in df_val.columns or "label" not in df_val.columns:
        raise ValueError("val.csv must have columns: path,label")

    # ---- Targets: TRAIN ONLY ----
    # Keep TETRA/PVC at 2000, bump PS higher to help recall
    targets = {
        "TETRA": 2000,
        "PVC": 2000,
        "PS": 2500,
    }

    val_targets = {
        "TETRA": 500,
        "PVC": 500,
        "PS": 650,
    }

    # ---- Clean previous aug for reruns ----
    for cls in set(targets.keys()) | set(val_targets.keys()):
        clear_aug_dir(cls)

    # ---- Generate ----
    all_aug = []
    for cls, tgt in targets.items():
        all_aug.append(generate_for_class(df_train, cls, tgt))

    df_aug = pd.concat(all_aug, axis=0).reset_index(drop=True)
    df_aug.to_csv(OUT_TRAIN_AUG, index=False)

    df_train_plus = pd.concat([df_train, df_aug], axis=0).reset_index(drop=True)
    df_train_plus.to_csv(OUT_TRAIN_PLUS, index=False)

    print("\nSaved:")
    print(f"  {OUT_TRAIN_AUG} ({len(df_aug)})")
    print(f"  {OUT_TRAIN_PLUS} ({len(df_train_plus)})")
    print(f"Aug images saved under: {AUG_ROOT}")

    # ---- Generate VAL aug ----
    all_val_aug = []
    for cls, tgt in val_targets.items():
        all_val_aug.append(generate_for_class_from_split(df_val, "val", cls, tgt))

    df_val_aug = pd.concat(all_val_aug, axis=0).reset_index(drop=True)
    df_val_aug.to_csv(OUT_VAL_AUG, index=False)

    df_val_plus = pd.concat([df_val, df_val_aug], axis=0).reset_index(drop=True)
    df_val_plus.to_csv(OUT_VAL_PLUS, index=False)

    print("\nSaved:")
    print(f"  {OUT_VAL_AUG} ({len(df_val_aug)})")
    print(f"  {OUT_VAL_PLUS} ({len(df_val_plus)})")


if __name__ == "__main__":
    main()