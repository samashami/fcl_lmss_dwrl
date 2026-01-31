# src/data/check_dwrl.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -----------------------------
# Config (edit these if needed)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "src" / "data" / "DWRL_clean"        # your 7-class folder
SPLITS_DIR = REPO_ROOT / "src" / "data" / "splits"           # where make_splits_dwrl.py saved CSVs

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV   = SPLITS_DIR / "val.csv"
TEST_CSV  = SPLITS_DIR / "test.csv"

CLASSES = ["PET", "PP", "PE", "TETRA", "PS", "PVC", "Other"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}

SEED = 42
IMG_SIZE = 224
PREVIEW_PER_CLASS = 6  # keep small


# -----------------------------
# Transforms
# -----------------------------
def make_transforms(img_size: int = 224):
    # "preprocessing": resize + normalize (ImageNet stats for ResNet/ImageNet-pretrained)
    base = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # training aug: reasonable + fast
    train_aug = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_aug, base  # train, eval


# -----------------------------
# Dataset (reads split CSV)
# -----------------------------
class DWRLDataset(Dataset):
    """
    Expects CSV with columns:
      - path: absolute or relative path to image
      - label: class string in CLASSES (e.g., "PET")
    """
    def __init__(self, csv_path: Path, transform=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        # basic validation
        assert "path" in self.df.columns and "label" in self.df.columns, \
            f"{csv_path} must contain columns: path,label"

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = Path(row["path"])
        label_str = row["label"]
        y = CLASS2IDX[label_str]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


# -----------------------------
# Helpers
# -----------------------------
def print_split_stats(name: str, df: pd.DataFrame):
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    print(f"{name:>5}: {total}")
    for c in CLASSES:
        print(f"  {c:<5} {counts.get(c, 0)}")


def plot_preview_from_csv(csv_path: Path, title: str, per_class: int = 6):
    df = pd.read_csv(csv_path)
    # sample up to per_class per label
    rows = []
    for c in CLASSES:
        sub = df[df["label"] == c]
        if len(sub) == 0:
            continue
        rows.append(sub.sample(n=min(per_class, len(sub)), random_state=SEED))
    samp = pd.concat(rows, axis=0)

    # plot: one row per class, per_class columns
    n_rows = len(CLASSES)
    n_cols = per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    fig.suptitle(title, fontsize=14)

    # convert axes to 2D list even if per_class==1
    if n_cols == 1:
        axes = [[axes[r]] for r in range(n_rows)]

    # fill
    for r, c in enumerate(CLASSES):
        sub = samp[samp["label"] == c]
        sub = sub.reset_index(drop=True)
        for j in range(n_cols):
            ax = axes[r][j]
            ax.axis("off")
            if j < len(sub):
                p = Path(sub.loc[j, "path"])
                try:
                    im = Image.open(p).convert("RGB")
                    ax.imshow(im)
                    if j == 0:
                        ax.set_title(c, fontsize=10, loc="left")
                except Exception as e:
                    ax.set_title(f"{c} (err)", fontsize=10)
            else:
                if j == 0:
                    ax.set_title(c, fontsize=10, loc="left")

    plt.tight_layout()
    plt.show()


def sanity_check_files_exist(csv_path: Path, max_missing: int = 20):
    df = pd.read_csv(csv_path)
    missing = []
    for p in df["path"].tolist():
        if not Path(p).exists():
            missing.append(p)
            if len(missing) >= max_missing:
                break
    if missing:
        print(f"[WARN] {len(missing)} missing files (showing up to {max_missing}):")
        for m in missing[:max_missing]:
            print("  ", m)
    else:
        print(f"[OK] all files exist for {csv_path.name}")


# -----------------------------
# Main
# -----------------------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # quick: show raw folder counts (what you already did)
    print("Raw folder counts (DWRL_clean):")
    for c in CLASSES:
        n = sum(1 for _ in (DATA_ROOT / c).glob("*") if _.suffix.lower() in [".jpg", ".jpeg", ".png"])
        print(f"{c:<5}: {n}")

    # check splits
    print("\nSplit CSV stats:")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print_split_stats("train", train_df)
    print_split_stats("val", val_df)
    print_split_stats("test", test_df)

    # existence checks
    print("\nFile existence checks:")
    sanity_check_files_exist(TRAIN_CSV)
    sanity_check_files_exist(VAL_CSV)
    sanity_check_files_exist(TEST_CSV)

    # preview a few images from each split
    print("\nPreview train samples:")
    plot_preview_from_csv(TRAIN_CSV, title="TRAIN (raw images, before tensor/normalize)", per_class=PREVIEW_PER_CLASS)

    print("\nPreview val samples:")
    plot_preview_from_csv(VAL_CSV, title="VAL (raw images, before tensor/normalize)", per_class=PREVIEW_PER_CLASS)

    print("\nPreview test samples:")
    plot_preview_from_csv(TEST_CSV, title="TEST (raw images, before tensor/normalize)", per_class=PREVIEW_PER_CLASS)

    # optional: make sure dataloader works with transforms
    train_tf, eval_tf = make_transforms(IMG_SIZE)
    train_ds = DWRLDataset(TRAIN_CSV, transform=train_tf)
    test_ds  = DWRLDataset(TEST_CSV,  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    xb, yb = next(iter(train_loader))
    print(f"\nDataloader OK. train batch: x={tuple(xb.shape)} y={tuple(yb.shape)} labels={yb[:8].tolist()}")

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    xb2, yb2 = next(iter(test_loader))
    print(f"test batch: x={tuple(xb2.shape)} y={tuple(yb2.shape)} labels={yb2[:8].tolist()}")


if __name__ == "__main__":
    main()