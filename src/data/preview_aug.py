from __future__ import annotations

from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Import from an existing file you DO have:
from src.data.check_dwrl import make_transforms  # <-- this must exist in check_dwrl.py

SEED = 42
IMG_SIZE = 224

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = REPO_ROOT / "src" / "data" / "splits" / "train.csv"


def preview_augmented_examples(csv_path: Path, n: int = 6):
    df = pd.read_csv(csv_path)
    n = min(n, len(df))
    df = df.sample(n=n, random_state=SEED).reset_index(drop=True)

    train_tf, _ = make_transforms(IMG_SIZE)

    fig, axes = plt.subplots(n, 2, figsize=(7, n * 2.2))
    fig.suptitle("Left: raw | Right: augmented (train tf)", fontsize=12)

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    for i in range(n):
        p = Path(df.loc[i, "path"])
        label = str(df.loc[i, "label"])
        img = Image.open(p).convert("RGB")

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(label, fontsize=9)

        x = train_tf(img)
        x_vis = (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        axes[i, 1].imshow(x_vis)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"train.csv not found at: {TRAIN_CSV}")
    preview_augmented_examples(TRAIN_CSV, n=6)


if __name__ == "__main__":
    main()