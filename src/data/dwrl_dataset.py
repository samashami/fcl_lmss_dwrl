# src/data/dwrl_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from src.data.dwrl_labels import CLASS2IDX


class DWRLDataset(Dataset):
    """
    CSV-based dataset.
    Expects CSV columns:
      - path: absolute/relative image path
      - label: string label in dwrl_labels.CLASSES
    Returns: (image_tensor, class_index)
    """
    def __init__(
        self,
        csv_path: Path | str,
        transform=None,
        verify_paths: bool = True,
        drop_missing: bool = True,
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # validate schema
        required = {"path", "label"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{self.csv_path} must contain columns {sorted(required)}; "
                f"found {list(df.columns)}"
            )

        # validate labels
        bad_labels = sorted(set(df["label"].unique()) - set(CLASS2IDX.keys()))
        if bad_labels:
            raise ValueError(
                f"{self.csv_path} contains unknown labels {bad_labels}. "
                f"Known: {sorted(CLASS2IDX.keys())}"
            )

        # verify paths (and optionally drop missing)
        if verify_paths:
            exists_mask = df["path"].apply(lambda p: Path(p).exists())
            n_missing = int((~exists_mask).sum())
            if n_missing > 0:
                example = df.loc[~exists_mask, "path"].iloc[0]
                msg = f"{n_missing} missing files in {self.csv_path.name}, example: {example}"
                if drop_missing:
                    print(f"[WARN] {msg} -> dropping missing rows")
                    df = df.loc[exists_mask].reset_index(drop=True)
                else:
                    raise FileNotFoundError(msg)

        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        p = Path(row["path"])
        y = CLASS2IDX[str(row["label"])]

        # keep failure mode explicit and fast
        if not p.exists():
            raise FileNotFoundError(f"Missing file at index {idx}: {p}")

        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, y

    def get_targets(self) -> torch.Tensor:
        """Convenience: return int targets for splitting."""
        ys = [CLASS2IDX[str(x)] for x in self.df["label"].tolist()]
        return torch.tensor(ys, dtype=torch.long)