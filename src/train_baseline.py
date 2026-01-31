# src/train_baseline.py
from __future__ import annotations

from pathlib import Path
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# -----------------------------
# Config
# -----------------------------
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 0  # mac safe

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "src" / "data" / "splits"
TRAIN_CSV = SPLITS_DIR / "train_plus_aug.csv"
VAL_CSV   = SPLITS_DIR / "val_plus_aug.csv"
TEST_CSV  = SPLITS_DIR / "test.csv"

CLASSES = ["PET", "PP", "PE", "TETRA", "PS", "PVC", "Other"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Transforms
# -----------------------------
def make_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


# -----------------------------
# Dataset
# -----------------------------
class DWRLDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        self.df = pd.read_csv(csv_path)
        assert "path" in self.df.columns and "label" in self.df.columns
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = Path(row["path"])
        y = CLASS2IDX[row["label"]]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, y


# -----------------------------
# Utils
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.numel()
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(yb.cpu().tolist())
    acc = correct / max(total, 1)
    return acc, all_targets, all_preds


def confusion_matrix(targets, preds, num_classes: int):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


# -----------------------------
# Main
# -----------------------------
def main():
    torch.manual_seed(SEED)

    train_tf, eval_tf = make_transforms(IMG_SIZE)

    train_ds = DWRLDataset(TRAIN_CSV, transform=train_tf)
    val_ds   = DWRLDataset(VAL_CSV,   transform=eval_tf)
    test_ds  = DWRLDataset(TEST_CSV,  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Device: {DEVICE}")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = 0.0
    best_path = REPO_ROOT / "results_best_baseline.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += yb.numel()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_acc, _, _ = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "val_acc": best_val}, best_path)

    # Test using best checkpoint
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(DEVICE)

    test_acc, targets, preds = evaluate(model, test_loader, DEVICE)
    cm = confusion_matrix(targets, preds, len(CLASSES))
    print(f"\nBest val acc: {best_val:.4f}")
    print(f"Test acc    : {test_acc:.4f}\n")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # per-class accuracy quick
    cm_float = cm.float()
    per_class = (cm_float.diag() / cm_float.sum(dim=1).clamp(min=1)).tolist()
    for i, a in enumerate(per_class):
        print(f"{IDX2CLASS[i]:<5}: {a:.3f}")


if __name__ == "__main__":
    main()