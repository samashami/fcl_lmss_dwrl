# src/data/dwrl_transforms.py
from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def make_dwrl_transforms(img_size: int = 224):
    """
    Keep training augmentation LIGHT because train_plus_aug.csv already
    contains saved augmented images.
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tf, eval_tf

# Backward-compatible alias for run_fcl_dwrl.py
def make_transforms(img_size: int = 224):
    return make_dwrl_transforms(img_size)