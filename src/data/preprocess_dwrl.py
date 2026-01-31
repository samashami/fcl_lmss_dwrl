import os, random
from PIL import Image
import matplotlib.pyplot as plt

BASE = "/Users/sama/Desktop/Study/fcl_lmss_dwrl/src/data/DWRL_clean"
CLASSES = ["PET","PP","PE","TETRA","PS","PVC","Other"]
EXTS = (".jpg",".jpeg",".png")

def list_images(folder):
    return [os.path.join(folder,f) for f in os.listdir(folder)
            if f.lower().endswith(EXTS)]

# 1) counts
for c in CLASSES:
    n = len(list_images(os.path.join(BASE, c)))
    print(f"{c:6s}: {n}")

# 2) tiny plot helper (minimal preprocessing)
def load_preprocess(path, size=224):
    img = Image.open(path).convert("RGB")
    img = img.resize((size,size))   # keep same as your previous baseline for now
    return img

def show_samples_per_class(k=8):
    fig, axes = plt.subplots(len(CLASSES), k, figsize=(k*1.3, len(CLASSES)*1.3))
    for r, c in enumerate(CLASSES):
        imgs = list_images(os.path.join(BASE, c))
        picks = random.sample(imgs, k=min(k, len(imgs)))
        for j in range(k):
            ax = axes[r, j]
            ax.axis("off")
            if j < len(picks):
                ax.imshow(load_preprocess(picks[j]))
            if j == 0:
                ax.set_title(c, loc="left", fontsize=10)
    plt.tight_layout()
    plt.show()

show_samples_per_class(k=6)  # small number, fast to run