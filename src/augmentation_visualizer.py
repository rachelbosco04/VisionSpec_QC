import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# VisionSpec QC - Final Dataset Visualizer + Augmentation
# =====================================================

DATASET_DIR = "data/final_dataset"
SAVE_DIR = "outputs/logs"

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# PCB-Friendly Augmentation Settings
# =====================================================

ROTATION_LIMIT = 10          # ±10 degrees
BRIGHTNESS = 0.15           # ±15%
CONTRAST = 0.15             # ±15%
ZOOM = 0.10                 # ±10%
NOISE_STD = 3               # low gaussian noise

# =====================================================
# Helpers
# =====================================================

def get_images(folder):
    files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        files.extend(Path(folder).glob(ext))
    return files


def adjust_brightness_contrast(img):
    alpha = 1.0 + random.uniform(-CONTRAST, CONTRAST)
    beta = random.uniform(-BRIGHTNESS, BRIGHTNESS) * 255
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def rotate_zoom(img):
    h, w = img.shape[:2]

    angle = random.uniform(-ROTATION_LIMIT, ROTATION_LIMIT)
    scale = 1.0 + random.uniform(-ZOOM, ZOOM)

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )


def add_noise(img):
    noise = np.random.normal(0, NOISE_STD, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(img):
    out = img.copy()

    out = rotate_zoom(out)
    out = adjust_brightness_contrast(out)

    if random.random() < 0.25:
        out = add_noise(out)

    return out


# =====================================================
# Count Images
# =====================================================

splits = ["train", "val", "test"]
classes = ["pass", "defect"]

counts = {}

for split in splits:
    counts[split] = {}
    for cls in classes:
        folder = os.path.join(DATASET_DIR, split, cls)
        imgs = get_images(folder)
        counts[split][cls] = len(imgs)

# =====================================================
# Final Class Distribution
# =====================================================

total_pass = sum(counts[s]["pass"] for s in splits)
total_defect = sum(counts[s]["defect"] for s in splits)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    ["PASS", "DEFECT"],
    [total_pass, total_defect],
    color=["green", "red"]
)

for b in bars:
    h = b.get_height()
    plt.text(
        b.get_x() + b.get_width() / 2,
        h + 10,
        str(int(h)),
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

plt.title("VisionSpec QC - Final Dataset Distribution")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/final_class_distribution.png", dpi=200)
plt.close()

# =====================================================
# Random Samples
# =====================================================

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for row, cls in enumerate(classes):

    imgs = get_images(os.path.join(DATASET_DIR, "train", cls))
    sample = random.sample(imgs, min(4, len(imgs)))

    for col, img_path in enumerate(sample):

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[row, col].imshow(img)
        axes[row, col].set_title(cls.upper())
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/final_random_samples.png", dpi=200)
plt.close()

# =====================================================
# Image Stats
# =====================================================

widths = []
heights = []
brightness = []

all_imgs = []

for split in splits:
    for cls in classes:
        all_imgs.extend(get_images(os.path.join(DATASET_DIR, split, cls)))

sample_imgs = random.sample(all_imgs, min(200, len(all_imgs)))

for path in sample_imgs:

    img = cv2.imread(str(path))
    h, w = img.shape[:2]

    widths.append(w)
    heights.append(h)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness.append(gray.mean())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].hist(widths, bins=15)
ax[0].set_title("Width")

ax[1].hist(heights, bins=15)
ax[1].set_title("Height")

ax[2].hist(brightness, bins=15)
ax[2].set_title("Brightness")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/final_image_stats.png", dpi=200)
plt.close()

# =====================================================
# AUGMENTATION VERIFICATION GRID
# =====================================================

fig, axes = plt.subplots(6, 4, figsize=(12, 18))

for row in range(6):

    cls = random.choice(classes)

    imgs = get_images(os.path.join(DATASET_DIR, "train", cls))
    img_path = random.choice(imgs)

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axes[row, 0].imshow(img)
    axes[row, 0].set_title("Original")
    axes[row, 0].axis("off")

    for col in range(1, 4):

        aug = augment(img)

        axes[row, col].imshow(aug)
        axes[row, col].set_title(f"Aug {col}")
        axes[row, col].axis("off")

plt.suptitle("VisionSpec QC - Final Augmentation Verification", fontsize=16)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/final_augmentation_grid.png", dpi=200)
plt.close()

# =====================================================
# Done
# =====================================================

print("Done.")
print("Saved:")
print(" final_class_distribution.png")
print(" final_random_samples.png")
print(" final_image_stats.png")
print(" final_augmentation_grid.png")