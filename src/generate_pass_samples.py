import cv2
import os
import random
from pathlib import Path

# ===============================
# CONFIG
# ===============================
DATA_DIR = "data/raw/Data_YOLO"
SAVE_DIR = "data/pass_images"

CROP_SIZE = 96
TARGET = 3000
MAX_TRIES_PER_IMAGE = 25

os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# READ LABELS
# ===============================
def read_labels(label_path, w, h):
    """
    Read YOLO labels and convert to pixel boxes
    """
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        vals = line.strip().split()

        if len(vals) < 5:
            continue

        _, xc, yc, bw, bh = map(float, vals)

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        boxes.append((x1, y1, x2, y2))

    return boxes


# ===============================
# CHECK OVERLAP
# ===============================
def overlap(crop, box):
    """
    True if crop overlaps defect box
    """
    x1, y1, x2, y2 = crop
    a1, b1, a2, b2 = box

    return not (
        x2 < a1 or
        x1 > a2 or
        y2 < b1 or
        y1 > b2
    )


# ===============================
# LOAD IMAGES
# ===============================
images = list(Path(DATA_DIR).glob("images/train/*.jpg"))
images += list(Path(DATA_DIR).glob("images/val/*.jpg"))
images += list(Path(DATA_DIR).glob("images/train/*.png"))
images += list(Path(DATA_DIR).glob("images/val/*.png"))

print("Images found:", len(images))


# ===============================
# GENERATE PASS SAMPLES
# ===============================
count = 0
skipped_small = 0

for img_path in images:

    if count >= TARGET:
        break

    img = cv2.imread(str(img_path))

    if img is None:
        continue

    h, w = img.shape[:2]

    # Skip too-small images
    if w < CROP_SIZE or h < CROP_SIZE:
        skipped_small += 1
        continue

    # labels/train/img.txt
    label_path = str(img_path).replace("images", "labels")
    label_path = os.path.splitext(label_path)[0] + ".txt"

    boxes = read_labels(label_path, w, h)

    tries = 0

    while tries < MAX_TRIES_PER_IMAGE and count < TARGET:

        x = random.randint(0, w - CROP_SIZE)
        y = random.randint(0, h - CROP_SIZE)

        crop_box = (x, y, x + CROP_SIZE, y + CROP_SIZE)

        bad = False

        for b in boxes:
            if overlap(crop_box, b):
                bad = True
                break

        if not bad:
            crop = img[y:y + CROP_SIZE, x:x + CROP_SIZE]

            # basic quality check
            if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
                tries += 1
                continue

            save_path = os.path.join(
                SAVE_DIR,
                f"pass_{count:05d}.jpg"
            )

            cv2.imwrite(save_path, crop)
            count += 1

            if count % 100 == 0:
                print(f"Generated {count} PASS samples")

        tries += 1


# ===============================
# FINAL REPORT
# ===============================
print("\n==========================")
print("PASS SAMPLE GENERATION DONE")
print("==========================")
print("Generated PASS samples :", count)
print("Skipped small images   :", skipped_small)
print("Saved to               :", SAVE_DIR)
print("==========================")