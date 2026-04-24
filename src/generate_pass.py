import cv2
import os
import random
import numpy as np
from pathlib import Path

DATA_DIR = "data/raw/Data_YOLO"
SAVE_DIR = "data/pass_images"
CROP_SIZE = 96
TARGET = 3500
TRIES_PER_IMAGE = 40

os.makedirs(SAVE_DIR, exist_ok=True)

# clear old files
for f in Path(SAVE_DIR).glob("*.*"):
    f.unlink()

def read_labels(label_path, w, h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 5:
                continue

            _, xc, yc, bw, bh = map(float, vals)

            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)

            boxes.append((x1,y1,x2,y2))

    return boxes

def overlap(crop, box):
    x1,y1,x2,y2 = crop
    a1,b1,a2,b2 = box
    return not (x2 < a1 or x1 > a2 or y2 < b1 or y1 > b2)

def edge_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges.mean()

images = list(Path(DATA_DIR).glob("images/train/*.jpg"))
images += list(Path(DATA_DIR).glob("images/val/*.jpg"))

count = 0

for img_path in images:

    if count >= TARGET:
        break

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    if w < CROP_SIZE or h < CROP_SIZE:
        continue

    label_path = str(img_path).replace("images", "labels").replace(".jpg", ".txt")
    boxes = read_labels(label_path, w, h)

    candidates = []

    for _ in range(TRIES_PER_IMAGE):

        x = random.randint(0, w - CROP_SIZE)
        y = random.randint(0, h - CROP_SIZE)

        crop_box = (x, y, x+CROP_SIZE, y+CROP_SIZE)

        bad = False
        for b in boxes:
            if overlap(crop_box, b):
                bad = True
                break

        if bad:
            continue

        crop = img[y:y+CROP_SIZE, x:x+CROP_SIZE]

        score = edge_score(crop)

        # prefer structured regions
        if 8 < score < 80:
            candidates.append((score, crop))

    # keep best few
    candidates.sort(reverse=True, key=lambda z: z[0])

    for _, crop in candidates[:3]:

        save_path = os.path.join(SAVE_DIR, f"pass_{count}.jpg")
        cv2.imwrite(save_path, crop)
        count += 1

        if count % 100 == 0:
            print(f"Generated {count}")

        if count >= TARGET:
            break

print("\nDONE")
print("Generated PASS:", count)