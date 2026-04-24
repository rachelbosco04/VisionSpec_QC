import cv2
import os
import shutil
import numpy as np
from pathlib import Path

INPUT_DIR = "data/pass_images"
REJECT_DIR = "data/rejected_pass"

os.makedirs(REJECT_DIR, exist_ok=True)

files = list(Path(INPUT_DIR).glob("*.jpg"))

kept = 0
removed = 0

for file in files:
    img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

    if img is None:
        shutil.move(str(file), os.path.join(REJECT_DIR, file.name))
        removed += 1
        continue

    mean = img.mean()
    std = img.std()

    # black pixels %
    black_ratio = np.sum(img < 25) / img.size

    reject = False

    # too dark
    if mean < 35:
        reject = True

    # too blank
    if std < 8:
        reject = True

    # too many dark blobs
    if black_ratio > 0.22:
        reject = True

    if reject:
        shutil.move(str(file), os.path.join(REJECT_DIR, file.name))
        removed += 1
    else:
        kept += 1

print("Kept:", kept)
print("Rejected:", removed)