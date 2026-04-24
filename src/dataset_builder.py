import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# =====================================================
# VisionSpec QC - Dataset Builder
# Creates final balanced dataset for Week 2 training
# =====================================================

random.seed(42)

# ---------------- CONFIG ----------------
PASS_DIR = "data/pass_images"
DEFECT_DIR = "data/raw/Data_YOLO/images/train"

OUTPUT_DIR = "data/final_dataset"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

IMG_EXTS = [".jpg", ".jpeg", ".png"]

# ----------------------------------------


def reset_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def get_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(Path(folder).glob(f"*{ext}"))
    return [str(x) for x in files]


def copy_files(files, target_folder):
    os.makedirs(target_folder, exist_ok=True)

    for file in files:
        shutil.copy(file, target_folder)


# =====================================================
# LOAD DATA
# =====================================================

print("=" * 55)
print(" VisionSpec QC - Dataset Builder")
print("=" * 55)

pass_images = get_images(PASS_DIR)
defect_images = get_images(DEFECT_DIR)

print(f"PASS images found   : {len(pass_images)}")
print(f"DEFECT images found : {len(defect_images)}")

if len(pass_images) == 0 or len(defect_images) == 0:
    print("\nERROR: Missing PASS or DEFECT images.")
    exit()

# =====================================================
# BALANCE CLASSES
# =====================================================

n = min(len(pass_images), len(defect_images))

random.shuffle(pass_images)
random.shuffle(defect_images)

pass_images = pass_images[:n]
defect_images = defect_images[:n]

print(f"Balanced each class : {n}")

# =====================================================
# SPLIT
# =====================================================

train_pass, temp_pass = train_test_split(
    pass_images,
    test_size=(1 - TRAIN_RATIO),
    random_state=42
)

val_pass, test_pass = train_test_split(
    temp_pass,
    test_size=0.50,
    random_state=42
)

train_defect, temp_defect = train_test_split(
    defect_images,
    test_size=(1 - TRAIN_RATIO),
    random_state=42
)

val_defect, test_defect = train_test_split(
    temp_defect,
    test_size=0.50,
    random_state=42
)

# =====================================================
# CREATE FOLDERS
# =====================================================

reset_folder(OUTPUT_DIR)

folders = [
    "train/pass",
    "train/defect",
    "val/pass",
    "val/defect",
    "test/pass",
    "test/defect"
]

for folder in folders:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

# =====================================================
# COPY FILES
# =====================================================

copy_files(train_pass,   os.path.join(OUTPUT_DIR, "train/pass"))
copy_files(train_defect, os.path.join(OUTPUT_DIR, "train/defect"))

copy_files(val_pass,     os.path.join(OUTPUT_DIR, "val/pass"))
copy_files(val_defect,   os.path.join(OUTPUT_DIR, "val/defect"))

copy_files(test_pass,    os.path.join(OUTPUT_DIR, "test/pass"))
copy_files(test_defect,  os.path.join(OUTPUT_DIR, "test/defect"))

# =====================================================
# SUMMARY
# =====================================================

print("\nDataset created successfully.\n")

print("TRAIN")
print(" PASS   :", len(train_pass))
print(" DEFECT :", len(train_defect))

print("\nVAL")
print(" PASS   :", len(val_pass))
print(" DEFECT :", len(val_defect))

print("\nTEST")
print(" PASS   :", len(test_pass))
print(" DEFECT :", len(test_defect))

print("\nSaved to:", OUTPUT_DIR)
print("=" * 55)