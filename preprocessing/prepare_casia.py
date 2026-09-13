from pathlib import Path
import random
import shutil

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CASIA_DIR = BASE_DIR / "datasets" / "CASIA2"
OUTPUT_DIR = BASE_DIR / "datasets" / "CASIA2_processed"

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ============================================================
# SETUP
# ============================================================

random.seed(SEED)

AU_DIR = CASIA_DIR / "Au"
TP_DIR = CASIA_DIR / "Tp"

if not AU_DIR.exists():
    raise FileNotFoundError(f"Missing folder: {AU_DIR}")

if not TP_DIR.exists():
    raise FileNotFoundError(f"Missing folder: {TP_DIR}")

# ============================================================
# COLLECT IMAGES
# ============================================================

real_images = [
    p for p in AU_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

edited_images = [
    p for p in TP_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"REAL images found:   {len(real_images)}")
print(f"EDITED images found: {len(edited_images)}")

# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(real_images)
random.shuffle(edited_images)

# ============================================================
# SPLIT FUNCTION
# ============================================================

def split_dataset(images):
    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


real_train, real_val, real_test = split_dataset(real_images)
edited_train, edited_val, edited_test = split_dataset(edited_images)

print("\nSplit sizes:")
print(f"REAL   -> train={len(real_train)}, val={len(real_val)}, test={len(real_test)}")
print(f"EDITED -> train={len(edited_train)}, val={len(edited_val)}, test={len(edited_test)}")

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    for label in ["REAL", "EDITED"]:
        (OUTPUT_DIR / split / label).mkdir(
            parents=True,
            exist_ok=True
        )

# ============================================================
# COPY FILES
# ============================================================

def copy_files(files, split, label):
    destination = OUTPUT_DIR / split / label

    for index, source in enumerate(files, start=1):
        destination_file = destination / source.name

        # Avoid accidental overwrite
        if destination_file.exists():
            destination_file = destination / f"{source.stem}_{index}{source.suffix}"

        shutil.copy2(source, destination_file)

    print(f"Copied {len(files)} {label} images -> {split}")


copy_files(real_train, "train", "REAL")
copy_files(real_val, "val", "REAL")
copy_files(real_test, "test", "REAL")

copy_files(edited_train, "train", "EDITED")
copy_files(edited_val, "val", "EDITED")
copy_files(edited_test, "test", "EDITED")

print("\n========================================")
print("CASIA2 preprocessing completed!")
print("========================================")
print(f"Output: {OUTPUT_DIR}")