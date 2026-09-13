from pathlib import Path
import hashlib
import shutil
import random
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "datasets" / "CASIA2_processed"
OUTPUT_DIR = BASE_DIR / "datasets" / "CASIA2_clean"

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
}


# ============================================================
# FUNCTIONS
# ============================================================

def md5_hash(path):
    """Return MD5 hash of an image."""

    hasher = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def collect_unique_images(class_dir):
    """
    Find images and remove exact byte-level duplicates.

    One representative image is kept for each unique hash.
    """

    hash_to_file = {}

    for path in class_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        file_hash = md5_hash(path)

        if file_hash not in hash_to_file:
            hash_to_file[file_hash] = path

    return list(hash_to_file.values())


# ============================================================
# START
# ============================================================

print("=" * 60)
print("CASIA 2.0 DATASET CLEANING")
print("=" * 60)

print(f"\nSource: {SOURCE_DIR}")
print(f"Output: {OUTPUT_DIR}")

random.seed(SEED)


# ============================================================
# COLLECT UNIQUE IMAGES
# ============================================================

classes = ["EDITED", "REAL"]

unique_images = {}

for class_name in classes:

    class_dir = SOURCE_DIR / "train" / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {class_dir}"
        )

    images = collect_unique_images(class_dir)

    unique_images[class_name] = images

    print(
        f"\n{class_name}: "
        f"{len(list(class_dir.iterdir()))} files → "
        f"{len(images)} unique images"
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:

    for class_name in classes:

        directory = OUTPUT_DIR / split / class_name

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# SPLIT DATA
# ============================================================

for class_name in classes:

    images = unique_images[class_name]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]

    val_images = images[train_end:val_end]

    test_images = images[val_end:]

    print(f"\n{class_name} split:")

    print(f"  Train: {len(train_images)}")
    print(f"  Val:   {len(val_images)}")
    print(f"  Test:  {len(test_images)}")

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        destination = (
            OUTPUT_DIR /
            split_name /
            class_name
        )

        for image_path in split_images:

            target = destination / image_path.name

            # Avoid filename collision
            if target.exists():

                target = destination / (
                    image_path.stem +
                    "_" +
                    md5_hash(image_path)[:8] +
                    image_path.suffix
                )

            shutil.copy2(
                image_path,
                target
            )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CLEAN DATASET CREATED")
print("=" * 60)

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}")

    for class_name in classes:

        directory = OUTPUT_DIR / split / class_name

        count = sum(
            1
            for p in directory.iterdir()
            if p.is_file()
        )

        print(
            f"{class_name}: {count}"
        )

print("\nOutput:")
print(OUTPUT_DIR)

print("\nCleaning completed successfully.")