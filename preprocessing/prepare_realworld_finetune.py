from pathlib import Path
import shutil
import random

BASE_DIR = Path(__file__).resolve().parent.parent

CIFAKE_DIR = BASE_DIR / "datasets" / "CIFAKE"
OUTPUT_DIR = BASE_DIR / "datasets" / "CIFAKE_finetune"

# Number of REAL images to use for fine-tuning.
# Keep this modest to avoid unnecessary training time.
REAL_SAMPLES = 5000

SEED = 42
random.seed(SEED)

print("=" * 80)
print("TRUSTIFY AI - PREPARE CIFAKE FINE-TUNING DATA")
print("=" * 80)

# Existing CIFAKE REAL images
source_real = CIFAKE_DIR / "train" / "REAL"

if not source_real.exists():
    raise FileNotFoundError(f"Could not find: {source_real}")

images = [
    p for p in source_real.iterdir()
    if p.is_file() and p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }
]

print(f"\nFound REAL images: {len(images)}")

if len(images) < REAL_SAMPLES:
    REAL_SAMPLES = len(images)

selected = random.sample(images, REAL_SAMPLES)

# Create output structure automatically
output_real = OUTPUT_DIR / "REAL"
output_fake = OUTPUT_DIR / "FAKE"

output_real.mkdir(parents=True, exist_ok=True)
output_fake.mkdir(parents=True, exist_ok=True)

print(f"Selected REAL images: {REAL_SAMPLES}")

# Copy selected REAL images
for i, src in enumerate(selected, 1):
    dst = output_real / src.name

    # Avoid duplicate filename problems
    if dst.exists():
        dst = output_real / f"real_{i:05d}{src.suffix}"

    shutil.copy2(src, dst)

# We don't need to copy fake images here.
# Fine-tuning will use the existing CIFAKE dataset for FAKE
# and this generated REAL subset for additional generalization.

print("\nCreated:")
print(f"  {output_real}")
print(f"  {output_fake}")

print("\nREAL images copied:", len(list(output_real.iterdir())))

print("\nIMPORTANT:")
print("The external evaluation set was NOT touched:")
print(BASE_DIR / "datasets" / "real_world_test" / "REAL")

print("\nPreparation complete.")