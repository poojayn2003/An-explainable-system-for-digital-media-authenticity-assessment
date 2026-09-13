from pathlib import Path
import random
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.amp import autocast, GradScaler

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CIFAKE_DIR = BASE_DIR / "datasets" / "CIFAKE"
REALWORLD_DIR = BASE_DIR / "datasets" / "realworld_finetune" / "REAL"

WORK_DIR = BASE_DIR / "datasets" / "realworld_ft_balanced"

MODEL_PATH = BASE_DIR / "models" / "efficientnet_best.pth"
OUTPUT_MODEL = BASE_DIR / "models" / "efficientnet_realworld_ft.pth"

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 1e-5
NUM_WORKERS = 0

NUM_IMAGES_PER_CLASS = 604
VAL_RATIO = 0.15

SEED = 42

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("BALANCED REAL-WORLD FINE-TUNING")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# CHECK PATHS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Original model not found:\n{MODEL_PATH}"
    )

if not REALWORLD_DIR.exists():
    raise FileNotFoundError(
        f"REAL-world dataset not found:\n{REALWORLD_DIR}"
    )

fake_dir = CIFAKE_DIR / "train" / "FAKE"

if not fake_dir.exists():
    raise FileNotFoundError(
        f"CIFAKE FAKE directory not found:\n{fake_dir}"
    )

# ============================================================
# COLLECT IMAGES
# ============================================================

random.seed(SEED)

real_images = [
    p for p in REALWORLD_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

fake_images = [
    p for p in fake_dir.iterdir()
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

print()
print("Available REAL-world images :", len(real_images))
print("Available CIFAKE FAKE images:", len(fake_images))

if len(real_images) < NUM_IMAGES_PER_CLASS:
    raise RuntimeError(
        f"Need {NUM_IMAGES_PER_CLASS} REAL images, "
        f"but only {len(real_images)} available."
    )

if len(fake_images) < NUM_IMAGES_PER_CLASS:
    raise RuntimeError(
        f"Need {NUM_IMAGES_PER_CLASS} FAKE images, "
        f"but only {len(fake_images)} available."
    )

random.shuffle(real_images)
random.shuffle(fake_images)

real_images = real_images[:NUM_IMAGES_PER_CLASS]
fake_images = fake_images[:NUM_IMAGES_PER_CLASS]

print()
print("Selected REAL images :", len(real_images))
print("Selected FAKE images :", len(fake_images))

# ============================================================
# CREATE BALANCED DATASET
#
# ImageFolder alphabetical order:
#
# FAKE -> 0
# REAL -> 1
# ============================================================

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)

train_fake = WORK_DIR / "train" / "FAKE"
train_real = WORK_DIR / "train" / "REAL"

val_fake = WORK_DIR / "val" / "FAKE"
val_real = WORK_DIR / "val" / "REAL"

for directory in [
    train_fake,
    train_real,
    val_fake,
    val_real
]:
    directory.mkdir(parents=True)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

val_count = int(NUM_IMAGES_PER_CLASS * VAL_RATIO)

real_val = real_images[:val_count]
real_train = real_images[val_count:]

fake_val = fake_images[:val_count]
fake_train = fake_images[val_count:]

# ============================================================
# COPY DATA
# ============================================================

for i, image in enumerate(fake_train):
    shutil.copy2(
        image,
        train_fake / f"fake_{i:04d}.jpg"
    )

for i, image in enumerate(real_train):
    shutil.copy2(
        image,
        train_real / f"real_{i:04d}.jpg"
    )

for i, image in enumerate(fake_val):
    shutil.copy2(
        image,
        val_fake / f"fake_{i:04d}.jpg"
    )

for i, image in enumerate(real_val):
    shutil.copy2(
        image,
        val_real / f"real_{i:04d}.jpg"
    )

print()
print("Training:")
print("  FAKE:", len(fake_train))
print("  REAL:", len(real_train))

print("Validation:")
print("  FAKE:", len(fake_val))
print("  REAL:", len(real_val))

# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    WORK_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    WORK_DIR / "val",
    transform=val_transform
)

print()
print("Classes:", train_dataset.classes)
print("Mapping:", train_dataset.class_to_idx)

# We MUST get:
# {'FAKE': 0, 'REAL': 1}

if train_dataset.class_to_idx != {
    "FAKE": 0,
    "REAL": 1
}:
    raise RuntimeError(
        "ERROR: Incorrect class mapping!"
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# ============================================================
# LOAD EXISTING CIFAKE MODEL
# ============================================================

print()
print("Loading original CIFAKE EfficientNet-B0...")

model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

model = model.to(device)

# ============================================================
# FREEZE FEATURE EXTRACTOR
# ============================================================

for param in model.features.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(
    f"Trainable parameters: {trainable_params:,}"
)

# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scaler = GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)

# ============================================================
# TRAIN
# ============================================================

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    print()
    print("=" * 50)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 50)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += (
            loss.item() * labels.size(0)
        )

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            with autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            val_loss += (
                loss.item() * labels.size(0)
            )

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_acc:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_acc:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            OUTPUT_MODEL
        )

        print()
        print(
            "✓ Saved:",
            OUTPUT_MODEL
        )

# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("FINE-TUNING COMPLETE")
print("=" * 60)
print()
print("Original model:")
print(MODEL_PATH)
print()
print("New model:")
print(OUTPUT_MODEL)
print()
print("Original model was NOT overwritten.")
print("=" * 60)