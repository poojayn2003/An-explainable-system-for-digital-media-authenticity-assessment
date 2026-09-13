import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "datasets/CIFAKE/train"
MODEL_DIR = "models"

IMG_SIZE = 224
BATCH_SIZE = 8

# IMPORTANT:
# Keep this 0 for Windows stability.
NUM_WORKERS = 0

VAL_SIZE = 0.10
RANDOM_SEED = 42

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 5

LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5

PATIENCE = 2

# ------------------------------------------------------------
# SAFETY / TESTING
# ------------------------------------------------------------

# Keep TRUE for the first test of the new checkpoint system.
SMOKE_TEST = False
RESUME_TRAINING = False
SMOKE_SAMPLES_PER_CLASS = 500
TRAIN_SAMPLES_PER_CLASS = 10000

# ------------------------------------------------------------
# CHECKPOINTS
# ------------------------------------------------------------

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "efficientnet_best.pth"
)

LAST_CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "efficientnet_last.pth"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TRUSTIFY AI - CHECKPOINT-SAFE EFFICIENTNET TRAINING")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# TRANSFORMS
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

imagenet_mean = weights.transforms().mean
imagenet_std = weights.transforms().std


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading CIFAKE training dataset...")

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=train_transform
)

print(f"Classes: {dataset.classes}")
print(f"Class mapping: {dataset.class_to_idx}")
print(f"Total images: {len(dataset)}")


# ============================================================
# LABELS
# ============================================================

labels = np.array([
    label
    for _, label in dataset.samples
])


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

indices = np.arange(len(dataset))

train_indices, val_indices = train_test_split(
    indices,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels
)


# ============================================================
# SMOKE TEST
# ============================================================

if SMOKE_TEST:

    print("\n" + "=" * 70)
    print("CHECKPOINT SYSTEM SMOKE TEST")
    print("=" * 70)

    fake_indices = [
        i for i in train_indices
        if labels[i] == dataset.class_to_idx["FAKE"]
    ]

    real_indices = [
        i for i in train_indices
        if labels[i] == dataset.class_to_idx["REAL"]
    ]

    fake_indices = fake_indices[
        :SMOKE_SAMPLES_PER_CLASS
    ]

    real_indices = real_indices[
        :SMOKE_SAMPLES_PER_CLASS
    ]

    train_indices = np.array(
        fake_indices + real_indices
    )

    # Small validation set for testing.
    val_indices = val_indices[:500]
# ============================================================
# CONTROLLED TRAINING SET
# ============================================================

if not SMOKE_TEST:

    print("\n" + "=" * 70)
    print("CONTROLLED TRAINING MODE")
    print("=" * 70)

    fake_indices = [
        i for i in train_indices
        if labels[i] == dataset.class_to_idx["FAKE"]
    ]

    real_indices = [
        i for i in train_indices
        if labels[i] == dataset.class_to_idx["REAL"]
    ]

    random.seed(RANDOM_SEED)

    random.shuffle(fake_indices)
    random.shuffle(real_indices)

    fake_indices = fake_indices[:TRAIN_SAMPLES_PER_CLASS]
    real_indices = real_indices[:TRAIN_SAMPLES_PER_CLASS]

    train_indices = np.array(
        fake_indices + real_indices
    )

    print(f"FAKE training images: {len(fake_indices)}")
    print(f"REAL training images: {len(real_indices)}")
    print(f"Total controlled training images: {len(train_indices)}")

print(
    f"\nTraining images: {len(train_indices)}"
)

print(
    f"Validation images: {len(val_indices)}"
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = Subset(
    dataset,
    train_indices
)


val_base_dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=eval_transform
)

val_dataset = Subset(
    val_base_dataset,
    val_indices
)


# ============================================================
# DATA LOADERS
# ============================================================

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

print(
    f"Training batches: {len(train_loader)}"
)

print(
    f"Validation batches: {len(val_loader)}"
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nLoading ImageNet-pretrained EfficientNet-B0..."
)

model = efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    2
)

model = model.to(device)

print("EfficientNet-B0 ready.")
print("Output classes: FAKE / REAL")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels_batch in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels_batch = labels_batch.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_batch
            )

            total_loss += (
                loss.item()
                * labels_batch.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels_batch
            ).sum().item()

            total += labels_batch.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels_batch) in enumerate(
        loader
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        labels_batch = labels_batch.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels_batch
                )

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_batch
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += (
            loss.item()
            * labels_batch.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels_batch
        ).sum().item()

        total += labels_batch.size(0)

        if (batch_idx + 1) % 25 == 0:

            print(
                f"  Batch "
                f"[{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    best_val_accuracy,
    stage
):

    checkpoint = {
        "epoch": epoch,
        "stage": stage,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "best_val_accuracy":
            best_val_accuracy,

        "classes":
            dataset.classes,

        "class_to_idx":
            dataset.class_to_idx
    }

    torch.save(
        checkpoint,
        LAST_CHECKPOINT_PATH
    )

    print(
        "\n✓ LAST CHECKPOINT SAVED"
    )

    print(
        f"  Epoch: {epoch}"
    )

    print(
        f"  Stage: {stage}"
    )

    print(
        f"  File: {LAST_CHECKPOINT_PATH}"
    )


# ============================================================
# BEST MODEL SAVE
# ============================================================

def save_best_model(
    model,
    val_accuracy
):

    torch.save(
        model.state_dict(),
        BEST_MODEL_PATH
    )

    print(
        "\n✓ BEST MODEL SAVED"
    )

    print(
        f"  Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"  File: {BEST_MODEL_PATH}"
    )


# ============================================================
# CHECKPOINT LOAD / RESUME
# ============================================================

def load_checkpoint(model, optimizer, scaler):

    if not os.path.exists(LAST_CHECKPOINT_PATH):
        print("\nNo checkpoint found.")
        print("Starting training from scratch.")

        return 0, None, 0.0

    print("\n" + "=" * 70)
    print("LOADING TRAINING CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(
        LAST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    epoch = checkpoint["epoch"]
    stage = checkpoint["stage"]
    best_accuracy = checkpoint["best_val_accuracy"]

    print("✓ Checkpoint loaded successfully")
    print(f"  Epoch: {epoch}")
    print(f"  Stage: {stage}")
    print(f"  Best validation accuracy: {best_accuracy:.4f}")

    return epoch, stage, best_accuracy

# ============================================================
# FREEZE BACKBONE
# ============================================================

print(
    "\nStage 1: Freezing EfficientNet backbone..."
)

for parameter in model.features.parameters():

    parameter.requires_grad = False


for parameter in model.classifier.parameters():

    parameter.requires_grad = True


# ============================================================
# STAGE 1 OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE_STAGE1,
    weight_decay=1e-4
)


# ============================================================
# MIXED PRECISION SCALER
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda")
)


# ============================================================
# TRAINING STAGE
# ============================================================

def run_training_stage(
    stage_name,
    epochs,
    optimizer,
    start_epoch=1,
    best_val_accuracy=0.0
):

    patience_counter = 0

    print("\n" + "=" * 70)
    print(stage_name)
    print("=" * 70)

    for epoch in range(
        start_epoch,
        epochs + 1
    ):

        print(
            f"\nEpoch {epoch}/{epochs}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler
        )

        print(
            f"\nTrain Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # CRITICAL SAFETY CHECKPOINT
        #
        # SAVE BEFORE VALIDATION
        # ----------------------------------------------------

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val_accuracy=best_val_accuracy,
            stage=stage_name
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        try:

            print(
                "\nStarting validation..."
            )

            val_loss, val_accuracy = validate(
                model,
                val_loader
            )

            print(
                f"\nValidation Loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation Accuracy: "
                f"{val_accuracy:.4f}"
            )

        except KeyboardInterrupt:

            print(
                "\n\n⚠ Validation interrupted."
            )

            print(
                "The completed training epoch "
                "has already been saved."
            )

            print(
                f"Resume checkpoint: "
                f"{LAST_CHECKPOINT_PATH}"
            )

            return best_val_accuracy

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            save_best_model(
                model,
                val_accuracy
            )

            patience_counter = 0

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping triggered."
                )

                break

    return best_val_accuracy

# ============================================================
# STAGE 1
# ============================================================

stage1_epochs = (
    1 if SMOKE_TEST
    else STAGE1_EPOCHS
)

start_epoch = 1
best_accuracy = 0.0

# ------------------------------------------------------------
# RESUME FROM CHECKPOINT
# ------------------------------------------------------------

if RESUME_TRAINING:

    saved_epoch, saved_stage, saved_best_accuracy = load_checkpoint(
        model,
        optimizer,
        scaler
    )

    best_accuracy = saved_best_accuracy

    # Only resume Stage 1 if the checkpoint belongs to Stage 1.
    if saved_stage == "STAGE 1 - CLASSIFIER TRAINING":

        start_epoch = saved_epoch + 1

        print("\nResuming Stage 1...")
        print(f"Starting from epoch: {start_epoch}")

    else:

        print(
            "\nCheckpoint belongs to another stage."
        )


best_accuracy = run_training_stage(
    stage_name="STAGE 1 - CLASSIFIER TRAINING",

    epochs=stage1_epochs,

    optimizer=optimizer,

    start_epoch=start_epoch,

    best_val_accuracy=best_accuracy
)


# ============================================================
# STAGE 2
# ============================================================

if not SMOKE_TEST:

    print("\n" + "=" * 70)
    print(
        "STAGE 2: FINE-TUNING UPPER "
        "EFFICIENTNET LAYERS"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Unfreeze final feature blocks
    # --------------------------------------------------------

    for parameter in model.features[-3:].parameters():
        parameter.requires_grad = True

    # --------------------------------------------------------
    # Stage 2 optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE_STAGE2,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # RESUME INFORMATION
    # --------------------------------------------------------

    stage2_start_epoch = 1

    if RESUME_TRAINING:

        # Check whether the checkpoint belongs to Stage 2.
        if (
            'saved_stage' in globals()
            and saved_stage == "STAGE 2 - FINE-TUNING"
        ):

            print("\nResuming Stage 2...")

            # Load Stage 2 optimizer/scaler state.
            checkpoint = torch.load(
                LAST_CHECKPOINT_PATH,
                map_location=device,
                weights_only=False
            )

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

            stage2_start_epoch = (
                checkpoint["epoch"] + 1
            )

            best_accuracy = (
                checkpoint["best_val_accuracy"]
            )

            print(
                f"Starting Stage 2 from epoch: "
                f"{stage2_start_epoch}"
            )

        elif (
            'saved_stage' in globals()
            and saved_stage == "STAGE 1 - CLASSIFIER TRAINING"
        ):

            # Stage 1 was completed.
            # Start Stage 2 from epoch 1.
            stage2_start_epoch = 1

            print(
                "\nStage 1 checkpoint detected."
            )

            print(
                "Starting Stage 2 from epoch 1."
            )

    # --------------------------------------------------------
    # TRAIN STAGE 2
    # --------------------------------------------------------

    best_accuracy = run_training_stage(
        stage_name="STAGE 2 - FINE-TUNING",

        epochs=STAGE2_EPOCHS,

        optimizer=optimizer,

        start_epoch=stage2_start_epoch,

        best_val_accuracy=best_accuracy
    )

# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation accuracy: "
    f"{best_accuracy:.4f}"
)

print(
    f"Best model: "
    f"{BEST_MODEL_PATH}"
)

print(
    f"Last checkpoint: "
    f"{LAST_CHECKPOINT_PATH}"
)

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 2)
    )

    print("\nGPU MEMORY")

    print(
        f"Allocated: "
        f"{allocated:.2f} MB"
    )

    print(
        f"Reserved : "
        f"{reserved:.2f} MB"
    )