"""
Fine-tune ResNet18 to classify bisexual lighting in MTG art.

Training is two-phase to prevent overfitting on the small dataset:
  Phase 1 — freeze backbone, train only the classifier head (faster, stable)
  Phase 2 — unfreeze all layers, fine-tune end-to-end with a lower LR

Best model (by val accuracy) is saved to models/lighting_classifier.pt.

Usage:
  python -m bisexual_lighting.train
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from .model import LightingClassifier

load_dotenv(override=True)

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bisexual_lighting"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "lighting_classifier.pt"

PHASE1_EPOCHS = int(os.environ.get("TRAIN_PHASE1_EPOCHS", "10"))
PHASE2_EPOCHS = int(os.environ.get("TRAIN_PHASE2_EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "16"))

# Keep color jitter modest — color is a primary signal for bisexual lighting
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def make_weighted_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    class_counts = [labels.count(c) for c in range(num_classes)]
    weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch. If optimizer is None, runs in eval mode (no grad)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = correct = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

    return total_loss / len(loader), correct / total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load dataset twice so each split gets its own transform
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=TRAIN_TRANSFORMS)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=VAL_TRANSFORMS)

    log.info("Classes: %s", train_dataset.classes)  # ['no', 'yes'] alphabetically
    log.info("Total images: %d", len(train_dataset))

    all_labels = [label for _, label in train_dataset.samples]
    indices = list(range(len(all_labels)))

    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=all_labels, random_state=42
    )
    log.info("Train: %d  Val: %d", len(train_idx), len(val_idx))

    train_labels = [all_labels[i] for i in train_idx]
    sampler = make_weighted_sampler(train_labels, num_classes=len(train_dataset.classes))

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Class-weighted loss as a second guard against imbalance
    class_counts = [train_labels.count(c) for c in range(len(train_dataset.classes))]
    class_weights = torch.tensor(
        [1.0 / c for c in class_counts], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = LightingClassifier().to(device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    # ── Phase 1: train head only ─────────────────────────────────────────────
    log.info("Phase 1: training classifier head only (%d epochs)", PHASE1_EPOCHS)
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.fc.requires_grad_(True)

    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=1e-3)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc = run_epoch(model, val_loader, criterion, None, device)
        log.info("P1 epoch %2d/%d  loss=%.4f  val_acc=%.4f", epoch, PHASE1_EPOCHS, train_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    # ── Phase 2: fine-tune entire network ────────────────────────────────────
    log.info("Phase 2: fine-tuning entire network (%d epochs)", PHASE2_EPOCHS)
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        log.info("P2 epoch %2d/%d  loss=%.4f  val_acc=%.4f", epoch, PHASE2_EPOCHS, train_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            log.info("  -> new best saved (%.4f)", best_val_acc)

    log.info("Training complete. Best val_acc=%.4f  Model: %s", best_val_acc, MODEL_PATH)


if __name__ == "__main__":
    main()
