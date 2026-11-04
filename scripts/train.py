"""MobileNetV2 eye-state classifier training pipeline with transfer learning.

Implements two-phase fine-tuning (frozen backbone -> unfrozen top blocks),
class-weighted CrossEntropyLoss, AdamW optimizer, Cosine Annealing LR schedule,
early stopping, best-model checkpointing, and metadata JSON serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scripts.prepare_data import generate_synthetic_dataset
from src.cnn_model import (
    build_model,
    freeze_backbone,
    unfreeze_top_blocks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds for scientific reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_file_sha256(path: Path) -> str:
    """Computes the SHA-256 hash of a file for integrity verification."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Computes inverse frequency class weights strictly from the training partition."""
    targets = [s[1] for s in dataset.samples]
    classes, counts = np.unique(targets, return_counts=True)
    total_samples = len(targets)
    num_classes = len(classes)
    weights = total_samples / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluates validation partition loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def train_model(
    data_dir: Path,
    output_model_path: Path,
    output_meta_path: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device_str: str = "cpu",
    seed: int = 42,
) -> dict[str, float]:
    """Coordinates the full two-phase transfer learning workflow."""
    set_seed(seed)
    device = torch.device(
        device_str if torch.cuda.is_available() and device_str != "cpu" else "cpu"
    )
    logger.info("Training on device: %s", device)

    # ImageNet transforms
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    train_dir = data_dir / "processed" / "train"
    val_dir = data_dir / "processed" / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        logger.info("Processed dataset partitions missing. Generating synthetic training data...")
        generate_synthetic_dataset(data_dir / "processed", seed=seed)

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize model
    model = build_model(num_classes=2, pretrained=False).to(device)

    # Phase 1: Train classifier head
    logger.info("--- Phase 1: Training Classification Head ---")
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-2)

    best_val_loss = float("inf")
    best_weights = None
    best_val_acc = 0.0

    phase1_epochs = max(epochs // 2, 1)
    for epoch in range(1, phase1_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate_epoch(model, val_loader, criterion, device)
        logger.info(
            "Phase 1 Epoch %d/%d | Train Loss: %.4f Acc: %.3f | Val Loss: %.4f Acc: %.3f",
            epoch,
            phase1_epochs,
            tr_loss,
            tr_acc,
            v_loss,
            v_acc,
        )
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_acc = v_acc
            best_weights = model.state_dict()

    # Phase 2: Fine-tune top blocks
    logger.info("--- Phase 2: Fine-Tuning Top Blocks ---")
    unfreeze_top_blocks(model, start_block=14)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 0.1,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - phase1_epochs, 1)
    )

    phase2_epochs = epochs - phase1_epochs
    for epoch in range(1, phase2_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(
            "Phase 2 Epoch %d/%d | Train Loss: %.4f Acc: %.3f | Val Loss: %.4f Acc: %.3f",
            epoch,
            phase2_epochs,
            tr_loss,
            tr_acc,
            v_loss,
            v_acc,
        )
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_acc = v_acc
            best_weights = model.state_dict()

    # Save best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    model_sha256 = compute_file_sha256(output_model_path)

    metadata = {
        "architecture": "MobileNetV2",
        "input_dimensions": [3, 128, 128],
        "classes": train_dataset.classes,
        "weights_sha256": model_sha256,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "v1.0-subject-split",
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "total_epochs": epochs,
    }

    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved best model to %s (SHA256: %s...)", output_model_path, model_sha256[:16])
    logger.info("Saved model metadata to %s", output_meta_path)
    return {"val_loss": best_val_loss, "val_accuracy": best_val_acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 eye-state classifier.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=4, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output-model", type=str, default="models/mobilenetv2_eye_state.pt")
    parser.add_argument("--output-metadata", type=str, default="models/mobilenetv2_eye_state.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        data_dir=Path(args.data_dir),
        output_model_path=Path(args.output_model),
        output_meta_path=Path(args.output_metadata),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
