"""MobileNetV2 eye-state classifier architecture and checkpoint management.

Implements transfer learning on MobileNetV2 with an adapted classification head,
freeze-then-unfreeze parameter management, and secure weights_only serialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class EyeStateMobileNetV2(nn.Module):
    """MobileNetV2 adapted for driver eye-state classification (open vs closed)."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = mobilenet_v2(weights=weights)

        # Feature extractor up to final 1x1 conv
        self.features = base.features

        # Custom lightweight classification head
        in_features = base.last_channel  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor, adaptive pooling, and head."""
        x = self.features(x)
        # Global average pooling (B, 1280, 1, 1) -> (B, 1280)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(num_classes: int = 2, pretrained: bool = True) -> EyeStateMobileNetV2:
    """Factory creating an EyeStateMobileNetV2 instance."""
    return EyeStateMobileNetV2(num_classes=num_classes, pretrained=pretrained)


def freeze_backbone(model: EyeStateMobileNetV2) -> None:
    """Freezes all feature extractor layers for Phase 1 head training."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_top_blocks(model: EyeStateMobileNetV2, start_block: int = 14) -> None:
    """Unfreezes top inverted residual blocks for Phase 2 fine-tuning."""
    # Unfreeze feature blocks from start_block onwards
    for i, block in enumerate(model.features):
        requires_grad = i >= start_block
        for param in block.parameters():
            param.requires_grad = requires_grad
    for param in model.classifier.parameters():
        param.requires_grad = True


def save_checkpoint(
    model: EyeStateMobileNetV2,
    checkpoint_path: str | Path,
    metadata: dict[str, Any],
    metadata_path: str | Path | None = None,
) -> None:
    """Saves model weights state_dict and companion JSON metadata."""
    ckpt_p = Path(checkpoint_path)
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_p)

    meta_p = Path(metadata_path) if metadata_path else ckpt_p.with_suffix(".json")
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_checkpoint(
    model: EyeStateMobileNetV2,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> EyeStateMobileNetV2:
    """Loads weights into model using secure weights_only=True deserialization."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_p}")

    state_dict = torch.load(ckpt_p, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
