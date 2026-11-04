"""Subject-wise dataset extraction, partitioning, and synthetic test generator.

Enforces strict subject-wise partitioning (70% train, 15% val, 15% test) to prevent
data leakage across video frames. Writes deterministic data/splits.json.
Provides a self-contained synthetic eye dataset generator for offline verification.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_eye_patch(is_closed: bool, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Generates an idealized synthetic eye patch for pipeline testing."""
    h, w = size
    # Skin-toned background with mild Gaussian noise
    base_color = np.array([180, 200, 230], dtype=np.uint8)  # BGR
    patch = np.ones((h, w, 3), dtype=np.uint8) * base_color
    noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    center = (w // 2, h // 2)

    if is_closed:
        # Draw a horizontal closed eyelid crease (dark slit)
        cv2.line(patch, (w // 6, h // 2), (5 * w // 6, h // 2), (40, 40, 60), 3)
        # Subtle curved lower shadow
        cv2.ellipse(patch, center, (w // 3, h // 8), 0, 0, 180, (80, 80, 100), 2)
    else:
        # Draw open eye sclera (elliptical white area)
        cv2.ellipse(patch, center, (w // 3, h // 4), 0, 0, 360, (240, 240, 245), -1)
        # Draw dark iris
        cv2.circle(patch, center, h // 6, (60, 40, 20), -1)
        # Draw black pupil
        cv2.circle(patch, center, h // 12, (10, 10, 10), -1)
        # Specular reflection dot
        cv2.circle(patch, (center[0] + 5, center[1] - 5), 3, (255, 255, 255), -1)

    return patch


def generate_synthetic_dataset(
    output_dir: Path,
    num_subjects: int = 10,
    samples_per_subject: int = 20,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Creates a deterministic synthetic eye-state dataset partitioned by subject."""
    random.seed(seed)
    np.random.seed(seed)

    subjects = [f"synth_subj_{i:02d}" for i in range(1, num_subjects + 1)]
    random.shuffle(subjects)

    n_train = int(num_subjects * 0.70)
    n_val = int(num_subjects * 0.15)
    splits = {
        "train": subjects[:n_train],
        "val": subjects[n_train : n_train + n_val],
        "test": subjects[n_train + n_val :],
    }

    for split_name, subj_list in splits.items():
        for subj in subj_list:
            open_dir = output_dir / split_name / "open"
            closed_dir = output_dir / split_name / "closed"
            open_dir.mkdir(parents=True, exist_ok=True)
            closed_dir.mkdir(parents=True, exist_ok=True)

            for i in range(samples_per_subject // 2):
                img_open = generate_synthetic_eye_patch(is_closed=False)
                img_closed = generate_synthetic_eye_patch(is_closed=True)
                cv2.imwrite(str(open_dir / f"{subj}_open_{i:03d}.png"), img_open)
                cv2.imwrite(str(closed_dir / f"{subj}_closed_{i:03d}.png"), img_closed)

    logger.info("Generated synthetic dataset across %d subjects in %s", num_subjects, output_dir)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and partition eye state dataset.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic fallback data")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    root_dir = Path(args.data_dir)
    processed_dir = root_dir / "processed"
    splits_file = root_dir / "splits.json"

    if args.synthetic or not (root_dir / "raw").is_dir():
        logger.info(
            "External raw dataset not found or --synthetic specified. Building synthetic dataset."
        )
        splits = generate_synthetic_dataset(processed_dir, seed=args.seed)
    else:
        # If external dataset exists, group files by subject prefix and partition
        raw_files = list((root_dir / "raw").glob("**/*.png")) + list(
            (root_dir / "raw").glob("**/*.jpg")
        )
        if not raw_files:
            logger.info("No raw images found. Generating synthetic dataset.")
            splits = generate_synthetic_dataset(processed_dir, seed=args.seed)
        else:
            # Group by subject ID (assuming sXXXX_ naming convention)
            subject_map: dict[str, list[Path]] = {}
            for p in raw_files:
                subj = p.stem.split("_")[0]
                subject_map.setdefault(subj, []).append(p)

            subjects = sorted(subject_map.keys())
            random.seed(args.seed)
            random.shuffle(subjects)
            n_train = max(int(len(subjects) * 0.70), 1)
            n_val = max(int(len(subjects) * 0.15), 1)
            splits = {
                "train": subjects[:n_train],
                "val": subjects[n_train : n_train + n_val],
                "test": subjects[n_train + n_val :],
            }

    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    logger.info("Saved subject-wise splits to %s", splits_file)


if __name__ == "__main__":
    main()
