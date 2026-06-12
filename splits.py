"""
Build and save train/val/test split at video level from Celeb-DF v2.

The test list format is:
    <label> <subfolder>/<video_stem>.mp4
e.g. "1 YouTube-real/00170.mp4" or "0 Celeb-synthesis/id1_id0_0007.mp4"

All videos from Celeb-real and YouTube-real are treated as real (label=0).
All videos from Celeb-synthesis are treated as fake (label=1).

The crops dirs contain one subdirectory per video (named after the video stem).

Usage:
    python -m data.splits --config configs/default.yaml
"""

import argparse
import os
import random

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from utils.io import save_json


def _parse_test_list(test_list_path):
    """Return set of video stem names listed in List_of_testing_videos.txt."""
    test_videos = set()
    with open(test_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "1 YouTube-real/00170.mp4"  or  "0 Celeb-synthesis/id1_id0_0007.mp4"
            parts = line.split()
            video_path = parts[-1]
            stem = os.path.splitext(os.path.basename(video_path))[0]
            test_videos.add(stem)
    return test_videos


def _collect_video_stems(crops_dir):
    """
    Returns sorted list of video stems present as subdirectories in crops_dir.
    Each subdirectory name is the video stem (e.g. "id0_0000").
    """
    if not os.path.isdir(crops_dir):
        return []
    return sorted(
        entry.name
        for entry in os.scandir(crops_dir)
        if entry.is_dir()
    )


def build_splits(cfg):
    seed = cfg["data"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)

    real_crops = cfg["paths"]["celebdf_crops_real"]
    fake_crops = cfg["paths"]["celebdf_crops_fake"]
    test_list_path = cfg["paths"]["celebdf_test_list"]
    splits_json_path = cfg["paths"]["splits_json"]
    train_ratio = cfg["data"]["train_val_ratio"]

    test_video_stems = _parse_test_list(test_list_path)

    real_stems = _collect_video_stems(real_crops)
    fake_stems = _collect_video_stems(fake_crops)

    all_videos = (
        [(s, 0) for s in real_stems] +
        [(s, 1) for s in fake_stems]
    )

    test_set   = [(s, l) for s, l in all_videos if s in test_video_stems]
    remaining  = [(s, l) for s, l in all_videos if s not in test_video_stems]

    rem_ids    = [s for s, _ in remaining]
    rem_labels = [l for _, l in remaining]

    train_ids, val_ids, train_labels, val_labels = train_test_split(
        rem_ids, rem_labels,
        train_size=train_ratio,
        stratify=rem_labels,
        random_state=seed,
    )

    splits = {
        "train": [{"video_id": v, "label": l} for v, l in zip(train_ids, train_labels)],
        "val":   [{"video_id": v, "label": l} for v, l in zip(val_ids,   val_labels)],
        "test":  [{"video_id": v, "label": l} for v, l in test_set],
    }

    os.makedirs(os.path.dirname(os.path.abspath(splits_json_path)), exist_ok=True)
    save_json(splits, splits_json_path)

    print(f"Train videos : {len(splits['train'])}")
    print(f"Val   videos : {len(splits['val'])}")
    print(f"Test  videos : {len(splits['test'])}")
    print(f"Splits saved to {splits_json_path}")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_splits(cfg)
