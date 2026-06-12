"""
Extract face crops from FF++ c23 videos.

Expected structure (flat .mp4 files):
    ffpp_root/
    ├── real/
    │   ├── 000.mp4, 001.mp4, ... (1000 videos)
    └── fake/
        ├── Deepfakes/    *.mp4
        ├── Face2Face/    *.mp4
        ├── FaceSwap/     *.mp4
        └── NeuralTextures/ *.mp4

Outputs (one subdir per video stem):
    ffpp_crops_real/<stem>/frame_000000.jpg ...
    ffpp_crops_fake/<method>_<stem>/frame_000000.jpg ...

Usage:
    python -m data.extract_crops_ffpp --config configs/default.yaml
"""

import argparse
import os

import yaml
from tqdm import tqdm

from data.extract_crops import extract_video
from utils.io import ensure_dir

# Only the 4 standard methods used by all papers
FFPP_MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def extract_ffpp_crops(cfg):
    ffpp_root  = cfg["paths"]["ffpp_root"]
    real_crops = cfg["paths"]["ffpp_crops_real"]
    fake_crops = cfg["paths"]["ffpp_crops_fake"]
    stride     = cfg["data"]["frame_stride"]
    max_frames = cfg["data"]["max_frames_per_video"]
    size       = cfg["data"]["image_size"]
    quality    = cfg["fsbi"]["jpeg_quality"]

    ensure_dir(real_crops)
    ensure_dir(fake_crops)

    tasks = []

    # Real videos — flat .mp4 files in real/
    real_vid_dir = os.path.join(ffpp_root, "real")
    if os.path.isdir(real_vid_dir):
        for fname in sorted(os.listdir(real_vid_dir)):
            if fname.lower().endswith(".mp4"):
                stem = os.path.splitext(fname)[0]
                tasks.append((
                    os.path.join(real_vid_dir, fname),
                    os.path.join(real_crops, stem),
                ))
    else:
        print(f"WARNING: real dir not found: {real_vid_dir}")

    # Fake videos — flat .mp4 files in fake/<Method>/
    for method in FFPP_MANIPULATIONS:
        fake_vid_dir = os.path.join(ffpp_root, "fake", method)
        if not os.path.isdir(fake_vid_dir):
            print(f"  Skipping missing: {fake_vid_dir}")
            continue
        for fname in sorted(os.listdir(fake_vid_dir)):
            if fname.lower().endswith(".mp4"):
                stem = os.path.splitext(fname)[0]
                tasks.append((
                    os.path.join(fake_vid_dir, fname),
                    os.path.join(fake_crops, f"{method}_{stem}"),
                ))

    real_count = sum(1 for _, out in tasks if "crops_ffpp/real" in out.replace("\\", "/"))
    fake_count = len(tasks) - real_count
    print(f"FF++ videos to process: {len(tasks)}  (real={real_count}, fake={fake_count})")

    for video_path, out_dir in tqdm(tasks, desc="FF++ crop extraction"):
        extract_video(video_path, out_dir, stride, max_frames, size, quality)

    print(f"Done. Crops saved to:")
    print(f"  Real: {real_crops}")
    print(f"  Fake: {fake_crops}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    extract_ffpp_crops(cfg)
