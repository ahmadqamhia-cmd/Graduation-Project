"""
Load image paths, labels, and video_ids for each split.
"""

import os

from utils.io import load_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _frames_for_video(crops_dir, video_id):
    video_dir = os.path.join(crops_dir, video_id)
    if not os.path.isdir(video_dir):
        return []
    frames = sorted(
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    return frames


def load_split(cfg, split_name, include_fsbi=False):
    """
    Returns (image_paths, labels, video_ids) for the requested split.

    split_name: 'train', 'val', or 'test'
    include_fsbi: if True and split_name == 'train', append FSBI pseudo-fake frames
    """
    splits_json = cfg["paths"]["splits_json"]
    real_dir = cfg["paths"]["celebdf_crops_real"]
    fake_dir = cfg["paths"]["celebdf_crops_fake"]
    fsbi_dir = cfg["paths"]["celebdf_crops_fsbi"]

    splits = load_json(splits_json)
    entries = splits[split_name]

    image_paths = []
    labels = []
    video_ids = []

    for entry in entries:
        vid = entry["video_id"]
        lbl = entry["label"]
        crops_dir = real_dir if lbl == 0 else fake_dir
        frames = _frames_for_video(crops_dir, vid)
        image_paths.extend(frames)
        labels.extend([lbl] * len(frames))
        video_ids.extend([vid] * len(frames))

    if include_fsbi and split_name == "train":
        # FSBI frames are stored flat or in subdirectories under fsbi_dir
        if os.path.isdir(fsbi_dir):
            fsbi_frames = []
            fsbi_vids = []
            for entry in os.scandir(fsbi_dir):
                if entry.is_dir():
                    frames = _frames_for_video(fsbi_dir, entry.name)
                    fsbi_frames.extend(frames)
                    fsbi_vids.extend([f"fsbi_{entry.name}"] * len(frames))
                elif os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS:
                    fsbi_frames.append(entry.path)
                    fsbi_vids.append("fsbi_flat")
            image_paths.extend(fsbi_frames)
            labels.extend([1] * len(fsbi_frames))
            video_ids.extend(fsbi_vids)

    return image_paths, labels, video_ids
