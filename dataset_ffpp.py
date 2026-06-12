"""
FF++ dataset loader for cross-dataset evaluation protocol.

FF++ directory structure after download:
    ffpp_root/
    ├── original_sequences/youtube/c23/videos/*.mp4       (real, 1000 videos)
    └── manipulated_sequences/
        ├── Deepfakes/c23/videos/*.mp4
        ├── Face2Face/c23/videos/*.mp4
        ├── FaceSwap/c23/videos/*.mp4
        └── NeuralTextures/c23/videos/*.mp4

Official splits JSON:
    https://github.com/ondyari/FaceForensics/blob/master/dataset/splits/train.json
    (720 train / 140 val / 140 test video pairs)

Cross-dataset protocol:
    Train + Val: FF++ c23
    Test:        Celeb-DF v2 (using existing test split)
"""

import json
import os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

FFPP_MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _frames_for_video(crops_dir, video_id):
    video_dir = os.path.join(crops_dir, video_id)
    if not os.path.isdir(video_dir):
        return []
    return sorted(
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


def load_ffpp_splits(cfg):
    """
    Returns dict with keys 'train', 'val', each being a list of
    {'video_id': str, 'label': int (0=real, 1=fake)}.
    Reads from ffpp_splits_json if it exists, else builds from crops dirs.
    """
    splits_json = cfg["paths"].get("ffpp_splits_json",
                                   "C:/Users/MYPC/deepfake_detection/data/ffpp_splits.json")

    if os.path.exists(splits_json):
        with open(splits_json) as f:
            return json.load(f)

    # Build from crops directories
    real_crops = cfg["paths"]["ffpp_crops_real"]
    fake_crops = cfg["paths"]["ffpp_crops_fake"]

    real_vids = sorted(
        e.name for e in os.scandir(real_crops) if e.is_dir()
    ) if os.path.isdir(real_crops) else []
    fake_vids = sorted(
        e.name for e in os.scandir(fake_crops) if e.is_dir()
    ) if os.path.isdir(fake_crops) else []

    all_vids = [(v, 0) for v in real_vids] + [(v, 1) for v in fake_vids]

    # 80/20 split
    from sklearn.model_selection import train_test_split
    ids = [v for v, _ in all_vids]
    labels = [l for _, l in all_vids]
    train_ids, val_ids, train_lbls, val_lbls = train_test_split(
        ids, labels, train_size=0.8, stratify=labels, random_state=42
    )

    splits = {
        "train": [{"video_id": v, "label": l} for v, l in zip(train_ids, train_lbls)],
        "val":   [{"video_id": v, "label": l} for v, l in zip(val_ids,   val_lbls)],
    }

    os.makedirs(os.path.dirname(os.path.abspath(splits_json)), exist_ok=True)
    with open(splits_json, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"FF++ splits saved to {splits_json}")
    return splits


def load_ffpp_split(cfg, split_name):
    """
    Returns (image_paths, labels, video_ids) for FF++ train or val.
    """
    real_crops = cfg["paths"]["ffpp_crops_real"]
    fake_crops = cfg["paths"]["ffpp_crops_fake"]

    splits = load_ffpp_splits(cfg)
    entries = splits[split_name]

    image_paths, labels, video_ids = [], [], []
    for entry in entries:
        vid = entry["video_id"]
        lbl = entry["label"]
        crops_dir = real_crops if lbl == 0 else fake_crops
        frames = _frames_for_video(crops_dir, vid)
        image_paths.extend(frames)
        labels.extend([lbl] * len(frames))
        video_ids.extend([vid] * len(frames))

    return image_paths, labels, video_ids
