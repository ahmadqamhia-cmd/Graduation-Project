"""
Extract face crops from raw Celeb-DF v2 videos.

For each video:
  - Extract every Nth frame (frame_stride), up to max_frames
  - Detect largest face with MTCNN; fall back to centre-crop if none found
  - Resize/pad to 224x224 and save as JPEG

Outputs three directories mirroring the video structure:
  <crops_real>/  (Celeb-real + YouTube-real videos)
  <crops_fake>/  (Celeb-synthesis videos)

Each video gets its own subdirectory named after the video stem.

Usage:
    python -m data.extract_crops --config configs/default.yaml
    python -m data.extract_crops --config configs/default.yaml --workers 4
"""

import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Lazily-initialised MTCNN (per-thread, CPU)
_mtcnn = None


def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
            import torch
            _mtcnn = MTCNN(
                image_size=224,
                margin=20,
                min_face_size=40,
                device=torch.device("cpu"),
                post_process=False,
                keep_all=False,
                select_largest=True,
            )
        except ImportError:
            _mtcnn = "haar"
            print("facenet-pytorch not available, falling back to Haar cascade.")
    return _mtcnn


def _detect_and_crop(frame_bgr, size=224):
    """
    Detect largest face and return a (size x size) PIL Image.
    Falls back to centre-crop if no face found.
    """
    detector = _get_mtcnn()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    if detector != "haar":
        try:
            import torch
            with torch.no_grad():
                crop = detector(pil_img)
            if crop is not None:
                # crop is a float tensor in [0,255], shape (3,H,W)
                arr = crop.permute(1, 2, 0).clamp(0, 255).byte().numpy()
                return Image.fromarray(arr).resize((size, size), Image.BILINEAR)
        except Exception:
            pass

    # Haar cascade fallback
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) > 0:
        # Largest face
        areas = [w * h for (x, y, w, h) in faces]
        x, y, w, h = faces[np.argmax(areas)]
        scale = 1.3
        cx, cy = x + w // 2, y + h // 2
        half = int(max(w, h) * scale / 2)
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(frame_bgr.shape[1], cx + half)
        y2 = min(frame_bgr.shape[0], cy + half)
        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size > 0:
            pil_crop = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            return pil_crop.resize((size, size), Image.BILINEAR)

    # Centre crop fallback
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    centre = frame_bgr[y0:y0 + side, x0:x0 + side]
    pil_centre = Image.fromarray(cv2.cvtColor(centre, cv2.COLOR_BGR2RGB))
    return pil_centre.resize((size, size), Image.BILINEAR)


def extract_video(video_path, out_dir, frame_stride=10, max_frames=300,
                  size=224, jpeg_quality=95):
    """Extract and save face crops from a single video. Returns number of frames saved."""
    os.makedirs(out_dir, exist_ok=True)

    # Skip if already extracted (idempotent)
    existing = len([f for f in os.listdir(out_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if existing > 0:
        return existing

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    frame_idx = 0
    saved = 0
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            try:
                crop = _detect_and_crop(frame, size=size)
                out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
                crop.save(out_path, "JPEG", quality=jpeg_quality)
                saved += 1
            except Exception:
                pass
        frame_idx += 1

    cap.release()
    return saved


def _process_one(args):
    video_path, out_dir, stride, max_f, size, quality = args
    n = extract_video(video_path, out_dir, stride, max_f, size, quality)
    return video_path, n


def extract_all_crops(cfg, workers=1):
    archive_real_dirs = [
        ("Celeb-real",    cfg["paths"]["celebdf_crops_real"]),
        ("YouTube-real",  cfg["paths"]["celebdf_crops_real"]),
    ]
    archive_fake_dirs = [
        ("Celeb-synthesis", cfg["paths"]["celebdf_crops_fake"]),
    ]

    archive_base = cfg["paths"].get("celebdf_archive", "C:/Users/MYPC/Desktop/archive")
    stride   = cfg["data"]["frame_stride"]
    max_f    = cfg["data"]["max_frames_per_video"]
    size     = cfg["data"]["image_size"]
    quality  = cfg["fsbi"]["jpeg_quality"]

    tasks = []
    for subdir, crops_dir in archive_real_dirs + archive_fake_dirs:
        src_dir = os.path.join(archive_base, subdir)
        if not os.path.isdir(src_dir):
            print(f"  Skipping missing directory: {src_dir}")
            continue
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith(".mp4"):
                continue
            video_path = os.path.join(src_dir, fname)
            stem = os.path.splitext(fname)[0]
            out_dir = os.path.join(crops_dir, stem)
            tasks.append((video_path, out_dir, stride, max_f, size, quality))

    print(f"Total videos to process: {len(tasks)}")
    total_frames = 0

    if workers <= 1:
        for args in tqdm(tasks, desc="Extracting crops"):
            _, n = _process_one(args)
            total_frames += n
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_one, a): a for a in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="Extracting crops"):
                _, n = fut.result()
                total_frames += n

    print(f"Done. Total frames saved: {total_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    extract_all_crops(cfg, workers=args.workers)
