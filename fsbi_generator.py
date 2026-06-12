"""
Offline FSBI pseudo-fake generation from Celeb-DF v2 real training frames.

Process per image:
  1. Affine distortion
  2. Face mask via landmark detection
  3. Poisson blend
  4. DWT LL sub-band averaging (sym2 wavelet, reflect padding, single level)

Usage:
    python -m data.fsbi_generator --config configs/default.yaml
"""

import argparse
import os
import random

import cv2
import numpy as np
import pywt
import yaml
from tqdm import tqdm

from data.dataset import load_split
from utils.io import ensure_dir


def _affine_distort(image, scale_range, rotation_range, seed_val):
    rng = np.random.RandomState(seed_val)
    h, w = image.shape[:2]
    scale = rng.uniform(*scale_range)
    angle = rng.uniform(*rotation_range)
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    distorted = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
    return distorted


def _simple_face_mask(image):
    """
    Elliptical mask approximating face region when landmark detection is unavailable.
    Falls back gracefully if RetinaFace is not installed.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    axes = (int(w * 0.42), int(h * 0.48))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask


def _face_mask_with_landmarks(image):
    try:
        from retinaface import RetinaFace
        faces = RetinaFace.detect_faces(image)
        if not faces or not isinstance(faces, dict):
            return _simple_face_mask(image)
        face_key = list(faces.keys())[0]
        landmarks = faces[face_key].get("landmarks", {})
        points = np.array(list(landmarks.values()), dtype=np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        return mask
    except Exception:
        return _simple_face_mask(image)


def _poisson_blend(source, target, mask):
    """Poisson blend source into target using the face mask."""
    h, w = target.shape[:2]
    center = (w // 2, h // 2)
    try:
        blended = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # Fallback: alpha blend
        alpha = (mask / 255.0)[..., np.newaxis]
        blended = (alpha * source + (1 - alpha) * target).astype(np.uint8)
    return blended


def _dwt_ll_averaging(image, wavelet="sym2", level=1, mode="reflect"):
    """
    Apply DWT LL sub-band averaging to each channel separately, then reconstruct.
    """
    result = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        channel = image[:, :, c].astype(np.float32)
        coeffs = pywt.dwt2(channel, wavelet=wavelet, mode=mode)
        LL, (LH, HL, HH) = coeffs
        LL_mean = np.full_like(LL, LL.mean())
        reconstructed = pywt.idwt2((LL_mean, (LH, HL, HH)), wavelet=wavelet, mode=mode)
        # Crop/pad to match original size
        rh, rw = reconstructed.shape
        oh, ow = channel.shape
        reconstructed = reconstructed[:oh, :ow]
        if reconstructed.shape != channel.shape:
            pad_h = oh - reconstructed.shape[0]
            pad_w = ow - reconstructed.shape[1]
            reconstructed = np.pad(reconstructed, ((0, pad_h), (0, pad_w)), mode="edge")
        result[:, :, c] = np.clip(reconstructed, 0, 255)
    return result.astype(np.uint8)


def generate_fsbi_images(cfg):
    seed = cfg["data"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)

    fsbi_dir = cfg["paths"]["celebdf_crops_fsbi"]
    scale_range = cfg["fsbi"]["affine_scale_range"]
    rotation_range = cfg["fsbi"]["affine_rotation_range"]
    jpeg_quality = cfg["fsbi"]["jpeg_quality"]
    wavelet = cfg["fsbi"]["wavelet"]
    level = cfg["fsbi"]["wavelet_level"]
    mode = cfg["fsbi"]["wavelet_mode"]

    ensure_dir(fsbi_dir)

    # Load only real training frames (label=0)
    image_paths, labels, video_ids = load_split(cfg, "train", include_fsbi=False)
    real_paths = [p for p, l in zip(image_paths, labels) if l == 0]

    print(f"Generating FSBI pseudo-fakes from {len(real_paths)} real training frames...")

    for idx, src_path in enumerate(tqdm(real_paths, desc="FSBI")):
        image = cv2.imread(src_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: Affine distortion
        distorted = _affine_distort(image, scale_range, rotation_range, seed_val=idx + seed)

        # Step 2: Face mask
        mask = _face_mask_with_landmarks(image)

        # Step 3: Poisson blend distorted into original
        blended = _poisson_blend(distorted, image, mask)

        # Step 4: DWT LL sub-band averaging
        result = _dwt_ll_averaging(blended, wavelet=wavelet, level=level, mode=mode)

        # Save
        video_subdir = os.path.join(fsbi_dir, os.path.basename(os.path.dirname(src_path)))
        ensure_dir(video_subdir)
        out_name = os.path.splitext(os.path.basename(src_path))[0] + "_fsbi.jpg"
        out_path = os.path.join(video_subdir, out_name)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    print(f"FSBI images saved to {fsbi_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generate_fsbi_images(cfg)
