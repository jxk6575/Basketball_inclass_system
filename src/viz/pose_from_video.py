"""Extract cam_03 pose and map to court-preview coordinates for 4D viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.action.halpe2h36m import wholebody133_to_h36m
from src.perception.rtmlib_backend import create_rtmlib_perception
from src.pose.reference_template import kpts133_to_pseudo3d


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def _pick_person(
    dets: list[dict],
    prev_bbox: list[float] | None,
) -> dict | None:
    if not dets:
        return None
    if prev_bbox is None:
        # largest box
        return max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
    scored = []
    for d in dets:
        iou = _bbox_iou(prev_bbox, d["bbox"])
        area = (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
        scored.append((iou * 3.0 + min(area / 1e5, 1.0), d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None


def h36m_image_to_three(
    h36m_xyz: np.ndarray,
    img_w: float,
    img_h: float,
    *,
    body_height_m: float = 1.75,
) -> np.ndarray:
    """
    Map single-view H36M (image xy + pseudo z) → three.js (X right, Y up, Z depth).

    Root (pelvis) image position drives court travel; limb offsets scaled to ~meters.
    NOT metric-accurate — preview only until multi-view triangulation.
    """
    j = np.asarray(h36m_xyz, dtype=np.float64)[:, :3].copy()
    root = j[0].copy()
    rel = j - root
    torso = float(np.linalg.norm(rel[9]))
    if torso < 1e-4:
        torso = 1.0
    # torso ≈ 0.30 of standing height
    scale = (body_height_m * 0.30) / torso
    rel_m = rel * scale

    # Image → court preview: x lateral, y (down) → depth along court
    court_x = (float(root[0]) / max(img_w, 1.0) - 0.5) * 12.0
    court_z = 1.2 + (float(root[1]) / max(img_h, 1.0)) * 12.0

    out = np.zeros((17, 3), dtype=np.float64)
    # image y down → up is -y; pseudo-z discarded for vertical (use image vertical)
    out[:, 0] = rel_m[:, 0] + court_x
    out[:, 1] = -rel_m[:, 1]  # up
    # lift so ankles near floor
    ankle_y = float(np.nanmean([out[3, 1], out[6, 1]]))
    out[:, 1] -= ankle_y
    out[:, 2] = rel_m[:, 2] * 0.15 + court_z
    return out


def extract_pose_sequence_from_video(
    video_path: Path,
    stride: int = 2,
    max_frames: int = 1200,
    score_thr: float = 0.35,
) -> dict[str, Any]:
    """
    Run detector+pose on video; return three.js joint sequence with root motion.
    """
    backend = create_rtmlib_perception()
    if backend is None:
        raise RuntimeError("RTMLib perception not available (models/cuda)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    img_w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0)
    img_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0)

    frames: list[dict[str, Any]] = []
    prev_bbox: list[float] | None = None
    idx = 0
    kept = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and (idx % stride) != 0:
            idx += 1
            continue

        dets = backend.detect_persons(frame, score_thr=score_thr)
        person = _pick_person(dets, prev_bbox)
        t_ms = idx * 1000.0 / fps

        if person is None:
            idx += 1
            if kept >= max_frames:
                break
            continue

        prev_bbox = person["bbox"]
        kpts, _scores = backend.estimate_pose133(frame, person["bbox"])
        k3 = kpts133_to_pseudo3d(np.asarray(kpts, dtype=np.float32))
        h36m = wholebody133_to_h36m(k3)
        three = h36m_image_to_three(h36m, img_w, img_h)
        conf = [float(k3[i, 2]) if i < len(k3) else 0.0 for i in range(17)]

        frames.append({
            "frame": idx,
            "t_ms": round(t_ms, 1),
            "joints": three,
            "conf": conf,
        })
        kept += 1
        idx += 1
        if kept >= max_frames:
            break

    cap.release()
    return {
        "fps": fps,
        "img_w": img_w,
        "img_h": img_h,
        "stride": stride,
        "video": str(video_path),
        "frames_raw": frames,
    }
