"""Build and persist 3D reference pose templates (H36M-17, phase-keyed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.action.halpe2h36m import wholebody133_to_h36m
from src.config import data_path
from src.pose.angles import compute_frame_angles
from src.types import ActionClip

H36M_JOINT_NAMES = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]


def kpts133_to_pseudo3d(kpts: np.ndarray) -> np.ndarray:
    """Single-camera pseudo depth lift for template building."""
    k3 = kpts.copy()
    if k3.shape[1] == 2:
        conf = np.ones((k3.shape[0], 1), dtype=np.float32)
        k3 = np.hstack([k3, conf])
    h = float(np.nanmax(k3[:, 1]) - np.nanmin(k3[:, 1]) + 1e-6)
    k3[:, 2] = (1.0 - (k3[:, 1] - np.nanmin(k3[:, 1])) / h) * 1.5 + 0.5
    return k3


def normalize_h36m(joints: np.ndarray, *, keep_root: bool = False) -> np.ndarray:
    """Torso-length scaled coordinates.

    By default pelvis-centered (templates / angle compare).
    With keep_root=True, restore approximate image-plane translation onto
    all joints so the body can move in court-preview viewers.
    """
    out = joints.astype(np.float64).copy()
    root = out[0].copy()
    out -= root
    torso = np.linalg.norm(out[9] - out[0])
    scale = torso if torso > 1e-4 else 1.0
    out /= scale
    if keep_root:
        # Image pixels → court-preview meters (viewer remaps lightly).
        # Lateral from image-x; depth proxy from image-y.
        tx = (float(root[0]) / 1920.0 - 0.5) * 12.0
        tz = 1.2 + (float(root[1]) / 1080.0) * 12.0
        out[:, 0] += tx
        out[:, 2] += tz
    return out


def _phase_mid_frame(clip: ActionClip, phase_name: str) -> int | None:
    for ph in clip.phases:
        if ph.name == phase_name:
            return (ph.start + ph.end) // 2
    return None


def _frame_kpts(seq: list[tuple[int, np.ndarray]], frame: int) -> np.ndarray | None:
    for fidx, k in seq:
        if fidx == frame:
            return k
    return None


def build_free_throw_template(
    video_path: Path,
    seq: list[tuple[int, np.ndarray]],
    clip: ActionClip,
    template_id: str = "curry_free_throw",
    display_name: str = "库里罚篮参考",
) -> dict:
    release_frame = next((p.start for p in clip.phases if p.name == "release"), None)
    phases_out: dict[str, dict] = {}

    for phase_name in ("load", "set", "release", "follow_through"):
        mid = _phase_mid_frame(clip, phase_name)
        if mid is None:
            continue
        kpts = _frame_kpts(seq, mid)
        if kpts is None:
            continue
        k3 = kpts133_to_pseudo3d(kpts)
        angles = compute_frame_angles(k3)
        h36m = wholebody133_to_h36m(kpts)
        h36m[:, 2] = k3[:17, 2]
        norm = normalize_h36m(h36m)

        phases_out[phase_name] = {
            "frame": mid,
            "joints_3d": norm.round(6).tolist(),
            "angles": {k: round(v, 2) for k, v in angles.items() if v == v},
        }

    return {
        "id": template_id,
        "display_name": display_name,
        "action_type": "free_throw",
        "subject": "Stephen Curry",
        "source_video": str(video_path),
        "release_frame": release_frame,
        "joint_format": "h36m_17",
        "joint_names": H36M_JOINT_NAMES,
        "coordinate_system": "pseudo3d_normalized",
        "normalization": {
            "root_joint": "pelvis",
            "scale": "neck_to_pelvis_distance",
        },
        "phases": phases_out,
    }


def save_template(template: dict, path: Path | None = None) -> Path:
    path = path or data_path("templates", f"{template['id']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_reference_template(template_id: str) -> dict:
    path = data_path("templates", f"{template_id}.json")
    if not path.exists():
        raise FileNotFoundError(f"Reference template not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
