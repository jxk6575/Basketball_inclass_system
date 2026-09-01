"""FIBA court landmark model for multi-camera extrinsic calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import ROOT

DEFAULT_LANDMARKS = ROOT / "configs" / "calibration" / "court_landmarks_fiba.yaml"


def load_court_model(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_LANDMARKS
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def landmark_xyz(model: dict[str, Any], point_id: str) -> np.ndarray:
    lm = model["landmarks"][point_id]
    return np.asarray(lm["xyz"], dtype=np.float64)


def all_landmark_ids(model: dict[str, Any], groups: list[str] | None = None) -> list[str]:
    out = []
    for pid, lm in model["landmarks"].items():
        if groups and lm.get("group") not in groups:
            continue
        out.append(pid)
    return out


def priority_ids(model: dict[str, Any]) -> list[str]:
    return list(model.get("priority_for_body_triangulation") or all_landmark_ids(model))


def annotation_order_for_camera(model: dict[str, Any], camera_id: str) -> list[str]:
    """Per-camera landmark list for the annotation GUI (falls back to priority)."""
    by_cam = model.get("annotation_order_by_camera") or {}
    order = by_cam.get(camera_id)
    if order:
        return list(order)
    # Fallback: landmarks that list this camera as typical
    ids = []
    for pid, lm in model.get("landmarks", {}).items():
        cams = lm.get("typical_cameras") or []
        if camera_id in cams:
            ids.append(pid)
    return ids or priority_ids(model)


def landmark_name(model: dict[str, Any], point_id: str) -> str:
    return str((model.get("landmarks") or {}).get(point_id, {}).get("name") or point_id)


def world_points_matrix(
    model: dict[str, Any],
    point_ids: list[str],
) -> np.ndarray:
    """(N, 3) world coordinates."""
    return np.stack([landmark_xyz(model, p) for p in point_ids], axis=0)
