"""Multi-view triangulation using court-landmark calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import ROOT


def load_camera_calibration(calib_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load per-camera K, R, t from data/calibration/v2_4cam_zoned/."""
    calib_dir = calib_dir or (ROOT / "data/calibration/v2_4cam_zoned")
    cams: dict[str, dict[str, Any]] = {}
    if not calib_dir.exists():
        return cams
    bundle = calib_dir / "cameras.json"
    if bundle.exists():
        doc = json.loads(bundle.read_text(encoding="utf-8"))
        for cam_id, res in (doc.get("solved") or {}).get("cameras", {}).items():
            if res.get("status") != "ok":
                continue
            cams[cam_id] = {
                "K": np.asarray(res["intrinsics"]["camera_matrix"], dtype=np.float64),
                "D": np.asarray(res["intrinsics"].get("dist_coeffs") or [0, 0, 0, 0, 0], dtype=np.float64),
                "R": np.asarray(res["rotation_matrix"], dtype=np.float64),
                "t": np.asarray(res["tvec"], dtype=np.float64).reshape(3, 1),
                "reproj_mean": (res.get("reproj_error_px") or {}).get("mean"),
            }
        return cams
    for path in sorted(calib_dir.glob("cam_*.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        cams[path.stem] = {
            "K": np.asarray(res["intrinsics"]["camera_matrix"], dtype=np.float64),
            "D": np.asarray(res["intrinsics"].get("dist_coeffs") or [0, 0, 0, 0, 0], dtype=np.float64),
            "R": np.asarray(res["rotation_matrix"], dtype=np.float64),
            "t": np.asarray(res["tvec"], dtype=np.float64).reshape(3, 1),
        }
    return cams


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def triangulate_joint(
    observations: dict[str, tuple[float, float]],
    cameras: dict[str, dict[str, Any]],
    min_views: int = 2,
) -> np.ndarray | None:
    """
    DLT triangulate one joint from {cam_id: (u,v)}.
    Returns (3,) world XYZ or None.
    """
    usable = [(cid, uv) for cid, uv in observations.items() if cid in cameras]
    if len(usable) < min_views:
        return None
    proj_mats = []
    pts = []
    for cid, (u, v) in usable:
        cam = cameras[cid]
        # undistort
        uv = np.array([[[u, v]]], dtype=np.float64)
        und = cv2.undistortPoints(uv, cam["K"], cam["D"], P=cam["K"])
        uu, vv = float(und[0, 0, 0]), float(und[0, 0, 1])
        proj_mats.append(projection_matrix(cam["K"], cam["R"], cam["t"]))
        pts.append([uu, vv])
    if len(proj_mats) == 2:
        X = cv2.triangulatePoints(proj_mats[0], proj_mats[1],
                                  np.array(pts[0], dtype=np.float64).reshape(2, 1),
                                  np.array(pts[1], dtype=np.float64).reshape(2, 1))
        X = (X[:3] / X[3]).reshape(3)
        return X
    # Multi-view DLT
    A = []
    for P, (u, v) in zip(proj_mats, pts):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.stack(A, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]
    return X


def triangulate_skeleton17(
    kpts_by_cam: dict[str, np.ndarray],
    cameras: dict[str, dict[str, Any]],
    conf_thr: float = 0.3,
    min_views: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    kpts_by_cam: cam_id -> (17, 2|3) with optional conf in col 2.
    Returns xyz (17,3) and conf (17,) — conf = fraction of views used.
    """
    xyz = np.full((17, 3), np.nan, dtype=np.float64)
    conf = np.zeros(17, dtype=np.float64)
    for j in range(17):
        obs = {}
        for cid, k in kpts_by_cam.items():
            if k is None or len(k) <= j:
                continue
            c = float(k[j, 2]) if k.shape[1] > 2 else 1.0
            if c < conf_thr:
                continue
            obs[cid] = (float(k[j, 0]), float(k[j, 1]))
        X = triangulate_joint(obs, cameras, min_views=min_views)
        if X is not None:
            xyz[j] = X
            conf[j] = len(obs) / max(1, len(cameras))
    return xyz, conf
