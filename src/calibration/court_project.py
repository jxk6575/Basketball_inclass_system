"""Court-plane projection helpers for cross-camera spatial priors."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# COCO-17 / WholeBody body ankles
L_ANKLE, R_ANKLE = 15, 16


def foot_pixel_from_person(
    bbox: list[float] | tuple[float, ...] | None,
    keypoints: np.ndarray | list | None = None,
    conf_thr: float = 0.25,
) -> tuple[float, float] | None:
    """
    Image foot contact estimate: mean of visible ankles, else bbox bottom-center.
    """
    if keypoints is not None:
        k = np.asarray(keypoints, dtype=np.float64)
        if k.ndim == 2 and k.shape[0] > R_ANKLE:
            pts = []
            for i in (L_ANKLE, R_ANKLE):
                c = float(k[i, 2]) if k.shape[1] >= 3 else 1.0
                if c >= conf_thr:
                    pts.append(k[i, :2])
            if pts:
                m = np.mean(pts, axis=0)
                return float(m[0]), float(m[1])
    if bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if x2 < x1:
        x1, y1, bw, bh = x1, y1, x2, y2
        x2, y2 = x1 + bw, y1 + bh
    return 0.5 * (x1 + x2), float(y2)


def pixel_to_court_xy(
    u: float,
    v: float,
    cam: dict[str, Any],
    *,
    z_plane: float = 0.0,
) -> np.ndarray | None:
    """
    Back-project pixel (u,v) through camera to the horizontal plane z = z_plane.

    ``cam`` needs keys K (3x3), D (dist), R (3x3), t (3,1) from court PnP.
    Returns (2,) court XY in metres, or None if ray is parallel / behind.
    """
    K = np.asarray(cam["K"], dtype=np.float64)
    D = np.asarray(cam.get("D") if cam.get("D") is not None else [0, 0, 0, 0, 0], dtype=np.float64)
    R = np.asarray(cam["R"], dtype=np.float64)
    t = np.asarray(cam["t"], dtype=np.float64).reshape(3, 1)

    uv = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.undistortPoints(uv, K, D)  # normalized camera coords
    x_n, y_n = float(und[0, 0, 0]), float(und[0, 0, 1])
    # Ray in camera frame: origin 0, direction d_c
    d_c = np.array([x_n, y_n, 1.0], dtype=np.float64)
    d_c = d_c / (np.linalg.norm(d_c) + 1e-12)
    # Camera center in world: C = -R^T t
    C = (-R.T @ t).reshape(3)
    # Direction in world: d_w = R^T d_c
    d_w = (R.T @ d_c.reshape(3, 1)).reshape(3)
    if abs(float(d_w[2])) < 1e-9:
        return None
    # C + λ d_w intersects z = z_plane
    lam = (z_plane - float(C[2])) / float(d_w[2])
    if lam <= 0:
        return None
    X = C + lam * d_w
    return np.array([float(X[0]), float(X[1])], dtype=np.float64)


def person_to_court_xy(
    cam: dict[str, Any],
    bbox: list[float] | None,
    keypoints: np.ndarray | list | None = None,
    *,
    conf_thr: float = 0.25,
) -> np.ndarray | None:
    """Foot pixel → court XY for one person observation."""
    foot = foot_pixel_from_person(bbox, keypoints, conf_thr=conf_thr)
    if foot is None:
        return None
    return pixel_to_court_xy(foot[0], foot[1], cam)
