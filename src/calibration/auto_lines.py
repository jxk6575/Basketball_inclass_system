"""Court line detection and intersection candidates for auto / semi-auto labeling."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LineSeg:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)


def _line_intersection(a: LineSeg, b: LineSeg) -> tuple[float, float] | None:
    """Infinite-line intersection; None if parallel or far from both segments."""
    x1, y1, x2, y2 = a.x1, a.y1, a.x2, a.y2
    x3, y3, x4, y4 = b.x1, b.y1, b.x2, b.y2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den

    def near_seg(x0, y0, xa, ya, xb, yb, pad=80.0) -> bool:
        xmin, xmax = min(xa, xb) - pad, max(xa, xb) + pad
        ymin, ymax = min(ya, yb) - pad, max(ya, yb) + pad
        return xmin <= x0 <= xmax and ymin <= y0 <= ymax

    if not near_seg(px, py, x1, y1, x2, y2):
        return None
    if not near_seg(px, py, x3, y3, x4, y4):
        return None
    return float(px), float(py)


def detect_court_lines(image: np.ndarray, min_length: float = 60.0) -> list[LineSeg]:
    """Detect long bright/white-ish line segments (court markings)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Court paint is bright; adaptive + Canny helps under gym lighting
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=int(min_length), maxLineGap=20,
    )
    segs: list[LineSeg] = []
    if lines is None:
        return segs
    for row in lines[:, 0]:
        x1, y1, x2, y2 = map(float, row)
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length < min_length:
            continue
        # Prefer roughly horizontal / vertical / 45° court lines (not random clutter)
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        if 20 < ang < 70 or 110 < ang < 160:
            # diagonal ok (sideline perspective)
            pass
        segs.append(LineSeg(x1, y1, x2, y2))
    # Deduplicate similar segments
    return _nms_lines(segs, dist_thresh=25.0)


def _nms_lines(segs: list[LineSeg], dist_thresh: float = 25.0) -> list[LineSeg]:
    kept: list[LineSeg] = []
    for s in sorted(segs, key=lambda t: -((t.x2 - t.x1) ** 2 + (t.y2 - t.y1) ** 2)):
        mid = np.array([(s.x1 + s.x2) / 2, (s.y1 + s.y2) / 2])
        dup = False
        for k in kept:
            kmid = np.array([(k.x1 + k.x2) / 2, (k.y1 + k.y2) / 2])
            if np.linalg.norm(mid - kmid) < dist_thresh:
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept[:80]


def intersection_candidates(
    segs: list[LineSeg],
    image_shape: tuple[int, int],
    max_points: int = 40,
) -> list[tuple[float, float]]:
    """Pairwise line intersections inside the image."""
    h, w = image_shape[:2]
    pts: list[tuple[float, float]] = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            hit = _line_intersection(segs[i], segs[j])
            if hit is None:
                continue
            x, y = hit
            if 0 <= x < w and 0 <= y < h:
                pts.append((x, y))
    # Cluster nearby intersections
    return _cluster_points(pts, radius=15.0)[:max_points]


def _cluster_points(
    pts: list[tuple[float, float]],
    radius: float = 15.0,
) -> list[tuple[float, float]]:
    if not pts:
        return []
    arr = np.asarray(pts, dtype=np.float64)
    used = np.zeros(len(arr), dtype=bool)
    out: list[tuple[float, float]] = []
    for i in range(len(arr)):
        if used[i]:
            continue
        d = np.linalg.norm(arr - arr[i], axis=1)
        idx = np.where(d <= radius)[0]
        used[idx] = True
        mean = arr[idx].mean(axis=0)
        out.append((float(mean[0]), float(mean[1])))
    return out


def snap_to_candidates(
    xy: tuple[float, float],
    candidates: list[tuple[float, float]],
    max_dist: float = 40.0,
) -> tuple[float, float] | None:
    if not candidates:
        return None
    p = np.asarray(xy, dtype=np.float64)
    best = None
    best_d = max_dist
    for c in candidates:
        d = float(np.linalg.norm(p - np.asarray(c)))
        if d < best_d:
            best_d = d
            best = c
    return best


def project_and_snap(
    world_xyz: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist: np.ndarray,
    candidates: list[tuple[float, float]],
    max_dist: float = 50.0,
) -> tuple[float, float] | None:
    """Project a world point and snap to nearest line-intersection candidate."""
    imgpts, _ = cv2.projectPoints(
        world_xyz.reshape(1, 1, 3), rvec, tvec, camera_matrix, dist,
    )
    u, v = float(imgpts[0, 0, 0]), float(imgpts[0, 0, 1])
    return snap_to_candidates((u, v), candidates, max_dist=max_dist)
