"""Shot geometry helpers — ported from ref_code/shot_utils.py."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# Track point: ((x, y), frame_idx, width, height, confidence)
TrackPoint = tuple[tuple[int, int], int, int, int, float]


def point_to_line_distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    target_point: tuple[float, float],
) -> float:
    line_vector = (point_b[0] - point_a[0], point_b[1] - point_a[1])
    target_vector = (target_point[0] - point_a[0], target_point[1] - point_a[1])
    line_length_sq = line_vector[0] ** 2 + line_vector[1] ** 2
    if line_length_sq < 1e-6:
        return math.sqrt(target_vector[0] ** 2 + target_vector[1] ** 2)

    t = max(0.0, min(1.0, (target_vector[0] * line_vector[0] + target_vector[1] * line_vector[1]) / line_length_sq))
    closest = (point_a[0] + t * line_vector[0], point_a[1] + t * line_vector[1])
    return math.sqrt((target_point[0] - closest[0]) ** 2 + (target_point[1] - closest[1]) ** 2)


def detect_down(ball_pos: list[TrackPoint], hoop_pos: list[TrackPoint]) -> bool:
    """Ball center below hoop lower boundary → shot attempt ending."""
    if not ball_pos or not hoop_pos:
        return False
    hoop_lower = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    return ball_pos[-1][0][1] > hoop_lower


def detect_up(ball_pos: list[TrackPoint], hoop_pos: list[TrackPoint]) -> bool:
    """Ball center above hoop lower boundary → potential shot start."""
    if not ball_pos or not hoop_pos:
        return True
    hoop_lower = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    return ball_pos[-1][0][1] < hoop_lower


def in_hoop_region(center: tuple[int, int], hoop_pos: list[TrackPoint]) -> bool:
    if not hoop_pos:
        return False
    x, y = center
    hx, hy = hoop_pos[-1][0]
    w, h = hoop_pos[-1][2], hoop_pos[-1][3]
    return (hx - w) < x < (hx + w) and (hy - h) < y < (hy + 0.5 * h)


def clean_ball_pos(ball_pos: list[TrackPoint], frame_count: int, max_age: int = 90) -> list[TrackPoint]:
    if len(ball_pos) > 1:
        w1, h1 = ball_pos[-2][2], ball_pos[-2][3]
        w2, h2 = ball_pos[-1][2], ball_pos[-1][3]
        x1, y1 = ball_pos[-2][0]
        x2, y2 = ball_pos[-1][0]
        f_dif = ball_pos[-1][1] - ball_pos[-2][1]
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_dist = 4 * math.sqrt(w1 ** 2 + h1 ** 2)
        if dist > max_dist and f_dif < 5:
            ball_pos.pop()
        elif (w2 * 1.4 < h2) or (h2 * 1.4 < w2):
            ball_pos.pop()

    if ball_pos and frame_count - ball_pos[0][1] > max_age:
        ball_pos.pop(0)
    return ball_pos


def clean_hoop_pos(hoop_pos: list[TrackPoint]) -> list[TrackPoint]:
    if len(hoop_pos) > 1:
        x1, y1 = hoop_pos[-2][0]
        x2, y2 = hoop_pos[-1][0]
        w1, h1 = hoop_pos[-2][2], hoop_pos[-2][3]
        w2, h2 = hoop_pos[-1][2], hoop_pos[-1][3]
        f_dif = hoop_pos[-1][1] - hoop_pos[-2][1]
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()
        if (w2 * 1.3 < h2) or (h2 * 1.3 < w2):
            hoop_pos.pop()
    if len(hoop_pos) > 25:
        hoop_pos.pop(0)
    return hoop_pos


def _rim_top_roi(
    hoop_pos: list[TrackPoint],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, w, h) for the strip covering the hoop rim's upper edge."""
    if not hoop_pos:
        return None
    hx, hy = int(hoop_pos[-1][0][0]), int(hoop_pos[-1][0][1])
    hoop_w, hoop_h = int(hoop_pos[-1][2]), int(hoop_pos[-1][3])
    hoop_x1 = hx - hoop_w // 2
    hoop_y1 = hy - hoop_h // 2
    trim = 0.12
    rim_x1 = hoop_x1 + int(hoop_w * trim)
    rim_w = int(hoop_w * (1 - 2 * trim))
    # Slightly above detected hoop top — bbox often sits a few px low vs paint
    rim_y1 = max(0, hoop_y1 - max(2, int(0.06 * hoop_h)))
    # Thin strip on the upper rim paint (avoid thick band diluting occlusion signal)
    rim_h = max(5, int(0.10 * hoop_h))
    if rim_w <= 0 or rim_h <= 0:
        return None
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    rim_x1 = max(0, min(rim_x1, fw - 2))
    rim_y1 = max(0, min(rim_y1, fh - 2))
    rim_w = max(1, min(rim_w, fw - rim_x1))
    rim_h = max(1, min(rim_h, fh - rim_y1))
    return rim_x1, rim_y1, rim_w, rim_h


def _orange_pixel_ratio(region_bgr: np.ndarray) -> float:
    """Fraction of pixels that look like rim paint (orange / red-orange)."""
    if region_bgr is None or region_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    # OpenCV H∈[0,180]: orange ≈ 5–25; gym lights often wrap rim to 160–180
    m_lo = cv2.inRange(hsv, np.array([0, 55, 60], np.uint8), np.array([30, 255, 255], np.uint8))
    m_hi = cv2.inRange(hsv, np.array([160, 55, 60], np.uint8), np.array([180, 255, 255], np.uint8))
    # BGR fallback: R dominates over B (washed-out orange)
    b, g, r = cv2.split(region_bgr)
    m_bgr = (
        (r.astype(np.int16) - b.astype(np.int16) > 35)
        & (r.astype(np.int16) - g.astype(np.int16) > 5)
        & (r > 90)
    ).astype(np.uint8) * 255
    return float(np.mean((m_lo > 0) | (m_hi > 0) | (m_bgr > 0)))


def _iter_shot_frames(
    shot_frames: list[np.ndarray] | dict[int, np.ndarray],
) -> list[tuple[int | None, np.ndarray]]:
    """Normalize list or {frame_idx: img} to [(idx_or_None, frame), …] in order."""
    if isinstance(shot_frames, dict):
        return [(int(k), shot_frames[k]) for k in sorted(shot_frames.keys())]
    return [(None, fr) for fr in shot_frames]


def check_rim_top_orange_occluded(
    shot_frames: list[np.ndarray] | dict[int, np.ndarray],
    hoop_pos: list[TrackPoint],
    ball_pos: list[TrackPoint] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Miss cue: while the ball descends near the rim, the **orange rim top**
    disappears (covered by the ball). A clean make leaves the top orange visible.

    No ball-color sampling — only whether rim orange is still present.
    """
    meta: dict[str, Any] = {"method": "rim_orange_visibility"}
    items = _iter_shot_frames(shot_frames)
    if len(items) < 3 or not hoop_pos:
        return False, {**meta, "reason": "too_few_frames"}

    roi = _rim_top_roi(hoop_pos, items[0][1].shape)
    if roi is None:
        return False, {**meta, "reason": "bad_rim_roi"}
    rx, ry, rw, rh = roi
    meta["rim_roi"] = [rx, ry, rw, rh]

    ball_by_f: dict[int, TrackPoint] = {}
    if ball_pos:
        for bp in ball_pos:
            ball_by_f[int(bp[1])] = bp

    ratios: list[float] = []
    near_rim_flags: list[bool] = []
    for fidx, frame in items:
        region = frame[ry: ry + rh, rx: rx + rw]
        ratios.append(_orange_pixel_ratio(region))
        near = True
        bp = None
        if fidx is not None and fidx in ball_by_f:
            bp = ball_by_f[fidx]
        elif ball_pos and fidx is None:
            # Sequential list (online tracker): approximate by list order later
            bp = None
        if bp is not None:
            (bx, by), _, _bw, bh, _ = bp
            # Ball still clearly above rim → not covering top edge yet
            if by + 0.5 * bh < ry - 4:
                near = False
            # Ball already well below rim plane → through-net, ignore
            elif by - 0.5 * bh > ry + rh + max(8, bh):
                near = False
        near_rim_flags.append(near)

    # Online path: list frames without frame ids — use temporal thirds + ball_pos order
    if ball_pos and all(f is None for f, _ in items) and len(ball_pos) == len(items):
        near_rim_flags = []
        for i, bp in enumerate(ball_pos):
            (bx, by), _, _bw, bh, _ = bp
            near = not (by + 0.5 * bh < ry - 4 or by - 0.5 * bh > ry + rh + max(8, bh))
            near_rim_flags.append(near)

    n = len(ratios)
    early = ratios[: max(2, n // 3)]
    baseline = float(np.median(early)) if early else 0.0
    meta["orange_baseline"] = round(baseline, 3)
    meta["orange_min"] = round(float(min(ratios)), 3)
    meta["orange_mean"] = round(float(np.mean(ratios)), 3)

    # Need a visible orange rim at start; otherwise lighting/cam angle is unreliable
    if baseline < 0.08:
        return False, {**meta, "reason": "no_orange_baseline"}

    # Sensitive to covering the top paint: make keeps ~baseline; miss drops it.
    drop_thr = max(0.06, 0.60 * baseline)
    abs_drop = 0.12
    occluded_idxs: list[int] = []
    for i, (r, near) in enumerate(zip(ratios, near_rim_flags)):
        if i < max(2, n // 3):
            continue  # never use early frames as occlusion evidence
        if not near:
            continue
        if r < drop_thr or (baseline - r) >= abs_drop:
            occluded_idxs.append(i)

    meta["orange_drop_thr"] = round(drop_thr, 3)
    meta["n_occluded_frames"] = len(occluded_idxs)
    # Require sustained occlusion (not a single noisy frame)
    occluded = len(occluded_idxs) >= 2
    meta["rim_occluded"] = occluded
    return occluded, meta


# Back-compat alias (old name used ball color; now orange-rim visibility)
def check_hoop_rim_occlusion(
    shot_frames: list[np.ndarray] | dict[int, np.ndarray],
    hoop_pos: list[TrackPoint],
    ball_color: np.ndarray | None = None,
    ball_pos: list[TrackPoint] | None = None,
) -> bool:
    occluded, _ = check_rim_top_orange_occluded(shot_frames, hoop_pos, ball_pos=ball_pos)
    return occluded


def rim_occlusion_indicates_miss(
    ball_pos: list[TrackPoint],
    hoop_pos: list[TrackPoint],
    shot_frames: list[np.ndarray] | dict[int, np.ndarray] | None,
) -> tuple[bool, dict]:
    """
    Ball moving down + orange rim top covered → hard miss.

    Makes leave the rim top orange visible; balls that sit on / clip the rim
    hide that orange strip. Does **not** sample ball color.
    """
    meta: dict[str, Any] = {"rim_occlusion_checked": False}
    if shot_frames is None:
        return False, meta
    n_frames = len(shot_frames) if not isinstance(shot_frames, dict) else len(shot_frames)
    if n_frames < 3 or len(ball_pos) < 2 or not hoop_pos:
        return False, meta

    moving_down = False
    for i in range(max(0, len(ball_pos) - 3), len(ball_pos)):
        if i > 0 and ball_pos[i][0][1] > ball_pos[i - 1][0][1]:
            moving_down = True
            break
    meta["ball_moving_down"] = moving_down
    if not moving_down:
        return False, meta

    occluded, occ_meta = check_rim_top_orange_occluded(shot_frames, hoop_pos, ball_pos=ball_pos)
    meta["rim_occlusion_checked"] = True
    meta.update(occ_meta)
    return bool(occluded), meta


def score(
    ball_pos: list[TrackPoint],
    hoop_pos: list[TrackPoint],
    shot_frames: list[np.ndarray] | None = None,
    rim_cross_px: float = 20.0,
) -> bool:
    """
    Determine make/miss (legacy online path).
    1) Rim occlusion by ball color while moving down → miss
    2) Trajectory segment crossing near hoop center → make
    """
    if len(ball_pos) < 2 or not hoop_pos:
        return False

    miss, _ = rim_occlusion_indicates_miss(ball_pos, hoop_pos, shot_frames)
    if miss:
        return False

    hoop_center = hoop_pos[-1][0]
    hoop_cy = hoop_center[1]
    point_a = None
    point_a_idx = -1
    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < hoop_cy:
            point_a = ball_pos[i][0]
            point_a_idx = i
            break
    if point_a is None or point_a_idx + 1 >= len(ball_pos):
        return False
    point_b = ball_pos[point_a_idx + 1][0]
    return point_to_line_distance(point_a, point_b, hoop_center) < rim_cross_px


def track_point_to_dict(pt: TrackPoint) -> dict[str, Any]:
    (x, y), frame, w, h, conf = pt
    return {
        "center": [int(x), int(y)],
        "frame": int(frame),
        "bbox_wh": [int(w), int(h)],
        "confidence": float(conf),
    }
