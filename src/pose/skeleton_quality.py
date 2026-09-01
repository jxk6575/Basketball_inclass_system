"""2D skeleton plausibility checks for perception / visualization."""

from __future__ import annotations

import numpy as np

# COCO-WholeBody (first 17)
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def _len2(k: np.ndarray, a: int, b: int, conf_thr: float) -> float | None:
    if k.shape[1] >= 3 and (float(k[a, 2]) < conf_thr or float(k[b, 2]) < conf_thr):
        return None
    return float(np.linalg.norm(k[a, :2] - k[b, :2]))


def clamp_keypoints_to_bbox(
    kpts: np.ndarray,
    bbox: list[float] | tuple[float, ...] | None,
    *,
    margin: float = 0.12,
) -> np.ndarray:
    """Zero confidence for joints outside an expanded person bbox."""
    k = np.asarray(kpts, dtype=np.float64).copy()
    if bbox is None or k.ndim != 2 or k.shape[0] < 17 or k.shape[1] < 3:
        return k
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    # Support xywh
    if x2 < x1 or y2 < y1:
        x1, y1, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x2, y2 = x1 + w, y1 + h
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= margin * bw
    x2 += margin * bw
    y1 -= margin * bh
    y2 += margin * bh
    for i in range(min(17, k.shape[0])):
        if not (x1 <= k[i, 0] <= x2 and y1 <= k[i, 1] <= y2):
            k[i, 2] = 0.0
    return k


def skeleton_plausible_2d(
    kpts: np.ndarray,
    conf_thr: float = 0.3,
    min_joints: int = 8,
    bbox: list[float] | tuple[float, ...] | None = None,
    bbox_margin: float = 0.12,
) -> tuple[bool, str]:
    """
    Reject grossly wrong 2D poses (crossed limbs, inverted body, absurd ratios,
    pole/edge-hugging non-humanoid detections, joints far outside person bbox).

    ``kpts``: (N, 2) or (N, 3) WholeBody / COCO-style, image coords (y down).
    ``bbox``: optional person box [x1,y1,x2,y2] or [x,y,w,h] for out-of-box reject.
    """
    k = np.asarray(kpts, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] < 17:
        return False, "too_few_joints"

    # Check out-of-box limbs on raw coords BEFORE clamping (ghost arms)
    if bbox is not None:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        if x2 < x1 or y2 < y1:
            x1, y1, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            x2, y2 = x1 + bw, y1 + bh
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        ex1, ey1 = x1 - bbox_margin * bw, y1 - bbox_margin * bh
        ex2, ey2 = x2 + bbox_margin * bw, y2 + bbox_margin * bh
        conf = k[:, 2] if k.shape[1] >= 3 else np.ones(k.shape[0])
        out_arm = 0
        for j in (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST):
            if float(conf[j]) >= conf_thr and not (
                ex1 <= float(k[j, 0]) <= ex2 and ey1 <= float(k[j, 1]) <= ey2
            ):
                out_arm += 1
        if out_arm >= 3:
            return False, "arms_outside_bbox"
        # Full arm span on raw coords
        for s, w, name in ((R_SHOULDER, R_WRIST, "r_arm"), (L_SHOULDER, L_WRIST, "l_arm")):
            if float(conf[s]) >= conf_thr and float(conf[w]) >= conf_thr:
                # Need torso estimate early
                if all(float(conf[i]) >= conf_thr for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)):
                    sm = 0.5 * (k[L_SHOULDER, :2] + k[R_SHOULDER, :2])
                    hm = 0.5 * (k[L_HIP, :2] + k[R_HIP, :2])
                    torso0 = float(np.linalg.norm(sm - hm)) + 1e-3
                    arm = float(np.linalg.norm(k[s, :2] - k[w, :2]))
                    # 3.2× torso: shooting release often stretches beyond 2.6×
                    if arm > 3.2 * torso0:
                        return False, f"armspan_{name}"

    if bbox is not None and k.shape[1] >= 3:
        k = clamp_keypoints_to_bbox(k, bbox, margin=bbox_margin)
    if k.shape[1] >= 3:
        valid = k[:, 2] >= conf_thr
    else:
        valid = np.isfinite(k[:, 0]) & np.isfinite(k[:, 1])
    if int(valid.sum()) < min_joints:
        return False, "too_few_valid"

    # Need torso + at least one leg
    need = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    if not all(bool(valid[i]) for i in need):
        return False, "missing_torso"

    shoulder_mid = 0.5 * (k[L_SHOULDER, :2] + k[R_SHOULDER, :2])
    hip_mid = 0.5 * (k[L_HIP, :2] + k[R_HIP, :2])
    torso = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso < 15.0:
        return False, "torso_too_small"

    # In image coords, hips should be below shoulders (larger y)
    if float(hip_mid[1]) < float(shoulder_mid[1]) - 0.15 * torso:
        return False, "inverted_torso"

    # Shoulder / hip width sanity
    sw = float(np.linalg.norm(k[L_SHOULDER, :2] - k[R_SHOULDER, :2]))
    hw = float(np.linalg.norm(k[L_HIP, :2] - k[R_HIP, :2]))
    if sw > 0 and hw > 0:
        ratio = max(sw, hw) / (min(sw, hw) + 1e-6)
        if ratio > 4.5:
            return False, "torso_width_skew"
    # Real humans have some shoulder or hip breadth vs torso length
    if max(sw, hw) < 0.18 * torso:
        return False, "torso_too_narrow"

    pts = k[valid, :2]
    xs, ys = pts[:, 0], pts[:, 1]
    span_x = float(xs.max() - xs.min())
    span_y = float(ys.max() - ys.min())
    # Pole / fence false positives: joints collapse to a thin vertical strip
    if span_y > 40.0 and span_x < max(12.0, 0.14 * span_y):
        return False, "pole_like"
    if span_x > 40.0 and span_y < max(12.0, 0.14 * span_x):
        return False, "flat_like"
    # Near-collinear keypoints (edge-hugging stick figure)
    if pts.shape[0] >= 6:
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(pts.shape[0] - 1, 1)
        eig = np.linalg.eigvalsh(cov)
        if float(eig[-1]) > 1.0 and float(eig[0]) / float(eig[-1]) < 0.025:
            return False, "collinear_joints"

    # Limb length vs torso (stricter than before)
    for a, b, name in (
        (R_SHOULDER, R_ELBOW, "r_upper_arm"),
        (R_ELBOW, R_WRIST, "r_forearm"),
        (L_SHOULDER, L_ELBOW, "l_upper_arm"),
        (L_ELBOW, L_WRIST, "l_forearm"),
        (R_HIP, R_KNEE, "r_thigh"),
        (R_KNEE, R_ANKLE, "r_shin"),
        (L_HIP, L_KNEE, "l_thigh"),
        (L_KNEE, L_ANKLE, "l_shin"),
    ):
        if not (valid[a] and valid[b]):
            continue
        L = _len2(k, a, b, conf_thr)
        if L is None:
            continue
        if L > 2.2 * torso or L < 0.08 * torso:
            return False, f"limb_{name}"

    # Full arm span shoulder→wrist must stay human-scale
    for s, w, name in ((R_SHOULDER, R_WRIST, "r_arm"), (L_SHOULDER, L_WRIST, "l_arm")):
        if valid[s] and valid[w]:
            arm = float(np.linalg.norm(k[s, :2] - k[w, :2]))
            if arm > 2.6 * torso:
                return False, f"armspan_{name}"

    # Ankles below hips (image y)
    for hip, ankle in ((R_HIP, R_ANKLE), (L_HIP, L_ANKLE)):
        if valid[hip] and valid[ankle]:
            if float(k[ankle, 1]) < float(k[hip, 1]) - 0.35 * torso:
                return False, "ankle_above_hip"

    # Knees roughly between hip and ankle in y
    for hip, knee, ankle in (
        (R_HIP, R_KNEE, R_ANKLE),
        (L_HIP, L_KNEE, L_ANKLE),
    ):
        if valid[hip] and valid[knee] and valid[ankle]:
            hy, ky, ay = float(k[hip, 1]), float(k[knee, 1]), float(k[ankle, 1])
            if ay > hy + 5:
                if not (hy - 0.2 * torso <= ky <= ay + 0.2 * torso):
                    return False, "knee_out_of_leg"

    return True, "ok"
