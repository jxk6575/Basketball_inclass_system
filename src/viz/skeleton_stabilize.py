"""Temporal cleanup for H36M-17 pseudo/true 3D skeletons in the 4D viewer."""

from __future__ import annotations

import numpy as np

# Swap pairs: right ↔ left limbs (H36M-17)
_LR_PAIRS = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))


def swap_left_right(joints: np.ndarray) -> np.ndarray:
    out = joints.copy()
    for a, b in _LR_PAIRS:
        out[a], out[b] = out[b].copy(), out[a].copy()
    return out


def _frame_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Mean L2 over finite joints (ignore near-identical pelvis)."""
    d = np.linalg.norm(a - b, axis=1)
    return float(np.mean(d))


def enforce_lr_consistency(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 17, 3). Prefer continuity vs previous frame; swap L/R when cheaper.
    Distinguishes full chirality flips from shoulder-line-only geometric jumps.
    """
    if len(seq) == 0:
        return seq
    out = seq.astype(np.float64).copy()
    upper = (8, 9, 10, 11, 12, 13, 14, 15, 16)
    arms = (11, 12, 13, 14, 15, 16)

    def shoulder_yaw(j: np.ndarray) -> float:
        v = j[14, :2] - j[11, :2]
        return float(np.arctan2(v[1], v[0]))

    def hip_yaw(j: np.ndarray) -> float:
        v = j[1, :2] - j[4, :2]
        return float(np.arctan2(v[1], v[0]))

    def yaw_delta(a: float, b: float) -> float:
        return abs(((a - b + np.pi) % (2 * np.pi)) - np.pi)

    def swap_arms(j: np.ndarray) -> np.ndarray:
        o = j.copy()
        for a, b in ((11, 14), (12, 15), (13, 16)):
            o[a], o[b] = o[b].copy(), o[a].copy()
        return o

    for t in range(1, len(out)):
        prev, cur = out[t - 1], out[t].copy()
        swapped = swap_left_right(cur)

        c0 = _frame_cost(cur, prev)
        c1 = _frame_cost(swapped, prev)
        dy0 = yaw_delta(shoulder_yaw(cur), shoulder_yaw(prev))
        dy1 = yaw_delta(shoulder_yaw(swapped), shoulder_yaw(prev))
        hy0 = yaw_delta(hip_yaw(cur), hip_yaw(prev))
        hy1 = yaw_delta(hip_yaw(swapped), hip_yaw(prev))

        if c1 < c0 * 0.92 or (dy0 > 1.2 and dy1 < dy0 * 0.55 and hy1 <= hy0 + 0.35):
            out[t] = swapped
            continue
        if dy0 > 1.5 and dy1 < dy0 * 0.5 and hy1 + 0.2 < hy0:
            out[t] = swapped
            continue
        if c1 < c0 and dy1 < dy0 and hy1 <= hy0 + 0.4:
            out[t] = swapped
            continue

        # Persistent shoulder↔hip disagreement (~opposite): swap arms only
        sh_hip = yaw_delta(shoulder_yaw(cur), hip_yaw(cur))
        if sh_hip > 2.0:
            arms_sw = swap_arms(cur)
            if yaw_delta(shoulder_yaw(arms_sw), hip_yaw(arms_sw)) < sh_hip * 0.5:
                cur = arms_sw
                dy0 = yaw_delta(shoulder_yaw(cur), shoulder_yaw(prev))

        # Shoulder-line glitch only (hips stable; full swap would wreck hips)
        if dy0 > 2.0 and hy0 < 0.6:
            for idx in upper:
                cur[idx] = 0.25 * cur[idx] + 0.75 * prev[idx]
            out[t] = cur
            continue

        out[t] = cur
    return out


def repair_outlier_joints(seq: np.ndarray, max_bone: float = 1.8) -> np.ndarray:
    """Clamp absurd bone lengths / flying head relative to torso scale."""
    out = seq.astype(np.float64).copy()
    edges = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16),
    ]
    for t in range(len(out)):
        j = out[t]
        torso = float(np.linalg.norm(j[9] - j[0]))
        if torso < 1e-4:
            torso = 1.0
        # head
        if float(np.linalg.norm(j[10] - j[9])) > 1.2 * torso:
            up = j[9] - j[8]
            n = float(np.linalg.norm(up))
            up = up / n * 0.32 * torso if n > 1e-6 else np.array([0.0, -0.25 * torso, 0.0])
            j[10] = j[9] + up
        for a, b in edges:
            bone = j[b] - j[a]
            L = float(np.linalg.norm(bone))
            lim = max_bone * torso
            if L > lim and L > 1e-6:
                j[b] = j[a] + bone / L * lim
        out[t] = j
    return out


def reject_spike_frames(seq: np.ndarray, thr: float = 1.15) -> np.ndarray:
    """Replace frames with huge mean jump or isolated facing flips by neighbor blend."""
    if len(seq) < 3:
        return seq
    out = seq.astype(np.float64).copy()

    def yaw(j: np.ndarray) -> float:
        v = j[14, :2] - j[11, :2]
        return float(np.arctan2(v[1], v[0]))

    def yaw_delta(a: np.ndarray, b: np.ndarray) -> float:
        return abs(((yaw(a) - yaw(b) + np.pi) % (2 * np.pi)) - np.pi)

    for t in range(1, len(out) - 1):
        d = _frame_cost(out[t], out[t - 1])
        dy = yaw_delta(out[t], out[t - 1])
        dy_skip = yaw_delta(out[t + 1], out[t - 1])
        spike = d > thr
        # Isolated ~180° facing flip while neighbors agree
        facing_glitch = dy > 1.2 and dy_skip < 0.7
        if spike or facing_glitch:
            if _frame_cost(out[t + 1], out[t - 1]) < max(d, 0.5) or facing_glitch:
                out[t] = 0.5 * (out[t - 1] + out[t + 1])
    return out


def ema_smooth(seq: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Causal EMA; alpha = weight on new frame (higher = less lag)."""
    if len(seq) == 0:
        return seq
    out = seq.astype(np.float64).copy()
    for t in range(1, len(out)):
        out[t] = alpha * out[t] + (1.0 - alpha) * out[t - 1]
    return out


def align_upper_to_hips(seq: np.ndarray) -> np.ndarray:
    """Sticky arm L/R fix when shoulders face opposite the hips (side-view ambiguity)."""
    out = seq.astype(np.float64).copy()

    def shoulder_yaw(j: np.ndarray) -> float:
        v = j[14, :2] - j[11, :2]
        return float(np.arctan2(v[1], v[0]))

    def hip_yaw(j: np.ndarray) -> float:
        v = j[1, :2] - j[4, :2]
        return float(np.arctan2(v[1], v[0]))

    def yaw_delta(a: float, b: float) -> float:
        return abs(((a - b + np.pi) % (2 * np.pi)) - np.pi)

    def swap_arms(j: np.ndarray) -> np.ndarray:
        o = j.copy()
        for a, b in ((11, 14), (12, 15), (13, 16)):
            o[a], o[b] = o[b].copy(), o[a].copy()
        return o

    mirrored = False
    for t in range(len(out)):
        sh_hip = yaw_delta(shoulder_yaw(out[t]), hip_yaw(out[t]))
        if sh_hip > 2.2:
            mirrored = True
        elif sh_hip < 0.6:
            mirrored = False
        if mirrored:
            cand = swap_arms(out[t])
            if yaw_delta(shoulder_yaw(cand), hip_yaw(cand)) <= sh_hip:
                out[t] = cand
    return out


def stabilize_skeleton_sequence(seq: np.ndarray) -> np.ndarray:
    """Full cleanup pipeline for viewer / pseudo3d preview."""
    x = np.asarray(seq, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 17:
        return x
    x = repair_outlier_joints(x)
    x = align_upper_to_hips(x)
    x = enforce_lr_consistency(x)
    x = reject_spike_frames(x)
    x = ema_smooth(x, alpha=0.50)
    x = repair_outlier_joints(x)
    return x
