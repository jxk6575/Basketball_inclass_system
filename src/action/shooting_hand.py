"""Infer anatomical shooting hand (left/right wrist) from pose + ball cues."""

from __future__ import annotations

from typing import Literal

import numpy as np

ShootingHand = Literal["left", "right"]

NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6
L_WRIST, R_WRIST = 9, 10
L_ELBOW, R_ELBOW = 7, 8
MIN_KPT_SCORE = 0.3

_SIDE = (
    ("left", L_WRIST, L_ELBOW, L_SHOULDER),
    ("right", R_WRIST, R_ELBOW, R_SHOULDER),
)


def wrist_idx(hand: ShootingHand) -> int:
    return L_WRIST if hand == "left" else R_WRIST


def elbow_idx(hand: ShootingHand) -> int:
    return L_ELBOW if hand == "left" else R_ELBOW


def shoulder_idx(hand: ShootingHand) -> int:
    return L_SHOULDER if hand == "left" else R_SHOULDER


def _elbow_angle_deg(k: np.ndarray, shoulder: int, elbow: int, wrist: int) -> float | None:
    if min(float(k[i, 2]) for i in (shoulder, elbow, wrist)) < MIN_KPT_SCORE:
        return None
    a, b, c = k[shoulder, :2], k[elbow, :2], k[wrist, :2]
    ba, bc = a - b, c - b
    n = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if n < 1e-6:
        return None
    cos = float(np.clip(np.dot(ba, bc) / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _wrist_raise_score(k: np.ndarray, shoulder: int, wrist: int) -> float:
    if k[wrist, 2] < MIN_KPT_SCORE or k[shoulder, 2] < MIN_KPT_SCORE:
        return 0.0
    dy = float(k[shoulder, 1]) - float(k[wrist, 1])
    if dy <= 8.0:
        return 0.0
    return min(1.0, dy / 80.0)


def infer_shooting_hand_at_frame(
    k: np.ndarray,
    ball_xy: tuple[float, float] | None = None,
) -> tuple[ShootingHand, dict]:
    """Score left vs right at one pose frame."""
    scores = {"left": 0.0, "right": 0.0}
    detail: dict[str, dict] = {}

    for hand, wi, ei, si in _SIDE:
        if k[wi, 2] < MIN_KPT_SCORE:
            continue
        raise_s = _wrist_raise_score(k, si, wi)
        scores[hand] += raise_s * 0.35
        ang = _elbow_angle_deg(k, si, ei, wi)
        if ang is not None:
            scores[hand] += min(1.0, ang / 160.0) * 0.25
        prox = 0.0
        if ball_xy is not None:
            dist = float(np.hypot(ball_xy[0] - k[wi, 0], ball_xy[1] - k[wi, 1]))
            prox = max(0.0, 1.0 - dist / 220.0)
            scores[hand] += prox * 0.45
        detail[hand] = {"raise": round(raise_s, 3), "elbow_deg": ang, "ball_prox": round(prox, 3)}

    total = scores["left"] + scores["right"]
    if total <= 1e-6:
        return "right", {"reason": "default_right", "scores": scores, "detail": detail}

    hand: ShootingHand = "left" if scores["left"] > scores["right"] else "right"
    margin = abs(scores["left"] - scores["right"]) / max(total, 1e-6)
    return hand, {
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "margin": round(float(margin), 4),
        "detail": detail,
    }


def infer_shooting_hand_from_window(
    seq: list[tuple[int, np.ndarray]],
    peak_idx: int,
    ball_by_frame: dict[int, dict] | None = None,
    pre: int = 12,
    post: int = 8,
) -> tuple[ShootingHand, dict]:
    """Aggregate shooting-hand evidence around a release peak index."""
    i0 = max(0, peak_idx - pre)
    i1 = min(len(seq), peak_idx + post + 1)
    agg = {"left": 0.0, "right": 0.0}
    frames_used = 0

    for i in range(i0, i1):
        frame, k = seq[i]
        ball_xy = _ball_xy_at_frame(ball_by_frame, frame)
        hand, meta = infer_shooting_hand_at_frame(k, ball_xy)
        w = 1.0 + (0.8 if i == peak_idx else 0.0)
        agg[hand] += w * (1.0 + float(meta.get("margin") or 0.0))
        frames_used += 1

    # Peak wrist height tie-break (image-y: lower = higher)
    peak_heights: dict[str, float] = {}
    for hand, wi, _, _ in _SIDE:
        ys = [
            float(seq[i][1][wi, 1])
            for i in range(i0, i1)
            if seq[i][1][wi, 2] >= MIN_KPT_SCORE
        ]
        if ys:
            peak_heights[hand] = min(ys)
    if peak_heights:
        best_hand = min(peak_heights, key=lambda h: peak_heights[h])
        agg[best_hand] += 0.6

    hand: ShootingHand = "left" if agg["left"] > agg["right"] else "right"
    if agg["left"] == agg["right"] == 0.0:
        hand = "right"
    return hand, {
        "agg_scores": {k: round(v, 4) for k, v in agg.items()},
        "peak_heights": {k: round(v, 2) for k, v in peak_heights.items()},
        "frames_used": frames_used,
    }


def shooting_side_angle_keys(hand: ShootingHand) -> tuple[str, str]:
    prefix = "left" if hand == "left" else "right"
    return f"{prefix}_elbow", f"{prefix}_wrist"


def _ball_xy_at_frame(
    ball_by_frame: dict[int, dict] | None,
    frame: int,
    window: int = 4,
) -> tuple[float, float] | None:
    if not ball_by_frame:
        return None
    if frame in ball_by_frame:
        ball = ball_by_frame[frame]
    else:
        ball = None
        for d in range(1, window + 1):
            if frame - d in ball_by_frame:
                ball = ball_by_frame[frame - d]
                break
            if frame + d in ball_by_frame:
                ball = ball_by_frame[frame + d]
                break
    if not ball or not ball.get("center"):
        return None
    c = ball["center"]
    return float(c[0]), float(c[1])
