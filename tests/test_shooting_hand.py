"""Unit tests for shooting-hand inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.action.shooting_hand import (
    L_WRIST,
    R_WRIST,
    infer_shooting_hand_at_frame,
    infer_shooting_hand_from_window,
    wrist_idx,
)
from src.pose.angles import compute_frame_angles


def _blank_kpts() -> np.ndarray:
    k = np.zeros((133, 3), dtype=np.float32)
    k[:, 2] = 0.9
    k[5, 1] = k[6, 1] = 250  # shoulders
    k[0, 1] = 230  # nose
    return k


def test_ball_near_left_wrist():
    k = _blank_kpts()
    k[L_WRIST, :2] = [100, 180]
    k[R_WRIST, :2] = [200, 220]
    hand, meta = infer_shooting_hand_at_frame(k, ball_xy=(105.0, 185.0))
    assert hand == "left"
    assert meta["scores"]["left"] > meta["scores"]["right"]


def test_right_wrist_higher_at_release():
    k = _blank_kpts()
    k[L_WRIST, :2] = [120, 240]
    k[R_WRIST, :2] = [180, 170]
    hand, _ = infer_shooting_hand_at_frame(k, ball_xy=None)
    assert hand == "right"


def test_window_prefers_shooting_side():
    seq = []
    for i in range(20):
        k = _blank_kpts()
        k[L_WRIST, :2] = [100, 240 - i]
        k[R_WRIST, :2] = [180, 200 - i * 2]
        seq.append((i, k))
    hand, _ = infer_shooting_hand_from_window(seq, peak_idx=10, ball_by_frame=None)
    assert hand == "right"


def test_compute_frame_angles_shooting_side():
    k = np.zeros((133, 3))
    k[6], k[8], k[10] = [0, 0, 0], [1, 0, 0], [1, 1, 0]
    k[5], k[7], k[9] = [0, 2, 0], [-1, 2, 0], [-1, 3, 0]
    ang = compute_frame_angles(k, shooting_hand="left")
    assert "shooting_elbow" in ang
    assert ang["shooting_elbow"] == ang["left_elbow"]
    assert wrist_idx("left") == L_WRIST
