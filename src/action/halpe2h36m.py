"""Convert COCO-WholeBody / Halpe 133 to H36M 17 (MotionBERT input)."""

from __future__ import annotations

import numpy as np

# Halpe26 -> H36M mapping (from MotionBERT dataset_wild.py)
def halpe26_to_h36m(x: np.ndarray) -> np.ndarray:
    """x: (T, V, C) Halpe26"""
    t, v, c = x.shape
    y = np.zeros((t, 17, c), dtype=x.dtype)
    y[:, 0] = x[:, 19]
    y[:, 1] = x[:, 12]
    y[:, 2] = x[:, 14]
    y[:, 3] = x[:, 16]
    y[:, 4] = x[:, 11]
    y[:, 5] = x[:, 13]
    y[:, 6] = x[:, 15]
    y[:, 7] = (x[:, 18] + x[:, 19]) * 0.5
    y[:, 8] = x[:, 18]
    y[:, 9] = x[:, 0]
    y[:, 10] = x[:, 17]
    y[:, 11] = x[:, 5]
    y[:, 12] = x[:, 7]
    y[:, 13] = x[:, 9]
    y[:, 14] = x[:, 6]
    y[:, 15] = x[:, 8]
    y[:, 16] = x[:, 10]
    return y


def coco17_from_133(kpts: np.ndarray) -> np.ndarray:
    """Take first 17 body joints from 133 layout."""
    return kpts[:17].copy()


def coco17_to_wholebody133(kpts17: np.ndarray) -> np.ndarray:
    """
    Expand COCO-17 (YOLO-Pose) to WholeBody-133 layout.

    Body joints 0–16 are copied; face/hand/foot extras stay at conf=0 so
    downstream code that indexes [:17] / H36M still works.
    """
    src = np.asarray(kpts17, dtype=np.float32)
    if src.ndim != 2 or src.shape[0] < 17:
        return np.zeros((133, 3), dtype=np.float32)
    if src.shape[1] == 2:
        out17 = np.zeros((17, 3), dtype=np.float32)
        out17[:, :2] = src[:17]
        out17[:, 2] = 1.0
        src = out17
    out = np.zeros((133, 3), dtype=np.float32)
    out[:17] = src[:17, :3]
    return out


def wholebody133_to_h36m(kpts: np.ndarray) -> np.ndarray:
    """Approximate: use COCO17 subset + synthetic hip/spine."""
    c17 = coco17_from_133(kpts)
    # Build pseudo Halpe26 from COCO17
    halpe = np.zeros((26, kpts.shape[1]), dtype=kpts.dtype)
    halpe[0] = c17[0]   # nose
    halpe[5], halpe[6] = c17[5], c17[6]
    halpe[7], halpe[8] = c17[7], c17[8]
    halpe[9], halpe[10] = c17[9], c17[10]
    halpe[11], halpe[12] = c17[11], c17[12]
    halpe[13], halpe[14] = c17[13], c17[14]
    halpe[15], halpe[16] = c17[15], c17[16]
    # Halpe 17 = head. Must set — otherwise H36M joint 10 stays at (0,0)
    # and after pelvis-centering becomes a far jittering outlier above the neck.
    halpe[17] = c17[0]  # head ← nose
    halpe[18] = (c17[5] + c17[6]) / 2  # neck
    halpe[19] = (c17[11] + c17[12]) / 2  # hip
    seq = halpe[np.newaxis, ...]
    return halpe26_to_h36m(seq)[0]
