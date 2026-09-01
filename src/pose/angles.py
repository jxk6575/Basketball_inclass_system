"""Joint angle utilities from WholeBody-133 or H36M-17 3D skeletons."""

from __future__ import annotations

import numpy as np

# COCO-WholeBody indices
NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
R_HAND_START = 112  # right hand 21 pts; index 8 approx MCP in hand block

# H36M-17
H36M_R_HIP, H36M_R_KNEE, H36M_R_ANKLE = 1, 2, 3
H36M_L_HIP, H36M_L_KNEE, H36M_L_ANKLE = 4, 5, 6
H36M_L_SHOULDER, H36M_L_ELBOW, H36M_L_WRIST = 11, 12, 13
H36M_R_SHOULDER, H36M_R_ELBOW, H36M_R_WRIST = 14, 15, 16


def angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC at point b in degrees."""
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _apply_shooting_side(angles: dict[str, float], shooting_hand: str | None) -> dict[str, float]:
    if shooting_hand not in ("left", "right"):
        return angles
    prefix = shooting_hand
    elbow_k, wrist_k = f"{prefix}_elbow", f"{prefix}_wrist"
    if elbow_k in angles:
        angles["shooting_elbow"] = angles[elbow_k]
    if wrist_k in angles:
        angles["shooting_wrist"] = angles[wrist_k]
    wi = L_WRIST if shooting_hand == "left" else R_WRIST
    return angles


def compute_frame_angles(
    kpts3d: np.ndarray,
    shooting_hand: str | None = None,
) -> dict[str, float]:
    """kpts3d: (133, 3) world coordinates in meters."""
    k = kpts3d
    rh_mcp = k[R_HAND_START + 8] if k.shape[0] > R_HAND_START + 8 else k[R_WRIST] + np.array([0.05, 0, 0])
    lh_mcp = k[L_WRIST] + np.array([-0.05, 0, 0])

    out = {
        "left_knee": angle_at_joint(k[L_HIP], k[L_KNEE], k[L_ANKLE]),
        "right_knee": angle_at_joint(k[R_HIP], k[R_KNEE], k[R_ANKLE]),
        "left_elbow": angle_at_joint(k[L_SHOULDER], k[L_ELBOW], k[L_WRIST]),
        "right_elbow": angle_at_joint(k[R_SHOULDER], k[R_ELBOW], k[R_WRIST]),
        "right_wrist": angle_at_joint(k[R_ELBOW], k[R_WRIST], rh_mcp),
        "left_wrist": angle_at_joint(k[L_ELBOW], k[L_WRIST], lh_mcp),
        "wrist_height_m": float(k[R_WRIST, 2]) if not np.isnan(k[R_WRIST, 2]) else float(k[R_WRIST, 1]),
    }
    if shooting_hand in ("left", "right"):
        wi = L_WRIST if shooting_hand == "left" else R_WRIST
        out["wrist_height_m"] = float(k[wi, 2]) if not np.isnan(k[wi, 2]) else float(k[wi, 1])
    return _apply_shooting_side(out, shooting_hand)


def compute_h36m_angles(
    joints17: np.ndarray,
    shooting_hand: str | None = None,
) -> dict[str, float]:
    """
    Court-world H36M-17 joints (N,3) in meters → dashboard angle keys.

    Wrist angle approximates forearm continuity (elbow–wrist–virtual tip).
    """
    k = np.asarray(joints17, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] < 17 or k.shape[1] < 3:
        return {}
    forearm = k[H36M_R_WRIST] - k[H36M_R_ELBOW]
    fn = float(np.linalg.norm(forearm))
    tip = k[H36M_R_WRIST] + (forearm / (fn + 1e-6)) * 0.08
    l_forearm = k[H36M_L_WRIST] - k[H36M_L_ELBOW]
    lfn = float(np.linalg.norm(l_forearm))
    ltip = k[H36M_L_WRIST] + (l_forearm / (lfn + 1e-6)) * 0.08
    out = {
        "left_knee": angle_at_joint(k[H36M_L_HIP], k[H36M_L_KNEE], k[H36M_L_ANKLE]),
        "right_knee": angle_at_joint(k[H36M_R_HIP], k[H36M_R_KNEE], k[H36M_R_ANKLE]),
        "left_elbow": angle_at_joint(k[H36M_L_SHOULDER], k[H36M_L_ELBOW], k[H36M_L_WRIST]),
        "right_elbow": angle_at_joint(k[H36M_R_SHOULDER], k[H36M_R_ELBOW], k[H36M_R_WRIST]),
        "right_wrist": angle_at_joint(k[H36M_R_ELBOW], k[H36M_R_WRIST], tip),
        "left_wrist": angle_at_joint(k[H36M_L_ELBOW], k[H36M_L_WRIST], ltip),
        "wrist_height_m": float(k[H36M_R_WRIST, 2]),
    }
    if shooting_hand in ("left", "right"):
        wi = H36M_L_WRIST if shooting_hand == "left" else H36M_R_WRIST
        out["wrist_height_m"] = float(k[wi, 2])
    return _apply_shooting_side(out, shooting_hand)


def compute_series_angles(frames: list[np.ndarray]) -> list[dict[str, float]]:
    return [compute_frame_angles(f) for f in frames]
