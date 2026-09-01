"""Unit tests for v2 action registry + jump_shot classification + enrollment helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.action.detect import classify_release_action, wholebody133_to_h36m
from src.action.registry import (
    KNOWN_ACTION_TYPES,
    SHOOTING_ACTION_TYPES,
    normalize_action_type,
)
from src.identity.sequential_enroll import EnrollSample, cluster_sequential_enrollments


def _synthetic_h36m_seq(
    n: int = 40,
    *,
    approach_px: float = 0.0,
    jump_px: float = 0.0,
    plant: bool = True,
) -> list[tuple[int, np.ndarray]]:
    """Build wholebody133 seq whose H36M projection encodes approach/jump cues."""
    seq = []
    for i in range(n):
        k = np.zeros((133, 3), dtype=np.float32)
        k[:, 2] = 0.9
        # Approximate COCO body that maps into H36M via wholebody133_to_h36m
        # pelvis ~ mid-hip, ankles, wrists
        t = i / max(n - 1, 1)
        # Horizontal approach: move pelvis toward hoop at x=200
        x0 = 800.0 - approach_px * t
        # Vertical jump: lower image-y near the end
        y0 = 500.0 - jump_px * max(0.0, (t - 0.45) / 0.55)
        # COCO-ish indices used by converter (see halpe2h36m / detect)
        # Use dense body keypoints: nose0, shoulders 5/6, hips 11/12, ankles 15/16, wrists 9/10
        k[0] = [x0, y0 - 80, 0.9]
        k[5] = [x0 - 30, y0 - 40, 0.9]
        k[6] = [x0 + 30, y0 - 40, 0.9]
        k[11] = [x0 - 20, y0, 0.9]
        k[12] = [x0 + 20, y0, 0.9]
        ankle_y = y0 + 120 - (0.6 * jump_px if jump_px else 0) * max(0.0, (t - 0.45) / 0.55)
        travel = 0.0 if plant else (80.0 * t)
        k[15] = [x0 - 15 + travel, ankle_y, 0.9]
        k[16] = [x0 + 15 + travel, ankle_y, 0.9]
        k[9] = [x0 - 40, y0 - 20 - 100 * t, 0.9]
        k[10] = [x0 + 40, y0 - 30 - 120 * t, 0.9]
        # elbows
        k[7] = [x0 - 35, y0 - 10, 0.9]
        k[8] = [x0 + 35, y0 - 15, 0.9]
        seq.append((i, k))
    return seq


def test_normalize_keeps_jump_shot():
    assert "jump_shot" in KNOWN_ACTION_TYPES
    assert "jump_shot" in SHOOTING_ACTION_TYPES
    assert normalize_action_type("jump_shot") == "jump_shot"
    assert normalize_action_type("jumper") == "jump_shot"
    assert normalize_action_type("dribble") == "triple_threat"


def test_classify_jump_shot_vs_free_throw():
    planted = _synthetic_h36m_seq(40, approach_px=0.0, jump_px=0.0, plant=True)
    # Force wrist motion for free_throw path
    for _, k in planted:
        k[10, 1] -= 50
    atype, meta = classify_release_action(planted, release_frame=35, hoop_xy=(200.0, 200.0))
    assert atype in {"free_throw", "jump_shot", "unknown"}
    # Strong jump should prefer jump_shot
    jumped = _synthetic_h36m_seq(40, approach_px=20.0, jump_px=90.0, plant=True)
    for _, k in jumped:
        k[10, 1] -= 40
    atype_j, meta_j = classify_release_action(jumped, release_frame=35, hoop_xy=(200.0, 200.0))
    assert atype_j == "jump_shot", meta_j
    assert meta_j.get("pelvis_up", 0) > 0.2 or meta_j.get("ankle_up", 0) > 0.2


def test_classify_layup_runup():
    layup = _synthetic_h36m_seq(40, approach_px=450.0, jump_px=10.0, plant=False)
    atype, meta = classify_release_action(layup, release_frame=35, hoop_xy=(200.0, 200.0))
    assert atype == "layup", meta


def test_cluster_drops_nested_short_id():
    rng = np.random.default_rng(0)

    def samp(t: float, emb: np.ndarray) -> EnrollSample:
        return EnrollSample(
            frame_idx=int(t * 30),
            timestamp_s=t,
            bbox=[100, 100, 300, 500],
            area_ratio=0.08,
            frontal_score=0.9,
            quality=0.8,
            body_emb=emb,
            color_desc=np.ones((6, 3), dtype=np.float32),
        )

    e0 = rng.standard_normal(512).astype(np.float32)
    e0 /= np.linalg.norm(e0)
    e1 = rng.standard_normal(512).astype(np.float32)
    e1 /= np.linalg.norm(e1)
    # long person A
    samples = [samp(1.0 + 0.2 * i, e0) for i in range(20)]
    # nested short B with different emb
    samples += [samp(2.5 + 0.15 * i, e1) for i in range(3)]
    # later person C
    e2 = rng.standard_normal(512).astype(np.float32)
    e2 /= np.linalg.norm(e2)
    samples += [samp(8.0 + 0.2 * i, e2) for i in range(10)]
    samples.sort(key=lambda s: s.timestamp_s)
    people = cluster_sequential_enrollments(samples, id_prefix="stu", min_samples=3)
    ids = [p.student_id for p in people]
    assert "stu_00" in ids
    # nested short should be dropped → expect 2 people (A and C)
    assert len(people) == 2, [(p.student_id, p.t0, p.t_end, len(p.samples)) for p in people]


if __name__ == "__main__":
    test_normalize_keeps_jump_shot()
    test_classify_jump_shot_vs_free_throw()
    test_classify_layup_runup()
    test_cluster_drops_nested_short_id()
    print("v2 unit tests passed.")
