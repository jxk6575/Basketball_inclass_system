"""Integration tests for core pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.action.pipeline import run_action_session_auto
from src.config import data_path
from src.identity.enrollment import EnrollmentGallery
from src.orchestrator.session_pipeline import create_session, register_student, run_pipeline
from src.pose.angles import compute_frame_angles
from src.pose.pose2sim_wrapper import run_pose3d_session
from src.privacy.consent import grant_consent, has_consent
from src.privacy.db import init_db
from src.privacy.retention import cascade_delete_student
from src.scoring.fusion import run_scoring_session
from src.types import ConsentScope


def _fake_pose2d(session_id: str, student_id: str, n_frames: int = 60):
    cam = "cam_03"
    out = data_path("sessions", session_id, "perception", cam)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n_frames):
        kpts = np.zeros((133, 3), dtype=np.float32)
        kpts[10, 1] = 200 + i * 2  # rising wrist for release detect
        kpts[10, 2] = 0.9
        kpts[0, 1] = 230  # nose / neck
        kpts[0, 2] = 0.9
        kpts[5, 1] = 255  # left shoulder
        kpts[5, 2] = 0.9
        kpts[8, 1] = 300
        kpts[6, 1] = 250  # right shoulder
        kpts[6, 2] = 0.9
        kpts[12, 1] = 400
        kpts[14, 1] = 500
        kpts[16, 1] = 600
        frames.append({
            "frame": i,
            "persons": [{
                "student_id": student_id,
                "track_id": 1,
                "keypoints": kpts.tolist(),
                "scores": [0.9] * 133,
            }],
        })
    (out / "pose2d.json").write_text(json.dumps({
        "camera_id": cam,
        "session_id": session_id,
        "fps": 30.0,
        "processing": "per_camera_isolated",
        "frames": frames,
    }))


def test_consent_and_cascade_delete():
    init_db()
    sid = create_session("test_class")
    register_student("stu1", "张三")
    grant_consent("stu1", sid, [ConsentScope.VIDEO, ConsentScope.FACE])
    assert has_consent("stu1", sid, ConsentScope.VIDEO)
    gallery = EnrollmentGallery(sid)
    gallery.save_meta("stu1", "张三")
    deleted = cascade_delete_student(sid, "stu1")
    assert isinstance(deleted, list)


def test_angles():
    k = np.zeros((133, 3))
    k[6], k[8], k[10] = [0, 0, 0], [1, 0, 0], [1, 1, 0]
    ang = compute_frame_angles(k)
    assert "right_elbow" in ang


def test_peak_merge_groups_single_shot_motion():
    from src.action.detect import _merge_nearby_peaks

    frames = [100, 146, 192, 500]
    wrist_y = [180.0, 175.0, 170.0, 200.0]
    peaks = [0, 1, 2, 3]
    merged = _merge_nearby_peaks(peaks, frames, wrist_y, merge_window=50)
    assert merged == [2, 3]


def test_shooting_wrist_above_shoulder_filter():
    from src.action.detect import _wrist_above_shoulder_and_neck, detect_shooting_phases

    shoot_k = np.zeros((133, 3), dtype=np.float32)
    shoot_k[10] = [100, 180, 0.9]  # wrist high
    shoot_k[6] = [100, 250, 0.9]   # right shoulder
    shoot_k[5] = [80, 255, 0.9]
    shoot_k[0] = [100, 220, 0.9]   # nose
    assert _wrist_above_shoulder_and_neck(shoot_k)

    dribble_k = shoot_k.copy()
    dribble_k[10, 1] = 400  # wrist below shoulder
    assert not _wrist_above_shoulder_and_neck(dribble_k)

    seq = []
    for i in range(40):
        k = dribble_k.copy()
        k[10, 1] = 350 + (i % 5) * 3
        seq.append((i, k))
    assert detect_shooting_phases(seq) == []


def test_end_to_end_pipeline():
    init_db()
    session_id = create_session("e2e_class")
    student_id = "stu_e2e"
    register_student(student_id, "李四")
    grant_consent(student_id, session_id, [ConsentScope.VIDEO, ConsentScope.FACE, ConsentScope.REPORT])
    _fake_pose2d(session_id, student_id)

    run_pose3d_session(session_id, ["cam_03"])
    done = run_action_session_auto(session_id, [student_id])
    # Fake pose has no cam_04 ball gate → shooting detector empty is OK.
    # Seed one clip so scoring/report path still exercises end-to-end.
    if not done:
        from src.types import ActionClip, ActionPhase, StudentActions

        clip = ActionClip(
            action_type="free_throw",
            start_frame=10,
            end_frame=50,
            confidence=0.8,
            phases=[
                ActionPhase(name="load", start=10, end=25),
                ActionPhase(name="release", start=25, end=30),
                ActionPhase(name="follow_through", start=30, end=50),
            ],
            student_id=student_id,
            participant_ids=[student_id],
        )
        out = data_path("sessions", session_id, "actions")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{student_id}.json").write_text(
            StudentActions(student_id=student_id, clips=[clip]).model_dump_json(indent=2),
            encoding="utf-8",
        )

    reports = run_scoring_session(session_id, [student_id])

    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["student_id"] == student_id
    assert report["total_score"] >= 0


if __name__ == "__main__":
    test_consent_and_cascade_delete()
    test_angles()
    test_peak_merge_groups_single_shot_motion()
    test_shooting_wrist_above_shoulder_filter()
    test_end_to_end_pipeline()
    print("All tests passed.")
