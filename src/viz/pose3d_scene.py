"""Build interactive 4D (3D+time) skeleton scene JSON for a group session."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.calibration.court_model import landmark_xyz, load_court_model, priority_ids
from src.pose.triangulate import load_camera_calibration
from src.viz.pose_from_video import extract_pose_sequence_from_video, h36m_image_to_three
from src.viz.skeleton_stabilize import stabilize_skeleton_sequence

# H36M-17 bones
H36M_EDGES = [
    [0, 1], [1, 2], [2, 3],
    [0, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13],
    [8, 14], [14, 15], [15, 16],
]

H36M_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "spine", "thorax", "neck", "head",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]


def _court_mesh() -> dict[str, Any]:
    model = load_court_model()
    lines = [
        ("corner_bl", "corner_br"),
        ("corner_br", "corner_mr"),
        ("corner_mr", "corner_ml"),
        ("corner_ml", "corner_bl"),
        ("paint_bl", "paint_br"),
        ("paint_br", "paint_fr"),
        ("paint_fr", "paint_fl"),
        ("paint_fl", "paint_bl"),
        ("ft_circle_l", "ft_circle_r"),
        ("center_circle_l", "center_circle_r"),
        ("corner_ml", "corner_mr"),
    ]
    pts = {pid: landmark_xyz(model, pid).tolist() for pid in priority_ids(model)}
    segs = []
    for a, b in lines:
        if a in pts and b in pts:
            segs.append({"a": pts[a], "b": pts[b]})
    return {"points": pts, "segments": segs, "standard": model.get("standard")}


def _repair_pseudo_joints(arr: np.ndarray) -> np.ndarray:
    """Fix known export bugs in existing motion.json pseudo3d skeletons."""
    out = arr.astype(np.float64).copy()
    pelvis, thorax, neck, head = out[0], out[8], out[9], out[10]
    torso = float(np.linalg.norm(neck - pelvis))
    if torso < 1e-4:
        torso = 1.0
    head_len = float(np.linalg.norm(head - neck))
    if head_len > 1.5 * torso or head_len < 1e-4:
        up = neck - thorax
        nrm = float(np.linalg.norm(up))
        if nrm < 1e-6:
            up = np.array([0.0, -0.25, 0.0])
        else:
            up = up / nrm * min(0.35 * torso, 0.35)
        out[10] = neck + up
    return out


def _joints_to_three(joints: list[list[float]], mode: str) -> np.ndarray:
    arr = np.asarray(joints, dtype=np.float64)
    if arr.shape != (17, 3):
        return np.full((17, 3), np.nan)
    if mode == "triangulated":
        out = np.zeros_like(arr)
        out[:, 0] = arr[:, 0]
        out[:, 1] = arr[:, 2]
        out[:, 2] = arr[:, 1]
        return out

    arr = _repair_pseudo_joints(arr)
    out = np.zeros_like(arr)
    out[:, 0] = arr[:, 0] * 0.55
    out[:, 1] = (-arr[:, 1]) * 0.55 + 1.0
    out[:, 2] = arr[:, 2] * 0.55 + 4.5
    return out


def _pack_frames(
    joints_seq: np.ndarray,
    meta_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frames_out: list[dict] = []
    for t, meta in enumerate(meta_frames):
        j = joints_seq[t]
        frames_out.append({
            "frame": meta.get("frame", t),
            "t_ms": meta.get("t_ms", t * 33.3),
            "joints": [
                None if not np.all(np.isfinite(p)) else [float(p[0]), float(p[1]), float(p[2])]
                for p in j
            ],
            "conf": meta.get("conf") or [1.0] * 17,
            "action_type": meta.get("action_type"),
            "action_phase": meta.get("action_phase"),
            "clip_id": meta.get("clip_id"),
        })
    return frames_out


def _clips_from_report(report: dict) -> list[dict]:
    clips = []
    for i, c in enumerate(report.get("clips") or [], start=1):
        clips.append({
            "i": i,
            "action_type": c.get("action_type"),
            "release_ms": c.get("release_ms"),
            "start_ms": c.get("start_ms"),
            "end_ms": c.get("end_ms"),
        })
    return clips


def build_group_pose3d_scene(
    group_dir: Path,
    calib_dir: Path | None = None,
    sample_every: int = 2,
    max_frames: int = 1200,
    video_path: Path | None = None,
    video_stride: int = 2,
) -> dict[str, Any]:
    summary = json.loads((group_dir / "summary.json").read_text(encoding="utf-8"))
    motion_path = group_dir / "motion.json"
    report_path = group_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    clips = _clips_from_report(report)

    cameras = load_camera_calibration(calib_dir) if calib_dir else {}
    fps = float(summary.get("fps") or 30.0)

    # --- Preferred: rebuild from cam_03 video (visible court travel) ---
    if video_path is not None and video_path.exists():
        raw = extract_pose_sequence_from_video(
            video_path, stride=video_stride, max_frames=max_frames,
        )
        fps = float(raw["fps"])
        meta = raw["frames_raw"]
        seq = np.array([f["joints"] for f in meta], dtype=np.float64)
        seq = stabilize_skeleton_sequence(seq)
        frames_out = _pack_frames(seq, meta)
        # attach clip labels by time
        for fr in frames_out:
            t = fr["t_ms"]
            for c in clips:
                if c.get("start_ms") is not None and c.get("end_ms") is not None:
                    if c["start_ms"] <= t <= c["end_ms"]:
                        fr["action_type"] = c.get("action_type")
                        fr["clip_id"] = c.get("i")
                        break
        return {
            "group_id": summary.get("group_id"),
            "session_id": summary.get("session_id"),
            "student_id": summary.get("student_id"),
            "mode": "pseudo3d_video",
            "mode_note": (
                "由 cam_03 视频重提 2D 姿态并映射到球场预览坐标（含根平移）；"
                "已做左右肢一致性与时序平滑。非标定三角化，位移为近似。"
            ),
            "fps": fps,
            "n_frames": len(frames_out),
            "joint_names": H36M_NAMES,
            "edges": H36M_EDGES,
            "court": _court_mesh(),
            "clips": clips,
            "calibration_cameras": list(cameras.keys()),
            "frames": frames_out,
            "sources": {
                "video": str(video_path),
                "motion": str(motion_path) if motion_path.exists() else None,
                "viz_2d": str(group_dir / "viz"),
            },
        }

    # --- Fallback: motion.json (often pelvis-pinned) + stabilize ---
    mode = "pseudo3d_export"
    mode_note = (
        "来自 motion.json 的伪3D（可能无根平移）。已做跳变抑制；"
        "要用球场上可见位移请加 --from-video 从 cam_03 重提。"
    )
    records = []
    if motion_path.exists():
        motion = json.loads(motion_path.read_text(encoding="utf-8"))
        records = motion.get("records") or []
        meta = motion.get("metadata") or {}
        fps = float(meta.get("fps") or fps)
        cs = motion.get("coordinate_system") or {}
        if cs.get("space") == "court_world" and len(cameras) >= 2:
            mode = "triangulated"
            mode_note = "球场坐标系真 3D（标定三角化）"
        elif (motion.get("metadata") or {}).get("skeleton_source") == "triangulated":
            mode = "triangulated"
            mode_note = "三角化 3D 骨架"

    meta_frames: list[dict] = []
    seq_list: list[np.ndarray] = []
    for i, rec in enumerate(records):
        if sample_every > 1 and (i % sample_every) != 0:
            continue
        sk = rec.get("skeleton_3d") or {}
        joints = sk.get("joints")
        if not joints or len(joints) < 17:
            continue
        three = _joints_to_three(joints[:17], mode)
        seq_list.append(three)
        meta_frames.append({
            "frame": i,
            "t_ms": round(float(rec.get("timestamp_ms", 0)), 1),
            "conf": sk.get("joint_scores") or [1.0] * 17,
            "action_type": rec.get("action_type"),
            "action_phase": rec.get("action_phase"),
            "clip_id": rec.get("clip_id"),
        })
        if len(seq_list) >= max_frames:
            break

    if seq_list:
        seq = stabilize_skeleton_sequence(np.stack(seq_list, axis=0))
        frames_out = _pack_frames(seq, meta_frames)
    else:
        frames_out = []

    return {
        "group_id": summary.get("group_id"),
        "session_id": summary.get("session_id"),
        "student_id": summary.get("student_id"),
        "mode": mode,
        "mode_note": mode_note,
        "fps": fps,
        "n_frames": len(frames_out),
        "joint_names": H36M_NAMES,
        "edges": H36M_EDGES,
        "court": _court_mesh(),
        "clips": clips,
        "calibration_cameras": list(cameras.keys()),
        "frames": frames_out,
        "sources": {
            "motion": str(motion_path) if motion_path.exists() else None,
            "viz_2d": str(group_dir / "viz"),
        },
    }


# re-export helper used by tests / scripts
__all__ = [
    "build_group_pose3d_scene",
    "H36M_EDGES",
    "H36M_NAMES",
    "h36m_image_to_three",
]
