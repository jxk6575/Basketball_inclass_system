"""Pose2Sim multi-view 3D reconstruction — time-aligned fusion (v2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cameras.registry import get_camera_ids
from src.cameras.temporal import build_per_camera_timelines, collect_student_kpts_at_time
from src.config import data_path, load_yaml
from src.pose.angles import compute_frame_angles


def _triangulate_stub(
    points_per_cam: dict[str, np.ndarray],
    student_id: str,
) -> np.ndarray:
    """
    Stub triangulation: average available 2D lifts to fake 3D.
    Replace with Pose2Sim triangulation when calibration present.
    """
    cams = list(points_per_cam.keys())
    if not cams:
        return np.zeros((133, 3), dtype=np.float32)
    base = points_per_cam[cams[0]].copy()
    if len(cams) > 1:
        for cam in cams[1:]:
            base[:, :2] = (base[:, :2] + points_per_cam[cam][:, :2]) / 2
    base[:, 2] = 1.5 + 0.001 * base[:, 1]
    return base.astype(np.float32)


def _collect_student_time_samples(
    session_id: str,
    student_id: str,
    camera_ids: list[str] | None = None,
    sample_stride_ms: float = 33.0,
) -> list[tuple[float, dict[str, np.ndarray]]]:
    """
    Build time-sampled multi-camera observations (NOT frame-index aligned).
    """
    camera_ids = camera_ids or get_camera_ids()
    timelines = build_per_camera_timelines(session_id, camera_ids)

    # Union of timestamps from anchor-capable cameras
    all_ts: set[float] = set()
    for cam_id, tl in timelines.items():
        for fr in tl.get("frames", []):
            for person in fr.get("persons", []):
                if person.get("student_id") == student_id:
                    all_ts.add(float(fr["timestamp_ms"]))
                    break

    if not all_ts:
        return []

    sorted_ts = sorted(all_ts)
    # optional downsample
    sampled = sorted_ts[:: max(1, int(sample_stride_ms / 33))]

    samples = []
    for ts in sampled:
        pts = collect_student_kpts_at_time(session_id, student_id, ts, camera_ids)
        if len(pts) >= 1:
            samples.append((ts, pts))
    return samples


def _write_trc(path: Path, frames: list[np.ndarray], fps: float = 30.0, timestamps: list[float] | None = None) -> None:
    n_markers = 133
    lines = [
        "PathFileType\t4\t(X/Y/Z)\tpose3d",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{fps}\t{fps}\t{len(frames)}\t{n_markers}\tm\t{fps}\t1\t{len(frames)}",
        "Frame#\tTime\t" + "\t".join(f"P{i}\t" for i in range(1, n_markers + 1)),
    ]
    for i, kpts in enumerate(frames, start=1):
        t = (timestamps[i - 1] / 1000.0) if timestamps else (i - 1) / fps
        coords = []
        for m in range(n_markers):
            x, y, z = kpts[m]
            coords.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        lines.append(f"{i}\t{t:.6f}\t" + "\t".join(coords))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pose3d_session(
    session_id: str,
    camera_ids: list[str] | None = None,
) -> list[str]:
    """
    Fuse per-camera isolated pose2d via timestamp alignment → 3D + angles.
    """
    camera_ids = camera_ids or get_camera_ids()

    student_ids: set[str] = set()
    timelines = build_per_camera_timelines(session_id, camera_ids)
    for tl in timelines.values():
        for fr in tl.get("frames", []):
            for p in fr.get("persons", []):
                if p.get("student_id"):
                    student_ids.add(p["student_id"])

    out3d = data_path("sessions", session_id, "pose3d")
    ang_dir = data_path("sessions", session_id, "angles")
    out3d.mkdir(parents=True, exist_ok=True)
    ang_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    for sid in sorted(student_ids):
        samples = _collect_student_time_samples(session_id, sid, camera_ids)
        if not samples:
            continue

        frames_3d = []
        angle_series = []
        timestamps = []
        for ts, pts_per_cam in samples:
            k3 = _triangulate_stub(pts_per_cam, sid)
            frames_3d.append(k3)
            timestamps.append(ts)
            ang = compute_frame_angles(k3)
            angle_series.append({"timestamp_ms": ts, **ang})

        trc_path = out3d / f"{sid}.trc"
        _write_trc(trc_path, frames_3d, timestamps=timestamps)
        ang_path = ang_dir / f"{sid}.json"
        ang_path.write_text(json.dumps({
            "student_id": sid,
            "session_id": session_id,
            "fusion": "timestamp_aligned",
            "frames": angle_series,
        }, ensure_ascii=False, indent=2))
        processed.append(sid)

    repair_path = data_path("sessions", session_id, "identity_repair.json")
    repair_path.write_text(json.dumps({"repairs": [], "fusion": "timestamp_aligned"}, indent=2))
    return processed


def try_run_pose2sim_cli(session_dir: Path) -> bool:
    try:
        from Pose2Sim import Pose2Sim  # type: ignore
        Pose2Sim.poseEstimation(str(session_dir / "pose2sim_config.toml"))
        Pose2Sim.personAssociation()
        Pose2Sim.synchronizeCams()
        Pose2Sim.triangulation()
        Pose2Sim.filtering()
        return True
    except Exception:
        return False
