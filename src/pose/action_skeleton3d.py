"""Triangulate action-clip skeletons from multi-view video + court calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.action.halpe2h36m import wholebody133_to_h36m
from src.cameras.temporal import frame_to_timestamp_ms
from src.pose.triangulate import (
    load_camera_calibration,
    projection_matrix,
    triangulate_skeleton17,
)
from src.viz.pose_from_video import _pick_person
from src.viz.pose3d_scene import H36M_EDGES, H36M_NAMES, _court_mesh
from src.viz.skeleton_stabilize import stabilize_skeleton_sequence

# H36M indices
PELVIS, R_ANKLE, L_ANKLE = 0, 3, 6
R_HIP, L_HIP = 1, 4
R_KNEE, L_KNEE = 2, 5


def _bone_len(xyz: np.ndarray, a: int, b: int) -> float:
    if np.any(~np.isfinite(xyz[a])) or np.any(~np.isfinite(xyz[b])):
        return float("nan")
    return float(np.linalg.norm(xyz[a] - xyz[b]))


def skeleton_plausible(xyz: np.ndarray, conf: np.ndarray, conf_thr: float = 0.2) -> tuple[bool, str]:
    """Reject grossly wrong triangulations."""
    valid = np.isfinite(xyz[:, 0]) & (conf >= conf_thr)
    if int(valid.sum()) < 8:
        return False, "too_few_joints"
    if not (valid[PELVIS] and (valid[R_ANKLE] or valid[L_ANKLE])):
        return False, "missing_pelvis_or_feet"
    # Court bounds (meters) — loose
    # Full-court friendly (drives / breakthrough may leave FT/paint zone)
    if abs(float(xyz[PELVIS, 0])) > 12.0 or float(xyz[PELVIS, 1]) < -6.0 or float(xyz[PELVIS, 1]) > 30.0:
        return False, "pelvis_out_of_court"
    if float(xyz[PELVIS, 2]) < 0.2 or float(xyz[PELVIS, 2]) > 2.5:
        return False, "pelvis_height_bad"
    for a, b, lo, hi in (
        (R_HIP, R_KNEE, 0.25, 0.75),
        (R_KNEE, R_ANKLE, 0.25, 0.75),
        (L_HIP, L_KNEE, 0.25, 0.75),
        (L_KNEE, L_ANKLE, 0.25, 0.75),
    ):
        if valid[a] and valid[b]:
            L = _bone_len(xyz, a, b)
            if not (lo <= L <= hi):
                return False, f"bone_{a}_{b}={L:.2f}"
    # Foot height relative to pelvis (should be below)
    feet = []
    if valid[R_ANKLE]:
        feet.append(float(xyz[R_ANKLE, 2]))
    if valid[L_ANKLE]:
        feet.append(float(xyz[L_ANKLE, 2]))
    if feet and float(xyz[PELVIS, 2]) < max(feet) - 0.05:
        return False, "pelvis_below_feet"
    return True, "ok"


def mean_reproj_error_px(
    xyz: np.ndarray,
    conf: np.ndarray,
    kpts_by_cam: dict[str, np.ndarray],
    cameras: dict[str, dict[str, Any]],
    conf_thr: float = 0.25,
) -> float:
    """Average |proj - observed| over joints/views with valid 3D."""
    errs: list[float] = []
    for j in range(17):
        if not np.isfinite(xyz[j, 0]) or conf[j] < conf_thr:
            continue
        X = xyz[j].reshape(3, 1)
        for cid, k in kpts_by_cam.items():
            if cid not in cameras or k is None or len(k) <= j:
                continue
            c = float(k[j, 2]) if k.shape[1] > 2 else 1.0
            if c < conf_thr:
                continue
            cam = cameras[cid]
            K = cam["K"].copy()
            # projectPoints prefers |fy|; chirality uses signed fy in solve export
            K_use = K.copy()
            R, t = cam["R"], cam["t"]
            D = cam["D"]
            rvec, _ = cv2.Rodrigues(R)
            imgpts, _ = cv2.projectPoints(X.reshape(1, 1, 3), rvec, t.reshape(3, 1), K_use, D)
            u, v = float(imgpts[0, 0, 0]), float(imgpts[0, 0, 1])
            errs.append(float(np.hypot(u - float(k[j, 0]), v - float(k[j, 1]))))
    return float(np.mean(errs)) if errs else 1e9


def ground_by_feet(
    seq: np.ndarray,
    *,
    percentile: float = 20.0,
) -> tuple[np.ndarray, float]:
    """
    Subtract a floor height estimated from ankle Z percentile (standing/contact frames).
    Returns (grounded_seq, floor_z_world_before).
    """
    out = seq.copy()
    foot_z = []
    for fr in out:
        zs = []
        if np.isfinite(fr[R_ANKLE, 2]):
            zs.append(fr[R_ANKLE, 2])
        if np.isfinite(fr[L_ANKLE, 2]):
            zs.append(fr[L_ANKLE, 2])
        if zs:
            foot_z.append(float(np.mean(zs)))
    if not foot_z:
        return out, 0.0
    floor_z = float(np.nanpercentile(np.asarray(foot_z), percentile))
    out[:, :, 2] -= floor_z
    return out, floor_z


def extract_cam_pose_window(
    video_path: Path,
    frame_indices: list[int],
    *,
    score_thr: float = 0.4,
    kpt_thr: float = 0.35,
) -> dict[int, np.ndarray]:
    """
    Extract H36M-17 (x,y,conf) for requested frames. Tracks largest/consistent person.
    Returns {frame_idx: (17,3)}.
    """
    from src.perception.rtmlib_backend import create_rtmlib_perception

    backend = create_rtmlib_perception()
    if backend is None:
        raise RuntimeError("RTMLib/YOLO perception unavailable")

    want = set(int(i) for i in frame_indices)
    if not want:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    out: dict[int, np.ndarray] = {}
    prev_bbox = None
    idx = 0
    max_need = max(want)

    while idx <= max_need:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            dets = backend.detect_persons(frame, score_thr=score_thr)
            # Filter tiny detections (noise)
            dets = [
                d for d in dets
                if (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) > 80 ** 2
            ]
            person = _pick_person(dets, prev_bbox)
            if person is not None:
                prev_bbox = person["bbox"]
                kpts, _ = backend.estimate_pose133(frame, person["bbox"])
                k = np.asarray(kpts, dtype=np.float64)
                # zero low-conf joints
                k[k[:, 2] < kpt_thr, 2] = 0.0
                h36m = wholebody133_to_h36m(k)
                # Require core joints
                core = [PELVIS, R_HIP, L_HIP, R_ANKLE, L_ANKLE]
                if sum(1 for j in core if float(h36m[j, 2]) >= kpt_thr) >= 3:
                    out[idx] = h36m
        idx += 1
    cap.release()
    return out


def estimate_offset_ms_from_wrist(
    anchor: dict[int, np.ndarray],
    other: dict[int, np.ndarray],
    fps_a: float,
    fps_o: float,
    *,
    max_lag_ms: float = 2500.0,
) -> float:
    """
    Align other → anchor by correlating right-wrist image-y (release = local min).
    Returns offset_ms such that local_ms = common_ms + offset.
    """
    if len(anchor) < 5 or len(other) < 5:
        return 0.0
    # Build dense signals on ms grid
    def series(store: dict[int, np.ndarray], fps: float):
        frames = sorted(store.keys())
        t = np.array([frame_to_timestamp_ms(f, fps) for f in frames], dtype=np.float64)
        # wrist = 16 in H36M
        y = np.array([
            float(store[f][16, 1]) if float(store[f][16, 2]) > 0.2 else np.nan
            for f in frames
        ], dtype=np.float64)
        return t, y

    ta, ya = series(anchor, fps_a)
    to, yo = series(other, fps_o)
    # Fill nan with linear interp
    def fill(y):
        n = np.isnan(y)
        if n.all():
            return y
        yi = y.copy()
        idx = np.arange(len(y))
        yi[n] = np.interp(idx[n], idx[~n], y[~n])
        return yi

    ya, yo = fill(ya), fill(yo)
    # Normalize
    ya = (ya - np.nanmean(ya)) / (np.nanstd(ya) + 1e-6)
    yo = (yo - np.nanmean(yo)) / (np.nanstd(yo) + 1e-6)
    # Sample both on common time
    t0 = max(float(ta[0]), float(to[0]))
    t1 = min(float(ta[-1]), float(to[-1]))
    if t1 - t0 < 500:
        return 0.0
    grid = np.arange(t0, t1, 33.0)
    sa = np.interp(grid, ta, ya)
    so = np.interp(grid, to, yo)
    max_lag = int(max_lag_ms / 33.0)
    best_lag, best_score = 0, -1e18
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = sa[-lag:], so[: len(so) + lag]
        elif lag > 0:
            a, b = sa[: len(sa) - lag], so[lag:]
        else:
            a, b = sa, so
        if len(a) < 10:
            continue
        score = float(np.dot(a, b) / len(a))
        if score > best_score:
            best_score = score
            best_lag = lag
    # other is shifted by +lag samples relative to anchor →
    # other_time ≈ anchor_time + lag*33 → offset = other - anchor = lag*33
    return float(best_lag * 33.0)


def triangulate_clip_sequence(
    poses_by_cam: dict[str, dict[int, np.ndarray]],
    fps_by_cam: dict[str, float],
    offsets_ms: dict[str, float],
    cameras: dict[str, dict[str, Any]],
    t0_ms: float,
    t1_ms: float,
    *,
    step_ms: float = 50.0,
    conf_thr: float = 0.3,
    max_reproj_px: float = 90.0,
    anchor: str = "cam_03",
) -> list[dict[str, Any]]:
    """Triangulate on anchor clock [t0,t1]; drop bad frames."""
    frames_out: list[dict[str, Any]] = []
    t = float(t0_ms)
    while t <= t1_ms + 1e-6:
        k17: dict[str, np.ndarray] = {}
        for cam, store in poses_by_cam.items():
            if cam not in cameras or not store:
                continue
            fps = fps_by_cam[cam]
            local_ms = t + float(offsets_ms.get(cam, 0.0))
            # nearest frame
            best_f, best_dt = None, 1e18
            for f in store:
                dt = abs(frame_to_timestamp_ms(f, fps) - local_ms)
                if dt < best_dt:
                    best_dt, best_f = dt, f
            if best_f is None or best_dt > step_ms * 1.5:
                continue
            k17[cam] = store[best_f]
        if len(k17) < 2:
            t += step_ms
            continue
        xyz, conf = triangulate_skeleton17(k17, cameras, conf_thr=conf_thr, min_views=2)
        ok, reason = skeleton_plausible(xyz, conf, conf_thr=0.15)
        if not ok:
            t += step_ms
            continue
        reproj = mean_reproj_error_px(xyz, conf, k17, cameras, conf_thr=conf_thr)
        if reproj > max_reproj_px:
            t += step_ms
            continue
        frames_out.append({
            "t_ms": round(t, 1),
            "joints": xyz.tolist(),
            "conf": conf.tolist(),
            "n_views": len(k17),
            "reproj_px": round(reproj, 1),
            "views": sorted(k17.keys()),
            "reject_reason": None,
        })
        t += step_ms
    return frames_out


def process_group_action_skeletons(
    group_dir: Path,
    *,
    calib_dir: Path | None = None,
    videos_dir: Path | None = None,
    group_id: int | None = None,
    stride: int = 2,
    pad_ms: float = 400.0,
) -> dict[str, Any]:
    """
    End-to-end: extract multi-cam 2D on action windows → sync → triangulate → ground feet.
    """
    from src.config import ROOT

    group_dir = Path(group_dir)
    motion = json.loads((group_dir / "motion.json").read_text(encoding="utf-8"))
    summary = {}
    if (group_dir / "summary.json").exists():
        summary = json.loads((group_dir / "summary.json").read_text(encoding="utf-8"))
    session_id = summary.get("session_id") or (motion.get("metadata") or {}).get("session_id")
    student_id = summary.get("student_id") or (motion["clips"][0].get("student_id") if motion.get("clips") else None)
    gid = group_id or int(str(summary.get("group_id") or group_dir.name).replace("group_", "") or "0")

    videos_dir = Path(videos_dir or (ROOT / "data/test_data_v1"))
    calib_dir = Path(calib_dir or (ROOT / "data/calibration/v2_4cam_zoned"))
    cameras = load_camera_calibration(calib_dir)
    if len(cameras) < 2:
        raise RuntimeError(f"Need ≥2 calibrated cameras in {calib_dir}")

    clips = list(motion.get("clips") or [])
    if not clips:
        raise RuntimeError(f"No clips in {group_dir}/motion.json")

    # Union of action windows (anchor cam_03 clock)
    windows: list[tuple[float, float]] = []
    for c in clips:
        windows.append((float(c["start_ms"]) - pad_ms, float(c["end_ms"]) + pad_ms))
    windows.sort()
    merged: list[list[float]] = []
    for a, b in windows:
        if not merged or a > merged[-1][1] + 100:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    cam_ids = [c for c in ("cam_01", "cam_02", "cam_03") if c in cameras]
    poses_by_cam: dict[str, dict[int, np.ndarray]] = {}
    fps_by_cam: dict[str, float] = {}

    for cam in cam_ids:
        cam_i = int(cam.split("_")[1])
        video = videos_dir / f"{gid}-{cam_i}.mkv"
        if not video.exists():
            video = videos_dir / f"{gid}-{cam_i}.mp4"
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        fps_by_cam[cam] = fps
        # Frame list for this cam: map anchor windows with provisional offset 0,
        # then re-extract is expensive — use wide pad; offset applied at triangulate time
        # so we need frames covering ±max_lag around windows.
        lag = 2500.0
        frame_set: set[int] = set()
        for a, b in merged:
            fa = max(0, int((a - lag) / 1000.0 * fps))
            fb = min(nframes - 1, int((b + lag) / 1000.0 * fps))
            for f in range(fa, fb + 1, max(1, stride)):
                frame_set.add(f)
        print(f"  [{group_dir.name}] extract {cam} frames={len(frame_set)} from {video.name}", flush=True)
        poses_by_cam[cam] = extract_cam_pose_window(video, sorted(frame_set))

    # Sync offsets vs cam_03
    offsets = {c: 0.0 for c in cam_ids}
    anchor = "cam_03" if "cam_03" in poses_by_cam else cam_ids[0]
    for cam in cam_ids:
        if cam == anchor:
            continue
        offsets[cam] = estimate_offset_ms_from_wrist(
            poses_by_cam[anchor], poses_by_cam[cam],
            fps_by_cam[anchor], fps_by_cam[cam],
        )
        print(f"  [{group_dir.name}] offset {cam}={offsets[cam]:.0f} ms", flush=True)

    # Per-clip triangulate
    all_frames: list[dict[str, Any]] = []
    clip_stats: list[dict[str, Any]] = []
    for ci, clip in enumerate(clips):
        seq = triangulate_clip_sequence(
            poses_by_cam, fps_by_cam, offsets, cameras,
            float(clip["start_ms"]), float(clip["end_ms"]),
            step_ms=50.0, anchor=anchor,
        )
        clip_stats.append({
            "clip_id": clip.get("clip_id"),
            "action_type": clip.get("action_type"),
            "start_ms": clip.get("start_ms"),
            "end_ms": clip.get("end_ms"),
            "release_ms": clip.get("release_ms"),
            "n_kept": len(seq),
        })
        for fr in seq:
            fr["clip_id"] = clip.get("clip_id")
            fr["action_type"] = clip.get("action_type")
            fr["clip_index"] = ci
            all_frames.append(fr)

    if not all_frames:
        return {
            "group_id": gid,
            "session_id": session_id,
            "student_id": student_id,
            "status": "empty",
            "offsets_ms": offsets,
            "clip_stats": clip_stats,
            "frames": [],
        }

    # Stabilize + ground by feet
    xyz = np.stack([np.asarray(f["joints"], dtype=np.float64) for f in all_frames], axis=0)
    xyz = stabilize_skeleton_sequence(xyz)
    xyz, floor_z = ground_by_feet(xyz, percentile=20.0)
    for i, fr in enumerate(all_frames):
        fr["joints"] = xyz[i].tolist()
        # foot height after grounding (should be ~0)
        fz = []
        for j in (R_ANKLE, L_ANKLE):
            if np.isfinite(xyz[i, j, 2]):
                fz.append(float(xyz[i, j, 2]))
        fr["foot_z_m"] = round(float(np.mean(fz)), 3) if fz else None
        fr["conf"] = all_frames[i]["conf"]

    # Camera centers for scene
    cam_scene = []
    for cid, cam in cameras.items():
        R, t = cam["R"], cam["t"].reshape(3)
        C = (-R.T @ t).reshape(3)
        forward = R.T @ np.array([0.0, 0.0, 1.0])
        up = R.T @ np.array([0.0, -1.0, 0.0])
        cam_scene.append({
            "id": cid,
            "center": C.tolist(),
            "forward": forward.tolist(),
            "up": up.tolist(),
            "z_below_ground": bool(C[2] < 0),
            "reproj_mean": cam.get("reproj_mean"),
        })

    return {
        "group_id": gid,
        "session_id": session_id,
        "student_id": student_id,
        "status": "ok",
        "mode": "triangulated",
        "mode_note": "多视 DLT + 脚底接地（踝关节 Z 分位）",
        "coordinate_system": {
            "space": "court_world",
            "unit": "meter",
            "origin": "near_baseline_midpoint",
            "axes": "x_right, y_toward_center_line, z_up",
        },
        "floor_z_subtracted_m": floor_z,
        "offsets_ms": offsets,
        "fps": 1000.0 / 50.0,
        "n_frames": len(all_frames),
        "joint_names": H36M_NAMES,
        "edges": H36M_EDGES,
        "court": _court_mesh(),
        "cameras": cam_scene,
        "clips": clips,
        "clip_stats": clip_stats,
        "frames": [
            {
                "frame": i,
                "t_ms": fr["t_ms"],
                "joints": fr["joints"],
                "conf": fr["conf"],
                "foot_z_m": fr.get("foot_z_m"),
                "reproj_px": fr.get("reproj_px"),
                "n_views": fr.get("n_views"),
                "clip_id": fr.get("clip_id"),
                "action_type": fr.get("action_type"),
            }
            for i, fr in enumerate(all_frames)
        ],
        "sources": {
            "calib": str(calib_dir),
            "motion": str(group_dir / "motion.json"),
            "videos": str(videos_dir),
        },
    }
