#!/usr/bin/env python3
"""Formal v1 batch runner for data/test_data_v1 (8 groups x 4 cameras)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_conda = os.environ.get("CONDA_PREFIX")
if _conda:
    os.environ["LD_LIBRARY_PATH"] = f"{_conda}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

from rtmlib import draw_skeleton  # noqa: E402

from src.action.detect import extract_student_sequence, load_ball_track  # noqa: E402
from src.cameras.registry import (
    camera_runs_pose2d,
    get_action_segment_camera,
    get_camera_ids,
    get_perception_config,
    get_shot_outcome_camera,
)  # noqa: E402
from src.cameras.temporal import frame_to_timestamp_ms, run_temporal_alignment  # noqa: E402
from src.config import data_path, load_models_config, model_path  # noqa: E402
from src.identity.enrollment import EnrollmentGallery  # noqa: E402
from src.identity.perception import _estimate_face_bbox  # noqa: E402
from src.orchestrator.session_pipeline import create_session, register_student  # noqa: E402
from src.output.export import build_group_report, write_session_output  # noqa: E402
from src.perception.camera_pipeline import run_single_camera_perception  # noqa: E402
from src.perception.rtmlib_backend import RTMLibPerception  # noqa: E402
from src.perception.yolo_pose_detector import create_yolo_pose_detector  # noqa: E402
from src.privacy.consent import grant_consent  # noqa: E402
from src.privacy.db import init_db  # noqa: E402
from src.shot.outcome import run_ball_tracking_on_video, run_shot_outcome_session  # noqa: E402
from src.types import ConsentScope, StudentActions  # noqa: E402
from src.utils.video_io import create_video_writer, ffmpeg_available  # noqa: E402
from src.viz.identity_style import (  # noqa: E402
    VizIdentitySticky,
    format_student_label,
    student_color_bgr,
)

PHASE_COLORS = {
    "load": (255, 180, 0),
    "set": (0, 200, 255),
    "release": (0, 80, 255),
    "follow_through": (180, 0, 255),
    "approach": (255, 160, 40),
    "gather": (0, 200, 255),
    "takeoff": (40, 180, 255),
    "finish": (180, 0, 255),
    "action": (60, 220, 120),
    "recover": (200, 120, 255),
    "full": (120, 120, 120),
}

CAM_MAP = {1: "cam_01", 2: "cam_02", 3: "cam_03", 4: "cam_04"}


def discover_groups(data_dir: Path) -> dict[int, dict[str, Path]]:
    groups: dict[int, dict[str, Path]] = {}
    for path in sorted(data_dir.glob("*-*.mkv")):
        stem = path.stem
        if "-" not in stem:
            continue
        g_s, c_s = stem.split("-", 1)
        if not g_s.isdigit() or not c_s.isdigit():
            continue
        g, c = int(g_s), int(c_s)
        if c not in CAM_MAP:
            continue
        groups.setdefault(g, {})[CAM_MAP[c]] = path
    return dict(sorted(groups.items()))


def remux_to_mp4(src: Path, dst: Path) -> Path:
    """Fast remux mkv→mp4 when possible; else re-encode."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1000:
        return dst
    if not ffmpeg_available():
        # OpenCV may still open mkv; copy as-is with .mp4 name won't work — symlink
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        return dst
    # try copy streams
    cmd_copy = [
        "ffmpeg", "-y", "-i", str(src),
        "-c", "copy", "-an", "-movflags", "+faststart", str(dst),
    ]
    r = subprocess.run(cmd_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000:
        return dst
    # re-encode
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dst


def _load_detections_by_frame(session_id: str, cam_id: str) -> dict[int, list[dict]]:
    det_path = data_path("sessions", session_id, "perception", cam_id, "detections.jsonl")
    by_frame: dict[int, list[dict]] = {}
    if not det_path.exists():
        return by_frame
    for line in det_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_frame.setdefault(int(rec["frame"]), []).append(rec)
    return by_frame


def _draw_track_label(
    vis,
    bbox,
    rec: dict,
) -> None:
    """Draw box colored by student_id (display sticky), not track_id."""
    x1, y1, x2, y2 = map(int, bbox[:4])
    tid = rec.get("track_id")
    raw_sid = rec.get("student_id")
    sid = rec.get("display_student_id") or raw_sid
    conf = rec.get("identity_confidence", "?")
    color = student_color_bgr(sid, track_id=tid)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    who = format_student_label(sid) if sid else "?"
    sticky_mark = "*" if sid and raw_sid and sid != raw_sid else ""
    label = f"{who}{sticky_mark} T{tid} [{conf}]"
    cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _find_enrollment_persons(
    video_path: Path,
    *,
    yolo,
    backend,
    enroll_thr: float,
    min_ratio: float,
    max_persons: int = 2,
) -> tuple[list[dict], np.ndarray | None]:
    """
    Scan video for a frame with up to ``max_persons`` large detections.
    Prefers the frame whose top-N bbox area sum is largest (multi-person first).
    """
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(5, int(round(fps * 0.5)))
    max_probes = 250
    indices = list(range(0, max(n, 1), step))
    if len(indices) > max_probes:
        sel = [int(i * (len(indices) - 1) / (max_probes - 1)) for i in range(max_probes)]
        indices = [indices[i] for i in sel]

    best_dets: list[dict] = []
    best_score = -1.0
    best_frame = None
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        if yolo is not None:
            dets = yolo.detect_persons(frame, score_thr=enroll_thr, best_only=False)
        else:
            dets = backend.detect_persons(frame, score_thr=enroll_thr)
        fh, fw = frame.shape[:2]
        frame_area = float(fh * fw)
        dets = [
            d for d in dets
            if (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) / frame_area >= min_ratio
        ]
        if not dets:
            continue
        dets = sorted(
            dets,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )[:max_persons]
        # Prefer more people, then larger total area
        area_sum = sum(
            (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) for d in dets
        )
        score = len(dets) * 1e9 + area_sum
        if score > best_score:
            best_score = score
            best_dets = dets
            best_frame = frame.copy()
    cap.release()
    return best_dets, best_frame


def _find_enrollment_person(
    video_path: Path,
    *,
    yolo,
    backend,
    enroll_thr: float,
    min_ratio: float,
) -> tuple[dict | None, np.ndarray | None]:
    dets, frame = _find_enrollment_persons(
        video_path, yolo=yolo, backend=backend,
        enroll_thr=enroll_thr, min_ratio=min_ratio, max_persons=1,
    )
    return (dets[0] if dets else None), frame


def enroll_from_cam03(
    session_id: str,
    video_path: Path,
    student_id: str,
    fallback_videos: list[Path] | None = None,
    *,
    max_persons: int = 2,
) -> list[str]:
    """
    Enroll up to ``max_persons`` people from cam_03 (or fallbacks).

    Returns student_ids. Primary keeps ``student_id``; extras get ``{student_id}_p2``, …
    """
    perc = get_perception_config()
    enroll_thr = float(perc.get("enrollment_detect_score_threshold", 0.45))
    min_ratio = float(perc.get("min_person_area_ratio", 0.015))
    yolo = create_yolo_pose_detector()
    backend = None
    if yolo is None:
        cfg = load_models_config()
        backend = RTMLibPerception(
            det_model=str(model_path(cfg["detector"]["path"])),
            pose_model=str(model_path(cfg["pose"]["path"])),
            device="cuda",
        )

    candidates = [video_path] + list(fallback_videos or [])
    best_dets: list[dict] = []
    best_frame = None
    used_path = video_path
    for thr, ratio in ((enroll_thr, min_ratio), (0.30, min(0.008, min_ratio))):
        for path in candidates:
            dets, frame = _find_enrollment_persons(
                path, yolo=yolo, backend=backend,
                enroll_thr=thr, min_ratio=ratio, max_persons=max_persons,
            )
            if dets and frame is not None:
                best_dets, best_frame, used_path = dets, frame, path
                break
        if best_dets and best_frame is not None:
            break

    if not best_dets or best_frame is None:
        raise RuntimeError(f"No person for enrollment in {video_path}")

    # Drop near-duplicate boxes (IoU)
    selected: list[dict] = []
    for d in best_dets:
        bb = d["bbox"]
        dup = False
        for s in selected:
            sb = s["bbox"]
            ix1, iy1 = max(bb[0], sb[0]), max(bb[1], sb[1])
            ix2, iy2 = min(bb[2], sb[2]), min(bb[3], sb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a = (bb[2] - bb[0]) * (bb[3] - bb[1])
            b = (sb[2] - sb[0]) * (sb[3] - sb[1])
            if inter / (a + b - inter + 1e-6) > 0.4:
                dup = True
                break
        if not dup:
            selected.append(d)
        if len(selected) >= max_persons:
            break

    gallery = EnrollmentGallery(session_id)
    student_ids: list[str] = []
    for i, det in enumerate(selected):
        sid = student_id if i == 0 else f"{student_id}_p{i + 1}"
        bbox = det["bbox"]
        face_bbox = _estimate_face_bbox(bbox, best_frame.shape)
        gallery.add_face_sample(sid, best_frame, face_bbox)
        gallery.add_body_sample(sid, best_frame, bbox)
        gallery.add_clothing_color_sample(sid, best_frame, bbox)
        gallery.save_meta(sid, f"v1_{sid}", {"source": str(used_path), "person_index": i})
        student_ids.append(sid)
    return student_ids


def _phase_at(clips, frame_idx: int):
    """Return (phase_name, 1-based clip index, action_type, clip) or empties."""
    for i, clip in enumerate(clips, start=1):
        sf = int(getattr(clip, "start_frame", -1))
        ef = int(getattr(clip, "end_frame", -1))
        if sf <= frame_idx <= ef:
            for ph in clip.phases:
                if ph.start <= frame_idx <= ph.end:
                    return ph.name, i, getattr(clip, "action_type", None) or "action", clip
            return "action", i, getattr(clip, "action_type", None) or "action", clip
    return None, None, None, None


def _format_student_tag(clip) -> str:
    parts = list(getattr(clip, "participant_ids", None) or [])
    sid = getattr(clip, "student_id", None)
    if not parts and sid:
        parts = [sid]
    if not parts:
        return ""
    if len(parts) >= 2:
        return f" [{parts[0]}→{parts[1]}]"
    return f" [{parts[0]}]"


def _action_overlay_label(action_type: str | None, clip_i: int, phase: str, clip=None) -> str:
    """Human-readable overlay: e.g. 'pass #2 - action [stu_a→stu_b]'."""
    atype = (action_type or "action").strip() or "action"
    tag = _format_student_tag(clip) if clip is not None else ""
    return f"{atype} #{clip_i} - {phase}{tag}"


def _compose_phases_quad(
    out_path: Path,
    cam_videos: dict[str, Path],
    *,
    anchor: str = "cam_03",
    clips: list | None = None,
    stride: int = 2,
    cell_size: tuple[int, int] = (960, 540),
) -> Path:
    """
    2x2 mosaic of four camera annotated videos, driven by anchor (cam_03) clock.
    Layout:
      [cam_01] [cam_02]
      [cam_03] [cam_04]
    """
    order = ["cam_01", "cam_02", "cam_03", "cam_04"]
    clips = clips or []
    cell_w, cell_h = cell_size
    out_w, out_h = cell_w * 2, cell_h * 2

    caps: dict[str, cv2.VideoCapture] = {}
    meta: dict[str, dict] = {}
    for cam in order:
        path = cam_videos.get(cam)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"missing annotated video for {cam}: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Annotated videos were written at fps/stride; each frame i ↔ source frame i*stride
        # Recover source fps ≈ annotated_fps * stride for timestamp mapping
        src_fps = fps * max(stride, 1)
        caps[cam] = cap
        meta[cam] = {"fps": fps, "src_fps": src_fps, "n": n, "path": str(path)}

    if anchor not in caps:
        raise ValueError(f"anchor {anchor} not in cam videos")

    out_fps = float(meta[anchor]["fps"])
    writer, _codec = create_video_writer(out_path, out_fps, (out_w, out_h))

    # Per-cam sequential readers: advance until timestamp catches anchor
    cursors = {cam: -1 for cam in order}
    last_frames: dict[str, np.ndarray] = {
        cam: np.zeros((cell_h, cell_w, 3), dtype=np.uint8) for cam in order
    }

    def _read_until(cam: str, target_ms: float) -> np.ndarray:
        cap = caps[cam]
        src_fps = meta[cam]["src_fps"]
        n = meta[cam]["n"]
        while cursors[cam] + 1 < n:
            next_i = cursors[cam] + 1
            next_ms = frame_to_timestamp_ms(next_i * stride, src_fps)
            # Peek: if next frame is still before/at target, consume it
            if next_ms <= target_ms + 1e-3 or cursors[cam] < 0:
                ok, fr = cap.read()
                if not ok:
                    break
                cursors[cam] = next_i
                last_frames[cam] = cv2.resize(fr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                # If we've gone past target, stop
                if next_ms >= target_ms:
                    break
            else:
                break
        return last_frames[cam]

    anchor_n = meta[anchor]["n"]
    anchor_src_fps = meta[anchor]["src_fps"]
    positions = {
        "cam_01": (0, 0),
        "cam_02": (cell_w, 0),
        "cam_03": (0, cell_h),
        "cam_04": (cell_w, cell_h),
    }

    for ai in range(anchor_n):
        t_ms = frame_to_timestamp_ms(ai * stride, anchor_src_fps)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for cam in order:
            tile = _read_until(cam, t_ms)
            x0, y0 = positions[cam]
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = tile
            # Corner cam label (overwrite small badge)
            cv2.putText(
                canvas, cam, (x0 + 12, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2,
            )

        # Thin grid lines
        cv2.line(canvas, (cell_w, 0), (cell_w, out_h), (80, 80, 80), 2)
        cv2.line(canvas, (0, cell_h), (out_w, cell_h), (80, 80, 80), 2)

        # Phase overlay from cam_03 action clips (source frame index)
        src_frame = ai * stride
        phase, clip_i, atype, clip = _phase_at(clips, src_frame)
        if phase:
            color = PHASE_COLORS.get(phase, (200, 200, 200))
            cv2.putText(
                canvas, _action_overlay_label(atype, clip_i, phase, clip), (24, out_h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
            )
        cv2.putText(
            canvas, f"{anchor} t={t_ms/1000:.2f}s", (out_w - 280, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )
        writer.write(canvas)

    for cap in caps.values():
        cap.release()
    writer.release()
    return out_path


def _lookup_hold(by_frame: dict[int, list], idx: int, max_gap: int) -> list:
    """Return data for idx, or most recent prior frame within max_gap (anti-flicker)."""
    if idx in by_frame and by_frame[idx]:
        return by_frame[idx]
    for back in range(1, max_gap + 1):
        prev = idx - back
        if prev < 0:
            break
        if prev in by_frame and by_frame[prev]:
            return by_frame[prev]
    return by_frame.get(idx) or []


def _load_ball_track_by_frame(track_path: Path) -> tuple[dict[int, dict], dict[int, dict]]:
    """Returns (ball_by_frame, events_by_frame). Frames keep ball/hoop even if ball missing."""
    ball_by_frame: dict[int, dict] = {}
    events: dict[int, dict] = {}
    if not track_path.exists():
        return ball_by_frame, events
    bdoc = json.loads(track_path.read_text(encoding="utf-8"))
    for e in bdoc.get("events", []):
        events[int(e["frame"])] = e
    for fr in bdoc.get("frames", []):
        ball_by_frame[int(fr["frame"])] = fr
    return ball_by_frame, events


def _draw_ball_hoop(vis, bf: dict | None, *, hoop_upper_half_only: bool = False) -> None:
    if not bf:
        return
    if bf.get("ball"):
        bb = bf["ball"].get("bbox")
        if bb and len(bb) >= 4:
            x, y, bw, bh = bb[:4]
            if bw > 0 and bh > 0:
                cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 2)
        cx, cy = bf["ball"]["center"]
        cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 255), -1)
        cv2.putText(vis, "ball", (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    hoop = bf.get("hoop")
    if hoop and hoop.get("bbox"):
        bb = hoop["bbox"]
        if len(bb) >= 4:
            x, y, bw, bh = bb[:4]
            cy = float(hoop["center"][1]) if hoop.get("center") else (y + bh / 2)
            if hoop_upper_half_only and cy >= 0.5 * vis.shape[0]:
                return
            cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 0, 255), 2)
            cv2.putText(vis, "hoop", (int(x), max(20, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def _ball_track_path(session_id: str, cam_id: str, shot_cam: str) -> Path:
    out = data_path("sessions", session_id, "shot_outcomes")
    if cam_id == shot_cam:
        return out / "ball_track.json"
    return out / f"ball_track_{cam_id}.json"


def render_group_visualizations(
    group_dir: Path,
    session_id: str,
    videos: dict[str, Path],
    student_id: str,
    stride: int = 1,
    hold_gap: int | None = None,
    student_ids: list[str] | None = None,
) -> dict[str, str]:
    out_viz = group_dir / "viz"
    out_viz.mkdir(parents=True, exist_ok=True)
    keyframes = group_dir / "keyframes"
    keyframes.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    # Load actions from all enrolled students
    clips = []
    sids = student_ids or [student_id]
    act_dir = data_path("sessions", session_id, "actions")
    for sid in sids:
        act_path = act_dir / f"{sid}.json"
        if not act_path.exists():
            continue
        doc = StudentActions.model_validate_json(act_path.read_text(encoding="utf-8"))
        clips.extend(doc.clips)
    clips.sort(key=lambda c: (c.start_frame, c.end_frame))

    perc = get_perception_config()
    kpt_thr = float(perc.get("skeleton_draw_threshold", 0.45))
    # Cover assist stride gaps (default 4) when viz uses smaller stride
    if hold_gap is None:
        hold_gap = max(stride * 2, int(perc.get("assist_camera_stride", 4)))

    anchor = get_action_segment_camera()
    shot_cam = get_shot_outcome_camera()

    for cam_id, video_path in videos.items():
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_name = f"{cam_id}_annotated.mp4"
        draw_pose = camera_runs_pose2d(cam_id)
        if cam_id == shot_cam:
            out_name = f"{cam_id}_ball.mp4"
            draw_pose = False
        out_path = out_viz / out_name
        writer, codec = create_video_writer(out_path, fps / max(stride, 1), (w, h))

        # pose + tracking/ReID overlays
        pose_by_frame: dict[int, list] = {}
        det_by_frame: dict[int, list[dict]] = {}
        if draw_pose:
            pose_path = data_path("sessions", session_id, "perception", cam_id, "pose2d.json")
            if pose_path.exists():
                pdoc = json.loads(pose_path.read_text(encoding="utf-8"))
                for fr in pdoc.get("frames", []):
                    pose_by_frame[int(fr["frame"])] = fr.get("persons") or []
            det_by_frame = _load_detections_by_frame(session_id, cam_id)

        # Per-camera ball/hoop track (cam_01~04)
        ball_by_frame, ball_events = _load_ball_track_by_frame(
            _ball_track_path(session_id, cam_id, shot_cam)
        )

        saved_kf: set[str] = set()
        last_persons: list = []
        last_tracks: list = []
        last_ball: dict | None = None
        id_sticky = VizIdentitySticky(iou_thr=0.45, sticky_frames=max(8, 16 // max(stride, 1)))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue
            vis = frame.copy()

            if draw_pose:
                persons = _lookup_hold(pose_by_frame, idx, hold_gap)
                track_recs = _lookup_hold(det_by_frame, idx, hold_gap)
                if track_recs:
                    last_tracks = track_recs
                else:
                    track_recs = last_tracks

                track_recs = id_sticky.resolve(list(track_recs or []))
                for rec in track_recs:
                    _draw_track_label(vis, rec["bbox"], rec)

                if not persons:
                    persons = last_persons
                drawn_persons: list = []
                for p in persons:
                    k = np.asarray(p["keypoints"], dtype=np.float32)
                    if k.ndim == 2 and k.shape[0] >= 17:
                        from src.pose.skeleton_quality import (
                            clamp_keypoints_to_bbox,
                            skeleton_plausible_2d,
                        )
                        pbbox = None
                        psid = p.get("student_id")
                        for rec in track_recs:
                            dsid = rec.get("display_student_id") or rec.get("student_id")
                            if dsid and psid and dsid == psid:
                                pbbox = rec.get("bbox")
                                break
                            if rec.get("student_id") and rec.get("student_id") == psid:
                                pbbox = rec.get("bbox")
                                break
                        if pbbox is None and track_recs:
                            pbbox = track_recs[0].get("bbox")
                        if pbbox is not None:
                            k = clamp_keypoints_to_bbox(k, pbbox, margin=0.12)
                        ok_skel, _ = skeleton_plausible_2d(
                            k, conf_thr=kpt_thr, bbox=pbbox,
                        )
                        if not ok_skel:
                            continue
                        sc = k[:, 2] if k.shape[1] >= 3 else np.ones(k.shape[0])
                        draw_skeleton(vis, k[None, :, :2], sc[None, :], kpt_thr=kpt_thr, radius=2, line_width=2)
                        drawn_persons.append(p)
                # Only hold plausible skeletons (avoid sticky pole/edge artifacts)
                if drawn_persons:
                    last_persons = drawn_persons
                elif persons and persons is not last_persons:
                    # Fresh pose existed but all rejected → clear sticky hold
                    last_persons = []

            # Ball / hoop on all cameras
            bf = ball_by_frame.get(idx)
            if bf and (bf.get("ball") or bf.get("hoop")):
                last_ball = bf
            elif last_ball is not None:
                # hold last ball/hoop briefly across stride gaps
                bf = last_ball
            _draw_ball_hoop(vis, bf, hoop_upper_half_only=(cam_id != shot_cam))

            if cam_id == shot_cam:
                ev = ball_events.get(idx)
                if ev:
                    label = "MADE" if ev.get("made") else "MISS"
                    color = (0, 255, 0) if ev.get("made") else (0, 140, 255)
                    cv2.putText(vis, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

            if cam_id == anchor and clips:
                phase, clip_i, atype, clip = _phase_at(clips, idx)
                if phase:
                    color = PHASE_COLORS.get(phase, (200, 200, 200))
                    label = _action_overlay_label(atype, clip_i, phase, clip)
                    cv2.putText(
                        vis, label, (24, h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    )
                    key = f"{atype}_{clip_i}_{phase}"
                    if key not in saved_kf:
                        cv2.imwrite(str(keyframes / f"{key}.jpg"), vis)
                        saved_kf.add(key)

            cv2.putText(vis, f"{cam_id} f{idx}", (w - 180, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            writer.write(vis)
            idx += 1

        cap.release()
        writer.release()
        outputs[cam_id] = str(out_path)

    # phases.mp4 = 2x2 mosaic synced to action-segment camera (cam_03) clock
    if all(c in outputs for c in ("cam_01", "cam_02", "cam_03", "cam_04")):
        phases_path = out_viz / "phases.mp4"
        print(f"  compose phases quad → {phases_path}")
        _compose_phases_quad(
            phases_path,
            {c: Path(outputs[c]) for c in ("cam_01", "cam_02", "cam_03", "cam_04")},
            anchor=anchor,
            clips=clips,
            stride=stride,
        )
        outputs["phases"] = str(phases_path)

    return outputs


def process_group(
    group_id: int,
    videos: dict[str, Path],
    out_root: Path,
    stride: int = 2,
    skip_viz: bool = False,
    fast: bool = False,
    shot_ball_only: bool = False,
) -> dict:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    group_name = f"group_{group_id:02d}"
    group_dir = out_root / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    perc = get_perception_config()
    anchor = get_action_segment_camera()
    assist_stride = int(perc.get("assist_camera_stride", 4)) if fast else stride
    shot_scale = float(perc.get("shot_camera_process_scale", 0.5))

    init_db()
    session_id = create_session("v1_testset", metadata={"group_id": group_id})
    student_id = f"stu_g{group_id:02d}"

    raw_dir = data_path("sessions", session_id, "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Path] = {}
    for cam_id, src in videos.items():
        dst = raw_dir / f"{cam_id}.mp4"
        print(f"  [{group_name}] prepare {cam_id} <- {src.name}")
        prepared[cam_id] = remux_to_mp4(src, dst)
    timings["prepare"] = time.perf_counter() - t0

    # enroll up to 2 people (pass needs two IDs)
    t_enroll = time.perf_counter()
    enroll_cam = get_action_segment_camera()
    if enroll_cam not in prepared:
        enroll_cam = next(iter(prepared))
    print(f"  [{group_name}] enroll from {enroll_cam} (max 2 persons)")
    fallback = [prepared[c] for c in ("cam_01", "cam_02", "cam_04") if c in prepared and c != enroll_cam]
    student_ids = enroll_from_cam03(
        session_id, prepared[enroll_cam], student_id, fallback_videos=fallback, max_persons=2,
    )
    for sid in student_ids:
        register_student(sid, f"Student {sid}", class_id="v1_testset")
        grant_consent(sid, session_id, [ConsentScope.VIDEO, ConsentScope.FACE, ConsentScope.REPORT])
    print(f"  [{group_name}] enrolled: {student_ids}")
    timings["enroll"] = time.perf_counter() - t_enroll

    # perception on pose cameras only (cam_04 = ball/hoop only)
    t_pose = time.perf_counter()
    for cam_id, path in prepared.items():
        if not camera_runs_pose2d(cam_id):
            print(f"  [{group_name}] skip pose perception {cam_id} (ball-only)")
            continue
        cam_stride = stride if cam_id == anchor else assist_stride
        print(f"  [{group_name}] perception {cam_id} (stride={cam_stride})")
        run_single_camera_perception(
            session_id, cam_id, path,
            student_ids_filter=student_ids,
            stride=cam_stride,
        )
    timings["perception"] = time.perf_counter() - t_pose

    # Ball/hoop: cam_04 always high-res for make/miss; cam_01–03 always tracked.
    # full mode: scale=1.0 on all cams; realtime/fast: cheaper scale on pose cams.
    shot_cam = get_shot_outcome_camera()
    t_ball = time.perf_counter()
    for cam_id, path in prepared.items():
        track_out = _ball_track_path(session_id, cam_id, shot_cam)
        if cam_id == shot_cam:
            ball_stride, ball_scale = 1, 1.0
        elif fast or shot_ball_only:
            ball_stride = max(stride, assist_stride)
            ball_scale = shot_scale  # e.g. 0.75
        else:
            # full mode: max resolution ball detection on pose cams
            ball_stride = stride
            ball_scale = 1.0
        print(f"  [{group_name}] ball track {cam_id} (scale={ball_scale}, stride={ball_stride})")
        run_ball_tracking_on_video(
            path, out_json=track_out, stride=ball_stride, process_scale=ball_scale,
            hoop_upper_half_only=(cam_id != shot_cam),
        )
    timings["ball_track"] = time.perf_counter() - t_ball

    # Event-based temporal sync (needs pose2d + optional cam_04 ball segments)
    t_align = time.perf_counter()
    print(f"  [{group_name}] temporal align (event_anchor)")
    run_temporal_alignment(
        session_id, list(prepared.keys()), student_ids=student_ids, use_events=True,
    )
    timings["align"] = time.perf_counter() - t_align

    # action — 始终自动判别动作类型（不按组号/预设注入）
    t_action = time.perf_counter()
    print(f"  [{group_name}] action (auto-classify) students={student_ids}")
    from src.action.pipeline import run_action_session_auto
    done = run_action_session_auto(session_id, student_ids)
    print(f"  [{group_name}] action students done={done}")

    timings["action"] = time.perf_counter() - t_action

    t_shot = time.perf_counter()
    print(f"  [{group_name}] shot outcome")
    run_shot_outcome_session(session_id)
    timings["shot_outcome"] = time.perf_counter() - t_shot

    t_export = time.perf_counter()
    print(f"  [{group_name}] export JSON")
    motion_path = group_dir / "motion.json"
    motion = write_session_output(
        session_id, motion_path, group_id=group_name, sample_stride=1,
    )
    report = build_group_report(session_id, motion, group_name)
    report_path = group_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    timings["export"] = time.perf_counter() - t_export

    # Triangulated 3D skeletons for dashboard joint angles (not pseudo-2D)
    t_skel = time.perf_counter()
    print(f"  [{group_name}] skeleton3d triangulate")
    try:
        from src.pose.action_skeleton3d import process_group_action_skeletons
        from scripts.extract_action_skeletons_3d import write_viewer
        scene = process_group_action_skeletons(
            group_dir,
            group_id=group_id,
            stride=max(2, stride),
        )
        skel_path = group_dir / "skeleton3d_triangulated.json"
        skel_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
        if scene.get("frames"):
            write_viewer(scene, group_dir / "skeleton3d_court_viewer.html")
            print(f"  [{group_name}] skeleton3d frames={scene.get('n_frames')} status={scene.get('status')}")
        else:
            print(f"  [{group_name}] skeleton3d empty status={scene.get('status')}")
    except Exception as e:
        print(f"  [{group_name}] skeleton3d failed: {e}")
    timings["skeleton3d"] = time.perf_counter() - t_skel

    viz_paths = {}
    if not skip_viz:
        t_viz = time.perf_counter()
        print(f"  [{group_name}] visualize")
        viz_paths = render_group_visualizations(
            group_dir, session_id, prepared, student_id, stride=stride,
            student_ids=student_ids,
        )
        timings["viz"] = time.perf_counter() - t_viz

    # Dashboard after 3D + viz so angle_source can be triangulated_3d
    t_dash = time.perf_counter()
    try:
        from scripts.build_group_dashboard import build_dashboard
        build_dashboard(group_dir)
        print(f"  [{group_name}] dashboard refreshed")
    except Exception as e:
        print(f"  [{group_name}] dashboard failed: {e}")
    timings["dashboard"] = time.perf_counter() - t_dash

    timings["total"] = time.perf_counter() - t0
    print(f"  [{group_name}] timing(s): " + ", ".join(f"{k}={v:.1f}" for k, v in timings.items()))

    # Dominant action type from auto-classified clips (not a pre-known label)
    act_types = [c.get("action_type") for c in (report.get("clips") or []) if c.get("action_type")]
    dominant = max(set(act_types), key=act_types.count) if act_types else "unknown"
    summary = {
        "group_id": group_name,
        "session_id": session_id,
        "student_id": student_id,
        "student_ids": student_ids,
        "action_type": dominant,
        "action_type_source": "auto_classify",
        "clip_count": report["clip_count"],
        "shot_stats": report["shot_stats"],
        "record_count": report["record_count"],
        "timings_sec": {k: round(v, 2) for k, v in timings.items()},
        "fast_mode": fast,
        "shot_ball_only": shot_ball_only,
        "mode": "realtime" if (fast and shot_ball_only) else ("full" if not fast else "custom"),
        "outputs": {
            "motion": str(motion_path),
            "report": str(report_path),
            "viz": viz_paths,
        },
    }
    (group_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def resolve_run_mode(args: argparse.Namespace) -> tuple[bool, bool, bool, str]:
    """
    Two primary modes:

    - realtime: 快速精简 — assist 跳帧 + pose 机位降本球轨，默认不渲 viz
      （课堂近实时 / ≤10s 教师反馈路径）
    - full: 离线全量 — 统一 stride、全分辨率球轨、默认渲 viz
      （归档 / 报告 / 可视化）

    Legacy ``--fast`` / ``--shot-ball-only`` / ``--skip-viz`` still work when
    ``--mode`` is omitted.
    """
    mode = getattr(args, "mode", None)
    if mode == "realtime":
        fast, shot_ball_only = True, True
        skip_viz = not bool(getattr(args, "with_viz", False))
        if getattr(args, "skip_viz", False):
            skip_viz = True
        return fast, shot_ball_only, skip_viz, "realtime"
    if mode == "full":
        fast, shot_ball_only = False, False
        skip_viz = bool(getattr(args, "skip_viz", False))
        return fast, shot_ball_only, skip_viz, "full"
    # legacy flag combo
    fast = bool(getattr(args, "fast", False))
    shot_ball_only = bool(getattr(args, "shot_ball_only", False))
    skip_viz = bool(getattr(args, "skip_viz", False))
    label = "legacy"
    if fast and shot_ball_only:
        label = "realtime_legacy"
    elif not fast and not shot_ball_only:
        label = "full_legacy"
    return fast, shot_ball_only, skip_viz, label


def main():
    parser = argparse.ArgumentParser(
        description="Formal v1 test_data_v1 batch runner "
        "(modes: realtime=快速精简, full=离线全量)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "test_data_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "outputs" / "v1",
    )
    parser.add_argument("--groups", type=str, default="all", help="e.g. 1,2,3 or all")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for anchor cam / ball / viz")
    parser.add_argument(
        "--mode",
        choices=["realtime", "full"],
        default=None,
        help="realtime=快速精简（fast+降本球轨，默认跳过viz）；"
             "full=离线全量（全分辨率球轨+默认渲viz）",
    )
    parser.add_argument("--fast", action="store_true",
                        help="[legacy] cam_01/02 用 assist_camera_stride；非 shot cam 球检半分辨率")
    parser.add_argument(
        "--shot-ball-only",
        action="store_true",
        help="[legacy] pose 机位球轨用半分辨率+跳帧加速（仍跑 cam_01–03，传球需要）；cam_04 仍全分辨率",
    )
    parser.add_argument("--skip-viz", action="store_true", help="跳过 viz 渲染")
    parser.add_argument(
        "--with-viz",
        action="store_true",
        help="realtime 模式下仍渲 viz（覆盖默认 skip）",
    )
    parser.add_argument(
        "--rerender-viz",
        action="store_true",
        help="仅用已有 session 重渲 viz（不重跑感知/动作）",
    )
    args = parser.parse_args()

    groups = discover_groups(args.data_dir)
    if not groups:
        raise SystemExit(f"No groups found in {args.data_dir}")

    if args.groups != "all":
        wanted = {int(x.strip()) for x in args.groups.split(",") if x.strip()}
        groups = {g: v for g, v in groups.items() if g in wanted}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fast, shot_ball_only, skip_viz, mode_label = resolve_run_mode(args)
    print(f"Groups: {sorted(groups.keys())}")
    print(f"Output: {args.out_dir}")
    print(f"Mode: {mode_label} (fast={fast}, shot_ball_only={shot_ball_only}, skip_viz={skip_viz})")

    if args.rerender_viz:
        results = []
        for gid in sorted(groups.keys()):
            gdir = args.out_dir / f"group_{gid:02d}"
            sp = gdir / "summary.json"
            if not sp.exists():
                print(f"SKIP group {gid}: no summary.json")
                continue
            summary = json.loads(sp.read_text(encoding="utf-8"))
            sid = summary["session_id"]
            stu = summary["student_id"]
            stus = summary.get("student_ids") or [stu]
            prepared = {
                cam: data_path("sessions", sid, "raw", f"{cam}.mp4")
                for cam in ("cam_01", "cam_02", "cam_03", "cam_04")
            }
            missing = [c for c, p in prepared.items() if not p.exists()]
            if missing:
                print(f"SKIP group {gid}: missing raw {missing}")
                continue
            print(f"\n=== Re-render viz group {gid} ===")
            viz_paths = render_group_visualizations(
                gdir, sid, prepared, stu, stride=args.stride, student_ids=stus,
            )
            summary.setdefault("outputs", {})["viz"] = viz_paths
            summary["mode"] = summary.get("mode") or mode_label
            sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(summary)
            print(f"OK group {gid}: viz={list(viz_paths.keys())}")
        print(f"\nRe-rendered viz for {len(results)} groups")
        return

    results = []
    for gid, cams in groups.items():
        print(f"\n=== Processing group {gid} ({len(cams)} cams) ===")
        try:
            summary = process_group(
                gid, cams, args.out_dir,
                stride=args.stride, skip_viz=skip_viz, fast=fast,
                shot_ball_only=shot_ball_only,
            )
            summary["mode"] = mode_label
            (args.out_dir / f"group_{gid:02d}" / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            results.append(summary)
            print(f"OK group {gid}: clips={summary['clip_count']} shots={summary['shot_stats']}")
        except Exception as e:
            err = {"group_id": f"group_{gid:02d}", "status": "failed", "error": str(e)}
            results.append(err)
            print(f"FAIL group {gid}: {e}")
            import traceback
            traceback.print_exc()

    # Merge with existing group summaries so partial runs keep a full manifest
    by_id: dict[str, dict] = {}
    for gdir in sorted(args.out_dir.glob("group_*")):
        sp = gdir / "summary.json"
        if sp.exists():
            try:
                by_id[gdir.name] = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                pass
    for r in results:
        gid = r.get("group_id")
        if gid:
            by_id[gid] = r
    merged = [by_id[k] for k in sorted(by_id.keys())]

    manifest = {
        "dataset": str(args.data_dir),
        "output": str(args.out_dir),
        "stride": args.stride,
        "mode": mode_label,
        "fast": fast,
        "shot_ball_only": shot_ball_only,
        "skip_viz": skip_viz,
        "groups": merged,
        "totals": {
            "groups_ok": sum(1 for r in merged if "error" not in r),
            "groups_failed": sum(1 for r in merged if "error" in r),
            "clips": sum(r.get("clip_count", 0) for r in merged if "error" not in r),
            "makes": sum(r.get("shot_stats", {}).get("makes", 0) for r in merged if "error" not in r),
            "attempts": sum(r.get("shot_stats", {}).get("attempts", 0) for r in merged if "error" not in r),
        },
    }
    man_path = args.out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest: {man_path}")
    print(json.dumps(manifest["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
