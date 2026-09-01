"""Per-camera perception: detect, track, pose2d export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.cameras.temporal import frame_to_timestamp_ms
from src.cameras.registry import camera_runs_pose2d, get_camera, get_perception_config
from src.config import data_path, load_yaml
from src.identity.embedders import create_body_embedder, create_face_embedder
from src.identity.enrollment import EnrollmentGallery
from src.identity.tracker import FaceBodyTracker
from src.privacy.consent import has_consent
from src.types import ConsentScope

_rtmlib_backend = None
_yolo_pose_backend = None


def _get_yolo_pose():
    global _yolo_pose_backend
    if _yolo_pose_backend is None:
        try:
            from src.perception.yolo_pose_detector import create_yolo_pose_detector
            _yolo_pose_backend = create_yolo_pose_detector()
        except Exception:
            _yolo_pose_backend = False
    return _yolo_pose_backend if _yolo_pose_backend is not False else None


def _get_rtmlib():
    global _rtmlib_backend
    if _rtmlib_backend is None:
        try:
            from src.perception.rtmlib_backend import create_rtmlib_perception
            _rtmlib_backend = create_rtmlib_perception()
        except Exception:
            _rtmlib_backend = False
    return _rtmlib_backend if _rtmlib_backend is not False else None


# COCO-WholeBody 133 keypoint names (subset for export)
POSE_NAMES_133 = [f"kpt_{i}" for i in range(133)]


def _estimate_face_bbox(body_bbox: list[float], frame_shape: tuple) -> list[float]:
    x1, y1, x2, y2 = body_bbox
    h = y2 - y1
    fx1 = max(0, x1 + (x2 - x1) * 0.25)
    fy1 = max(0, y1)
    fx2 = min(frame_shape[1], x2 - (x2 - x1) * 0.25)
    fy2 = min(frame_shape[0], y1 + h * 0.35)
    return [fx1, fy1, fx2, fy2]


def _stub_person_detections(frame: np.ndarray) -> list[list[float]]:
    """Placeholder detector; swap with RTMDet/YOLO."""
    h, w = frame.shape[:2]
    return [[w * 0.35, h * 0.15, w * 0.65, h * 0.92]]


def _stub_pose133(body_bbox: list[float]) -> np.ndarray:
    """Generate plausible 133x3 keypoints from bbox for pipeline testing."""
    x1, y1, x2, y2 = body_bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    kpts = np.zeros((133, 3), dtype=np.float32)
    # body 17 — minimal skeleton
    joints = {
        0: (cx, y1 + 0.08 * h),
        5: (cx - 0.15 * w, y1 + 0.22 * h),
        6: (cx + 0.15 * w, y1 + 0.22 * h),
        7: (cx - 0.2 * w, y1 + 0.4 * h),
        8: (cx + 0.2 * w, y1 + 0.4 * h),
        9: (cx - 0.22 * w, y1 + 0.55 * h),
        10: (cx + 0.22 * w, y1 + 0.55 * h),
        11: (cx - 0.1 * w, y1 + 0.5 * h),
        12: (cx + 0.1 * w, y1 + 0.5 * h),
        13: (cx - 0.1 * w, y1 + 0.75 * h),
        14: (cx + 0.1 * w, y1 + 0.75 * h),
        15: (cx - 0.1 * w, y2 - 0.02 * h),
        16: (cx + 0.1 * w, y2 - 0.02 * h),
    }
    for idx, (px, py) in joints.items():
        kpts[idx] = [px, py, 0.9]
    # right hand 112-132 simplified
    for i, off in enumerate(np.linspace(0, 0.08 * w, 21)):
        kpts[112 + i] = [cx + 0.22 * w + off, y1 + 0.55 * h + off * 0.5, 0.7]
    return kpts


def _identity_cfg() -> dict:
    return load_yaml("cameras.yaml").get("identity", {})


def _perception_cfg() -> dict:
    return get_perception_config()


def _bbox_area_ratio(bbox: list[float], frame_shape: tuple) -> float:
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    fh, fw = frame_shape[:2]
    return area / max(float(fh * fw), 1.0)


def _filter_person_detections(detections: list[dict], frame_shape: tuple) -> list[dict]:
    cfg = _perception_cfg()
    min_ratio = float(cfg.get("min_person_area_ratio", 0.015))
    return [
        d for d in detections
        if _bbox_area_ratio(d["bbox"], frame_shape) >= min_ratio
    ]


def _person_detect_score_threshold() -> float:
    return float(_perception_cfg().get("person_detect_score_threshold", 0.55))


def _person_kpt_score_threshold() -> float:
    return float(_perception_cfg().get("person_kpt_score_threshold", 0.45))


def _apply_kpt_threshold(kpts: np.ndarray, kpt_thr: float) -> np.ndarray:
    out = kpts.copy()
    if out.shape[1] >= 3:
        out[out[:, 2] < kpt_thr, 2] = 0.0
    return out


def _pose_has_enough_keypoints(kpts: np.ndarray, kpt_thr: float) -> bool:
    cfg = _perception_cfg()
    min_kpts = int(cfg.get("min_valid_keypoints", 8))
    if kpts.ndim != 2 or kpts.shape[1] < 3:
        return False
    return int(np.sum(kpts[:, 2] >= kpt_thr)) >= min_kpts


def _iou_bbox(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-8)


def _match_yolo_keypoints(
    track_bbox: list[float],
    detections: list[dict],
    *,
    min_iou: float = 0.35,
) -> np.ndarray | None:
    """Pick COCO-17 keypoints from the detection overlapping this track."""
    best_iou, best = 0.0, None
    for d in detections:
        k = d.get("keypoints")
        if k is None:
            continue
        iou = _iou_bbox(track_bbox, d["bbox"])
        if iou > best_iou:
            best_iou, best = iou, k
    if best is None or best_iou < min_iou:
        return None
    return np.asarray(best, dtype=np.float32)


def _accept_pose133(
    kpts: np.ndarray,
    bbox: list[float],
    kpt_thr: float,
    *,
    min_joints: int | None = None,
) -> tuple[np.ndarray | None, str]:
    """Clamp + threshold + plausibility; return cleaned kpts or None."""
    from src.pose.skeleton_quality import clamp_keypoints_to_bbox, skeleton_plausible_2d

    k = clamp_keypoints_to_bbox(kpts, bbox, margin=0.25)
    k = _apply_kpt_threshold(k, kpt_thr)
    if min_joints is None:
        if not _pose_has_enough_keypoints(k, kpt_thr):
            return None, "too_few_kpts"
        mj = int(_perception_cfg().get("min_valid_keypoints", 8))
    else:
        if int(np.sum(k[:, 2] >= kpt_thr)) < min_joints:
            return None, "too_few_kpts"
        mj = min_joints
    # Wider bbox margin / arm span for shooting (arms above head leave bbox)
    ok, reason = skeleton_plausible_2d(
        k, conf_thr=kpt_thr, min_joints=mj, bbox=bbox, bbox_margin=0.28,
    )
    if not ok:
        return None, reason
    return k, "ok"


def _estimate_pose_with_fallback(
    frame: np.ndarray,
    bbox: list[float],
    detections: list[dict],
    kpt_thr: float,
) -> tuple[np.ndarray | None, str]:
    """
    Prefer RTMW-133; if quality fails, fall back to YOLO COCO-17 expanded to 133.
    Returns (kpts133, pose_source) or (None, reason).
    """
    rtmlib = _get_rtmlib()
    if rtmlib is not None:
        try:
            kpts, _ = rtmlib.estimate_pose133(frame, bbox)
            accepted, reason = _accept_pose133(kpts, bbox, kpt_thr)
            if accepted is not None:
                return accepted, "rtmw"
        except Exception:
            reason = "rtmw_error"
    else:
        reason = "no_rtmw"

    # YOLO-Pose fallback (reuse detection keypoints — no extra inference)
    yolo_k17 = _match_yolo_keypoints(bbox, detections)
    if yolo_k17 is not None:
        from src.action.halpe2h36m import coco17_to_wholebody133
        k133 = coco17_to_wholebody133(yolo_k17)
        # Slightly looser joint count for recall (classroom occlusion)
        fb_thr = max(0.20, float(kpt_thr) - 0.05)
        fb_min = max(6, int(_perception_cfg().get("min_valid_keypoints", 8)) - 2)
        accepted, fb_reason = _accept_pose133(
            k133, bbox, fb_thr, min_joints=fb_min,
        )
        if accepted is not None:
            return accepted, "yolo_pose"
        reason = f"yolo_reject:{fb_reason}"
    else:
        reason = f"{reason}|no_yolo_kpts"

    # Last resort stub only when neither backend produced usable pose
    if rtmlib is None and yolo_k17 is None:
        accepted, _ = _accept_pose133(_stub_pose133(bbox), bbox, kpt_thr)
        if accepted is not None:
            return accepted, "stub"
    return None, reason


def _detect_persons(frame: np.ndarray, det_score_thr: float) -> list[dict]:
    """Person detections with bbox + score; prefer YOLO-Pose human validation."""
    cfg = _perception_cfg()
    backend = cfg.get("person_detector", "yolo_pose")

    if backend == "yolo_pose":
        yolo = _get_yolo_pose()
        if yolo is not None:
            detections = yolo.detect_persons(frame, score_thr=det_score_thr)
            return _filter_person_detections(detections, frame.shape)

    rtmlib = _get_rtmlib()
    if rtmlib is not None:
        detections = rtmlib.detect_persons(frame, score_thr=det_score_thr)
        return _filter_person_detections(detections, frame.shape)

    return [{"bbox": b, "score": 1.0} for b in _stub_person_detections(frame)]


def _write_empty_pose2d(
    out_dir: Path,
    camera_id: str,
    session_id: str,
    video_path: Path,
    stride: int,
    reason: str,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or float(get_camera(camera_id).get("fps", 30))
    cap.release()
    pose_path = out_dir / "pose2d.json"
    pose_path.write_text(json.dumps({
        "camera_id": camera_id,
        "session_id": session_id,
        "fps": fps,
        "stride": stride,
        "processing": "skipped",
        "skip_reason": reason,
        "frames": [],
    }, ensure_ascii=False, indent=2))
    (out_dir / "detections.jsonl").write_text("", encoding="utf-8")
    return out_dir


def compute_alpha(face_bbox: list[float] | None, face_score: float = 1.0) -> float:
    cfg = _identity_cfg()
    th = cfg.get("face_score_threshold", 0.72)
    low = float(cfg.get("face_alpha_low", 0.0))
    high = float(cfg.get("face_alpha_high", 0.20))
    if face_bbox is None:
        return low
    if face_score >= th:
        return high
    # Weak face: keep body-dominant blend
    return min(high, max(low, high * 0.5))


def _create_tracker(gallery: EnrollmentGallery) -> FaceBodyTracker:
    cfg = _identity_cfg()
    return FaceBodyTracker(
        gallery=gallery,
        match_threshold=cfg.get("gallery_match_cost_threshold", 0.65),
    )


def run_perception_on_video(
    session_id: str,
    camera_id: str,
    video_path: Path,
    student_ids_filter: list[str] | None = None,
    stride: int = 1,
    force_student_id: str | None = None,
) -> Path:
    out_dir = data_path("sessions", session_id, "perception", camera_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "detections.jsonl"
    pose_path = out_dir / "pose2d.json"
    stride = max(1, int(stride))

    if not camera_runs_pose2d(camera_id):
        return _write_empty_pose2d(
            out_dir, camera_id, session_id, video_path, stride,
            reason="camera_has_no_pose2d_role",
        )

    det_score_thr = _person_detect_score_threshold()
    kpt_score_thr = _person_kpt_score_threshold()

    gallery = EnrollmentGallery(session_id)
    tracker = _create_tracker(gallery)
    face_emb = create_face_embedder()
    body_emb = create_body_embedder()
    # Single-student drills: if ReID fails (cross-cam appearance shift), still
    # export pose so action segmentation is not empty.
    if force_student_id is None and student_ids_filter and len(student_ids_filter) == 1:
        force_student_id = student_ids_filter[0]
    if not gallery.list_students():
        print(
            f"  [perception {camera_id}] WARNING: empty enrollment gallery "
            f"for session {session_id}",
            flush=True,
        )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or float(get_camera(camera_id).get("fps", 30))
    pose_frames: list[dict] = []
    frame_idx = 0

    with open(det_path, "w", encoding="utf-8") as det_f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            detections = _detect_persons(frame, det_score_thr)
            dets = []

            for det in detections:
                bbox = det["bbox"]
                idcfg = _identity_cfg()
                match_mode = str(idcfg.get("match_mode", "body_color"))
                be = body_emb.embed(frame, bbox)
                from src.identity.clothing_color import extract_clothing_color
                color_desc = extract_clothing_color(
                    frame, bbox, keypoints=det.get("keypoints"),
                )
                if match_mode == "body_color":
                    # g08 winner: body + clothing color; skip face embed for speed
                    fe = None
                    alpha = 0.0
                else:
                    fb = _estimate_face_bbox(bbox, frame.shape)
                    fe = face_emb.embed(frame, fb)
                    alpha = compute_alpha(fb if fe is not None else None)
                dets.append({
                    "bbox": bbox,
                    "face_emb": fe,
                    "body_emb": be,
                    "alpha": alpha,
                    "color_desc": color_desc,
                    "score": float(det.get("score", 1.0)),
                })

            tracks = tracker.update(dets)
            if force_student_id:
                owned = any(t.student_id == force_student_id and t.age == 0 for t in tracks)
                if not owned:
                    live = [t for t in tracks if t.age == 0]
                    if live:
                        best = max(
                            live,
                            key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]),
                        )
                        best.student_id = force_student_id
                        best.sticky_student_id = force_student_id
                        best.identity_confidence = "forced"
            frame_poses = []
            for t in tracks:
                # Skip coasting / unmatched tracks — only live detections this frame
                if t.age > 0:
                    continue
                sid = t.student_id
                if sid and student_ids_filter and sid not in student_ids_filter:
                    continue
                if sid and not has_consent(sid, session_id, ConsentScope.VIDEO):
                    continue
                rec = {
                    "frame": frame_idx,
                    "timestamp_ms": frame_to_timestamp_ms(frame_idx, fps),
                    "bbox": t.bbox,
                    "track_id": t.track_id,
                    "student_id": sid,
                    "face_sim": t.face_sim,
                    "body_sim": t.body_sim,
                    "alpha": t.alpha,
                    "identity_confidence": (
                        (t.identity_confidence or "medium") if sid else "low"
                    ),
                }
                det_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # Always estimate pose for live tracks (do NOT gate on student_id).
                # Missing ID used to drop entire shooting skeletons on side cams.
                kpts, pose_source = _estimate_pose_with_fallback(
                    frame, t.bbox, detections, kpt_score_thr,
                )
                if kpts is None:
                    continue
                # If filter is set and ID unknown, still keep pose for viz continuity
                # but skip when we know it's an out-of-gallery bystander with no sticky.
                if student_ids_filter and sid is None:
                    # Keep unlabeled poses — action stage filters by student_id later
                    pass
                try:
                    from src.identity.clothing_color import extract_clothing_color
                    t_color = extract_clothing_color(frame, t.bbox, keypoints=kpts)
                except Exception:
                    t_color = None
                frame_poses.append({
                    "student_id": sid,
                    "track_id": t.track_id,
                    "keypoints": kpts.tolist(),
                    "scores": kpts[:, 2].tolist(),
                    "pose_source": pose_source,
                    "bbox": t.bbox,
                    "identity_confidence": (
                        (t.identity_confidence or "medium") if sid else "low"
                    ),
                    "face_sim": float(t.face_sim or 0.0),
                    "body_sim": float(t.body_sim or 0.0),
                    "gallery_cost": float(t.gallery_cost or 1.0),
                })
                _ = t_color  # reserved for future online gallery update
            pose_frames.append({
                "frame": frame_idx,
                "timestamp_ms": frame_to_timestamp_ms(frame_idx, fps),
                "persons": frame_poses,
            })
            frame_idx += 1

    cap.release()
    pose_path.write_text(json.dumps({
        "camera_id": camera_id,
        "session_id": session_id,
        "fps": fps,
        "stride": stride,
        "processing": "per_camera_isolated",
        "frames": pose_frames,
    }, ensure_ascii=False, indent=2))
    return out_dir


def run_perception_session(session_id: str, camera_ids: list[str] | None = None) -> list[Path]:
    """Delegate to per-camera isolated pipeline."""
    from src.perception.camera_pipeline import run_perception_all_cameras

    results = run_perception_all_cameras(session_id, camera_ids)
    return [Path(p) for p in results.values() if p]
