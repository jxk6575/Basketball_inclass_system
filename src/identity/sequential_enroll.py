"""Sequential frontal enrollment from a dedicated registration video (v2).

Classroom protocol: each student briefly faces the enrollment camera (cam_01)
one-by-one. We scan the video, score YOLO-Pose frontal visibility, segment
sustained dominant presentations, and write multi-sample galleries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.cameras.registry import get_perception_config
from src.config import load_models_config, model_path
from src.identity.clothing_color import clothing_color_similarity, extract_clothing_color
from src.identity.embedders import create_body_embedder, create_face_embedder
from src.identity.enrollment import EnrollmentGallery
from src.identity.perception import _estimate_face_bbox


@dataclass
class EnrollSample:
    frame_idx: int
    timestamp_s: float
    bbox: list[float]
    area_ratio: float
    frontal_score: float
    quality: float
    body_emb: np.ndarray
    color_desc: np.ndarray
    keypoints: np.ndarray | None = None
    frame_bgr: np.ndarray | None = None


@dataclass
class EnrollPerson:
    student_id: str
    samples: list[EnrollSample] = field(default_factory=list)
    t0: float = 0.0
    t_end: float = 0.0

    @property
    def best(self) -> EnrollSample:
        return max(self.samples, key=lambda s: s.frontal_score * s.area_ratio)


def _frontal_score(kpts_xy: np.ndarray, confs: np.ndarray) -> float:
    """0–1 score from COCO-17 face/shoulder visibility (frontal preference)."""
    face_ids = (0, 1, 2, 3, 4)
    face_ok = sum(1 for i in face_ids if float(confs[i]) > 0.40)
    core = sum(1 for i in (0, 1, 2) if float(confs[i]) > 0.45)
    sh = 0.0
    if float(confs[5]) > 0.30 and float(confs[6]) > 0.30:
        dy = abs(float(kpts_xy[5, 1]) - float(kpts_xy[6, 1]))
        dx = abs(float(kpts_xy[5, 0]) - float(kpts_xy[6, 0])) + 1e-3
        sh = float(np.clip(1.0 - (dy / dx) / 2.0, 0.0, 1.0))
    return 0.45 * (core / 3.0) + 0.35 * (face_ok / 5.0) + 0.20 * sh


def _load_yolo():
    from src.perception.yolo_pose_detector import create_yolo_pose_detector

    yolo = create_yolo_pose_detector()
    if yolo is not None:
        return yolo, "yolo_pose"
    from ultralytics import YOLO

    cfg = load_models_config()
    return YOLO(str(model_path(cfg["yolo_pose"]["path"]))), "ultralytics"


def _detect_persons_pose(backend, kind: str, frame: np.ndarray, score_thr: float) -> list[dict]:
    """Return [{bbox, score, keypoints(17x3)}]."""
    out: list[dict] = []
    if kind == "yolo_pose":
        # Prefer raw ultralytics for keypoints if wrapper lacks them
        try:
            from ultralytics import YOLO
            from src.config import load_models_config, model_path as mp

            cfg = load_models_config()
            model = YOLO(str(mp(cfg["yolo_pose"]["path"])))
            r = model.predict(frame, verbose=False, conf=score_thr)[0]
        except Exception:
            dets = backend.detect_persons(frame, score_thr=score_thr, best_only=False)
            for d in dets:
                out.append({
                    "bbox": list(map(float, d["bbox"])),
                    "score": float(d.get("score", 0.5)),
                    "keypoints": None,
                })
            return out
    else:
        r = backend.predict(frame, verbose=False, conf=score_thr)[0]

    if r.boxes is None:
        return out
    kps = r.keypoints
    for bi, b in enumerate(r.boxes):
        xyxy = b.xyxy[0].detach().cpu().numpy()
        bbox = [float(x) for x in xyxy[:4]]
        kp = None
        if kps is not None and kps.data is not None and bi < len(kps.data):
            kp = kps.data[bi].detach().cpu().numpy().astype(np.float32)
        out.append({
            "bbox": bbox,
            "score": float(b.conf[0]),
            "keypoints": kp,
        })
    return out


def scan_frontal_candidates(
    video_path: Path,
    *,
    sample_hz: float = 5.0,
    min_area_ratio: float = 0.035,
    min_frontal: float = 0.40,
    score_thr: float = 0.35,
    cx_range: tuple[float, float] = (0.18, 0.82),
    min_dominance: float = 1.35,
    closeup_area: float = 0.085,
    stop_on_group_lineup: bool = True,
    group_lineup_persons: int = 4,
    group_lineup_seconds: float = 2.5,
    keep_frames: bool = False,
) -> list[EnrollSample]:
    """Scan video for frontal enrollment candidates.

    Stops early when a sustained multi-person lineup is detected (typical
    after individual walk-up registration).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open enrollment video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / max(sample_hz, 0.5))))
    backend, kind = _load_yolo()
    body = create_body_embedder()

    samples: list[EnrollSample] = []
    lineup_streak_s = 0.0
    prev_t: float | None = None

    for i in range(0, max(n, 1), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        t = float(i) / fps
        dt = (t - prev_t) if prev_t is not None else (step / fps)
        prev_t = t
        fh, fw = frame.shape[:2]
        frame_area = float(fh * fw)
        dets = _detect_persons_pose(backend, kind, frame, score_thr)

        sized: list[tuple[float, dict]] = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            ar = (x2 - x1) * (y2 - y1) / frame_area
            if ar >= 0.022:
                sized.append((ar, d))
        sized.sort(key=lambda x: x[0], reverse=True)

        # Group lineup: many similarly sized people → end of individual enroll
        if stop_on_group_lineup and len(sized) >= group_lineup_persons:
            top_ar = sized[0][0]
            similar = sum(1 for ar, _ in sized if ar >= 0.55 * top_ar)
            if similar >= group_lineup_persons and top_ar < closeup_area:
                lineup_streak_s += dt
                if lineup_streak_s >= group_lineup_seconds and samples:
                    break
            else:
                lineup_streak_s = 0.0
        else:
            lineup_streak_s = 0.0

        best: tuple[float, dict, float, float] | None = None
        second_ar = sized[1][0] if len(sized) > 1 else 0.0
        for ar, d in sized:
            x1, y1, x2, y2 = d["bbox"]
            cx = 0.5 * (x1 + x2) / fw
            if ar < min_area_ratio or not (cx_range[0] <= cx <= cx_range[1]):
                continue
            dominance = ar / (second_ar + 1e-6)
            if ar < closeup_area and dominance < min_dominance:
                continue
            kp = d.get("keypoints")
            if kp is not None and kp.shape[0] >= 17:
                fs = _frontal_score(kp[:, :2], kp[:, 2])
            else:
                fs = 0.25
            q = (
                fs * 0.50
                + min(ar / 0.12, 1.0) * 0.30
                + min(dominance / 3.0, 1.0) * 0.10
                + (1.0 - abs(cx - 0.5)) * 0.10
            )
            if best is None or q > best[0]:
                best = (q, d, ar, fs)
        if best is None:
            continue
        q, d, ar, fs = best
        if fs < min_frontal or ar < min_area_ratio:
            continue
        bbox = list(map(float, d["bbox"]))
        kp = d.get("keypoints")
        emb = body.embed(frame, bbox)
        color = extract_clothing_color(frame, bbox, keypoints=kp)
        samples.append(EnrollSample(
            frame_idx=i,
            timestamp_s=t,
            bbox=bbox,
            area_ratio=float(ar),
            frontal_score=float(fs),
            quality=float(q),
            body_emb=emb,
            color_desc=color,
            keypoints=kp,
            frame_bgr=frame.copy() if keep_frames else None,
        ))
    cap.release()
    return samples


def cluster_sequential_enrollments(
    samples: list[EnrollSample],
    *,
    id_prefix: str = "stu",
    min_samples: int = 3,
    min_max_frontal: float = 0.45,
    min_max_area: float = 0.045,
    gap_split_s: float = 1.5,
    same_body_thr: float = 0.58,
    revisit_body_thr: float = 0.78,
    revisit_color_thr: float = 0.70,
    revisit_max_gap_s: float = 5.0,
    max_persons: int = 16,
) -> list[EnrollPerson]:
    """Segment then assign sequential student IDs (prefer new ID over over-merge)."""
    if not samples:
        return []

    # 1) temporal segments — also split on embedding drift within a run
    segs: list[list[EnrollSample]] = []
    cur: list[EnrollSample] = [samples[0]]
    for s in samples[1:]:
        gap = s.timestamp_s - cur[-1].timestamp_s
        mean = np.mean([x.body_emb for x in cur], axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)
        e = s.body_emb / (np.linalg.norm(s.body_emb) + 1e-8)
        bs = float(np.dot(e, mean))
        cs = clothing_color_similarity(s.color_desc, cur[-1].color_desc)
        same = gap <= gap_split_s and (
            bs >= same_body_thr or (bs >= same_body_thr - 0.08 and cs >= 0.60)
        )
        if same:
            cur.append(s)
        else:
            segs.append(cur)
            cur = [s]
    segs.append(cur)

    # 2) quality filter
    kept = [
        seg for seg in segs
        if len(seg) >= min_samples
        and max(x.frontal_score for x in seg) >= min_max_frontal
        and max(x.area_ratio for x in seg) >= min_max_area
    ]

    # 3) sequential IDs — merge only confident short revisits
    people: list[EnrollPerson] = []
    for seg in kept:
        mean = np.mean([x.body_emb for x in seg], axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)
        matched: EnrollPerson | None = None
        for ent in reversed(people):
            em = np.mean([x.body_emb for x in ent.samples], axis=0)
            em = em / (np.linalg.norm(em) + 1e-8)
            bs = float(np.dot(mean, em))
            mid_s = seg[len(seg) // 2]
            mid_e = ent.samples[len(ent.samples) // 2]
            cs = clothing_color_similarity(mid_s.color_desc, mid_e.color_desc)
            gap = seg[0].timestamp_s - ent.t_end
            if gap < revisit_max_gap_s and bs >= revisit_body_thr and cs >= revisit_color_thr:
                matched = ent
                break
        if matched is None:
            if len(people) >= max_persons:
                continue
            sid = f"{id_prefix}_{len(people):02d}"
            matched = EnrollPerson(
                student_id=sid, t0=seg[0].timestamp_s, t_end=seg[-1].timestamp_s,
            )
            people.append(matched)
        matched.samples.extend(seg)
        matched.t_end = seg[-1].timestamp_s

    # 4) drop nested short false-positives (e.g. brief bystander inside another's window)
    if len(people) <= 1:
        return people
    keep_flags = [True] * len(people)
    for i, a in enumerate(people):
        if not keep_flags[i]:
            continue
        dur_a = max(0.01, a.t_end - a.t0)
        # Drop fleeting segments that are unlikely to be full frontal enrollments
        if len(a.samples) < 4 and dur_a < 1.2:
            keep_flags[i] = False
            continue
        for j, b in enumerate(people):
            if i == j or not keep_flags[j]:
                continue
            # b nested in a and much shorter
            if b.t0 >= a.t0 - 0.2 and b.t_end <= a.t_end + 0.2:
                dur_b = max(0.01, b.t_end - b.t0)
                if dur_b < 0.45 * dur_a and len(b.samples) <= max(4, len(a.samples) // 3):
                    keep_flags[j] = False
    filtered = [p for p, ok in zip(people, keep_flags) if ok]
    # re-index ids (caller may further trim to expected_persons)
    for i, p in enumerate(filtered):
        p.student_id = f"{id_prefix}_{i:02d}"
    return filtered


def write_enrollment_gallery(
    session_id: str,
    people: list[EnrollPerson],
    video_path: Path,
    *,
    max_samples_per_person: int = 8,
    preview_dir: Path | None = None,
) -> list[str]:
    """Persist face/body/color samples into EnrollmentGallery; return student_ids."""
    if not people:
        return []
    gallery = EnrollmentGallery(session_id)
    face_emb = create_face_embedder()
    # Reload frames for best samples if not cached
    cap = cv2.VideoCapture(str(video_path))
    student_ids: list[str] = []

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for person in people:
        # Rank samples by quality; take diverse timestamps
        ranked = sorted(person.samples, key=lambda s: s.frontal_score * s.area_ratio, reverse=True)
        chosen: list[EnrollSample] = []
        for s in ranked:
            if len(chosen) >= max_samples_per_person:
                break
            if any(abs(s.timestamp_s - c.timestamp_s) < 0.35 for c in chosen):
                continue
            chosen.append(s)
        if not chosen:
            chosen = ranked[:1]

        for s in chosen:
            if s.frame_bgr is not None:
                frame = s.frame_bgr
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, s.frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
            face_bbox = _estimate_face_bbox(s.bbox, frame.shape)
            # Try real face embed; still store body/color always
            emb = face_emb.embed(frame, face_bbox)
            if emb is not None:
                d = gallery.student_dir(person.student_id)
                idx = len(list(d.glob("face_*.npy")))
                np.save(d / f"face_{idx:03d}.npy", emb)
                gallery._cache.pop(person.student_id, None)
            else:
                gallery.add_face_sample(person.student_id, frame, face_bbox)
            gallery.add_body_sample(person.student_id, frame, s.bbox)
            gallery.add_clothing_color_sample(
                person.student_id, frame, s.bbox, keypoints=s.keypoints,
            )

        best = person.best
        gallery.save_meta(
            person.student_id,
            person.student_id,
            {
                "source": str(video_path),
                "enroll_mode": "sequential_frontal",
                "t0": round(person.t0, 2),
                "t_end": round(person.t_end, 2),
                "n_samples": len(chosen),
                "best_frame": best.frame_idx,
                "best_frontal": round(best.frontal_score, 3),
                "best_area_ratio": round(best.area_ratio, 3),
            },
        )
        student_ids.append(person.student_id)

        if preview_dir is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, best.frame_idx)
            ok, frame = cap.read()
            if ok:
                x1, y1, x2, y2 = map(int, best.bbox)
                vis = frame.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    vis, person.student_id, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2,
                )
                cv2.imwrite(str(preview_dir / f"{person.student_id}.jpg"), vis)

    cap.release()
    return student_ids


def enroll_sequential_from_video(
    session_id: str,
    video_path: Path,
    *,
    id_prefix: str = "stu",
    preview_dir: Path | None = None,
    max_persons: int = 16,
    expected_persons: int | None = None,
    **scan_kwargs: Any,
) -> list[str]:
    """
    End-to-end: scan → cluster → write gallery.

    Returns enrolled student_ids in appearance order.
    """
    perc = get_perception_config()
    min_ar = float(scan_kwargs.pop(
        "min_area_ratio", max(0.03, float(perc.get("min_person_area_ratio", 0.015))),
    ))
    # Defaults tuned for v2 group0 (six frontal walk-ups before lineup)
    scan_defaults = dict(
        min_dominance=1.25,
        closeup_area=0.075,
        group_lineup_seconds=3.5,
        group_lineup_persons=5,
        min_frontal=0.38,
    )
    scan_defaults.update(scan_kwargs)
    samples = scan_frontal_candidates(
        video_path,
        min_area_ratio=min_ar,
        keep_frames=False,
        **scan_defaults,
    )
    cluster_max = max(max_persons, (expected_persons or 0) + 4)
    people = cluster_sequential_enrollments(
        samples,
        id_prefix=id_prefix,
        max_persons=cluster_max,
        min_samples=2,
        min_max_frontal=0.42,
        min_max_area=0.040,
        gap_split_s=1.2,
        same_body_thr=0.62,
        revisit_body_thr=0.82,
        revisit_color_thr=0.75,
        revisit_max_gap_s=4.0,
    )
    if expected_persons is not None and len(people) > expected_persons:
        # Keep earliest walk-ups (appearance order), not just largest clusters
        people = sorted(people, key=lambda p: p.t0)[:expected_persons]
        for i, p in enumerate(people):
            p.student_id = f"{id_prefix}_{i:02d}"
    elif expected_persons is not None and len(people) < expected_persons:
        print(
            f"  [enroll] WARNING: expected {expected_persons} persons, got {len(people)}",
            flush=True,
        )
    return write_enrollment_gallery(
        session_id, people, video_path, preview_dir=preview_dir,
    )
