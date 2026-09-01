"""Lineup / group frontal enrollment (v3 group0).

Protocol: several students stand facing the enrollment camera together.
We pick the best multi-person frontal frame(s), sort left→right, and write
one gallery entry per person (A..D ≈ stu_00..).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.identity.embedders import create_face_embedder
from src.identity.enrollment import EnrollmentGallery
from src.identity.perception import _estimate_face_bbox
from src.identity.sequential_enroll import _detect_persons_pose, _frontal_score, _load_yolo


def _frame_lineup_candidates(
    frame: np.ndarray,
    dets: list[dict],
    *,
    min_area: float = 0.018,
    min_frontal: float = 0.35,
    cx_lo: float = 0.05,
    cx_hi: float = 0.95,
) -> list[dict]:
    fh, fw = frame.shape[:2]
    area = float(fh * fw)
    out: list[dict] = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        ar = (x2 - x1) * (y2 - y1) / area
        cx = 0.5 * (x1 + x2) / fw
        if ar < min_area or not (cx_lo <= cx <= cx_hi):
            continue
        kp = d.get("keypoints")
        if kp is not None and getattr(kp, "shape", None) is not None and kp.shape[0] >= 17:
            fs = _frontal_score(kp[:, :2], kp[:, 2])
        else:
            fs = 0.25
        if fs < min_frontal:
            continue
        out.append({
            "bbox": list(map(float, d["bbox"])),
            "score": float(d.get("score", 0.5)),
            "keypoints": kp,
            "area_ratio": float(ar),
            "frontal": float(fs),
            "cx": float(cx),
        })
    out.sort(key=lambda x: x["cx"])
    return out


def pick_best_lineup_frame(
    video_path: Path,
    *,
    expected_persons: int = 4,
    sample_hz: float = 2.0,
    score_thr: float = 0.35,
    min_frontal: float = 0.40,
) -> tuple[int, float, np.ndarray, list[dict]] | None:
    """Return (frame_idx, t_s, frame_bgr, persons_ltr) or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open enrollment video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / max(sample_hz, 0.5))))
    backend, kind = _load_yolo()

    best: tuple[float, int, float, np.ndarray, list[dict]] | None = None
    for i in range(0, max(n, 1), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = _detect_persons_pose(backend, kind, frame, score_thr)
        cands = _frame_lineup_candidates(frame, dets, min_frontal=min_frontal)
        if len(cands) < expected_persons:
            continue
        # Prefer similarly sized court players (drop tiny bystanders / huge outliers)
        cands = sorted(cands, key=lambda x: x["area_ratio"], reverse=True)
        top = cands[: max(expected_persons + 2, expected_persons)]
        med = float(np.median([x["area_ratio"] for x in top[:expected_persons]]))
        similar = [x for x in top if x["area_ratio"] >= 0.45 * med]
        if len(similar) < expected_persons:
            continue
        chosen = sorted(similar, key=lambda x: x["cx"])[:expected_persons]
        # If still more than expected after cx sort window, take expected contiguous in x
        if len(similar) > expected_persons:
            similar_ltr = sorted(similar, key=lambda x: x["cx"])
            # sliding window of size expected with max frontal*area
            best_win = None
            for a in range(0, len(similar_ltr) - expected_persons + 1):
                win = similar_ltr[a : a + expected_persons]
                sc = sum(p["frontal"] * p["area_ratio"] for p in win)
                if best_win is None or sc > best_win[0]:
                    best_win = (sc, win)
            chosen = best_win[1] if best_win else chosen

        score = sum(p["frontal"] * p["area_ratio"] for p in chosen)
        # Prefer more centered group
        mean_cx = float(np.mean([p["cx"] for p in chosen]))
        score *= 1.0 - 0.25 * abs(mean_cx - 0.5)
        t = float(i) / fps
        if best is None or score > best[0]:
            best = (score, i, t, frame.copy(), chosen)
    cap.release()
    if best is None:
        return None
    _, fi, t, frame, persons = best
    return fi, t, frame, persons


def enroll_lineup_from_video(
    session_id: str,
    video_path: Path,
    *,
    id_prefix: str = "stu",
    expected_persons: int = 4,
    preview_dir: Path | None = None,
    n_extra_frames: int = 4,
    **_kwargs: Any,
) -> list[str]:
    """Enroll expected_persons from a facing-camera lineup. Returns student_ids L→R."""
    picked = pick_best_lineup_frame(
        video_path,
        expected_persons=expected_persons,
        sample_hz=2.5,
        min_frontal=0.38,
    )
    if picked is None:
        print(
            f"  [enroll-lineup] WARNING: could not find {expected_persons}-person frontal lineup",
            flush=True,
        )
        return []
    fi, t0, frame0, persons = picked
    print(
        f"  [enroll-lineup] best frame={fi} t={t0:.2f}s n={len(persons)} "
        f"cx={[round(p['cx'],2) for p in persons]}",
        flush=True,
    )

    # Collect a few nearby frames for multi-sample galleries
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    backend, kind = _load_yolo()
    face = create_face_embedder()
    gallery = EnrollmentGallery(session_id)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    offsets = [0] + [
        int(round(df * fps))
        for df in (-1.0, -0.5, 0.5, 1.0, 1.5, 2.0)[:n_extra_frames]
    ]
    # For each person slot, gather matched bboxes across frames by nearest cx
    slots: list[list[tuple[np.ndarray, list[float], Any]]] = [[] for _ in persons]
    for off in offsets:
        idx = max(0, fi + off)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            continue
        dets = _detect_persons_pose(backend, kind, fr, 0.35)
        cands = _frame_lineup_candidates(fr, dets, min_frontal=0.32)
        if len(cands) < expected_persons:
            if off == 0:
                cands = persons  # fallback to primary
            else:
                continue
        cands = sorted(cands, key=lambda x: x["cx"])
        # Match each slot to nearest remaining cand by cx
        used: set[int] = set()
        for si, ref in enumerate(persons):
            best_j, best_d = None, 1e9
            for j, c in enumerate(cands):
                if j in used:
                    continue
                d = abs(c["cx"] - ref["cx"])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is None or best_d > 0.12:
                continue
            used.add(best_j)
            c = cands[best_j]
            slots[si].append((fr, c["bbox"], c.get("keypoints")))

    student_ids: list[str] = []
    for si, samples in enumerate(slots):
        sid = f"{id_prefix}_{si:02d}"
        if not samples:
            # at least primary frame
            samples = [(frame0, persons[si]["bbox"], persons[si].get("keypoints"))]
        for fr, bbox, kp in samples[:6]:
            face_bbox = _estimate_face_bbox(bbox, fr.shape)
            emb = face.embed(fr, face_bbox)
            if emb is not None:
                d = gallery.student_dir(sid)
                idx = len(list(d.glob("face_*.npy")))
                np.save(d / f"face_{idx:03d}.npy", emb)
                gallery._cache.pop(sid, None)
            else:
                gallery.add_face_sample(sid, fr, face_bbox)
            gallery.add_body_sample(sid, fr, bbox)
            gallery.add_clothing_color_sample(sid, fr, bbox, keypoints=kp)
        gallery.save_meta(
            sid,
            sid,
            {
                "source": str(video_path),
                "enroll_mode": "lineup_frontal",
                "frame_idx": fi,
                "t0": round(t0, 2),
                "order": "left_to_right",
                "slot": si,
                "cx": round(float(persons[si]["cx"]), 3),
                "n_samples": len(samples[:6]),
                "best_frontal": round(float(persons[si]["frontal"]), 3),
                "best_area_ratio": round(float(persons[si]["area_ratio"]), 3),
            },
        )
        student_ids.append(sid)
        if preview_dir is not None:
            vis = frame0.copy()
            for j, p in enumerate(persons):
                x1, y1, x2, y2 = map(int, p["bbox"])
                color = (0, 255, 0) if j == si else (80, 80, 80)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    vis, f"{id_prefix}_{j:02d}", (x1, max(30, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
                )
            cv2.imwrite(str(preview_dir / f"{sid}.jpg"), vis)

    cap.release()
    if len(student_ids) < expected_persons:
        print(
            f"  [enroll-lineup] WARNING: expected {expected_persons}, got {len(student_ids)}",
            flush=True,
        )
    return student_ids
