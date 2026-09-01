"""Assign shooting-clip identity from video evidence at release.

Appearance ReID alone is unreliable when kits are similar. At release we prefer
the person whose wrists/bbox are nearest the ball (same-camera ball track when
available), otherwise mild spatial prominence: larger bbox + elevated wrists.
No drill-cycle or free-throw-spot priors.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.action.registry import is_shooting_action
from src.cameras.registry import get_action_segment_camera
from src.config import data_path, load_yaml
from src.types import ActionClip


def _identity_cfg() -> dict:
    return dict((load_yaml("cameras.yaml").get("identity") or {}))


def _release_ms(clip: ActionClip) -> float | None:
    """Pose / release clock for identity — never raw rim (rebounder trap)."""
    meta = clip.metadata or {}
    mc = meta.get("multicam") or {}
    # cam_03 pose peak is the shooter; rim time is often a rebounder/spotter
    if mc.get("pose_timestamp_ms") is not None:
        return float(mc["pose_timestamp_ms"])
    for ph in clip.phases or []:
        if ph.name == "release" and ph.start_ms is not None:
            return float(ph.start_ms)
    if clip.start_ms is not None and clip.end_ms is not None:
        return 0.5 * (float(clip.start_ms) + float(clip.end_ms))
    if mc.get("rim_timestamp_ms") is not None:
        return float(mc["rim_timestamp_ms"]) - 900.0
    if clip.start_ms is not None:
        return float(clip.start_ms)
    return None


def _wrist_raise_score(person: dict) -> float:
    """Higher when wrists are clearly above shoulders (release / follow-through)."""
    kps = person.get("keypoints") or []
    if len(kps) < 11:
        return 0.0

    def _xy(i: int) -> tuple[float, float, float] | None:
        if i >= len(kps):
            return None
        k = kps[i]
        if not isinstance(k, (list, tuple)) or len(k) < 2:
            return None
        conf = float(k[2]) if len(k) > 2 else 1.0
        if conf < 0.25:
            return None
        return float(k[0]), float(k[1]), conf

    ls, rs = _xy(5), _xy(6)
    lw, rw = _xy(9), _xy(10)
    if not ls or not rs:
        return 0.0
    sh_y = 0.5 * (ls[1] + rs[1])
    score = 0.0
    for w in (lw, rw):
        if w is None:
            continue
        # image-y decreases upward
        dy = sh_y - w[1]
        if dy > 8.0:
            score += min(1.0, dy / 80.0)
    return float(score)


def _person_area(person: dict) -> float:
    bb = person.get("bbox") or [0, 0, 0, 0]
    if len(bb) < 4:
        return 0.0
    return abs(float(bb[2] - bb[0]) * float(bb[3] - bb[1]))


def _wrist_points(person: dict) -> list[tuple[float, float]]:
    kps = person.get("keypoints") or []
    out: list[tuple[float, float]] = []
    for i in (9, 10):
        if i >= len(kps):
            continue
        k = kps[i]
        if not isinstance(k, (list, tuple)) or len(k) < 2:
            continue
        conf = float(k[2]) if len(k) > 2 else 1.0
        if conf < 0.2:
            continue
        out.append((float(k[0]), float(k[1])))
    return out


def _bbox_center(person: dict) -> tuple[float, float] | None:
    bb = person.get("bbox") or [0, 0, 0, 0]
    if len(bb) < 4:
        return None
    return 0.5 * (float(bb[0]) + float(bb[2])), 0.5 * (float(bb[1]) + float(bb[3]))


def _dist_to_ball(person: dict, ball_xy: tuple[float, float]) -> float:
    """Min distance from ball to wrists, else to bbox center."""
    bx, by = ball_xy
    wrists = _wrist_points(person)
    if wrists:
        return min(float(((wx - bx) ** 2 + (wy - by) ** 2) ** 0.5) for wx, wy in wrists)
    c = _bbox_center(person)
    if c is None:
        return 1e9
    return float(((c[0] - bx) ** 2 + (c[1] - by) ** 2) ** 0.5)


def pick_spatial_shooter(
    persons: list[dict],
    *,
    frame_w: float = 1920.0,
    ball_xy: tuple[float, float] | None = None,
) -> dict | None:
    """Score persons from pose (+ optional ball proximity); return best shooter."""
    del frame_w  # kept for API compatibility; no court-spot prior
    if not persons:
        return None
    areas = [_person_area(p) for p in persons]
    max_a = max(areas) if areas else 1.0
    ball_dists = [_dist_to_ball(p, ball_xy) for p in persons] if ball_xy else None
    min_bd = min(ball_dists) if ball_dists else None

    best, best_s = None, -1.0
    for i, (p, area) in enumerate(zip(persons, areas)):
        if area < 0.25 * max_a:
            continue  # far / tiny bystanders
        raise_s = _wrist_raise_score(p)
        # Closer + arms up; ball near wrist/bbox is strongest video cue
        score = (area / max_a) * (1.0 + 1.35 * raise_s)
        if ball_dists is not None and min_bd is not None:
            d = ball_dists[i]
            # Soft proximity: within ~200px strongly preferred
            prox = max(0.0, 1.0 - d / 220.0)
            score = score * (0.55 + 0.9 * prox)
            if d <= min_bd + 8.0:
                score += 0.15
        if p.get("student_id"):
            score += 0.05
        if score > best_s:
            best_s, best = score, p
    return best


@lru_cache(maxsize=32)
def _load_pose_frames(session_id: str, camera_id: str) -> tuple[list[dict], float]:
    path = data_path("sessions", session_id, "perception", camera_id, "pose2d.json")
    if not path.exists():
        return [], 1920.0
    doc = json.loads(path.read_text(encoding="utf-8"))
    frames = list(doc.get("frames") or [])
    fw = 1920.0
    for fr in frames[:20]:
        for p in fr.get("persons") or []:
            bb = p.get("bbox") or []
            if len(bb) >= 4 and float(bb[2]) > fw * 0.5:
                fw = max(fw, float(bb[2]) * 1.05)
    return frames, fw


@lru_cache(maxsize=32)
def _load_ball_frames_by_ms(session_id: str, camera_id: str) -> list[tuple[float, tuple[float, float]]]:
    """Ball centers keyed for nearest-time lookup (same camera when possible)."""
    out_dir = data_path("sessions", session_id, "shot_outcomes")
    candidates = []
    if camera_id == "cam_04" or str(camera_id).endswith("04"):
        candidates.append(out_dir / "ball_track.json")
    candidates.append(out_dir / f"ball_track_{camera_id}.json")
    # cam_04 official track as last resort (coords may not match other cams)
    if camera_id not in ("cam_04",) and not str(camera_id).endswith("04"):
        pass  # do not mix cam_04 xy onto other-cam pose
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[float, tuple[float, float]]] = []
    for fr in doc.get("frames") or []:
        tm = fr.get("timestamp_ms")
        ball = fr.get("ball") or {}
        c = ball.get("center")
        if tm is None or not c or len(c) < 2:
            continue
        out.append((float(tm), (float(c[0]), float(c[1]))))
    return out


def _nearest_ball_xy(
    balls: list[tuple[float, tuple[float, float]]],
    t_ms: float,
    max_dt_ms: float = 120.0,
) -> tuple[float, float] | None:
    best, best_d = None, None
    for tm, xy in balls:
        d = abs(tm - t_ms)
        if d > max_dt_ms:
            continue
        if best_d is None or d < best_d:
            best_d, best = d, xy
    return best


def _nearest_frame(frames: list[dict], t_ms: float, max_dt_ms: float = 450.0) -> dict | None:
    best, best_d = None, None
    for fr in frames:
        tm = fr.get("timestamp_ms")
        if tm is None:
            continue
        d = abs(float(tm) - float(t_ms))
        if d > max_dt_ms:
            continue
        if best_d is None or d < best_d:
            best_d, best = d, fr
    return best


def _best_ball_person_in_window(
    frames: list[dict],
    balls: list[tuple[float, tuple[float, float]]],
    t_ms: float,
    *,
    pre_ms: float = 900.0,
    post_ms: float = 150.0,
    step_ms: float = 33.0,
    near_px: float = 160.0,
) -> tuple[dict | None, dict[str, Any]]:
    """
    Search around release for the person whose wrist/bbox is nearest the ball.

    At the annotated release instant the ball has often already left the hand;
    scanning a short pre-release window recovers the holder from video only.
    Prefer raised wrists (shooter) over nearby passers/rebounders.
    """
    info: dict[str, Any] = {"window_pre_ms": pre_ms, "window_post_ms": post_ms}
    best_person: dict | None = None
    # Sort key: prefer raised wrists, then closer ball, then nearer to t_ms
    best_key: tuple[float, float, float] | None = None
    best_tm = t_ms
    best_xy: tuple[float, float] | None = None
    best_dist = 1e9
    # Also track best raised-wrist person independent of ball (release pose)
    best_raise_person: dict | None = None
    best_raise_key: tuple[float, float, float] | None = None
    best_raise_tm = t_ms
    best_raise_xy: tuple[float, float] | None = None
    best_raise_dist = 1e9

    t0 = t_ms - pre_ms
    t1 = t_ms + post_ms
    tm = t0
    while tm <= t1 + 1e-6:
        fr = _nearest_frame(frames, tm, max_dt_ms=min(80.0, step_ms + 20.0))
        bxy = _nearest_ball_xy(balls, tm, max_dt_ms=min(80.0, step_ms + 20.0))
        if fr is not None:
            for p in fr.get("persons") or []:
                raise_s = _wrist_raise_score(p)
                d = _dist_to_ball(p, bxy) if bxy is not None else 1e9
                if raise_s >= 0.45:
                    # Joint raise×proximity (Basketball-SORT / GIF: motion alone fails)
                    dd = d if d < 1e8 else 420.0
                    joint = float(raise_s) / (1.0 + dd / 160.0)
                    rkey = (-joint, dd, abs(tm - t_ms))
                    if best_raise_key is None or rkey < best_raise_key:
                        best_raise_key = rkey
                        best_raise_person = p
                        best_raise_tm = tm
                        best_raise_xy = bxy
                        best_raise_dist = d
                if bxy is not None:
                    if d > near_px * 2.5:
                        continue
                    # Penalize flat arms (passer/catcher); reward release pose
                    effective_d = d / (1.0 + 2.5 * raise_s)
                    key = (0.0 if raise_s >= 0.35 else 1.0, effective_d, abs(tm - t_ms))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_person = p
                        best_tm = tm
                        best_xy = bxy
                        best_dist = d
        tm += step_ms

    raise_ok = (
        best_raise_person is not None
        and best_raise_key is not None
        and (-best_raise_key[0]) >= 0.28  # joint score threshold
        and _wrist_raise_score(best_raise_person) >= 0.55
    )
    # Prefer raise when ball left the hand; fall back if a near-ball raised person exists
    if raise_ok:
        raise_s_best = _wrist_raise_score(best_raise_person)
        # Only steal from raise-pose with a near-ball candidate that is also a
        # strong release pose near t_ms (avoid early passer/inbounder).
        if (
            best_person is not None
            and best_dist <= near_px * 1.2
            and _wrist_raise_score(best_person) >= max(0.55, raise_s_best - 0.15)
            and best_raise_dist > best_dist + 80.0
            and abs(best_tm - t_ms) <= 350.0
        ):
            info.update({
                "reason": "ball_person_window",
                "assoc_t_ms": best_tm,
                "ball_xy": [round(best_xy[0], 1), round(best_xy[1], 1)] if best_xy else None,
                "ball_dist": round(best_dist, 1),
                "wrist_raise": round(_wrist_raise_score(best_person), 3),
            })
            return best_person, info
        info.update({
            "reason": "raise_pose_window",
            "assoc_t_ms": best_raise_tm,
            "ball_xy": [round(best_raise_xy[0], 1), round(best_raise_xy[1], 1)] if best_raise_xy else None,
            "ball_dist": round(best_raise_dist, 1) if best_raise_dist < 1e8 else None,
            "wrist_raise": round(raise_s_best, 3),
        })
        return best_raise_person, info

    if best_person is not None and best_dist <= near_px * 1.5:
        info.update({
            "reason": "ball_person_window",
            "assoc_t_ms": best_tm,
            "ball_xy": [round(best_xy[0], 1), round(best_xy[1], 1)] if best_xy else None,
            "ball_dist": round(best_dist, 1),
            "wrist_raise": round(_wrist_raise_score(best_person), 3),
        })
        return best_person, info
    info["reason"] = "no_near_ball"
    return None, info


_CONF_VOTE_W = {
    "high": 3.0,
    "medium": 2.0,
    "sticky": 1.5,
    "lr_order": 0.8,
    "spatial_prior": 1.2,
    "low": 0.2,
}

# Lazy singletons for release-time gallery rematch (GIF-style ID module)
_REMATCH_CACHE: dict[str, Any] = {}


def _read_video_frame(video_path: Path, t_ms: float):
    import cv2

    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    idx = int(round(float(t_ms) / 1000.0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def _gallery_rematch_shooter(
    session_id: str,
    camera_id: str,
    person: dict,
    t_ms: float,
) -> tuple[str | None, dict[str, Any]]:
    """
    Re-query enrollment gallery on the shooter crop at release.

    Sticky track IDs often freeze the wrong student during occlusion; a fresh
    OSNet + clothing match on the raised-wrist crop recovers identity
    (inspired by GIF WACV'26 global ID fusion, without heavy transformers).
    Body margin is preferred over fused clothing when face is weak.
    """
    cfg = _identity_cfg()
    meta: dict[str, Any] = {"enabled": bool(cfg.get("release_gallery_rematch", True))}
    if not meta["enabled"]:
        return None, meta
    bb = person.get("bbox")
    if not isinstance(bb, (list, tuple)) or len(bb) < 4:
        meta["reason"] = "no_bbox"
        return None, meta

    gallery_id = session_id
    enroll_root = data_path("enrollment", session_id)
    if not enroll_root.exists() or not any(enroll_root.iterdir()):
        # Fall back to shared gallery manifest
        man = data_path("outputs", "v3", "gallery_manifest.json")
        if not man.exists():
            man = data_path("outputs", "v2", "gallery_manifest.json")
        if man.exists():
            try:
                gallery_id = str(json.loads(man.read_text(encoding="utf-8")).get("session_id") or session_id)
            except Exception:
                gallery_id = session_id
    meta["gallery_id"] = gallery_id

    cache_key = f"tracker:{gallery_id}"
    if cache_key not in _REMATCH_CACHE:
        from src.identity.enrollment import EnrollmentGallery
        from src.identity.tracker import FaceBodyTracker

        gal = EnrollmentGallery(gallery_id)
        thr = float(cfg.get("gallery_match_cost_threshold", 0.58))
        _REMATCH_CACHE[cache_key] = FaceBodyTracker(gallery=gal, match_threshold=thr)
        _REMATCH_CACHE[f"gallery:{gallery_id}"] = gal
    tracker = _REMATCH_CACHE[cache_key]
    gallery = _REMATCH_CACHE[f"gallery:{gallery_id}"]

    if "body" not in _REMATCH_CACHE:
        from src.identity.embedders import create_body_embedder, create_face_embedder

        _REMATCH_CACHE["body"] = create_body_embedder()
        _REMATCH_CACHE["face"] = create_face_embedder()
    body_embeder = _REMATCH_CACHE["body"]
    face_embeder = _REMATCH_CACHE["face"]

    video = data_path("sessions", session_id, "raw", f"{camera_id}.mp4")
    img = _read_video_frame(video, t_ms)
    if img is None:
        meta["reason"] = "no_frame"
        return None, meta

    from src.identity.clothing_color import extract_clothing_color
    from src.identity.embedders import cosine_sim
    from src.identity.perception import _estimate_face_bbox

    kps = person.get("keypoints")
    body_emb = body_embeder.embed(img, list(bb))
    if body_emb is None:
        meta["reason"] = "no_body_emb"
        return None, meta
    color = extract_clothing_color(img, list(bb), keypoints=kps)
    face_bb = _estimate_face_bbox(list(bb), img.shape)
    face_emb = face_embeder.embed(img, face_bb) if face_bb else None
    alpha = float(cfg.get("face_alpha_high", 0.20)) if face_emb is not None else 0.0
    fused_sid, fs, bs, conf, cost = tracker._match_gallery(
        face_emb, body_emb, alpha=alpha, color_desc=color,
    )

    # Body-primary ranking (release faces are unreliable)
    body_ranked: list[tuple[str, float]] = []
    for sid in gallery.list_students():
        data = gallery.load_student(sid)
        bodies = data.get("body") or []
        if not bodies:
            continue
        score = max(float(cosine_sim(body_emb, b)) for b in bodies)
        body_ranked.append((sid, score))
    body_ranked.sort(key=lambda x: -x[1])
    meta["body_top"] = [
        {"sid": s, "sim": round(v, 3)} for s, v in body_ranked[:3]
    ]
    meta["fused"] = {
        "sid": fused_sid, "conf": conf,
        "cost": round(float(cost), 3),
        "face": round(float(fs), 3),
        "body": round(float(bs), 3),
    }

    min_body = float(cfg.get("release_rematch_min_body", 0.55))
    body_margin = float(cfg.get("release_rematch_body_margin", 0.045))
    sticky = person.get("student_id")
    if len(body_ranked) >= 1 and body_ranked[0][1] >= min_body:
        best_s, best_v = body_ranked[0]
        second_v = body_ranked[1][1] if len(body_ranked) > 1 else 0.0
        if best_v - second_v >= body_margin:
            meta["reason"] = "body_margin"
            meta["sid"] = best_s
            return best_s, meta
        # Soft: accept fused when it agrees with body top
        if fused_sid == best_s and conf in ("high", "medium"):
            meta["reason"] = "fused_agrees_body"
            meta["sid"] = fused_sid
            return str(fused_sid), meta
        # Soft: body top agrees with sticky/instant ID → reinforce
        if sticky and best_s == str(sticky) and best_v - second_v >= body_margin * 0.5:
            meta["reason"] = "body_agrees_sticky"
            meta["sid"] = best_s
            return best_s, meta
    if fused_sid and conf == "high" and float(bs) >= min_body:
        meta["reason"] = "fused_high"
        meta["sid"] = fused_sid
        return str(fused_sid), meta
    meta["reason"] = "ambiguous"
    return None, meta


def _track_raise_majority_sid(
    frames: list[dict],
    track_id: Any,
    t_ms: float,
    *,
    pre_ms: float = 500.0,
    post_ms: float = 2000.0,
    min_raise: float = 0.35,
) -> tuple[str | None, float, float, dict[str, float]]:
    """
    Resolve student_id for a track from frames where wrists are raised.

    Instantaneous ReID at release is noisy; the same track often recovers the
    correct gallery ID during the follow-through. Returns
    (sid, margin, vote_mass, raw_votes). Falls back to all-frame majority in the
    window when no raised-wrist votes exist.

    When the short window is ambiguous (low margin / mixed IDs), bias toward the
    full-track conf-weighted majority — same physical track, longer evidence.
    """
    from collections import Counter

    raise_votes: Counter[str] = Counter()
    all_votes: Counter[str] = Counter()
    life_votes: Counter[str] = Counter()
    for fr in frames:
        tm = fr.get("timestamp_ms")
        for p in fr.get("persons") or []:
            if p.get("track_id") != track_id:
                continue
            sid = p.get("student_id")
            if not sid:
                continue
            conf = str(p.get("identity_confidence") or "low")
            cw = float(_CONF_VOTE_W.get(conf, 0.5))
            life_votes[str(sid)] += cw
            if tm is None or not (t_ms - pre_ms <= float(tm) <= t_ms + post_ms):
                continue
            all_votes[str(sid)] += cw
            raise_s = _wrist_raise_score(p)
            if raise_s >= min_raise:
                raise_votes[str(sid)] += cw * (1.0 + 2.5 * raise_s)
    votes = raise_votes if raise_votes else all_votes
    if not votes and life_votes:
        votes = life_votes
    if not votes:
        return None, 0.0, 0.0, {}
    ranked = votes.most_common(2)
    best, w1 = ranked[0]
    w2 = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    margin = (float(w1) - w2) / (float(w1) + 1e-6)
    # Ambiguous local window → prefer clear lifetime majority on this track
    if life_votes and (margin < 0.30 or len(votes) >= 3):
        lr = life_votes.most_common(2)
        lb, lw1 = lr[0]
        lw2 = float(lr[1][1]) if len(lr) > 1 else 0.0
        lmargin = (float(lw1) - lw2) / (float(lw1) + 1e-6)
        lshare = float(lw1) / max(float(sum(life_votes.values())), 1e-6)
        if lmargin >= 0.20 and lshare >= 0.45:
            best, w1, w2 = lb, lw1, lw2
            margin = lmargin
            votes = life_votes
    return best, margin, float(sum(votes.values())), {k: float(v) for k, v in votes.items()}

def _pose_cameras(session_id: str, preferred: str) -> list[str]:
    """Cameras with pose2d, preferred (action) cam first."""
    root = data_path("sessions", session_id, "perception")
    if not root.exists():
        return [preferred]
    cams = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "pose2d.json").exists())
    if preferred in cams:
        cams = [preferred] + [c for c in cams if c != preferred]
    return cams or [preferred]


def _resolve_ball_handler_on_cam(
    session_id: str,
    camera_id: str,
    t_ms: float,
) -> dict[str, Any] | None:
    """Ball–person at release on one camera + track raise-majority ID."""
    frames, fw = _load_pose_frames(session_id, camera_id)
    balls = _load_ball_frames_by_ms(session_id, camera_id)
    if not frames or not balls:
        return None
    shooter, win_meta = _best_ball_person_in_window(
        # Prefer pre-release holder; long post window often grabs rebounders.
        # Slightly longer pre-window helps TT/gather when release clock is late.
        frames, balls, t_ms, pre_ms=1100.0, post_ms=80.0, near_px=160.0,
    )
    if shooter is None:
        return None
    raise_s = float(win_meta.get("wrist_raise") or _wrist_raise_score(shooter))
    dist = float(win_meta.get("ball_dist") or 999.0)
    tid = shooter.get("track_id")
    maj_sid, margin, mass, votes = _track_raise_majority_sid(frames, tid, t_ms)
    inst_sid = shooter.get("student_id")
    sid = maj_sid or (str(inst_sid) if inst_sid else None)
    rematch_meta: dict[str, Any] = {}
    assoc_t = float(win_meta.get("assoc_t_ms") or t_ms)
    rematch_sid, rematch_meta = _gallery_rematch_shooter(
        session_id, camera_id, shooter, assoc_t,
    )
    if rematch_sid:
        # Clear body rematch overrides sticky/majority flicker
        sid = rematch_sid
        margin = max(margin, 0.55)
        mass = max(mass, 12.0)
    return {
        "camera_id": camera_id,
        "sid": sid,
        "inst_sid": str(inst_sid) if inst_sid else None,
        "maj_sid": maj_sid,
        "rematch_sid": rematch_sid,
        "rematch": rematch_meta,
        "track_id": tid,
        "wrist_raise": raise_s,
        "ball_dist": dist,
        "raise_margin": margin,
        "raise_mass": mass,
        "raise_votes": votes,
        "bbox": shooter.get("bbox"),
        "identity_confidence": (
            "gallery_rematch" if rematch_sid else shooter.get("identity_confidence")
        ),
        "area": round(_person_area(shooter), 1),
        "frame_w": fw,
        "win_meta": win_meta,
    }


def spatial_shooter_sid_at(
    session_id: str,
    t_ms: float,
    *,
    camera_id: str | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """
    Pick shooter student_id from video evidence only.

    1) Per-camera: associate ball with nearest raised-wrist person near release.
    2) Resolve that track's ID via raise-frame majority (sticky vs flicker).
    3) Prefer action camera when its raise-majority is clear; else multi-cam vote.
    """
    from collections import Counter

    anchor = camera_id or get_action_segment_camera()
    cams = _pose_cameras(session_id, anchor) if camera_id is None else [camera_id]
    meta: dict[str, Any] = {"camera_id": anchor, "t_ms": t_ms, "cams": cams}

    per_cam: dict[str, dict[str, Any]] = {}
    for cam in cams:
        info = _resolve_ball_handler_on_cam(session_id, cam, t_ms)
        if info is not None:
            per_cam[cam] = info
    meta["per_cam"] = {
        c: {
            "sid": i.get("sid"),
            "inst_sid": i.get("inst_sid"),
            "maj_sid": i.get("maj_sid"),
            "rematch_sid": i.get("rematch_sid"),
            "wrist_raise": round(float(i.get("wrist_raise") or 0), 3),
            "ball_dist": round(float(i.get("ball_dist") or 0), 1),
            "raise_margin": round(float(i.get("raise_margin") or 0), 3),
            "raise_mass": round(float(i.get("raise_mass") or 0), 1),
        }
        for c, i in per_cam.items()
    }

    def _pack(info: dict[str, Any], reason: str) -> tuple[str | None, dict[str, Any]]:
        sid = info.get("sid")
        out = dict(meta)
        out.update({
            "reason": reason,
            "bbox": info.get("bbox"),
            "identity_confidence": info.get("identity_confidence"),
            "wrist_raise": round(float(info.get("wrist_raise") or 0), 3),
            "ball_dist": round(float(info.get("ball_dist") or 0), 1),
            "area": info.get("area"),
            "track_id": info.get("track_id"),
            "chosen_camera": info.get("camera_id"),
            "raise_margin": round(float(info.get("raise_margin") or 0), 3),
            "rematch_sid": info.get("rematch_sid"),
            "maj_sid": info.get("maj_sid"),
        })
        wm = info.get("win_meta") or {}
        if wm.get("assoc_t_ms") is not None:
            out["assoc_t_ms"] = wm["assoc_t_ms"]
        if wm.get("ball_xy") is not None:
            out["ball_xy"] = wm["ball_xy"]
        if info.get("rematch"):
            out["rematch"] = {
                "reason": (info.get("rematch") or {}).get("reason"),
                "body_top": (info.get("rematch") or {}).get("body_top"),
            }
        return (str(sid) if sid else None), out

    idcfg = _identity_cfg()
    prefer_action_raise = bool(idcfg.get("release_prefer_action_cam_raise", True))
    action_min_raise = float(idcfg.get("release_action_cam_min_raise", 0.70))

    # Prefer action-cam when raise-majority is clear and association is strong
    anchor_info = per_cam.get(anchor)
    if anchor_info and anchor_info.get("sid"):
        r = float(anchor_info.get("wrist_raise") or 0)
        d = float(anchor_info.get("ball_dist") or 999)
        margin = float(anchor_info.get("raise_margin") or 0)
        mass = float(anchor_info.get("raise_mass") or 0)
        other_raises = [
            float(i.get("wrist_raise") or 0)
            for c, i in per_cam.items()
            if c != anchor and i.get("sid")
        ]
        best_other_r = max(other_raises) if other_raises else 0.0
        # Only trust solo action-cam when its raise clearly leads / ties other cams
        raise_leads = r >= best_other_r - 0.05 or not other_raises
        strong_pose = r >= 0.5 or (d <= 100.0 and r >= 0.35)
        # At release the ball often leaves the hand: high raise + clear ID on
        # action cam beats side-cam ball proximity (rebounder / passer trap).
        # But do NOT lock action-cam when another cam has clearly higher raise.
        if (
            prefer_action_raise
            and r >= action_min_raise
            and margin >= 0.12
            and mass >= 6.0
            and raise_leads
        ):
            return _pack(anchor_info, "cam_raise_majority")
        if margin >= 0.15 and mass >= 8.0 and strong_pose and raise_leads:
            return _pack(anchor_info, "cam_raise_majority")
        # Low-raise (TT / gather / layup at rim): trust action-cam holder only when
        # ball is clearly closer than other cams' candidates with different IDs.
        if r < 0.35 and d <= 100.0 and margin >= 0.35 and mass >= 20.0:
            # Side cams often see the driver with raised arms while action-cam
            # tracks the ball at the rim (no wrist raise).
            side_votes: Counter[str] = Counter()
            for c, i in per_cam.items():
                if c == anchor or not i.get("sid"):
                    continue
                rr = float(i.get("wrist_raise") or 0)
                if rr < 0.45:
                    continue
                sid_v = str(i.get("rematch_sid") or i.get("sid"))
                w = 1.6 if i.get("rematch_sid") else 1.0
                side_votes[sid_v] += w
            if side_votes:
                ranked = side_votes.most_common(2)
                best_side, w1 = ranked[0]
                w2 = float(ranked[1][1]) if len(ranked) > 1 else 0.0
                if (
                    w1 >= 2.0
                    and best_side != anchor_info.get("sid")
                    and w1 - w2 >= 0.4
                ):
                    chosen_side = None
                    best_rr = -1.0
                    for c, i in per_cam.items():
                        if c == anchor:
                            continue
                        sid_v = str(i.get("rematch_sid") or i.get("sid") or "")
                        if sid_v != best_side:
                            continue
                        rr = float(i.get("wrist_raise") or 0)
                        if rr > best_rr:
                            best_rr, chosen_side = rr, i
                    if chosen_side is not None:
                        return _pack(chosen_side, "multicam_rematch_consensus")
            rival = False
            for c, i in per_cam.items():
                if c == anchor or not i.get("sid") or i.get("sid") == anchor_info.get("sid"):
                    continue
                if float(i.get("ball_dist") or 999) + 25.0 < d:
                    rival = True
                    break
            if not rival:
                return _pack(anchor_info, "cam_holder_majority")

    # Strong single-cam evidence anywhere (raised + near ball + clear maj)
    # Do not let a side cam steal from a clear action-cam raise of another ID.
    anchor_raise = float((anchor_info or {}).get("wrist_raise") or 0)
    anchor_sid = (anchor_info or {}).get("sid")
    strong: list[tuple[float, str, dict[str, Any]]] = []
    for cam, info in per_cam.items():
        if not info.get("sid"):
            continue
        r = float(info.get("wrist_raise") or 0)
        d = float(info.get("ball_dist") or 999)
        margin = float(info.get("raise_margin") or 0)
        mass = float(info.get("raise_mass") or 0)
        if r >= 0.7 and d <= 180.0 and margin >= 0.10 and mass >= 5.0:
            if (
                prefer_action_raise
                and cam != anchor
                and anchor_sid
                and info.get("sid") != anchor_sid
                and anchor_raise >= action_min_raise
            ):
                continue
            score = r * (1.25 if cam == anchor else 1.0) / (1.0 + d / 200.0)
            if info.get("rematch_sid"):
                score *= 1.15
            strong.append((score, cam, info))
    if strong:
        strong.sort(key=lambda x: -x[0])
        return _pack(strong[0][2], f"strong_cam:{strong[0][1]}")

    # Multi-cam weighted vote — boost action cam heavily when raise is weak
    # (TT / gather), because side cams often associate the inbound passer.
    score: Counter[str] = Counter()
    max_raise = max(
        (float(i.get("wrist_raise") or 0) for i in per_cam.values()),
        default=0.0,
    )
    low_raise_session = max_raise < 0.40
    for cam, info in per_cam.items():
        sid = info.get("sid")
        if not sid:
            continue
        r = float(info.get("wrist_raise") or 0)
        d = float(info.get("ball_dist") or 999)
        margin = float(info.get("raise_margin") or 0)
        if low_raise_session:
            w = 3.0 if cam == anchor else 0.55
        else:
            w = 1.5 if cam == anchor else 1.0
        w *= 0.5 + 0.7 * min(r, 2.0) / 2.0 + 0.5 * max(0.0, 1.0 - d / 200.0)
        w *= 0.7 + 0.6 * max(margin, 0.0)
        if info.get("rematch_sid"):
            w *= 1.25
        score[str(sid)] += w
    if score:
        best_sid = score.most_common(1)[0][0]
        # Prefer the cam that voted for best_sid with highest local weight
        chosen = None
        best_local = -1.0
        for cam, info in per_cam.items():
            if info.get("sid") != best_sid:
                continue
            r = float(info.get("wrist_raise") or 0)
            d = float(info.get("ball_dist") or 999)
            cam_w = 3.0 if (low_raise_session and cam == anchor) else (
                1.5 if cam == anchor else 1.0
            )
            local = cam_w * (0.5 + r) / (1.0 + d / 250.0)
            if info.get("rematch_sid"):
                local *= 1.2
            if local > best_local:
                best_local, chosen = local, info
        meta["vote"] = dict(score)
        meta["low_raise"] = low_raise_session
        if chosen is not None:
            return _pack(chosen, "multicam_vote")
        return best_sid, {**meta, "reason": "multicam_vote"}

    # Fallback: single-frame spatial on anchor (no ball / no multi-cam)
    frames, fw = _load_pose_frames(session_id, anchor)
    balls = _load_ball_frames_by_ms(session_id, anchor)
    fr = _nearest_frame(frames, t_ms)
    if fr is None:
        meta["reason"] = "no_frame"
        return None, meta
    meta["frame"] = fr.get("frame")
    ball_xy = _nearest_ball_xy(balls, t_ms) if balls else None
    if ball_xy is not None:
        meta["ball_xy"] = [round(ball_xy[0], 1), round(ball_xy[1], 1)]
    shooter = pick_spatial_shooter(
        list(fr.get("persons") or []),
        frame_w=fw,
        ball_xy=ball_xy,
    )
    if shooter is None:
        meta["reason"] = "no_person"
        return None, meta
    tid = shooter.get("track_id")
    maj_sid, margin, mass, _votes = _track_raise_majority_sid(frames, tid, t_ms)
    sid = maj_sid or shooter.get("student_id")
    meta.update({
        "reason": "ball_person" if ball_xy is not None else "spatial_shooter",
        "bbox": shooter.get("bbox"),
        "identity_confidence": shooter.get("identity_confidence"),
        "wrist_raise": round(_wrist_raise_score(shooter), 3),
        "area": round(_person_area(shooter), 1),
        "raise_margin": round(margin, 3),
        "track_id": tid,
    })
    if ball_xy is not None:
        meta["ball_dist"] = round(_dist_to_ball(shooter, ball_xy), 1)
    return (str(sid) if sid else None), meta


def reassign_shooting_clips_by_spatial(
    session_id: str,
    by_student: dict[str, list[ActionClip]],
) -> dict[str, list[ActionClip]]:
    """
    Re-own shooting and breakthrough clips from video evidence.

    Shooting: ball–person / raise-majority at release.
    Triple-threat: same association at clip start (ball often still on the driver).
    """
    owned: list[ActionClip] = []
    others: dict[str, list[ActionClip]] = {sid: [] for sid in by_student}
    for sid, clips in by_student.items():
        for c in clips:
            if is_shooting_action(c.action_type) or c.action_type == "triple_threat":
                owned.append(c)
            else:
                others.setdefault(sid, []).append(c)

    out: dict[str, list[ActionClip]] = {sid: list(others.get(sid, [])) for sid in by_student}
    n_changed = 0
    for c in owned:
        if is_shooting_action(c.action_type):
            t = _release_ms(c)
        else:
            t = float(c.start_ms) if c.start_ms is not None else None
        if t is None:
            owner = c.student_id or "unknown"
            out.setdefault(owner, []).append(c)
            continue
        sid, meta = spatial_shooter_sid_at(session_id, t)
        new_meta = dict(c.metadata or {})
        new_meta["spatial_shooter"] = meta
        if sid and sid != c.student_id:
            n_changed += 1
            c = c.model_copy(update={"student_id": sid, "metadata": new_meta})
        else:
            c = c.model_copy(update={"metadata": new_meta})
        owner = c.student_id or sid or "unknown"
        out.setdefault(owner, []).append(c)

    for sid in list(out.keys()):
        out[sid].sort(key=lambda c: (c.start_frame, c.end_frame))
    print(f"  [action] spatial shooter reassign changed={n_changed}/{len(owned)}", flush=True)
    return out
