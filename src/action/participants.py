"""Attribute student_id / participant_ids on action clips (pass needs two)."""

from __future__ import annotations

from collections import defaultdict

from src.action.detect import load_ball_track, load_pose2d_for_camera
from src.cameras.registry import get_action_segment_camera
from src.types import ActionClip


def _person_center(person: dict) -> tuple[float, float] | None:
    bb = person.get("bbox")
    if bb and len(bb) >= 4:
        return 0.5 * (float(bb[0]) + float(bb[2])), 0.5 * (float(bb[1]) + float(bb[3]))
    kpts = person.get("keypoints")
    if not kpts:
        return None
    xs, ys, n = 0.0, 0.0, 0
    for kp in kpts:
        if len(kp) < 2:
            continue
        x, y = float(kp[0]), float(kp[1])
        if x == 0 and y == 0:
            continue
        conf = float(kp[2]) if len(kp) > 2 else 1.0
        if conf < 0.2:
            continue
        xs += x
        ys += y
        n += 1
    if n < 4:
        return None
    return xs / n, ys / n


def _ball_xy(ball: dict | None) -> tuple[float, float] | None:
    if not ball:
        return None
    c = ball.get("center")
    if c and len(c) >= 2:
        return float(c[0]), float(c[1])
    return None


def _mean_ball_dist(
    frames: list[dict],
    student_id: str,
    ball_by_frame: dict[int, dict],
) -> float | None:
    dists: list[float] = []
    for fr in frames:
        fidx = int(fr.get("frame", -1))
        bxy = _ball_xy(ball_by_frame.get(fidx))
        if bxy is None:
            continue
        for p in fr.get("persons") or []:
            if p.get("student_id") != student_id:
                continue
            cxy = _person_center(p)
            if cxy is None:
                continue
            dists.append(((cxy[0] - bxy[0]) ** 2 + (cxy[1] - bxy[1]) ** 2) ** 0.5)
    if not dists:
        return None
    return sum(dists) / len(dists)


def _presence_counts(frames: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for fr in frames:
        seen: set[str] = set()
        for p in fr.get("persons") or []:
            sid = p.get("student_id")
            if sid and sid not in seen:
                counts[sid] += 1
                seen.add(sid)
    return dict(counts)


def resolve_clip_participants(
    session_id: str,
    clip: ActionClip,
    *,
    primary_student_id: str,
    camera_id: str | None = None,
) -> ActionClip:
    """
    Stamp ``student_id`` + ``participant_ids`` on a clip.

    - Most actions: primary actor only.
    - ``pass``: try [passer, receiver] via early/late ball proximity among
      students visible in the window (falls back to primary + next present).
    """
    cam = camera_id or clip.anchor_camera or get_action_segment_camera()
    doc = load_pose2d_for_camera(session_id, cam)
    f0, f1 = int(clip.start_frame), int(clip.end_frame)
    window = [
        fr for fr in doc.get("frames", [])
        if f0 <= int(fr.get("frame", -1)) <= f1
    ]
    presence = _presence_counts(window)
    if primary_student_id not in presence:
        presence[primary_student_id] = presence.get(primary_student_id, 0)

    student_id = primary_student_id
    participant_ids = [primary_student_id]
    meta = dict(clip.metadata or {})
    meta["participants_camera"] = cam

    if clip.action_type == "pass" and window:
        # Pass requires ≥2 distinct people visible in the clip window.
        min_frames = max(2, len(window) // 15)
        visible = [s for s, n in presence.items() if n >= min_frames]
        if len(visible) < 2:
            meta["pass_rejected"] = "need_two_people_in_frame"
            meta["pass_visible_ids"] = visible
            participant_ids = [primary_student_id]
            student_id = primary_student_id
        else:
            ball = load_ball_track(session_id, cam)
            if not ball:
                for alt in ("cam_01", "cam_02", "cam_03"):
                    ball = load_ball_track(session_id, alt)
                    if ball:
                        break
            n = len(window)
            early = window[: max(1, n // 3)]
            late = window[max(0, (2 * n) // 3) :]
            sids = sorted(visible, key=lambda s: -presence[s])
            early_dist = {s: _mean_ball_dist(early, s, ball) for s in sids}
            late_dist = {s: _mean_ball_dist(late, s, ball) for s in sids}

            def _best(dist_map: dict[str, float | None], exclude: set[str] | None = None) -> str | None:
                exclude = exclude or set()
                ranked = [
                    (d, s) for s, d in dist_map.items()
                    if d is not None and s not in exclude
                ]
                if not ranked:
                    return None
                ranked.sort(key=lambda t: t[0])
                return ranked[0][1]

            passer = _best(early_dist) or primary_student_id
            receiver = _best(late_dist, exclude={passer})
            if receiver is None or receiver == passer:
                others = [s for s in sids if s != passer]
                receiver = others[0] if others else None

            # Sequential ownership: soft check — classroom ball tracks are noisy.
            # Require two distinct people; prefer early-near-passer and/or late-near-receiver.
            ed_p = early_dist.get(passer) if passer else None
            ed_r = early_dist.get(receiver) if receiver else None
            ld_p = late_dist.get(passer) if passer else None
            ld_r = late_dist.get(receiver) if receiver else None
            has_roles = (
                passer is not None
                and receiver is not None
                and passer != receiver
            )
            early_ok = ed_p is not None and (ed_r is None or ed_p <= ed_r * 1.65)
            late_ok = ld_r is not None and (ld_p is None or ld_r <= ld_p * 1.65)
            dists = (ed_p, ed_r, ld_p, ld_r)
            no_ball_dist = all(v is None for v in dists)
            # Classroom ball tracks are noisy: ≥2 people in frame + roles is enough.
            sequential_ok = has_roles and (
                early_ok or late_ok or no_ball_dist or len(visible) >= 2
            )
            if not sequential_ok:
                meta["pass_rejected"] = "need_sequential_passer_receiver"
                participant_ids = [primary_student_id]
                student_id = primary_student_id
            else:
                participant_ids = [passer, receiver]
                student_id = passer
            meta["passer_id"] = passer
            meta["receiver_id"] = receiver
            meta["pass_early_ball_dist"] = {
                s: (round(d, 1) if d is not None else None) for s, d in early_dist.items()
            }
            meta["pass_late_ball_dist"] = {
                s: (round(d, 1) if d is not None else None) for s, d in late_dist.items()
            }
            meta["pass_visible_ids"] = visible
            meta["pass_early_ok"] = early_ok
            meta["pass_late_ok"] = late_ok
    else:
        # Prefer most-present id if primary barely appears
        if presence:
            top = max(presence.items(), key=lambda kv: kv[1])[0]
            if presence.get(primary_student_id, 0) < max(1, presence[top] // 2):
                student_id = top
            participant_ids = [student_id]

    # Deduplicate preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for s in participant_ids:
        if s and s not in seen:
            ordered.append(s)
            seen.add(s)
    if not ordered:
        ordered = [primary_student_id]

    return clip.model_copy(update={
        "student_id": student_id,
        "participant_ids": ordered,
        "metadata": meta,
    })


def annotate_student_actions(
    session_id: str,
    clips: list[ActionClip],
    primary_student_id: str,
) -> list[ActionClip]:
    return [
        resolve_clip_participants(session_id, c, primary_student_id=primary_student_id)
        for c in clips
    ]
