"""Shared rule-based action detection helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.action.halpe2h36m import wholebody133_to_h36m
from src.cameras.registry import get_action_segment_camera
from src.config import data_path
from src.types import ActionClip, ActionPhase

MAX_CLIP_LEN = 243

# COCO-WholeBody body indices (image Y grows downward)
NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6
L_WRIST, R_WRIST = 9, 10
MIN_KPT_SCORE = 0.3


def _wrist_above_shoulder_and_neck(k: np.ndarray, wrist_idx: int = R_WRIST) -> bool:
    """Shooting release: wrist must be above shoulder line and neck (nose) in image coords."""
    if k[wrist_idx, 2] < MIN_KPT_SCORE:
        return False

    wrist_y = float(k[wrist_idx, 1])
    shoulder_ys: list[float] = []
    for idx in (L_SHOULDER, R_SHOULDER):
        if k[idx, 2] >= MIN_KPT_SCORE:
            shoulder_ys.append(float(k[idx, 1]))
    if not shoulder_ys:
        return False

    shoulder_y = min(shoulder_ys)
    if k[NOSE, 2] >= MIN_KPT_SCORE:
        neck_y = float(k[NOSE, 1])
    else:
        neck_y = shoulder_y

    return wrist_y < shoulder_y and wrist_y < neck_y


def _elbow_angle_deg(k: np.ndarray) -> float | None:
    """Right elbow angle (shoulder–elbow–wrist) in degrees; None if low confidence."""
    r_elbow = 8
    if min(float(k[i, 2]) for i in (R_SHOULDER, r_elbow, R_WRIST)) < MIN_KPT_SCORE:
        return None
    a, b, c = k[R_SHOULDER, :2], k[r_elbow, :2], k[R_WRIST, :2]
    ba, bc = a - b, c - b
    n = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if n < 1e-6:
        return None
    cos = float(np.clip(np.dot(ba, bc) / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _combined_wrist_y(k: np.ndarray) -> float:
    """Image-y of the higher wrist (min y) when either side is visible."""
    ys: list[float] = []
    for wi in (L_WRIST, R_WRIST):
        if k[wi, 2] >= MIN_KPT_SCORE:
            ys.append(float(k[wi, 1]))
    return min(ys) if ys else 1e9


def _combined_wrist_y_series(seq: list[tuple[int, np.ndarray]]) -> list[float]:
    return [_combined_wrist_y(k) for _, k in seq]


def _pick_best_release_peak(
    group: list[int],
    wrist_y: list[float],
    seq: list[tuple[int, np.ndarray]] | None = None,
    ball_by_frame: dict[int, dict] | None = None,
) -> int:
    """Keep the highest wrist (min image-Y) in a shot-cycle group."""
    del seq, ball_by_frame
    return min(group, key=lambda i: wrist_y[i])


def _merge_nearby_peaks(
    peaks: list[int],
    frames: list[int],
    wrist_y: list[float],
    merge_window: int,
    seq: list[tuple[int, np.ndarray]] | None = None,
    ball_by_frame: dict[int, dict] | None = None,
) -> list[int]:
    """Keep one release per shot cycle within merge_window."""
    if not peaks:
        return []
    merged: list[int] = []
    group = [peaks[0]]
    for p in peaks[1:]:
        if frames[p] - frames[group[-1]] < merge_window:
            group.append(p)
        else:
            merged.append(_pick_best_release_peak(group, wrist_y, seq, ball_by_frame))
            group = [p]
    merged.append(_pick_best_release_peak(group, wrist_y, seq, ball_by_frame))
    return merged


def load_ball_track(session_id: str, camera_id: str | None = None) -> dict[int, dict]:
    """
    Load ball detections keyed by frame index.
    Prefer same-camera track (ball_track_{cam}.json) so spatial wrist–ball
    checks are valid; fall back to official cam_04 ball_track.json.
    """
    out_dir = data_path("sessions", session_id, "shot_outcomes")
    candidates: list[Path] = []
    if camera_id:
        if camera_id == "cam_04" or camera_id.endswith("04"):
            candidates.append(out_dir / "ball_track.json")
        candidates.append(out_dir / f"ball_track_{camera_id}.json")
    candidates.append(out_dir / "ball_track.json")

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for fr in doc.get("frames", []):
        ball = fr.get("ball")
        if ball and ball.get("center"):
            out[int(fr["frame"])] = ball
    return out


def _nearest_ball(ball_by_frame: dict[int, dict], frame: int, window: int = 3) -> dict | None:
    if frame in ball_by_frame:
        return ball_by_frame[frame]
    for d in range(1, window + 1):
        if frame - d in ball_by_frame:
            return ball_by_frame[frame - d]
        if frame + d in ball_by_frame:
            return ball_by_frame[frame + d]
    return None


def _ball_supports_release(
    seq: list[tuple[int, np.ndarray]],
    peak_idx: int,
    ball_by_frame: dict[int, dict],
    near_wrist_px: float = 120.0,
    leave_rise_px: float = 25.0,
    shooting_hand: str | None = None,
) -> tuple[bool, float]:
    """
    Use ball track to confirm a wrist-peak is a real shot release.
    - Near release: ball should be close to shooting wrist
    - After release: ball should rise / move away from wrist (leave hand)
    Returns (supported, confidence_boost).
    """
    if not ball_by_frame:
        return True, 0.0  # no ball data → keep pose-only decision

    from src.action.shooting_hand import infer_shooting_hand_from_window, wrist_idx

    frames = [f for f, _ in seq]
    f_rel = frames[peak_idx]
    k = seq[peak_idx][1]
    if shooting_hand is None:
        shooting_hand, _ = infer_shooting_hand_from_window(seq, peak_idx, ball_by_frame)
    wi = wrist_idx(shooting_hand)  # type: ignore[arg-type]
    wrist = k[wi, :2]
    if k[wi, 2] < MIN_KPT_SCORE:
        # Fall back to whichever wrist is visible / nearer ball
        candidates: list[tuple[int, np.ndarray]] = []
        for widx in (L_WRIST, R_WRIST):
            if k[widx, 2] >= MIN_KPT_SCORE:
                candidates.append((widx, k[widx, :2]))
        if not candidates:
            return True, 0.0
        if ball_at := _nearest_ball(ball_by_frame, f_rel, window=4):
            bx, by = ball_at["center"]
            widx, wrist = min(
                candidates,
                key=lambda item: float(np.hypot(bx - item[1][0], by - item[1][1])),
            )
        else:
            widx, wrist = candidates[0]

    ball_at = _nearest_ball(ball_by_frame, f_rel, window=4)
    if ball_at is None:
        # Ball invisible at peak — weak negative, still allow pose rule
        return True, -0.05

    bx, by = ball_at["center"]
    dist = float(np.hypot(bx - wrist[0], by - wrist[1]))

    # Look ahead: ball should move upward (smaller Y) after release
    future_ys: list[float] = []
    for off in range(1, 12):
        b = _nearest_ball(ball_by_frame, f_rel + off, window=1)
        if b:
            future_ys.append(float(b["center"][1]))

    if dist > near_wrist_px:
        # Ball already left the hand (or other-cam coords) — do NOT reject
        if future_ys and (by - min(future_ys)) >= leave_rise_px:
            return True, 0.08
        return True, -0.02

    if not future_ys:
        return True, 0.05  # near wrist but no future track

    rise = by - min(future_ys)  # positive if ball went higher
    if rise >= leave_rise_px:
        return True, 0.1
    # Ball stayed at hand height → likely dribble / hold
    if rise < 5 and dist < near_wrist_px * 0.5:
        return False, 0.0
    return True, 0.02


def _wrist_peak_indices(
    seq: list[tuple[int, np.ndarray]],
    wrist_idx: int,
) -> list[int]:
    wrist_y = [float(k[wrist_idx, 1]) for _, k in seq]
    peaks: list[int] = []
    for i in range(1, len(wrist_y) - 1):
        if wrist_y[i] < wrist_y[i - 1] and wrist_y[i] < wrist_y[i + 1]:
            if _wrist_above_shoulder_and_neck(seq[i][1], wrist_idx):
                peaks.append(i)
    return peaks


def _shooting_release_candidates(
    seq: list[tuple[int, np.ndarray]],
    min_peak_distance: int,
    peak_merge_window: int = 100,
    ball_by_frame: dict[int, dict] | None = None,
) -> list[tuple[int, float, str, dict]]:
    """
    Local wrist-height peaks on the shooting side (right default; left fallback).
    Optional ball_by_frame filters dribble false positives.
    Returns list of (seq_index, confidence, shooting_hand, hand_meta).
    """
    from src.action.shooting_hand import infer_shooting_hand_from_window

    frames = [f for f, _ in seq]
    ball_by_frame = ball_by_frame or {}

    peaks_r = _wrist_peak_indices(seq, R_WRIST)
    peaks_l = _wrist_peak_indices(seq, L_WRIST)
    # Primary: right-wrist peaks (legacy timing). Add left-only peaks far from any right peak.
    peaks: list[int] = list(peaks_r)
    for p in peaks_l:
        if not any(abs(frames[p] - frames[r]) < peak_merge_window for r in peaks_r):
            peaks.append(p)
    peaks.sort()

    wrist_y = _combined_wrist_y_series(seq)
    peaks = _merge_nearby_peaks(
        peaks, frames, wrist_y, peak_merge_window, seq=seq, ball_by_frame=ball_by_frame,
    )

    spaced: list[tuple[int, float, str, dict]] = []
    for p in peaks:
        if spaced and (frames[p] - frames[spaced[-1][0]]) < min_peak_distance:
            continue
        hand, hand_meta = infer_shooting_hand_from_window(seq, p, ball_by_frame)
        ok, boost = _ball_supports_release(seq, p, ball_by_frame, shooting_hand=hand)
        if not ok:
            continue
        spaced.append((
            p,
            float(np.clip(0.85 + boost, 0.5, 0.98)),
            hand,
            hand_meta,
        ))

    if not spaced:
        valid: list[int] = []
        for i, (_, k) in enumerate(seq):
            for wi in (R_WRIST, L_WRIST):
                if _wrist_above_shoulder_and_neck(k, wi):
                    valid.append(i)
                    break
        if valid:
            best = int(min(valid, key=lambda i: _combined_wrist_y(seq[i][1])))
            hand, hand_meta = infer_shooting_hand_from_window(seq, best, ball_by_frame)
            ok, boost = _ball_supports_release(seq, best, ball_by_frame, shooting_hand=hand)
            if ok or not ball_by_frame:
                spaced = [(
                    best,
                    float(np.clip(0.75 + boost, 0.5, 0.95)),
                    hand,
                    hand_meta,
                )]

    return spaced


def detect_shooting_phases(
    seq: list[tuple[int, np.ndarray]],
    action_type: str = "free_throw",
    pre_frames: int = 40,
    post_frames: int = 25,
    min_peak_distance: int = 45,
    ball_by_frame: dict[int, dict] | None = None,
) -> list[ActionClip]:
    if len(seq) < 30:
        return []

    frames = [f for f, _ in seq]
    peaks = _shooting_release_candidates(
        seq, min_peak_distance, ball_by_frame=ball_by_frame,
    )

    clips: list[ActionClip] = []
    for peak_idx, conf, shooting_hand, hand_meta in peaks:
        release = frames[peak_idx]
        start = frames[max(0, peak_idx - pre_frames)]
        end = frames[min(len(frames) - 1, peak_idx + post_frames)]
        if action_type == "layup":
            span = max(1, release - start)
            g1 = start + int(0.40 * span)
            g2 = start + int(0.70 * span)
            phases = [
                ActionPhase(name="approach", start=start, end=g1),
                ActionPhase(name="gather", start=g1, end=g2),
                ActionPhase(name="takeoff", start=g2, end=max(g2, release - 3)),
                ActionPhase(name="release", start=release - 2, end=release + 2),
                ActionPhase(name="finish", start=release + 2, end=end),
            ]
        else:
            mid = start + (release - start) // 2
            phases = [
                ActionPhase(name="load", start=start, end=mid),
                ActionPhase(name="set", start=mid, end=release - 3),
                ActionPhase(name="release", start=release - 2, end=release + 2),
                ActionPhase(name="follow_through", start=release + 2, end=end),
            ]
        clips.append(ActionClip(
            action_type=action_type,
            start_frame=start,
            end_frame=end,
            phases=phases,
            confidence=conf,
            metadata={
                "shooting_hand": shooting_hand,
                "shooting_hand_meta": hand_meta,
            },
        ))
    return clips


def load_pose2d_for_camera(session_id: str, camera_id: str) -> dict:
    p = data_path("sessions", session_id, "perception", camera_id, "pose2d.json")
    if not p.exists():
        return {"frames": [], "fps": 30.0}
    return json.loads(p.read_text(encoding="utf-8"))


def load_master_pose2d(session_id: str) -> dict:
    return load_pose2d_for_camera(session_id, get_action_segment_camera())


def extract_student_sequence(pose2d: dict, student_id: str) -> list[tuple[int, np.ndarray]]:
    seq = []
    for fr in pose2d.get("frames", []):
        fidx = fr["frame"]
        for person in fr.get("persons", []):
            if person.get("student_id") == student_id:
                kpts = np.array(person["keypoints"], dtype=np.float32)
                seq.append((fidx, kpts))
                break
    return seq


def resolve_pose_camera_for_student(
    session_id: str,
    student_id: str,
    preferred: str | None = None,
    min_frames: int = 40,
) -> tuple[str, dict, list[tuple[int, np.ndarray]]]:
    """
    Prefer the action-segment camera; if that student has no usable pose there,
    fall back to the pose camera with the most frames for that student.
    """
    preferred = preferred or get_action_segment_camera()
    order = [preferred] + [c for c in ("cam_01", "cam_02", "cam_03") if c != preferred]
    best_cam, best_doc, best_seq = preferred, {"frames": [], "fps": 30.0}, []
    for cam in order:
        doc = load_pose2d_for_camera(session_id, cam)
        seq = extract_student_sequence(doc, student_id)
        if cam == preferred and len(seq) >= min_frames:
            return cam, doc, seq
        if len(seq) > len(best_seq):
            best_cam, best_doc, best_seq = cam, doc, seq
    return best_cam, best_doc, best_seq


def detect_free_throw_phases(seq: list[tuple[int, np.ndarray]]) -> ActionClip | None:
    clips = detect_shooting_phases(seq, action_type="free_throw")
    return clips[0] if clips else None


def classify_action_stub(h36m_seq: np.ndarray) -> str:
    """Legacy stub — prefer classify_release_action on wholebody sequences."""
    if h36m_seq.shape[0] < 10:
        return "unknown"
    wrist_motion = np.std(h36m_seq[:, 10, :2])
    knee_flex = np.mean(h36m_seq[:, 13, 1] - h36m_seq[:, 0, 1])
    if wrist_motion > 8:
        return "free_throw"
    if knee_flex > 50:
        return "layup"
    return "triple_threat"


def classify_release_action(
    seq: list[tuple[int, np.ndarray]],
    release_frame: int,
    pre_frames: int = 50,
    hoop_xy: tuple[float, float] | None = None,
) -> tuple[str, dict]:
    """
    Infer action type from pose motion *before* a release (no prior label needed).

    Heuristic (cam_03 / free-throw-lane view):
    - Layup: run-up **toward** hoop (person→hoop distance decreases) + travel
    - Jump shot: little approach, clear vertical jump (pelvis/ankle lift)
    - Free throw: planted release without approaching the hoop
    """
    if not seq:
        return "unknown", {"reason": "empty_seq"}

    frames = [f for f, _ in seq]
    try:
        peak_i = frames.index(release_frame)
    except ValueError:
        peak_i = min(range(len(frames)), key=lambda i: abs(frames[i] - release_frame))

    window = seq[max(0, peak_i - pre_frames): peak_i + 1]
    if len(window) < 12:
        return "unknown", {"reason": "short_window", "n": len(window)}

    h_list = []
    for _, k in window:
        try:
            h_list.append(wholebody133_to_h36m(k))
        except Exception:
            continue
    if len(h_list) < 12:
        return "unknown", {"reason": "h36m_fail", "n": len(h_list)}

    h = np.stack(h_list)  # T,17,3 — 0 pelvis, 3 RAnkle, 6 LAnkle, 16 RWrist
    torso = float(np.linalg.norm(h[len(h) // 2, 8, :2] - h[len(h) // 2, 0, :2]) + 1e-3)

    def _path_len(xy: np.ndarray) -> float:
        if len(xy) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))

    pelvis = h[:, 0, :2]
    rankle, lankle = h[:, 3, :2], h[:, 6, :2]
    rwrist = h[:, 16, :2]

    pelvis_travel = _path_len(pelvis) / torso
    ankle_travel = 0.5 * (_path_len(rankle) + _path_len(lankle)) / torso
    pelvis_dx = float(abs(pelvis[-1, 0] - pelvis[0, 0])) / torso
    ankle_dy = 0.5 * (
        float(rankle[:, 1].max() - rankle[:, 1].min())
        + float(lankle[:, 1].max() - lankle[:, 1].min())
    ) / torso
    wrist_std = float(np.std(rwrist)) / torso

    # Vertical jump cue: image-y decreases as athlete rises (crouch → apex near release)
    mid = max(4, len(h) // 2)
    early_pelvis_y = float(np.median(pelvis[:mid, 1]))
    late_pelvis_y = float(np.min(pelvis[mid:, 1]))
    pelvis_up = (early_pelvis_y - late_pelvis_y) / torso
    early_rankle_y = float(np.median(rankle[:mid, 1]))
    early_lankle_y = float(np.median(lankle[:mid, 1]))
    early_ankle_y = 0.5 * (early_rankle_y + early_lankle_y)
    late_ankle_y = 0.5 * (
        float(np.min(rankle[mid:, 1])) + float(np.min(lankle[mid:, 1]))
    )
    ankle_up = (early_ankle_y - late_ankle_y) / torso
    # Per-foot peak lift (torso-normalized); used with simultaneous off-ground
    rankle_up = (early_rankle_y - float(np.min(rankle[mid:, 1]))) / torso
    lankle_up = (early_lankle_y - float(np.min(lankle[mid:, 1]))) / torso
    both_feet_up = float(min(rankle_up, lankle_up))

    # Both feet off the ground at the *same* late-window frames
    # (planted baseline − current ankle y ≥ lift_thr). Filters single-foot
    # pivot / FT tip-toe that inflate ankle_up without a real jump.
    lift_thr = 0.16 * torso
    r_lift = early_rankle_y - rankle[mid:, 1]
    l_lift = early_lankle_y - lankle[mid:, 1]
    both_off = (r_lift >= lift_thr) & (l_lift >= lift_thr)
    both_feet_off_n = int(np.sum(both_off))
    both_feet_off_frac = float(np.mean(both_off)) if len(both_off) else 0.0
    # Longest consecutive both-off run (noise-resistant)
    both_feet_off_run = 0
    _run = 0
    for flag in both_off.tolist():
        _run = _run + 1 if flag else 0
        if _run > both_feet_off_run:
            both_feet_off_run = _run
    # Require simultaneous lift; allow a single clear frame when both feet
    # and pelvis show a strong jump (sparse pose / short apex).
    both_feet_airborne = (
        both_feet_off_run >= 2
        or (both_feet_off_n >= 3 and both_feet_off_frac >= 0.12)
        or (
            both_feet_off_n >= 1
            and both_feet_up >= 0.35
            and pelvis_up >= 0.50
        )
    )

    # Person→hoop distance change (full window + late window before release)
    approach_ratio = 0.0
    late_approach = 0.0
    dist_start = dist_end = 0.0
    if hoop_xy is not None:
        hx, hy = float(hoop_xy[0]), float(hoop_xy[1])
        d0 = float(np.linalg.norm(pelvis[0] - np.array([hx, hy])))
        d1 = float(np.linalg.norm(pelvis[-1] - np.array([hx, hy])))
        dist_start, dist_end = d0, d1
        if d0 > 1.0:
            approach_ratio = (d0 - d1) / d0  # >0 means approaching hoop
        late_n = min(25, max(5, len(pelvis) // 2))
        lp = pelvis[-late_n:]
        ld0 = float(np.linalg.norm(lp[0] - np.array([hx, hy])))
        ld1 = float(np.linalg.norm(lp[-1] - np.array([hx, hy])))
        if ld0 > 1.0:
            late_approach = (ld0 - ld1) / ld0

    meta = {
        "pelvis_travel": round(pelvis_travel, 3),
        "ankle_travel": round(ankle_travel, 3),
        "pelvis_dx": round(pelvis_dx, 3),
        "ankle_dy": round(ankle_dy, 3),
        "pelvis_up": round(pelvis_up, 3),
        "ankle_up": round(ankle_up, 3),
        "rankle_up": round(rankle_up, 3),
        "lankle_up": round(lankle_up, 3),
        "both_feet_up": round(both_feet_up, 3),
        "both_feet_off_n": both_feet_off_n,
        "both_feet_off_run": both_feet_off_run,
        "both_feet_off_frac": round(both_feet_off_frac, 3),
        "both_feet_airborne": both_feet_airborne,
        "wrist_std": round(wrist_std, 3),
        "approach_ratio": round(approach_ratio, 3),
        "late_approach": round(late_approach, 3),
        "dist_start": round(dist_start, 1),
        "dist_end": round(dist_end, 1),
        "n": len(h),
    }

    # --- Decision tree for rim-gated release ---
    leaving = approach_ratio <= -0.12 if hoop_xy is not None else False
    near_rim = hoop_xy is not None and dist_end > 0 and dist_end <= 460.0
    closing = hoop_xy is not None and approach_ratio >= 0.12 and dist_end <= dist_start * 0.85
    planted = ankle_travel < 2.0 and pelvis_travel < 2.5 and approach_ratio < 0.12
    # Jump shot requires both feet off the ground together (not FT tip-toe / pivot).
    # Soften only when pelvis + weaker foot both show a clear lift.
    lift_ok = (
        (pelvis_up >= 0.45 and ankle_up >= 0.18 and both_feet_up >= 0.18)
        or pelvis_up >= 0.70
        or both_feet_up >= 0.40
    )
    hopped = both_feet_airborne and lift_ok

    # 1) Clear near-rim finish with run-up → layup
    #    Exception: big hop near rim → treat as pull-up jumper (g5)
    if near_rim and not leaving and ankle_travel >= 2.8 and (closing or approach_ratio >= 0.10):
        # Big hop with little approach → pull-up; clear approach → layup
        if hopped and pelvis_up >= 0.55 and approach_ratio < 0.20:
            return "jump_shot", {**meta, "reason": "near_rim_pullup"}
        return "layup", {**meta, "reason": "near_rim_finish"}

    # 2) Strong close-out toward hoop (even if hoop xy is a bit off)
    if (
        hoop_xy is not None
        and not leaving
        and approach_ratio >= 0.22
        and dist_end <= dist_start * 0.70
        and ankle_travel >= 3.0
        and dist_end <= 520.0
    ):
        if hopped and pelvis_up >= 0.55 and dist_end > 400.0:
            return "jump_shot", {**meta, "reason": "drive_pullup"}
        return "layup", {**meta, "reason": "drive_toward_hoop"}

    # 3) High travel finishing near basket → layup
    if (
        ankle_travel >= 4.0
        and wrist_std >= 0.8
        and not leaving
        and dist_end <= 650.0
        and (approach_ratio >= 0.05 or near_rim or dist_end <= 560.0 or ankle_travel >= 5.0)
    ):
        return "layup", {**meta, "reason": "high_travel_layup"}
    # Very high travel farther out only with clear approach + not too far
    if (
        ankle_travel >= 8.0
        and not leaving
        and approach_ratio >= 0.08
        and dist_end <= 800.0
    ):
        return "layup", {**meta, "reason": "extreme_drive_layup"}

    # Also: wrist_std gate may fail — still layup when travel is clear
    if ankle_travel >= 5.5 and not leaving and dist_end <= 650.0:
        return "layup", {**meta, "reason": "high_travel_layup_soft"}

    # 5) Vertical hop, little approach → jump_shot
    if hopped and approach_ratio < 0.12 and pelvis_dx < 2.0 and ankle_travel < 3.5:
        return "jump_shot", {**meta, "reason": "vertical_jump_shot"}

    # 6) Planted free throw
    if planted and wrist_std >= 0.8:
        return "free_throw", {**meta, "reason": "planted_shooting"}

    # 7) Leaving the hoop after release window
    if leaving:
        # Mild leave + high travel near basket → layup finish; hard leave is noise/FT
        if ankle_travel >= 5.0 and dist_end <= 700.0 and approach_ratio > -0.40:
            return "layup", {**meta, "reason": "leave_after_drive"}
        if hopped and ankle_travel < 6.0:
            return "jump_shot", {**meta, "reason": "leave_after_jump"}
        # Pull-up after drive: leave window with moderate lift / travel — not planted FT
        if (
            ankle_travel < 5.5
            and wrist_std >= 0.75
            and (
                pelvis_up >= 0.32
                or ankle_up >= 0.22
                or both_feet_up >= 0.18
                or both_feet_airborne
            )
        ):
            return "jump_shot", {**meta, "reason": "leave_after_pullup"}
        return "free_throw", {**meta, "reason": "leave_after_shot"}

    if hopped:
        return "jump_shot", {**meta, "reason": "soft_jump_shot"}
    if wrist_std >= 0.8 or hoop_xy is not None:
        return "free_throw", {**meta, "reason": "default_free_throw"}
    return "unknown", {**meta, "reason": "low_confidence"}


# Public alias — prefer this name in new code
classify_action_from_pose = classify_release_action
