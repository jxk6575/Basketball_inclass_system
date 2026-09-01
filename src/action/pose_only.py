"""Non-shooting actions: pass (cam ball leave) + triple_threat (pose-first, ball aux)."""

from __future__ import annotations

import numpy as np

from src.action.detect import (
    R_WRIST,
    _nearest_ball,
    _wrist_above_shoulder_and_neck,
    extract_student_sequence,
    load_ball_track,
    load_pose2d_for_camera,
)
from src.action.halpe2h36m import wholebody133_to_h36m
from src.cameras.registry import get_action_segment_camera
from src.types import ActionClip, ActionPhase

L_WRIST = 9
L_HIP, R_HIP = 11, 12


def _path_len(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def _h36m_window(seq: list[tuple[int, np.ndarray]], i0: int, i1: int) -> np.ndarray | None:
    hs = []
    for _, k in seq[i0:i1]:
        try:
            hs.append(wholebody133_to_h36m(k))
        except Exception:
            continue
    if len(hs) < 10:
        return None
    return np.stack(hs, axis=0)


def _bbox_aspect_series(h: np.ndarray) -> np.ndarray:
    """Per-frame keypoint bbox aspect = height / width (image coords)."""
    xy = h[:, :, :2].astype(np.float64)
    valid = np.isfinite(xy).all(axis=-1) & (np.abs(xy).sum(axis=-1) > 1.0)
    aspects = np.zeros(len(h), dtype=np.float64)
    for t in range(len(h)):
        pts = xy[t, valid[t]]
        if pts.shape[0] < 6:
            aspects[t] = np.nan
            continue
        w = float(pts[:, 0].max() - pts[:, 0].min())
        hh = float(pts[:, 1].max() - pts[:, 1].min())
        aspects[t] = hh / max(w, 1.0)
    return aspects


def _pose_features(h: np.ndarray) -> dict[str, float]:
    torso = float(np.linalg.norm(h[len(h) // 2, 8, :2] - h[len(h) // 2, 0, :2]) + 1e-3)
    pelvis = h[:, 0, :2]
    rankle, lankle = h[:, 3, :2], h[:, 6, :2]
    rwrist = h[:, 16, :2]
    lwrist = h[:, 13, :2]

    aspects = _bbox_aspect_series(h)
    good = aspects[np.isfinite(aspects) & (aspects > 0.3)]
    if len(good) >= 5:
        n0 = max(3, len(good) // 5)
        aspect_start = float(np.median(good[:n0]))
        aspect_min = float(np.min(good))
        aspect_drop = (aspect_start - aspect_min) / max(aspect_start, 1e-3)
        aspect_mean = float(np.median(good))
    else:
        aspect_start = aspect_min = aspect_mean = aspect_drop = 0.0

    pelvis_y = pelvis[:, 1]
    cog_drop = float(pelvis_y.max() - np.median(pelvis_y[: max(3, len(pelvis_y) // 5)])) / torso

    # Two-hand raise to chest (pass / catch cue): wrists in chest band, rising
    lsho, rsho = h[:, 11, :2], h[:, 14, :2]
    sho_y = 0.5 * (lsho[:, 1] + rsho[:, 1])
    hip_y = pelvis[:, 1]
    chest_lo = sho_y
    chest_hi = sho_y + 0.45 * np.maximum(hip_y - sho_y, torso * 0.5)
    both_chest = []
    palms_out = []
    for t in range(len(h)):
        ry, ly = float(rwrist[t, 1]), float(lwrist[t, 1])
        in_band = (
            float(chest_lo[t]) - 0.05 * torso <= ry <= float(chest_hi[t]) + 0.1 * torso
            and float(chest_lo[t]) - 0.05 * torso <= ly <= float(chest_hi[t]) + 0.1 * torso
        )
        both_chest.append(in_band)
        # palms-out proxy: wrists outside / at shoulders laterally
        out = (
            abs(float(rwrist[t, 0]) - float(rsho[t, 0])) >= 0.15 * torso
            and abs(float(lwrist[t, 0]) - float(lsho[t, 0])) >= 0.15 * torso
        )
        palms_out.append(out and in_band)
    n0 = max(3, len(h) // 5)
    n1 = max(n0 + 1, 2 * len(h) // 5)
    n2 = max(n1 + 1, 3 * len(h) // 5)
    chest_early = float(np.mean(both_chest[:n0]))
    chest_mid = float(np.mean(both_chest[n0:n2]))
    chest_frac = float(np.mean(both_chest))
    palms_frac = float(np.mean(palms_out))
    # Rise: wrists move up (smaller y) from early → mid
    early_wy = 0.5 * (float(np.mean(rwrist[:n0, 1])) + float(np.mean(lwrist[:n0, 1])))
    mid_wy = 0.5 * (float(np.mean(rwrist[n0:n2, 1])) + float(np.mean(lwrist[n0:n2, 1])))
    wrist_raise = (early_wy - mid_wy) / max(torso, 1.0)

    return {
        "torso": torso,
        "pelvis_travel": _path_len(pelvis) / torso,
        "ankle_travel": 0.5 * (_path_len(rankle) + _path_len(lankle)) / torso,
        "pelvis_dx": float(abs(pelvis[-1, 0] - pelvis[0, 0])) / torso,
        "pelvis_dy": float(abs(pelvis[-1, 1] - pelvis[0, 1])) / torso,
        "wrist_std": 0.5 * (float(np.std(rwrist)) + float(np.std(lwrist))) / torso,
        "wrist_dy": 0.5 * (
            float(rwrist[:, 1].max() - rwrist[:, 1].min())
            + float(lwrist[:, 1].max() - lwrist[:, 1].min())
        ) / torso,
        "wrist_dx": 0.5 * (
            float(abs(rwrist[-1, 0] - rwrist[0, 0]))
            + float(abs(lwrist[-1, 0] - lwrist[0, 0]))
        ) / torso,
        "aspect_start": aspect_start,
        "aspect_min": aspect_min,
        "aspect_mean": aspect_mean,
        "aspect_drop": float(max(0.0, aspect_drop)),
        "cog_drop": float(max(0.0, cog_drop)),
        "chest_frac": chest_frac,
        "chest_early": chest_early,
        "chest_mid": chest_mid,
        "palms_frac": palms_frac,
        "wrist_raise": float(wrist_raise),
    }


def _ball_features(
    seq: list[tuple[int, np.ndarray]],
    i0: int,
    i1: int,
    ball_by_frame: dict[int, dict],
    torso: float,
) -> dict[str, float]:
    """
    Ball cues from the *same* pose camera track (cam_01–03).

    Pass: ball starts near hands, then leaves with lateral flight.
    Triple-threat: ball stays held near hands/torso (not floor dribble, not leave).
    """
    empty = {
        "ball_seen": 0.0,
        "held_frac": 0.0,
        "held_frac_early": 0.0,
        "held_frac_late": 0.0,
        "dribble_frac": 0.0,
        "ball_leave": 0.0,
        "ball_travel": 0.0,
        "ball_dx": 0.0,
        "ball_dy": 0.0,
        "ball_fly_mid": 0.0,
    }
    if not ball_by_frame:
        return empty

    near_px = max(90.0, 1.35 * torso)
    mid = i0 + (i1 - i0) // 2
    mid0 = i0 + (i1 - i0) // 3
    mid1 = i0 + 2 * (i1 - i0) // 3
    dists: list[float] = []
    centers: list[tuple[float, float]] = []
    near_flags: list[bool] = []
    dribble_flags: list[bool] = []
    early_near: list[bool] = []
    late_near: list[bool] = []
    mid_centers: list[tuple[float, float]] = []

    for i in range(i0, i1):
        fr, k = seq[i]
        ball = _nearest_ball(ball_by_frame, int(fr), window=2)
        if ball is None or not ball.get("center"):
            continue
        bx, by = float(ball["center"][0]), float(ball["center"][1])
        wrists = []
        for wi in (L_WRIST, R_WRIST):
            if float(k[wi, 2]) >= 0.25:
                wrists.append(k[wi, :2])
        if not wrists:
            continue
        d = min(float(np.hypot(bx - float(w[0]), by - float(w[1]))) for w in wrists)
        hips = []
        for hi in (L_HIP, R_HIP):
            if float(k[hi, 2]) >= 0.25:
                hips.append(float(k[hi, 1]))
        hip_y = float(np.mean(hips)) if hips else by
        near = d <= near_px
        # Floor dribble: ball well below hips while still somewhat near body x
        dribble = by > hip_y + 0.35 * torso and d < near_px * 1.8

        dists.append(d)
        centers.append((bx, by))
        near_flags.append(near)
        dribble_flags.append(dribble)
        if i < mid:
            early_near.append(near)
        else:
            late_near.append(near)
        if mid0 <= i < mid1:
            mid_centers.append((bx, by))

    n = len(dists)
    if n < 4:
        return empty

    early_d = dists[: max(2, n // 3)]
    late_d = dists[-max(2, n // 3) :]
    xy = np.asarray(centers, dtype=np.float64)
    fly_mid = 0.0
    if len(mid_centers) >= 2:
        mxy = np.asarray(mid_centers, dtype=np.float64)
        fly_mid = _path_len(mxy) / max(torso, 1.0)
    return {
        "ball_seen": float(n) / max(i1 - i0, 1),
        "held_frac": float(sum(near_flags)) / n,
        "held_frac_early": float(sum(early_near)) / max(len(early_near), 1),
        "held_frac_late": float(sum(late_near)) / max(len(late_near), 1),
        "dribble_frac": float(sum(dribble_flags)) / n,
        "ball_leave": float(np.mean(late_d) - np.mean(early_d)) / max(torso, 1.0),
        "ball_travel": _path_len(xy) / max(torso, 1.0),
        "ball_dx": float(abs(xy[-1, 0] - xy[0, 0])) / max(torso, 1.0),
        "ball_dy": float(abs(xy[:, 1].max() - xy[:, 1].min())) / max(torso, 1.0),
        "ball_fly_mid": float(fly_mid),
    }


def _has_shooting_release(seq: list[tuple[int, np.ndarray]], i0: int, i1: int) -> bool:
    for wi in (L_WRIST, R_WRIST):
        wrist_y = [float(seq[i][1][wi, 1]) for i in range(i0, i1)]
        if len(wrist_y) < 5:
            continue
        for j in range(1, len(wrist_y) - 1):
            if wrist_y[j] < wrist_y[j - 1] and wrist_y[j] < wrist_y[j + 1]:
                if _wrist_above_shoulder_and_neck(seq[i0 + j][1], wi):
                    return True
    return False


def _has_crouch(feat: dict[str, float]) -> bool:
    """
    Triple-threat / ready stance: lowered CoG → bbox aspect (h/w) shrinks.

    Classroom kids often only dip CoG slightly before a cut — keep thresholds soft.
    Side-view crouches often drop amin below 1.5; only reject degenerate flat boxes.
    """
    drop = float(feat.get("aspect_drop", 0.0))
    amin = float(feat.get("aspect_min", 0.0))
    a0 = float(feat.get("aspect_start", 0.0))
    cog = float(feat.get("cog_drop", 0.0))
    # Degenerate / cling detections only
    if amin > 0 and amin < 0.55:
        return False
    if drop >= 0.10:
        return True
    if a0 > 0.5 and amin > 0 and amin / a0 <= 0.90 and cog >= 0.08:
        return True
    if cog >= 0.12:
        return True
    return False


def _has_cut_or_cog(feat: dict[str, float]) -> bool:
    """Weak breakthrough: CoG dip + lateral cut / direction change (not textbook TT)."""
    cog = float(feat.get("cog_drop", 0.0))
    drop = float(feat.get("aspect_drop", 0.0))
    at = float(feat.get("ankle_travel", 0.0))
    pdx = float(feat.get("pelvis_dx", 0.0))
    # Lower CoG then move sideways / change direction
    if cog >= 0.08 and (pdx >= 0.55 or at >= 1.2) and at < 11.0:
        return True
    if drop >= 0.08 and at >= 1.6 and at < 11.0:
        return True
    # Clear lateral cut even with mild CoG signal
    if pdx >= 1.0 and at >= 2.0 and at < 10.0 and cog >= 0.05:
        return True
    return False


def classify_pose_only_window(feat: dict[str, float]) -> tuple[str, float, str]:
    """
    Label a non-shooting window.

    - pass: two-hand raise to chest + ball fly/leave (still needs ball cues)
    - triple_threat: **pose-first** (crouch / CoG drop / cut); ball is
      auxiliary boost / soft veto only — do not require high held_frac
      (ball detector recall is too low for a hard gate)
    """
    ws = feat["wrist_std"]
    wdy = feat["wrist_dy"]
    at = float(feat.get("ankle_travel", 0.0))
    ball_seen = float(feat.get("ball_seen", 0.0))
    held = float(feat.get("held_frac", 0.0))
    held_e = float(feat.get("held_frac_early", 0.0))
    held_l = float(feat.get("held_frac_late", 0.0))
    drib = float(feat.get("dribble_frac", 0.0))
    leave = float(feat.get("ball_leave", 0.0))
    bdx = float(feat.get("ball_dx", 0.0))
    bdy = float(feat.get("ball_dy", 0.0))
    btr = float(feat.get("ball_travel", 0.0))
    fly_mid = float(feat.get("ball_fly_mid", 0.0))
    chest_mid = float(feat.get("chest_mid", 0.0))
    chest_early = float(feat.get("chest_early", 0.0))
    chest_frac = float(feat.get("chest_frac", 0.0))
    palms = float(feat.get("palms_frac", 0.0))
    raise_ = float(feat.get("wrist_raise", 0.0))
    drop = float(feat.get("aspect_drop", 0.0))
    cog = float(feat.get("cog_drop", 0.0))
    pdx = float(feat.get("pelvis_dx", 0.0))
    crouch = _has_crouch(feat)
    cut = _has_cut_or_cog(feat)

    hands_pass = (
        chest_mid >= 0.24
        and raise_ >= 0.06
        and palms >= 0.10
        and chest_early <= chest_mid + 0.10
        and drib < 0.40
        and at < 7.5
    )
    # Soft pass: clear ball flight + any chest/raise cue (classroom form varies)
    hands_pass_soft = (
        (chest_mid >= 0.15 or raise_ >= 0.04 or palms >= 0.06)
        and drib < 0.45
        and at < 8.5
        and fly_mid >= 0.34
    )
    ball_flyby = fly_mid >= 0.50 or (ball_seen >= 0.08 and btr >= 0.70 and bdx >= 0.50)
    # Legacy lateral leave still counts if hands also raise
    ball_leave_pass = (
        ball_seen >= 0.12
        and held_e >= 0.30
        and held_l <= 0.65
        and leave >= 0.25
        and bdx >= 0.70
        and bdy <= 1.15 * max(bdx, 1e-3)
        and drib < 0.35
    )

    # Pass candidate — do not steal breakthrough (CoG/cut) windows unless ball clearly flies.
    pass_ball_ok = ball_flyby or ball_leave_pass or fly_mid >= 0.28
    clear_pass_ball = (
        fly_mid >= 0.50
        or ball_leave_pass
        or (ball_flyby and fly_mid >= 0.36)
    )
    breakthrough_like = crouch or cut
    if (hands_pass or hands_pass_soft) and pass_ball_ok:
        # Prefer pass when ball flight is clear — light crouch/cut is common in
        # pass drills and must not steal every chest-pass window to TT.
        if breakthrough_like and not clear_pass_ball and fly_mid < 0.40:
            pass  # fall through to TT
        elif (
            hands_pass_soft
            and not hands_pass
            and breakthrough_like
            and fly_mid < 0.45
            and not ball_leave_pass
        ):
            # Soft pass + crouch/cut without clear ball flight → prefer TT
            pass
        else:
            conf = float(np.clip(
                0.60 + 0.12 * min(chest_mid, 0.8) + 0.08 * min(max(fly_mid, leave), 2.0),
                0.60, 0.93,
            ))
            reason = (
                "hands_chest_ball_fly"
                if ball_flyby or fly_mid >= 0.32
                else "hands_chest_ball_leave"
            )
            if hands_pass_soft and not hands_pass:
                reason = reason + "_soft"
            return "pass", conf, reason

    # --- Triple-threat: crouch OR weak CoG+cut; ball only soft veto / boost ---
    if not crouch and not cut:
        return "unknown", 0.0, "no_crouch_or_cut"

    # Free-throw / jump-shot load: planted crouch + vertical hand path — not breakthrough
    if raise_ >= 0.14 and at < 2.2 and pdx < 0.6:
        return "unknown", 0.0, "shooting_load_not_tt"
    if wdy >= 0.60 and at < 2.5 and raise_ >= 0.10 and pdx < 0.6:
        return "unknown", 0.0, "vertical_release_motion"
    # Shot ball flight (cam ball track sees arc) + only mild footwork → not breakthrough
    # Group1 false TT: ankle~2, pelvis_dx~2 while ball_fly_mid ≫ 1 during FT.
    # Allow clear CoG/aspect crouch with moderate travel (hesitation / double TT).
    strong_drive = (
        (at >= 3.5 and pdx >= 1.1 and cog >= 0.12)
        or (cog >= 0.18 and drop >= 0.14 and at >= 1.8 and pdx >= 0.35)
        or (at >= 2.8 and pdx >= 1.5 and cog >= 0.10)
    )
    if fly_mid >= 2.5 and not strong_drive:
        return "unknown", 0.0, "shot_ball_flight_not_tt"
    if fly_mid >= 1.2 and at < 2.6 and pdx < 1.6 and raise_ < 0.08 and not crouch:
        return "unknown", 0.0, "shot_context_weak_drive"

    # Full-court sprint / track-jump artifact — not a TT setup.
    # Pull-up drives usually sit below ~20; huge travel is pose ID flicker.
    if at >= 22.0:
        return "unknown", 0.0, "looks_like_drive_not_tt"

    # Planted idle / FT crouch without a real cut
    if at < 0.9 and pdx < 0.45 and drop < 0.14 and cog < 0.12:
        return "unknown", 0.0, "idle_crouch"
    if at < 2.8 and pdx < 1.0 and (raise_ >= 0.05 or wdy >= 0.25) and fly_mid >= 0.8:
        return "unknown", 0.0, "planted_shot_prep_not_tt"

    # Ready hands optional when CoG+cut is clear (kids often just dip + change direction)
    hands_ready = (
        chest_frac >= 0.08
        or palms >= 0.06
        or (ws >= 0.50 and wdy >= 0.18)
        or (ws >= 0.65)
    )
    strong_cut = cut and (at >= 1.5 or pdx >= 0.70 or cog >= 0.12)
    if not hands_ready and not strong_cut:
        return "unknown", 0.0, "tt_upper_body_idle"

    # Strong breakthrough footwork wins over soft ball leave/pass vetoes
    # (ball track often flickers off the driver mid-cut).
    strong_breakthrough = (at >= 3.5 and pdx >= 1.5 and cog >= 0.10) or (
        at >= 5.0 and pdx >= 2.0
    )

    # Soft ball veto: clear pass fly-away (not dribble/breakthrough ball motion)
    if (
        not strong_breakthrough
        and ball_seen >= 0.18
        and fly_mid >= 1.0
        and leave >= 0.55
        and held < 0.20
        and held_l <= 0.25
    ):
        return "unknown", 0.0, "tt_ball_passing"

    # Soft ball veto only when detector sees enough frames to be trustworthy
    if ball_seen >= 0.22 and not strong_breakthrough:
        if drib >= 0.50:
            return "unknown", 0.0, "tt_looks_like_dribble"
        if leave >= 0.70 and held_l <= 0.20 and held < 0.25:
            # Crouch/CoG dip with mild travel is still TT (double-cut / hesitate)
            if crouch and cog >= 0.12 and at >= 1.4:
                pass
            else:
                return "unknown", 0.0, "tt_ball_left_hand"

    ball_aux = ball_seen >= 0.12 and (held >= 0.25 or held_e >= 0.20)
    conf = float(np.clip(
        0.58
        + 0.14 * min(drop, 0.50)
        + 0.10 * min(cog, 0.40)
        + 0.06 * min(ws, 1.5)
        + 0.04 * min(pdx, 2.0)
        + (0.05 if ball_aux else 0.0),
        0.58,
        0.93,
    ))
    if cut and not crouch:
        reason = "cog_cut_breakthrough"
    elif at >= 1.6 or pdx >= 0.70:
        reason = "breakthrough_drive_pose"
    else:
        reason = "planted_crouch_pose"
    if ball_aux:
        reason = reason + "+ball_aux"
    return "triple_threat", conf, reason



def _merge_adjacent_same_type(
    clips: list[ActionClip],
    gap_frames: int = 25,
    max_span_frames: int = 110,
) -> list[ActionClip]:
    """Merge abutting same-type windows, but cap span so double-TT stays two events."""
    if not clips:
        return []
    ordered = sorted(clips, key=lambda c: c.start_frame)
    out: list[ActionClip] = [ordered[0]]
    for c in ordered[1:]:
        prev = out[-1]
        merged_span = max(prev.end_frame, c.end_frame) - prev.start_frame
        # TT: keep distinct cuts separate (group3 double TT / TT→shot pairs)
        # Pass: never glue — dense classroom exchanges are ~1–2s apart.
        tt_pair = prev.action_type == "triple_threat" and c.action_type == "triple_threat"
        pass_pair = prev.action_type == "pass" and c.action_type == "pass"
        if pass_pair:
            out.append(c)
            continue
        gap_ok = c.start_frame <= prev.end_frame + (
            12 if tt_pair else gap_frames
        )
        span_cap = 75 if tt_pair else max_span_frames
        span_ok = merged_span <= span_cap
        if (
            c.action_type == prev.action_type
            and gap_ok
            and span_ok
        ):
            end = max(prev.end_frame, c.end_frame)
            conf = max(prev.confidence, c.confidence)
            mid = (prev.start_frame + end) // 2
            phases = [
                ActionPhase(name="load", start=prev.start_frame, end=mid),
                ActionPhase(name="action", start=mid, end=max(mid, end - 2)),
                ActionPhase(name="recover", start=max(mid, end - 2), end=end),
            ]
            meta = dict(prev.metadata or {})
            meta["merged_from"] = meta.get("merged_from", 1) + 1
            out[-1] = prev.model_copy(update={
                "end_frame": end,
                "confidence": conf,
                "phases": phases,
                "metadata": meta,
            })
        else:
            out.append(c)
    return out



def _subdivide_passes_by_ball_peaks(
    clips: list[ActionClip],
    seq: list[tuple[int, np.ndarray]],
    ball_by_frame: dict[int, dict],
    *,
    min_gap_frames: int = 18,
) -> list[ActionClip]:
    """Split long pass windows into multiple events at ball-flight peaks."""
    if not clips or not ball_by_frame:
        return clips
    frames = [f for f, _ in seq]
    frame_to_i = {f: i for i, f in enumerate(frames)}
    out: list[ActionClip] = []
    for c in clips:
        if c.action_type != "pass":
            out.append(c)
            continue
        span = int(c.end_frame) - int(c.start_frame)
        if span < 40:
            out.append(c)
            continue
        # Ball travel series inside the window
        xs: list[tuple[int, float]] = []
        prev = None
        for fr in range(int(c.start_frame), int(c.end_frame) + 1):
            b = ball_by_frame.get(fr) or {}
            if not b.get("center"):
                continue
            x, y = float(b["center"][0]), float(b["center"][1])
            speed = 0.0 if prev is None else ((x - prev[0]) ** 2 + (y - prev[1]) ** 2) ** 0.5
            prev = (x, y)
            xs.append((fr, speed))
        if len(xs) < 6:
            out.append(c)
            continue
        speeds = [s for _, s in xs]
        thr = max(1.8, sorted(speeds)[int(0.55 * len(speeds))])
        peaks: list[int] = []
        for i in range(1, len(xs) - 1):
            fr, sp = xs[i]
            if sp >= thr and sp >= xs[i - 1][1] and sp >= xs[i + 1][1]:
                if not peaks or fr - peaks[-1] >= min_gap_frames:
                    peaks.append(fr)
                elif sp > dict(xs).get(peaks[-1], 0):
                    peaks[-1] = fr
        if len(peaks) <= 1:
            out.append(c)
            continue
        # Emit a short pass centered on each peak
        for fr in peaks:
            start_f = max(int(c.start_frame), fr - 12)
            end_f = min(int(c.end_frame), fr + 18)
            mid = (start_f + end_f) // 2
            meta = dict(c.metadata or {})
            meta["reason"] = str(meta.get("reason") or "pass") + "+ball_peak_split"
            meta["ball_peak_frame"] = fr
            out.append(c.model_copy(update={
                "start_frame": start_f,
                "end_frame": end_f,
                "confidence": float(c.confidence),
                "phases": [
                    ActionPhase(name="load", start=start_f, end=mid),
                    ActionPhase(name="action", start=mid, end=max(mid, end_f - 2)),
                    ActionPhase(name="recover", start=max(mid, end_f - 2), end=end_f),
                ],
                "metadata": meta,
            }))
    out.sort(key=lambda c: c.start_frame)
    return out


def detect_pose_only_segments(
    seq: list[tuple[int, np.ndarray]],
    *,
    ball_by_frame: dict[int, dict] | None = None,
    win: int = 50,
    step: int = 8,
    min_conf: float = 0.52,
    max_clips: int = 90,
) -> list[ActionClip]:
    """
    Sliding-window discovery on one camera sequence.
    Pass / triple_threat use same-camera ball track when available.
    """
    if len(seq) < win:
        return []
    frames = [f for f, _ in seq]
    ball_by_frame = ball_by_frame or {}
    candidates: list[tuple[float, ActionClip]] = []

    for i0 in range(0, len(seq) - win + 1, step):
        i1 = i0 + win
        start_f, end_f = frames[i0], frames[i1 - 1]
        # Reject windows with a large mid-window track hole (not mild stride gaps)
        local_gaps = [frames[j + 1] - frames[j] for j in range(i0, i1 - 1)]
        if local_gaps and max(local_gaps) > 90:
            continue
        # Also reject absurd wall-time span (many seconds of sparse IDs)
        if end_f - start_f > int(win * 4.5):
            continue
        h = _h36m_window(seq, i0, i1)
        if h is None:
            continue
        feat = _pose_features(h)
        feat.update(_ball_features(seq, i0, i1, ball_by_frame, feat["torso"]))
        at = float(feat.get("ankle_travel", 0.0))
        pdx = float(feat.get("pelvis_dx", 0.0))
        # Wrist-release gate: skip planted shot loads, but keep breakthrough→pull-up
        if _has_shooting_release(seq, i0, i1) and not (
            at >= 3.5 and (pdx >= 1.2 or float(feat.get("cog_drop", 0.0)) >= 0.12)
        ):
            continue
        atype, conf, reason = classify_pose_only_window(feat)
        if conf < min_conf or atype == "unknown":
            continue
        mid = frames[i0 + win // 2]
        phases = [
            ActionPhase(name="load", start=start_f, end=mid),
            ActionPhase(name="action", start=mid, end=max(mid, end_f - 2)),
            ActionPhase(name="recover", start=max(mid, end_f - 2), end=end_f),
        ]
        clip = ActionClip(
            action_type=atype,
            start_frame=start_f,
            end_frame=end_f,
            confidence=conf,
            phases=phases,
            metadata={
                "detector": "pose_only",
                "action_type_source": "pose_ball_window",
                "reason": reason,
                "features": {k: round(float(v), 3) for k, v in feat.items() if k != "torso"},
            },
        )
        candidates.append((conf, clip))

    candidates.sort(key=lambda x: -x[0])
    kept: list[ActionClip] = []
    for conf, clip in candidates:
        overlap = False
        mid_b = 0.5 * (clip.start_frame + clip.end_frame)
        for k in kept:
            a0, a1 = k.start_frame, k.end_frame
            b0, b1 = clip.start_frame, clip.end_frame
            inter = max(0, min(a1, b1) - max(a0, b0))
            # Passes: allow denser events — only suppress near-identical midpoints
            if clip.action_type == "pass" and k.action_type == "pass":
                mid_a = 0.5 * (a0 + a1)
                if abs(mid_a - mid_b) < 10:  # ~0.4s at 25–30fps — keep dense exchanges
                    overlap = True
                    break
                continue
            if inter > 0.35 * min(a1 - a0, b1 - b0):
                overlap = True
                break
        if not overlap:
            kept.append(clip)
    kept.sort(key=lambda c: c.start_frame)
    kept = _merge_adjacent_same_type(kept)
    # Peak-split disabled: was fragmenting dense drills and hurting recall
    # kept = _subdivide_passes_by_ball_peaks(kept, seq, ball_by_frame)
    if len(kept) > max_clips:
        # Prefer retaining passes over truncating a dense pass drill
        passes = [c for c in kept if c.action_type == "pass"]
        others = [c for c in kept if c.action_type != "pass"]
        if len(passes) >= 12:
            budget_pass = min(len(passes), max_clips - min(8, len(others)))
            passes = sorted(passes, key=lambda c: -c.confidence)[:budget_pass]
            others = sorted(others, key=lambda c: -c.confidence)[: max(0, max_clips - len(passes))]
            kept = sorted(passes + others, key=lambda c: c.start_frame)
        else:
            kept = sorted(kept, key=lambda c: -c.confidence)[:max_clips]
            kept.sort(key=lambda c: c.start_frame)
    return kept


def _merge_ball_tracks(*tracks: dict[int, dict]) -> dict[int, dict]:
    """Prefer first track's frames; fill gaps from later tracks (same-ish timeline)."""
    out: dict[int, dict] = {}
    for tr in tracks:
        for fr, ball in tr.items():
            if fr not in out:
                out[fr] = ball
    return out


def detect_pose_only_for_session(session_id: str, student_id: str) -> list[ActionClip]:
    """
    Action-segment camera pose + ball on cam_01–03.

    Pass recognition **requires** pose-camera ball tracks (not cam_04 alone).
    Falls back to another pose camera if the preferred anchor has no student pose.
    """
    from src.action.detect import resolve_pose_camera_for_student

    preferred = get_action_segment_camera()
    cam, doc, seq = resolve_pose_camera_for_student(session_id, student_id, preferred)
    if not seq or len(seq) < 40:
        return []
    if cam != preferred:
        print(
            f"  [action] pose-only fallback {preferred}→{cam} for {student_id} "
            f"(n={len(seq)})",
            flush=True,
        )

    # Prefer same-camera ball; merge other pose cams for coverage
    ball_anchor = load_ball_track(session_id, cam)
    ball_01 = load_ball_track(session_id, "cam_01")
    ball_02 = load_ball_track(session_id, "cam_02")
    ball_03 = load_ball_track(session_id, "cam_03")
    if cam == "cam_03":
        ball = _merge_ball_tracks(ball_anchor, ball_03, ball_01, ball_02)
    elif cam == "cam_01":
        ball = _merge_ball_tracks(ball_anchor, ball_01, ball_03, ball_02)
    else:
        ball = _merge_ball_tracks(ball_anchor, ball_02, ball_03, ball_01)

    part = detect_pose_only_segments(seq, ball_by_frame=ball)
    out: list[ActionClip] = []
    for c in part:
        meta = dict(c.metadata or {})
        meta["source_camera"] = cam
        meta["ball_frames"] = len(ball)
        if cam != preferred:
            meta["pose_camera_fallback_from"] = preferred
        out.append(c.model_copy(update={
            "anchor_camera": cam,
            "metadata": meta,
        }))
    return out
