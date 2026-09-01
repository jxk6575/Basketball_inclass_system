"""Multi-camera release fusion — pose peaks × rim-ball segments."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src.action.detect import (
    L_WRIST,
    R_WRIST,
    _merge_nearby_peaks,
    _wrist_peak_indices,
    extract_student_sequence,
    load_ball_track,
    load_pose2d_for_camera,
)
from src.cameras.registry import get_action_segment_camera
from src.cameras.temporal import frame_to_timestamp_ms
from src.config import data_path
from src.shot.track_geometry import hoop_geometry, multi_peak_above_hoop, shot_like_segments
from src.types import ActionClip, ActionPhase


POSE_CAMERAS = ("cam_01", "cam_02", "cam_03")


@dataclass
class ReleasePeak:
    camera_id: str
    frame: int
    timestamp_ms: float
    wrist_y: float
    seq_index: int = -1
    shooting_hand: str = "right"


def _raw_peaks_for_camera(
    session_id: str,
    camera_id: str,
    student_id: str,
    merge_window: int = 100,
) -> list[ReleasePeak]:
    from src.action.shooting_hand import infer_shooting_hand_from_window

    doc = load_pose2d_for_camera(session_id, camera_id)
    seq = extract_student_sequence(doc, student_id)
    if not seq or len(seq) < 30:
        return []
    fps = float(doc.get("fps", 30.0))
    frames = [f for f, _ in seq]
    ball_by_frame = load_ball_track(session_id, camera_id)
    wrist_y = [float(k[R_WRIST, 1]) for _, k in seq]

    peaks_r = _wrist_peak_indices(seq, R_WRIST)
    peaks_l = _wrist_peak_indices(seq, L_WRIST)
    peaks: list[int] = list(peaks_r)
    for p in peaks_l:
        if not any(abs(frames[p] - frames[r]) < merge_window for r in peaks_r):
            peaks.append(p)
    peaks.sort()
    peaks = _merge_nearby_peaks(
        peaks, frames, wrist_y, merge_window, seq=seq, ball_by_frame=ball_by_frame,
    )
    out: list[ReleasePeak] = []
    for i in peaks:
        hand, _ = infer_shooting_hand_from_window(seq, i, ball_by_frame)
        fr = frames[i]
        out.append(ReleasePeak(
            camera_id=camera_id,
            frame=fr,
            timestamp_ms=frame_to_timestamp_ms(fr, fps),
            wrist_y=wrist_y[i],
            seq_index=i,
            shooting_hand=hand,
        ))
    return out


def _cluster_peaks(
    peaks: list[ReleasePeak],
    gap_ms: float = 900.0,
) -> list[list[ReleasePeak]]:
    if not peaks:
        return []
    ordered = sorted(peaks, key=lambda p: p.timestamp_ms)
    clusters: list[list[ReleasePeak]] = [[ordered[0]]]
    for p in ordered[1:]:
        if p.timestamp_ms - clusters[-1][-1].timestamp_ms <= gap_ms:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def _cluster_repr(cluster: list[ReleasePeak], anchor_cam: str) -> dict:
    cams = sorted({p.camera_id for p in cluster})
    # Prefer anchor-cam peak; else highest wrist (min y)
    anchor_peaks = [p for p in cluster if p.camera_id == anchor_cam]
    if anchor_peaks:
        best = min(anchor_peaks, key=lambda p: p.wrist_y)
    else:
        best = min(cluster, key=lambda p: p.wrist_y)
    return {
        "timestamp_ms": float(best.timestamp_ms),
        "frame": int(best.frame),
        "camera_id": best.camera_id,
        "wrist_y": float(best.wrist_y),
        "n_cameras": len(cams),
        "cameras": cams,
        "support": len(cluster),
    }


def _cam04_segment_times(session_id: str) -> list[float]:
    """
    Midpoint timestamps (ms) of cam_04 shot-like ball segments.

    Used by event-based temporal sync to align cam_04 with pose cameras.
    Falls back to ball-above-hoop peak times when segments are unavailable.
    """
    path = data_path("sessions", session_id, "shot_outcomes", "ball_track.json")
    if not path.exists():
        # Also accept nested cam path used by some runners
        from src.cameras.registry import get_shot_outcome_camera
        alt = data_path(
            "sessions", session_id, "shot_outcomes",
            get_shot_outcome_camera(), "ball_track.json",
        )
        path = alt if alt.exists() else path
    if not path.exists():
        return []
    track = json.loads(path.read_text(encoding="utf-8"))
    times: list[float] = []
    try:
        for seg in shot_like_segments(track, min_points=3, max_min_dist=900.0):
            ts = [float(p["timestamp_ms"]) for p in seg if p.get("timestamp_ms") is not None]
            if not ts:
                continue
            times.append(0.5 * (min(ts) + max(ts)))
    except Exception:
        times = []
    if times:
        return times
    # Fallback: peak frames where ball is above hoop
    for ev in _cam04_ball_above_hoop_events(session_id):
        times.append(float(ev["timestamp_ms"]))
    return times


def _cam04_near_rim_approach_events(
    track: dict,
    hoop_cx: float,
    hoop_cy: float,
    hoop_w: float,
    *,
    max_approach: float,
    min_gap_ms: float = 2800.0,
) -> list[dict]:
    """
    Near-rim approaches where the ball never clears the rim (short misses).

    Uses local minima of ball–hoop distance on cam_04 samples. Above-hoop peaks
    remain primary; these fill gaps when the arc is lost / below the rim plane.
    """
    frames = track.get("frames") or []
    samples: list[tuple[float, float, float, float, int]] = []
    for f in frames:
        balls = f.get("balls") or []
        if not balls:
            continue
        b = balls[0] if isinstance(balls[0], dict) else None
        if not b or not b.get("center"):
            continue
        cx, cy = float(b["center"][0]), float(b["center"][1])
        d = ((cx - hoop_cx) ** 2 + (cy - hoop_cy) ** 2) ** 0.5
        samples.append((float(f.get("timestamp_ms") or 0.0), d, cx, cy, int(f.get("frame") or 0)))
    if len(samples) < 5:
        return []

    cands: list[dict] = []
    for i in range(2, len(samples) - 2):
        t, d, cx, cy, fr = samples[i]
        if d > max_approach:
            continue
        # Local distance minimum (among nearby samples)
        if not (
            d <= samples[i - 1][1]
            and d <= samples[i + 1][1]
            and d <= samples[i - 2][1]
            and d <= samples[i + 2][1]
        ):
            continue
        # Require a clear approach from farther out within the prior 0.4–2.8s
        # (sample-count windows are unreliable when detections are sparse).
        early = [
            s for s in samples[max(0, i - 40): i]
            if (t - s[0]) >= 400.0 and (t - s[0]) <= 2800.0
        ]
        early_d = min(s[1] for s in early) if early else None
        approached = early_d is not None and early_d >= d + 70.0
        # Isolated near-rim touch after a long quiet gap (short miss with lost arc)
        quiet_gap = True
        for s in samples[max(0, i - 40): i]:
            if (t - s[0]) > 2800.0:
                continue
            if (t - s[0]) >= 400.0 and s[1] <= max_approach * 0.85:
                quiet_gap = False
                break
        # Also: first ball sample after ≥2.5s silence that is already near-rim
        prev_t = samples[i - 1][0] if i > 0 else None
        first_after_silence = prev_t is not None and (t - prev_t) >= 2500.0 and d <= max_approach * 0.75
        if not approached and not (quiet_gap and d <= max_approach * 0.80) and not first_after_silence:
            continue
        cands.append({
            "timestamp_ms": t,
            "frame": fr,
            "ball_cy": cy,
            "hoop_cy": float(hoop_cy),
            "rim_dist": round(d, 1),
            "approach_min_dist": round(d, 1),
            "source": "cam04_near_rim_approach",
        })

    # First near-rim sample after long silence (may not be a local min yet)
    for i in range(1, len(samples)):
        t, d, cx, cy, fr = samples[i]
        if d > max_approach * 0.75:
            continue
        prev_t = samples[i - 1][0]
        if (t - prev_t) < 2500.0:
            continue
        cands.append({
            "timestamp_ms": t,
            "frame": fr,
            "ball_cy": cy,
            "hoop_cy": float(hoop_cy),
            "rim_dist": round(d, 1),
            "approach_min_dist": round(d, 1),
            "source": "cam04_near_rim_approach",
        })

    kept: list[dict] = []
    for ev in sorted(cands, key=lambda e: float(e["timestamp_ms"])):
        if kept and float(ev["timestamp_ms"]) - float(kept[-1]["timestamp_ms"]) < min_gap_ms:
            if float(ev.get("approach_min_dist") or 1e9) < float(
                kept[-1].get("approach_min_dist") or 1e9
            ):
                kept[-1] = ev
            continue
        kept.append(ev)
    return kept


def _cam04_ball_above_hoop_events(session_id: str) -> list[dict]:
    """
    Shooting anchors from cam_04: ball center above hoop center (image-y).

    Multiple peaks per glued trajectory are kept (≥~2.8s apart) so long tracks
    do not collapse several free-throws into one event.
    Peaks far from the rim in image space are rejected (bounce / ceiling clutter).
    Also merges near-rim approaches that never clear the rim (short misses).
    """
    path = data_path("sessions", session_id, "shot_outcomes", "ball_track.json")
    if not path.exists():
        return []
    track = json.loads(path.read_text(encoding="utf-8"))
    try:
        hoop_cx, hoop_cy, hoop_w, hoop_h = hoop_geometry(track)
    except Exception:
        return []
    hoop_area = max(1.0, float(hoop_w) * float(hoop_h))
    # Apex can sit far above/beside the rim in image space; gate primarily on
    # approach proximity (a sample near the hoop within ±1.2s of the peak).
    max_lateral = max(650.0, 3.8 * max(float(hoop_w), 80.0))
    max_approach = max(520.0, 2.4 * max(float(hoop_w), 80.0))
    events: list[dict] = []
    for seg in shot_like_segments(track, min_points=3, max_min_dist=900.0):
        for peak in multi_peak_above_hoop(
            seg, float(hoop_cy), hoop_area, min_peak_gap_ms=2800.0,
        ):
            cx, cy = float(peak["center"][0]), float(peak["center"][1])
            if abs(cx - float(hoop_cx)) > max_lateral:
                continue
            # Require a real approach: at least one near-rim sample in ±1.2s
            t = float(peak["timestamp_ms"])
            near = [
                p for p in seg
                if abs(float(p["timestamp_ms"]) - t) <= 1200.0
            ]
            if not near:
                continue
            min_d = min(
                ((p["center"][0] - hoop_cx) ** 2 + (p["center"][1] - hoop_cy) ** 2) ** 0.5
                for p in near
            )
            if min_d > max_approach:
                continue
            events.append({
                "timestamp_ms": t,
                "frame": int(peak["frame"]),
                "ball_cy": cy,
                "hoop_cy": float(hoop_cy),
                "rim_dist": round(min_d, 1),
                "approach_min_dist": round(min_d, 1),
                "source": "cam04_ball_above_hoop",
            })
    # Fill short-miss gaps: ball approaches rim without clearing apex
    approach_ev = _cam04_near_rim_approach_events(
        track, float(hoop_cx), float(hoop_cy), float(hoop_w),
        max_approach=max_approach, min_gap_ms=2800.0,
    )
    for ev in approach_ev:
        t = float(ev["timestamp_ms"])
        # Only suppress if a *prior* finish already covers this slot.
        # A later above-hoop peak must not erase the preceding short-miss attempt.
        if any(0.0 <= t - float(e["timestamp_ms"]) < 2600.0 for e in events):
            continue
        events.append(ev)

    events.sort(key=lambda e: float(e["timestamp_ms"]))
    # Collapse near-duplicate apex/approach pairs (<1.5s) — keep above-hoop.
    if len(events) >= 2:
        collapsed: list[dict] = [events[0]]
        for ev in events[1:]:
            prev = collapsed[-1]
            if float(ev["timestamp_ms"]) - float(prev["timestamp_ms"]) < 1500.0:
                prev_src = str(prev.get("source") or "")
                cur_src = str(ev.get("source") or "")
                if "above_hoop" in cur_src and "above_hoop" not in prev_src:
                    collapsed[-1] = ev
                elif "above_hoop" not in cur_src and "above_hoop" in prev_src:
                    pass
                elif float(ev.get("approach_min_dist") or 1e9) < float(
                    prev.get("approach_min_dist") or 1e9
                ):
                    collapsed[-1] = ev
                continue
            collapsed.append(ev)
        events = collapsed
    # Adaptive temporal NMS only for very sparse sessions where rebound
    # double-peaks are common. Dense attempt trains (median gap <~5s) must
    # keep consecutive finishes ~2.8–4s apart.
    if len(events) >= 4:
        gaps = [
            float(events[i + 1]["timestamp_ms"]) - float(events[i]["timestamp_ms"])
            for i in range(len(events) - 1)
        ]
        med = sorted(gaps)[len(gaps) // 2]
        if med >= 6500.0:
            merge_gap = min(3200.0, max(2800.0, 0.40 * med))
            kept: list[dict] = []
            for ev in events:
                if kept and float(ev["timestamp_ms"]) - float(kept[-1]["timestamp_ms"]) < merge_gap:
                    prev_src = str(kept[-1].get("source") or "")
                    cur_src = str(ev.get("source") or "")
                    if "above_hoop" in cur_src and "above_hoop" not in prev_src:
                        kept[-1] = ev
                    elif "above_hoop" not in cur_src and "above_hoop" in prev_src:
                        pass
                    elif float(ev.get("approach_min_dist") or 1e9) < float(
                        kept[-1].get("approach_min_dist") or 1e9
                    ):
                        kept[-1] = ev
                    continue
                kept.append(ev)
            events = kept
    return events


def fuse_release_clusters(
    session_id: str,
    student_id: str,
    anchor_camera: str | None = None,
    max_align_ms: float = 2800.0,
) -> list[dict]:
    """
    Fuse wrist-release peaks with **required** cam_04 ball-above-hoop events.

    Uses greedy nearest-neighbor matching (O(n·m)), not combinatorial search —
    critical for realtime / many-shot sessions.
    """
    rim_events = _cam04_ball_above_hoop_events(session_id)
    if not rim_events:
        return []

    anchor = anchor_camera or get_action_segment_camera()
    all_peaks: list[ReleasePeak] = []
    for cam in POSE_CAMERAS:
        all_peaks.extend(_raw_peaks_for_camera(session_id, cam, student_id))

    clusters = [_cluster_repr(c, anchor) for c in _cluster_peaks(all_peaks)]

    # Prefer temporal-alignment offsets (cam_04 rim → anchor-cam clock).
    # Falling back to median pair offset is brittle and can invent ±2s drift.
    sync_off = 0.0
    used_sync = False
    align_path = data_path("sessions", session_id, "sync", "alignment.json")
    if align_path.exists():
        try:
            align_doc = json.loads(align_path.read_text(encoding="utf-8"))
            offs = align_doc.get("camera_time_offsets_ms") or {}
            # common = local - offset  →  anchor_local = rim_local - off_rim + off_anchor
            sync_off = float(offs.get(anchor, 0.0)) - float(offs.get("cam_04", 0.0))
            used_sync = True
        except Exception:
            used_sync = False

    if not clusters:
        return [
            {
                "timestamp_ms": float(e["timestamp_ms"]) + sync_off,
                "rim_timestamp_ms": float(e["timestamp_ms"]),
                "pose_timestamp_ms": None,
                "frame": int(e.get("frame", 0)),
                "camera_id": "cam_04",
                "wrist_y": 0.0,
                "n_cameras": 0,
                "cameras": [],
                "support": 0,
                "clock_offset_ms": sync_off,
                "source": e.get("source", "cam04_ball_above_hoop"),
                "ball_cy": e.get("ball_cy"),
                "hoop_cy": e.get("hoop_cy"),
            }
            for e in rim_events
        ]

    # Adaptive align budget: never reach past half the typical inter-shot gap
    rim_ts = sorted(float(e["timestamp_ms"]) + sync_off for e in rim_events)
    gaps = [rim_ts[i + 1] - rim_ts[i] for i in range(len(rim_ts) - 1) if rim_ts[i + 1] > rim_ts[i]]
    med_gap = float(np.median(gaps)) if gaps else max_align_ms
    align_budget = float(min(max_align_ms, max(900.0, 0.45 * med_gap)))

    if used_sync:
        offset = sync_off
    else:
        # Estimate global clock offset: median of best rim↔pose pairs within ±2s
        pair_offs: list[float] = []
        for ev in rim_events:
            rt = float(ev["timestamp_ms"])
            best_dt, best_off = 1e18, 0.0
            for c in clusters:
                off = rt - float(c["timestamp_ms"])
                if abs(off) < best_dt and abs(off) <= 2000.0:
                    best_dt, best_off = abs(off), off
            if best_dt <= 2000.0:
                pair_offs.append(best_off)
        offset = float(np.median(pair_offs)) if pair_offs else 0.0

    # Greedy: for each rim event, pick nearest unused pose cluster after offset.
    # Primary identity is ALWAYS the rim timestamp so session NMS can collapse
    # the same attempt across students without swallowing adjacent shots.
    used_pose: set[int] = set()
    chosen: list[dict] = []
    for ev in sorted(rim_events, key=lambda e: float(e["timestamp_ms"])):
        rt_raw = float(ev["timestamp_ms"])
        # Rim expressed on anchor / pose clock
        rt = rt_raw + offset if used_sync else rt_raw
        best_i, best_err = -1, 1e18
        for i, c in enumerate(clusters):
            if i in used_pose:
                continue
            pose_t = float(c["timestamp_ms"])
            # When not using sync file, pose_t is compared to rim via +offset in err
            pose_cmp = pose_t if used_sync else (pose_t + offset)
            rim_cmp = rt if used_sync else rt_raw
            # Release must precede (or barely meet) ball-at-rim on the same clock
            if pose_cmp > rim_cmp + 400.0:
                continue
            if pose_cmp < rim_cmp - align_budget - 300.0:
                continue
            err = abs(rim_cmp - pose_cmp)
            lead = max(0.0, rim_cmp - pose_cmp)
            score = err - 0.15 * float(c.get("n_cameras", 0)) - 0.05 * min(lead, 1200.0) / 1200.0
            if score < best_err and err <= align_budget:
                best_err, best_i = score, i
        if best_i >= 0:
            pose = clusters[best_i]
            pose_ms = float(pose["timestamp_ms"])
            chosen.append({
                # Clip timing on pose/anchor clock (matches GT cam_03)
                "timestamp_ms": pose_ms,
                "rim_timestamp_ms": rt_raw,
                "pose_timestamp_ms": pose_ms,
                "frame": int(pose["frame"]),
                "camera_id": pose.get("camera_id", anchor),
                "wrist_y": float(pose.get("wrist_y") or 0.0),
                "n_cameras": int(pose.get("n_cameras") or 0),
                "cameras": list(pose.get("cameras") or []),
                "support": int(pose.get("support") or 0),
                "clock_offset_ms": offset,
                "source": "multicam_pose_x_cam04_ball_above_hoop",
                "ball_cy": ev.get("ball_cy"),
                "hoop_cy": ev.get("hoop_cy"),
                "align_error_ms": round(best_err if best_err < 1e17 else 0.0, 1),
            })
            used_pose.add(best_i)
        else:
            # Rim-only: project rim onto pose clock for timing
            pose_guess = rt_raw + offset if used_sync else (rt_raw - abs(offset) if offset else rt_raw)
            chosen.append({
                "timestamp_ms": pose_guess,
                "rim_timestamp_ms": rt_raw,
                "pose_timestamp_ms": None,
                "frame": int(ev.get("frame", 0)),
                "camera_id": "cam_04",
                "wrist_y": 0.0,
                "n_cameras": 0,
                "cameras": [],
                "support": 0,
                "clock_offset_ms": offset,
                "source": ev.get("source", "cam04_ball_above_hoop"),
                "ball_cy": ev.get("ball_cy"),
                "hoop_cy": ev.get("hoop_cy"),
            })

    chosen.sort(key=lambda c: float(c.get("rim_timestamp_ms") or c["timestamp_ms"]))
    return chosen


def clips_from_fused_releases(
    session_id: str,
    student_id: str,
    action_type: str | None = None,
    pre_frames: int | None = None,
    post_frames: int | None = None,
) -> list[ActionClip]:
    """Build ActionClips on the action-segment camera from fused releases.

    Production: pass ``action_type=None`` so each clip is classified from pose
    (run-up → layup, jump → jump_shot, planted → free_throw). A non-None value
    is a debug override only and should not be used by the session pipeline.
    """
    from src.action.detect import classify_release_action, resolve_pose_camera_for_student
    from src.action.registry import normalize_action_type
    from src.shot.track_geometry import hoop_geometry

    preferred = get_action_segment_camera()
    anchor, doc, seq = resolve_pose_camera_for_student(session_id, student_id, preferred)
    if not seq:
        return []
    if anchor != preferred:
        print(
            f"  [action] pose fallback {preferred}→{anchor} for {student_id} "
            f"(n={len(seq)})",
            flush=True,
        )
    fps = float(doc.get("fps", 30.0))
    frames = [f for f, _ in seq]
    fused = fuse_release_clusters(session_id, student_id, anchor_camera=anchor)
    if not fused:
        # No cam_04 ball-above-hoop → no shooting clips
        return []

    # Prefer anchor-cam hoop for person→hoop approach; never mix cam_04 coords
    # into cam_03 pose (breaks approach_ratio → false layups on free throws).
    hoop_xy: tuple[float, float] | None = None
    anchor_track = data_path(
        "sessions", session_id, "shot_outcomes", f"ball_track_{anchor}.json",
    )
    if anchor_track.exists():
        try:
            tdoc = json.loads(anchor_track.read_text(encoding="utf-8"))
            hx, hy, _, _ = hoop_geometry(tdoc)
            hoop_xy = (float(hx), float(hy))
        except Exception:
            hoop_xy = None

    clips: list[ActionClip] = []
    for item in fused:
        n_cam = int(item.get("n_cameras", 0))
        # Rim timestamp is the attempt identity (pose time is only for kinematics).
        rim_ms = float(item.get("rim_timestamp_ms") or item["timestamp_ms"])
        pose_ms_raw = item.get("pose_timestamp_ms")
        has_pose = n_cam > 0 or bool(item.get("cameras")) or pose_ms_raw is not None
        # Per-student rim-only (no pose peak) is usually "wrong person standing still".
        # Session-level coverage synthesizes missing rim events once.
        if not has_pose:
            continue

        # Map release time → nearest pose frame on anchor cam
        off = float(item.get("clock_offset_ms") or 0.0)
        if item.get("camera_id") == anchor and n_cam > 0 and item.get("frame") is not None:
            release = int(item["frame"])
        else:
            # Prefer matched pose peak time; else rim clock corrected by offset
            if pose_ms_raw is not None:
                pose_ms = float(pose_ms_raw)
            else:
                pose_ms = rim_ms - off
            best_i = min(
                range(len(frames)),
                key=lambda i: abs(frame_to_timestamp_ms(frames[i], fps) - pose_ms),
            )
            release = frames[best_i]
        try:
            peak_idx = frames.index(release)
        except ValueError:
            peak_idx = min(range(len(frames)), key=lambda i: abs(frames[i] - release))
            release = frames[peak_idx]

        if action_type:
            atype = action_type
            cls_meta: dict = {"source": "override"}
        else:
            atype, cls_meta = classify_release_action(seq, release, hoop_xy=hoop_xy)
            atype = normalize_action_type(atype)
            if atype not in ("free_throw", "jump_shot", "layup"):
                # Rim-gated attempt with ambiguous pose for *this* student:
                # still emit a planted default so session NMS/spatial can keep it.
                # (Other students' detectors may classify better; weak ones lose NMS.)
                atype = "free_throw"
                cls_meta = {**(cls_meta or {}), "source": "rim_gated_default_ft"}
            # Soft noise filters only when pose support is weak AND classification
            # itself looks non-shooting — never drop a clear multi-cam release.
            if (
                atype == "free_throw"
                and n_cam <= 0
                and float(cls_meta.get("ankle_travel") or 0.0) < 0.4
                and float(cls_meta.get("wrist_raise") or cls_meta.get("raise") or 1.0) < 0.15
            ):
                # Truly empty pose near this rim for this student — skip; peers keep it
                continue

        # Window in *video frame* units (not sequence index) so sparse pose
        # tracking cannot inflate a free-throw clip to 20+ seconds.
        use_pre = pre_frames if pre_frames is not None else (90 if atype == "layup" else 55)
        use_post = post_frames if post_frames is not None else (40 if atype == "layup" else 30)
        start_target = release - use_pre
        end_target = release + use_post
        start = min(frames, key=lambda f: abs(f - start_target))
        end = min(frames, key=lambda f: abs(f - end_target))
        if start > release:
            start = frames[max(0, peak_idx - 1)]
        if end < release:
            end = frames[min(len(frames) - 1, peak_idx + 1)]
        start = min(start, release)
        end = max(end, release)
        conf = float(np.clip(0.75 + 0.05 * n_cam, 0.7, 0.98))
        phases = _phases_for_action(atype, start, release, end)
        from src.action.shooting_hand import infer_shooting_hand_from_window

        ball_by_frame = load_ball_track(session_id, anchor)
        shooting_hand, hand_meta = infer_shooting_hand_from_window(
            seq, peak_idx, ball_by_frame,
        )
        clips.append(ActionClip(
            action_type=atype,
            start_frame=start,
            end_frame=end,
            phases=phases,
            confidence=conf,
            metadata={
                "shooting_hand": shooting_hand,
                "shooting_hand_meta": hand_meta,
                "multicam": {
                    "cameras": item.get("cameras"),
                    "n_cameras": n_cam,
                    "source": item.get("source"),
                    "clock_offset_ms": item.get("clock_offset_ms"),
                    "rim_timestamp_ms": rim_ms,
                    "pose_timestamp_ms": pose_ms_raw,
                    "align_error_ms": item.get("align_error_ms"),
                    "has_pose_peak": has_pose,
                },
                "action_classify": cls_meta,
            },
        ))
    return clips


def _phases_for_action(
    action_type: str,
    start: int,
    release: int,
    end: int,
) -> list[ActionPhase]:
    """Phase names follow configs/actions/*.yaml."""
    if action_type == "layup":
        # approach → gather → takeoff → release → finish
        span = max(1, release - start)
        g1 = start + int(0.40 * span)
        g2 = start + int(0.70 * span)
        takeoff_end = max(g2, release - 3)
        return [
            ActionPhase(name="approach", start=start, end=g1),
            ActionPhase(name="gather", start=g1, end=g2),
            ActionPhase(name="takeoff", start=g2, end=takeoff_end),
            ActionPhase(name="release", start=release - 2, end=release + 2),
            ActionPhase(name="finish", start=release + 2, end=end),
        ]
    if action_type == "jump_shot":
        # load → takeoff → release → follow_through
        span = max(1, release - start)
        load_end = start + int(0.45 * span)
        takeoff_end = max(load_end, release - 3)
        return [
            ActionPhase(name="load", start=start, end=load_end),
            ActionPhase(name="takeoff", start=load_end, end=takeoff_end),
            ActionPhase(name="release", start=release - 2, end=release + 2),
            ActionPhase(name="follow_through", start=release + 2, end=end),
        ]
    mid = start + (release - start) // 2
    return [
        ActionPhase(name="load", start=start, end=mid),
        ActionPhase(name="set", start=mid, end=max(mid, release - 3)),
        ActionPhase(name="release", start=release - 2, end=release + 2),
        ActionPhase(name="follow_through", start=release + 2, end=end),
    ]
