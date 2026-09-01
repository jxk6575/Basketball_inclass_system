"""Shot outcome pipeline — ball/hoop YOLO + make/miss on rim camera."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.action.clip_io import load_session_action_clip_dicts, release_ms_from_clip_dict
from src.cameras.registry import get_perception_config, get_shot_outcome_camera
from src.cameras.temporal import frame_to_timestamp_ms
from src.config import data_path
from src.shot.track_geometry import hoop_geometry, shot_like_segments, shot_peak_segments
from src.shot.tracker import ShotTracker
from src.shot.yolo_detector import YoloBallHoopDetector
from src.types import ShotOutcome, ShotOutcomeRecord


def _find_video(session_id: str, camera_id: str) -> Path | None:
    raw = data_path("sessions", session_id, "raw", f"{camera_id}.mp4")
    if raw.exists():
        return raw
    return None


def _load_action_clips(session_id: str) -> list[dict]:
    """Flatten student action clips for timestamp association."""
    return load_session_action_clip_dicts(session_id)


def _seg_to_track_points(seg: list[dict]) -> list:
    """Convert offline segment samples to geometry.TrackPoint list."""
    from src.shot.geometry import TrackPoint

    out: list[TrackPoint] = []
    for p in seg:
        cx, cy = float(p["center"][0]), float(p["center"][1])
        bb = p.get("bbox") or [0, 0, 40, 40]
        if len(bb) >= 4 and float(bb[2]) > float(bb[0]) and float(bb[3]) > float(bb[1]):
            w = abs(float(bb[2]) - float(bb[0]))
            h = abs(float(bb[3]) - float(bb[1]))
        else:
            w = float(bb[2]) if len(bb) >= 3 else 40.0
            h = float(bb[3]) if len(bb) >= 4 else 40.0
        if w < 1:
            w = 40.0
        if h < 1:
            h = 40.0
        out.append((
            (int(round(cx)), int(round(cy))),
            int(p.get("frame", 0)),
            int(round(w)),
            int(round(h)),
            float(p.get("confidence", 1.0)),
        ))
    return out


def _load_video_frames(
    video_path: Path,
    frame_indices: list[int],
    *,
    max_frames: int = 48,
) -> dict[int, object]:
    """Load BGR frames keyed by source frame index (sorted unique, capped)."""
    if not frame_indices or not video_path.exists():
        return {}
    uniq = sorted(set(int(f) for f in frame_indices))
    if len(uniq) > max_frames:
        step = max(1, len(uniq) // max_frames)
        uniq = uniq[::step][:max_frames]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    wanted = set(uniq)
    got: dict[int, object] = {}
    idx = 0
    max_need = max(uniq)
    while idx <= max_need:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            got[idx] = frame
            if len(got) >= len(wanted):
                break
        idx += 1
    cap.release()
    return got


def evaluate_make_miss(
    seg: list[dict],
    hoop_cx: float,
    hoop_cy: float,
    hoop_w: float,
    hoop_h: float,
    shot_frames: list | dict | None = None,
) -> tuple[bool, float, dict]:
    """
    Combined make/miss (production):

    1. **Rim top orange occlusion**: ball descending + orange rim top covered
       by the ball → **hard miss** (a make leaves the top orange visible)
    2. Else **trajectory** evidence (`_evaluate_trajectory_make`)
    """
    from src.shot.geometry import rim_occlusion_indicates_miss

    if len(seg) < 2:
        return False, 0.0, {"reason": "too_few_points"}

    ball_pos = _seg_to_track_points(seg)
    hoop_frame = int(seg[len(seg) // 2].get("frame", 0))
    hoop_pos = [((int(round(hoop_cx)), int(round(hoop_cy))), hoop_frame,
                 int(round(hoop_w)), int(round(hoop_h)), 1.0)]

    occluded, occ_meta = rim_occlusion_indicates_miss(ball_pos, hoop_pos, shot_frames)
    made, conf, meta = _evaluate_trajectory_make(seg, hoop_cx, hoop_cy, hoop_w, hoop_h)
    meta = {**occ_meta, **meta}

    # Occlusion is a strong miss cue, but do not override a clear centered
    # through/settle trajectory (angled cams often flash orange during makes).
    if occluded:
        strong_make = bool(made) and (
            str(meta.get("reason") or "") in {
                "near_rim_center",
                "near_rim_then_through",
                "near_rim_then_through_wide",
                "above_to_below_near_hoop",
                "centered_approach_below_exit",
            }
            or int(meta.get("n_through_tight") or 0) >= 4
        )
        if not strong_make:
            return False, 0.90, {
                **meta,
                "reason": "rim_occlusion_miss",
                "segment_frames": [seg[0]["frame"], seg[-1]["frame"]],
            }

    return made, conf, meta


def _evaluate_trajectory_make(
    seg: list[dict],
    hoop_cx: float,
    hoop_cy: float,
    hoop_w: float,
    hoop_h: float,
) -> tuple[bool, float, dict]:
    """
    Trajectory-only make/miss near the rim (no pixel occlusion).

    A make needs through-rim evidence (pass under near center, or settle into the
    rim cylinder). Merely approaching the rim from above — especially with a
    bounce/rebound — is treated as a miss.
    """
    if len(seg) < 2:
        return False, 0.0, {"reason": "too_few_points"}

    rim_r = max(0.7 * hoop_w, 50.0)
    near_x = max(1.0 * hoop_w, 70.0)
    tight_x = max(0.55 * hoop_w, 55.0)
    hoop_bot = hoop_cy + 0.55 * hoop_h

    dists = [
        ((p["center"][0] - hoop_cx) ** 2 + (p["center"][1] - hoop_cy) ** 2) ** 0.5
        for p in seg
    ]
    min_dist = min(dists)
    closest_i = int(dists.index(min_dist))
    closest = seg[closest_i]
    closest_x, closest_y = float(closest["center"][0]), float(closest["center"][1])
    above = [p for p in seg if p["center"][1] < hoop_cy]
    below = [p for p in seg if p["center"][1] > hoop_bot]
    near_above = [p for p in above if abs(p["center"][0] - hoop_cx) <= near_x]

    closing = False
    if closest_i >= 3:
        early = sum(dists[: closest_i // 2 + 1]) / max(1, closest_i // 2 + 1)
        late = sum(dists[max(0, closest_i - 3): closest_i + 1]) / min(4, closest_i + 1)
        closing = late + 40 < early

    # Rim bounce: after first approach, ball rises significantly (miss / rim-out).
    rebound_rise = 0.0
    first_near = next((i for i, d in enumerate(dists) if d <= rim_r * 1.25), None)
    if first_near is not None:
        y0 = float(seg[first_near]["center"][1])
        window = seg[first_near: min(len(seg), first_near + 18)]
        ymin = min(float(p["center"][1]) for p in window)
        rebound_rise = max(0.0, y0 - ymin)
    rebound = rebound_rise >= 55.0

    after = seg[closest_i:]
    below_near = [
        p for p in after
        if p["center"][1] > hoop_bot and abs(p["center"][0] - hoop_cx) <= near_x * 1.2
    ]
    # Centered pass under the rim (true through-net signal)
    through_tight = [
        p for p in after
        if p["center"][1] > hoop_cy + 0.2 * hoop_h
        and abs(p["center"][0] - hoop_cx) <= tight_x
    ]
    through_wide = [
        p for p in after
        if p["center"][1] > hoop_cy + 0.2 * hoop_h
        and abs(p["center"][0] - hoop_cx) <= near_x * 1.15
    ]
    last = after[-1] if after else closest
    last_dx = abs(float(last["center"][0]) - hoop_cx)

    meta = {
        "min_dist": round(min_dist, 1),
        "closest": closest["center"],
        "n_above": len(above),
        "n_below": len(below),
        "n_near_above": len(near_above),
        "n_through_tight": len(through_tight),
        "rebound_rise": round(rebound_rise, 1),
        "rebound": rebound,
        "closing": closing,
        "segment_frames": [seg[0]["frame"], seg[-1]["frame"]],
    }

    lateral_escape = (
        not through_tight
        and last_dx > max(near_x * 1.25, hoop_w * 1.05)
        and last["center"][1] > hoop_cy - 0.3 * hoop_h
        and dists[-1] > min_dist + 70
    )
    if lateral_escape and (rebound or not below_near) and len(through_wide) < 2:
        # Dense above-rim approach + below samples: ball worked the cylinder
        # even if it later exits laterally (net / floor roll).
        if not (closing and len(near_above) >= 8 and len(below) >= 6):
            return False, 0.86, {**meta, "reason": "lateral_escape_miss"}

    # Strong rim-out / bounce → miss unless clear through/below evidence.
    if rebound and len(through_tight) == 0 and len(through_wide) == 0 and not below_near:
        if not (closing and len(near_above) >= 8 and len(below) >= 6):
            return False, 0.86, {**meta, "reason": "rim_rebound_miss"}
    # Rebound with only sparse through samples and no below-rim settle → miss
    if rebound and len(through_tight) < 2 and not below_near and len(through_wide) < 3:
        if not (closing and len(near_above) >= 8 and len(below) >= 6):
            return False, 0.86, {**meta, "reason": "rim_rebound_miss"}
    # Tall bounce after rim contact: rim-out unless strong through+below settle
    # (bounce can fake a few "through" samples; real makes keep ≥8 tight + below).
    if rebound_rise >= 120.0 and (len(through_tight) < 8 or len(below) < 4):
        return False, 0.86, {**meta, "reason": "rim_rebound_miss"}

    # Ball reaches rim cylinder at/below rim plane
    in_rim_cylinder = min_dist <= rim_r and closest_y <= hoop_bot
    at_or_below_rim = closest_y >= hoop_cy - 0.12 * hoop_h
    if (
        in_rim_cylinder
        and at_or_below_rim
        and abs(closest_x - hoop_cx) <= 0.55 * hoop_w
        and (not rebound or len(through_tight) >= 4 or (bool(below_near) and len(through_tight) >= 2))
    ):
        return True, 0.92, {**meta, "reason": "near_rim_center"}

    # Approach from above: need through-net evidence (tight preferred)
    if in_rim_cylinder and closest_y < hoop_cy:
        if through_tight and (not rebound or len(through_tight) >= 4 or bool(below_near)):
            return True, 0.9, {**meta, "reason": "near_rim_then_through"}
        if through_wide and not rebound and last_dx <= tight_x * 1.35:
            return True, 0.86, {**meta, "reason": "near_rim_then_through_wide"}

    if near_above and through_tight:
        return True, 0.88, {**meta, "reason": "above_to_below_near_hoop"}

    # Angled rim cams: after a centered approach, the ball often exits the
    # net laterally in image space — do not require tight through samples.
    # Require at least one below sample still inside the near-x cylinder so
    # pure airballs / wide rim-outs are not counted as makes.
    if (
        near_above
        and len(near_above) >= 5
        and len(below) >= 3
        and min_dist <= rim_r * 1.05
        and rebound_rise < 90.0
        and closing
        and any(abs(float(p["center"][0]) - hoop_cx) <= near_x * 1.2 for p in below)
    ):
        return True, 0.78, {**meta, "reason": "centered_approach_below_exit"}

    # Prefer the most centered below point (ball may later roll away on the floor)
    if near_above and through_wide and not rebound:
        best_through_dx = min(abs(float(p["center"][0]) - hoop_cx) for p in through_wide)
        if best_through_dx <= tight_x * 1.4:
            return True, 0.84, {**meta, "reason": "above_to_below_near_hoop",
                                "best_through_dx": round(best_through_dx, 1)}
        return False, 0.82, {**meta, "reason": "wide_below_miss",
                             "best_through_dx": round(best_through_dx, 1)}

    # Occlusion make: ball approaches rim center and track ends there (net swallows ball).
    # Do not require "no below" — early below points can exist before the approach.
    ends_at_rim = closest_i >= len(seg) - 3 or len(after) <= 3
    if (
        not rebound
        and not through_tight
        and min_dist <= rim_r * 1.15
        and last_dx <= tight_x * 1.45
        and closest_y <= hoop_bot
        and ends_at_rim
        and closing
    ):
        return True, 0.85, {**meta, "reason": "approach_then_occlude"}

    if (
        near_above
        and min_dist <= rim_r * 1.15
        and not through_tight
        and not rebound
        and last_dx <= tight_x * 1.3
        and last["center"][1] < hoop_bot + 0.25 * hoop_h
        and ends_at_rim
    ):
        return True, 0.83, {**meta, "reason": "approach_then_occlude"}

    if (
        closing and above and not rebound and min_dist <= max(1.6 * hoop_w, rim_r * 2.2)
        and (through_tight or (through_wide and last_dx <= tight_x * 1.35) or at_or_below_rim)
        and last_dx <= near_x * 1.2
    ):
        return True, 0.8, {**meta, "reason": "closing_approach"}

    if above and below and min_dist > rim_r * 1.8:
        wide_below = all(abs(p["center"][0] - hoop_cx) > near_x for p in below[:3])
        if wide_below and not closing:
            return False, 0.8, {**meta, "reason": "wide_miss"}

    if above and min_dist <= rim_r * 1.7 and through_tight:
        return True, 0.72, {**meta, "reason": "near_approach"}

    if above and min_dist <= rim_r * 1.7 and not through_tight:
        return False, 0.72, {**meta, "reason": "near_but_no_through"}

    return False, 0.55, {**meta, "reason": "no_rim_interaction"}


def _segment_rim_quality(
    seg: list[dict],
    hoop_cx: float,
    hoop_cy: float,
    hoop_w: float,
) -> float:
    """Higher = more likely a real rim interaction (vs ghost/sideline ball)."""
    if not seg:
        return 0.0
    rim_r = max(0.7 * hoop_w, 50.0)
    dists = [
        ((p["center"][0] - hoop_cx) ** 2 + (p["center"][1] - hoop_cy) ** 2) ** 0.5
        for p in seg
        if p.get("center")
    ]
    if not dists:
        return 0.0
    min_d = min(dists)
    near = sum(1 for d in dists if d <= rim_r * 1.8)
    # Prefer close approach + several near-rim samples
    return float(near) / (1.0 + min_d / max(rim_r, 1.0))


def _greedy_align_clips_to_segments(
    valid: list[tuple[int, float]],
    seg_ts: list[float],
    max_align_ms: float,
    seg_quality: list[float] | None = None,
) -> tuple[float, list[tuple[int, int, float]]] | None:
    """
    Greedy clip↔segment alignment under a shared clock offset.

    When ``seg_quality`` is provided (rim-interaction scores), prefer segments
    that actually approach the rim among time-plausible candidates — avoids
    locking onto ghost/far ball tracks.
    """
    if not valid or not seg_ts:
        return None
    clips_sorted = sorted(valid, key=lambda t: t[1])
    best: tuple[tuple[float, float, float], float, list[tuple[int, int, float]]] | None = None
    for off in range(-4000, 4001, 50):
        used: set[int] = set()
        mapping: list[tuple[int, int, float]] = []
        sum_err = 0.0
        sum_q = 0.0
        for ci, ct in clips_sorted:
            target = ct + off
            best_j: int | None = None
            best_key: tuple[float, float] | None = None
            best_e: float | None = None
            for j, st in enumerate(seg_ts):
                if j in used:
                    continue
                e = abs(st - target)
                if e > max_align_ms:
                    continue
                q = float(seg_quality[j]) if seg_quality is not None else 0.0
                # Primary: lower time error; tie-break: higher rim quality
                key = (e - 180.0 * min(q, 8.0), -q)
                if best_key is None or key < best_key:
                    best_j, best_key, best_e = j, key, e
            if best_j is None or best_e is None:
                continue
            used.add(best_j)
            mapping.append((ci, best_j, best_e))
            sum_err += best_e
            if seg_quality is not None:
                sum_q += float(seg_quality[best_j])
        if not mapping:
            continue
        # Prefer more matches, higher total rim quality, then lower error
        rank = (-float(len(mapping)), -sum_q, sum_err)
        if best is None or rank < best[0]:
            best = (rank, float(off), mapping)
    if best is None:
        return None
    return best[1], best[2]


def _synthetic_segment_near_release(
    track_doc: dict,
    release_ms: float,
    *,
    window_ms: float = 2200.0,
) -> list[dict]:
    """Build a ball segment from raw cam frames around a release (short-miss fill)."""
    frames = track_doc.get("frames") or []
    out: list[dict] = []
    for f in frames:
        t = float(f.get("timestamp_ms") or 0.0)
        if abs(t - release_ms) > window_ms:
            continue
        balls = f.get("balls") or []
        if not balls:
            continue
        b = balls[0] if isinstance(balls[0], dict) else None
        if not b or not b.get("center"):
            continue
        out.append({
            "frame": int(f.get("frame") or 0),
            "timestamp_ms": t,
            "center": list(b["center"]),
            "bbox": b.get("bbox"),
            "confidence": float(b.get("confidence") or 0.5),
        })
    return out


def outcomes_from_clips_and_track(
    clips: list[dict],
    track_doc: dict,
    max_align_ms: float = 3200.0,
    video_path: Path | None = None,
) -> list[dict]:
    """
    Align shooting action clips to shot-like ball segments and score make/miss.

    When ``video_path`` is provided, each segment loads cam frames so
    **rim top orange occlusion** can veto misses (ball covering orange rim)
    before trajectory scoring.

    Non-shooting clips (triple_threat, pass, …) are skipped — cam_04 outcome
    only applies to the shooting family.
    """
    from src.action.registry import is_shooting_action

    shooting_clips = [c for c in clips if is_shooting_action(c.get("action_type"))]
    # Preserve original indices for assign map relative to shooting_clips
    clips = shooting_clips

    hoop_cx, hoop_cy, hoop_w, hoop_h = hoop_geometry(track_doc)
    # Prefer per-peak windows so glued free-throw tracks align 1:1 with clips
    segments = shot_peak_segments(track_doc)
    if not segments:
        segments = shot_like_segments(track_doc)
    releases = [release_ms_from_clip_dict(c) for c in clips]
    valid = [(i, float(r)) for i, r in enumerate(releases) if r is not None]

    if not valid:
        return []

    if not segments:
        # No above-hoop peaks — still score from synthetic near-release windows
        results = []
        for i, ts in valid:
            seg = _synthetic_segment_near_release(track_doc, ts)
            if len(seg) < 2:
                results.append({
                    "clip": clips[i],
                    "clip_id": clips[i].get("clip_id"),
                    "made": False,
                    "confidence": 0.55,
                    "timestamp_ms": ts,
                    "frame": seg[0]["frame"] if seg else None,
                    "metadata": {"source": "clip_synthetic", "reason": "no_shot_segments"},
                    "ball_trajectory": seg,
                })
                continue
            made, conf, meta = evaluate_make_miss(seg, hoop_cx, hoop_cy, hoop_w, hoop_h)
            results.append({
                "clip": clips[i],
                "clip_id": clips[i].get("clip_id"),
                "made": made,
                "confidence": conf,
                "timestamp_ms": float(seg[len(seg) // 2]["timestamp_ms"]),
                "frame": int(seg[len(seg) // 2]["frame"]),
                "metadata": {"source": "clip_synthetic", "scoring": "rim_orange+trajectory", **meta},
                "ball_trajectory": seg,
            })
        return results

    seg_ts = [float(seg[len(seg) // 2]["timestamp_ms"]) for seg in segments]
    seg_quality = [
        _segment_rim_quality(seg, hoop_cx, hoop_cy, hoop_w) for seg in segments
    ]
    aligned = _greedy_align_clips_to_segments(
        valid, seg_ts, max_align_ms, seg_quality=seg_quality,
    )

    assign: dict[int, tuple[int, float]] = {}
    offset = 0.0
    if aligned is not None:
        offset, mapping = aligned
        for ci, si, err in mapping:
            assign[ci] = (si, err)

    # Local rim-aware fallback for unmatched clips (or ghost-aligned ones)
    used_segs = {si for si, _ in assign.values()}
    for ci, release in valid:
        need = ci not in assign
        if ci in assign:
            si, _ = assign[ci]
            if seg_quality[si] < 0.35:
                need = True
        if not need:
            continue
        target = float(release) + offset
        best_j, best_score = None, -1e18
        for j, st in enumerate(seg_ts):
            if j in used_segs and (ci not in assign or assign[ci][0] != j):
                continue
            e = abs(st - target)
            if e > max_align_ms * 1.35:
                continue
            q = seg_quality[j]
            score = 4.0 * q - e / 400.0
            if score > best_score:
                best_score, best_j = score, j
        if best_j is not None and best_score > -5.0:
            if ci in assign:
                old = assign[ci][0]
                used_segs.discard(old)
            assign[ci] = (best_j, abs(seg_ts[best_j] - target))
            used_segs.add(best_j)

    results: list[dict] = []
    for ci, clip in enumerate(clips):
        release = releases[ci]
        if release is None:
            continue
        if ci in assign:
            best_i, best_d = assign[ci]
            seg = segments[best_i]
            shot_frames = None
            if video_path is not None:
                # Pad a few frames before first ball sample for rim base color
                f0 = max(0, int(seg[0]["frame"]) - 2)
                f1 = int(seg[-1]["frame"])
                idxs = list(range(f0, f1 + 1))
                shot_frames = _load_video_frames(video_path, idxs, max_frames=48)
            made, conf, meta = evaluate_make_miss(
                seg, hoop_cx, hoop_cy, hoop_w, hoop_h, shot_frames=shot_frames,
            )
            anchor = seg[len(seg) // 2]
            above = [p for p in seg if p["center"][1] < hoop_cy]
            if above:
                anchor = above[-1]
            results.append({
                "clip": clip,
                "clip_id": clip.get("clip_id"),
                "made": made,
                "confidence": conf,
                "timestamp_ms": float(anchor["timestamp_ms"]),
                "frame": int(anchor["frame"]),
                "metadata": {
                    "source": "clip_segment_aligned",
                    "scoring": "rim_orange+trajectory",
                    "clock_offset_ms": offset,
                    "align_error_ms": round(best_d, 1),
                    "rim_quality": round(float(seg_quality[best_i]), 3),
                    "n_shot_frames": len(shot_frames or []),
                    **meta,
                },
                "ball_trajectory": seg,
            })
        else:
            # Short miss / lost apex: score a synthetic window around the clip
            seg = _synthetic_segment_near_release(track_doc, float(release))
            if len(seg) >= 2:
                made, conf, meta = evaluate_make_miss(
                    seg, hoop_cx, hoop_cy, hoop_w, hoop_h,
                )
                anchor = seg[len(seg) // 2]
                results.append({
                    "clip": clip,
                    "clip_id": clip.get("clip_id"),
                    "made": made,
                    "confidence": conf,
                    "timestamp_ms": float(anchor["timestamp_ms"]),
                    "frame": int(anchor["frame"]),
                    "metadata": {
                        "source": "clip_synthetic_near_release",
                        "scoring": "rim_orange+trajectory",
                        "clock_offset_ms": offset,
                        **meta,
                    },
                    "ball_trajectory": seg,
                })
            else:
                results.append({
                    "clip": clip,
                    "clip_id": clip.get("clip_id"),
                    "made": False,
                    "confidence": 0.5,
                    "timestamp_ms": float(release),
                    "frame": None,
                    "metadata": {
                        "source": "clip_no_ball",
                        "reason": "no_aligned_segment_default_miss",
                        "clock_offset_ms": offset,
                    },
                    "ball_trajectory": [],
                })
    return results


def _scale_detection(det: dict, factor: float) -> dict:
    """Map detection coords from scaled inference frame back to full resolution."""
    if factor == 1.0:
        return det
    x, y, w, h = det["bbox"]
    cx, cy = det["center"]
    out = dict(det)
    out["bbox"] = (int(round(x * factor)), int(round(y * factor)),
                   int(round(w * factor)), int(round(h * factor)))
    out["center"] = (int(round(cx * factor)), int(round(cy * factor)))
    if "area" in det:
        out["area"] = int(det["area"] * factor * factor)
    return out


def _scale_hoop_dict(hoop: dict | None, factor: float) -> dict | None:
    if hoop is None or factor == 1.0:
        return hoop
    out = dict(hoop)
    if "center" in hoop:
        cx, cy = hoop["center"]
        out["center"] = [int(round(cx * factor)), int(round(cy * factor))]
    if "bbox" in hoop:
        bb = hoop["bbox"]
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            out["bbox"] = [int(round(v * factor)) for v in bb]
    return out


def _default_shot_process_scale() -> float:
    return float(get_perception_config().get("shot_camera_process_scale", 0.5))


def run_ball_tracking_on_video(
    video_path: Path,
    out_json: Path | None = None,
    max_frames: int | None = None,
    calibrate_frames: int | None = None,
    stride: int = 1,
    process_scale: float | None = None,
    hoop_upper_half_only: bool = False,
    keep_all_balls: bool = True,
) -> dict:
    """
    Run ball/hoop tracking on a video. Returns trajectory + shot events.
    Used by shot outcome stage and optionally by action enhancement.

    ``hoop_upper_half_only``: for cam_01–03, hoop must lie in the upper half
    of the frame (rejects ground clutter / false positives).
    ``calibrate_frames``: average the first N hoop detections then freeze
    (default from ``perception.hoop_calibrate_frames``, typically 2).
    ``keep_all_balls``: keep multiple basketballs per frame (needed on cam_01–03).
    """
    if calibrate_frames is None:
        calibrate_frames = int(get_perception_config().get("hoop_calibrate_frames", 2))
    calibrate_frames = max(1, int(calibrate_frames))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    tracker = ShotTracker(
        detector=YoloBallHoopDetector(),
        calibrate_hoop_frames=calibrate_frames,
        keep_shot_frames=True,
        hoop_upper_half_only=hoop_upper_half_only,
        keep_all_balls=keep_all_balls,
    )
    stride = max(1, int(stride))
    scale = float(process_scale if process_scale is not None else _default_shot_process_scale())
    scale = max(0.25, min(1.0, scale))
    inv_scale = 1.0 / scale

    frame_log: list[dict] = []
    events_out: list[dict] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and idx >= max_frames:
            break
        if idx % stride != 0:
            idx += 1
            continue
        infer_frame = frame
        if scale < 1.0:
            infer_frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR,
            )
        result = tracker.process_frame(infer_frame, idx)
        ts = frame_to_timestamp_ms(idx, fps)
        balls_scaled = []
        for b in (result["ball"] or []):
            sb = _scale_detection(b, inv_scale)
            balls_scaled.append({
                "center": list(sb["center"]),
                "bbox": list(sb["bbox"]),
                "confidence": sb["confidence"],
            })
        ball = balls_scaled[0] if balls_scaled else None
        hoop = None
        if result["hoop"]:
            h = result["hoop"][0]
            h = _scale_detection(h, inv_scale)
            if "center" in h:
                hoop = {"center": list(h["center"]), "confidence": h.get("confidence", 1.0)}
                if "bbox" in h:
                    bb = h["bbox"]
                    hoop["bbox"] = list(bb) if not isinstance(bb, list) else bb
        frame_log.append({
            "frame": idx,
            "timestamp_ms": ts,
            "ball": ball,
            "balls": balls_scaled,
            "hoop": hoop,
        })
        if result["event"] is not None:
            ev = result["event"]
            scaled_traj: list[dict] = []
            for p in ev.ball_trajectory or []:
                if not isinstance(p, dict):
                    continue
                sp = dict(p)
                if "center" in sp and isinstance(sp["center"], (list, tuple)):
                    sp["center"] = [float(c * inv_scale) for c in sp["center"]]
                scaled_traj.append(sp)
            events_out.append({
                "frame": ev.frame,
                "timestamp_ms": frame_to_timestamp_ms(ev.frame, fps),
                "made": ev.made,
                "confidence": ev.confidence,
                "ball_trajectory": scaled_traj,
                "hoop": _scale_hoop_dict(ev.hoop if isinstance(ev.hoop, dict) else None, inv_scale),
                "metadata": ev.metadata,
            })
        idx += 1

    cap.release()
    fixed_hoop = tracker.fixed_hoop
    if fixed_hoop and inv_scale != 1.0:
        fixed_hoop = _scale_detection(fixed_hoop, inv_scale)
    doc = {
        "video": str(video_path),
        "fps": fps,
        "process_scale": scale,
        "frames": frame_log,
        "events": events_out,
        "stats": {"makes": tracker.makes, "attempts": tracker.attempts},
        "fixed_hoop": fixed_hoop,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def ensure_ball_track(session_id: str, camera_id: str | None = None) -> Path | None:
    """Run ball tracking once if missing; return path to ball_track.json."""
    cam_id = camera_id or get_shot_outcome_camera()
    out_dir = data_path("sessions", session_id, "shot_outcomes")
    track_path = out_dir / "ball_track.json"
    if track_path.exists():
        return track_path

    video = _find_video(session_id, cam_id)
    if video is None:
        raw_dir = data_path("sessions", session_id, "raw")
        candidates = list(raw_dir.glob("*.mp4")) if raw_dir.exists() else []
        video = candidates[0] if candidates else None
    if video is None or not video.exists():
        return None

    try:
        run_ball_tracking_on_video(video, out_json=track_path)
        return track_path
    except Exception:
        return None


_WEAK_PRIMARY_MISS_REASONS = frozenset({
    "near_but_no_through",
    "no_rim_interaction",
    "wide_miss",
})
_STRONG_SIDE_MAKE_REASONS = frozenset({
    "near_rim_center",
    "near_rim_then_through",
    "near_rim_then_through_wide",
    "above_to_below_near_hoop",
    "near_approach",
})
_WEAK_PRIMARY_MAKE_REASONS = frozenset({
    "above_to_below_near_hoop",
    "near_approach",
    "closing_approach",
    "centered_approach_below_exit",
    # note: approach_then_occlude is a strong make cue (net swallow) — do not veto
})
_STRONG_SIDE_MISS_REASONS = frozenset({
    "rim_rebound_miss",
    "rim_occlusion_miss",
    "lateral_escape_miss",
    "near_but_no_through",
    "wide_below_miss",
    "wide_miss",
    # note: no_rim_interaction is weak evidence (occlusion / lost track) — not a veto vote
})


def _fuse_side_cam_outcomes(
    scored: list[dict],
    clips: list[dict],
    out_dir: Path,
    primary_cam: str,
) -> list[dict]:
    """
    Fuse rim-camera outcomes with side-camera trajectory votes.

    - Rescue weak primary misses when a side cam has strong through-net evidence.
    - Veto weak primary makes when ≥2 side cams agree on a strong miss.
    """
    side_paths = sorted(
        p for p in out_dir.glob("ball_track_*.json")
        if p.name != "ball_track.json"
    )
    if not side_paths or not scored:
        return scored

    side_scored: list[tuple[str, list[dict]]] = []
    for path in side_paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            side_scored.append((path.stem, outcomes_from_clips_and_track(clips, doc)))
        except Exception:
            continue
    if not side_scored:
        return scored

    out: list[dict] = []
    for i, item in enumerate(scored):
        made = item.get("made")
        meta = dict(item.get("metadata") or {})
        reason = str(meta.get("reason") or "")
        thr = int(meta.get("n_through_tight") or 0)

        # --- rescue weak miss ---
        if made is False and reason in _WEAK_PRIMARY_MISS_REASONS:
            best: tuple[str, dict] | None = None
            for cam_name, rows in side_scored:
                if i >= len(rows):
                    continue
                side = rows[i]
                if side.get("made") is not True:
                    continue
                sm = side.get("metadata") or {}
                s_reason = str(sm.get("reason") or "")
                s_thr = int(sm.get("n_through_tight") or 0)
                if s_reason not in _STRONG_SIDE_MAKE_REASONS or s_thr < 6:
                    continue
                if best is None or s_thr > int((best[1].get("metadata") or {}).get("n_through_tight") or 0):
                    best = (cam_name, side)
            if best is not None:
                cam_name, side = best
                sm = dict(side.get("metadata") or {})
                new_item = dict(item)
                new_item["made"] = True
                new_item["confidence"] = float(side.get("confidence") or 0.8)
                new_item["metadata"] = {
                    **meta,
                    "reason": sm.get("reason") or "side_cam_through",
                    "scoring": f"{primary_cam}+side_rescue",
                    "rescued_from": reason,
                    "rescue_camera": cam_name,
                    "rescue_through_tight": sm.get("n_through_tight"),
                    "primary_reason": reason,
                }
                out.append(new_item)
                continue

        # --- veto weak / contested make ---
        if made is True and (
            (reason in _WEAK_PRIMARY_MAKE_REASONS and thr < 4)
            or (reason == "closing_approach" and thr < 20)
        ):
            # Keep make when a side cam has strong through AND primary thr is not tiny
            side_strong_make = False
            if thr >= 2:
                for _cam_name, rows in side_scored:
                    if i >= len(rows):
                        continue
                    side = rows[i]
                    if side.get("made") is not True:
                        continue
                    sm = side.get("metadata") or {}
                    if (
                        str(sm.get("reason") or "") in _STRONG_SIDE_MAKE_REASONS
                        and int(sm.get("n_through_tight") or 0) >= 6
                    ):
                        side_strong_make = True
                        break
            if side_strong_make:
                out.append(item)
                continue

            miss_votes = 0
            miss_reasons: list[str] = []
            for cam_name, rows in side_scored:
                if i >= len(rows):
                    continue
                side = rows[i]
                if side.get("made") is not False:
                    continue
                sm = side.get("metadata") or {}
                s_reason = str(sm.get("reason") or "")
                counts = s_reason in _STRONG_SIDE_MISS_REASONS
                if s_reason == "no_rim_interaction" and (
                    reason == "closing_approach" or (thr <= 1 and reason != "centered_approach_below_exit")
                ):
                    counts = True
                if counts:
                    miss_votes += 1
                    miss_reasons.append(f"{cam_name}:{s_reason}")
            decisive = [v for v in miss_reasons if "no_rim_interaction" not in v]
            if reason == "centered_approach_below_exit":
                should_veto = len(decisive) >= 2
            else:
                should_veto = miss_votes >= 2 and (len(decisive) >= 1 or reason == "closing_approach")
            if should_veto:
                new_item = dict(item)
                new_item["made"] = False
                new_item["confidence"] = max(0.72, float(item.get("confidence") or 0.7))
                new_item["metadata"] = {
                    **meta,
                    "reason": "side_cam_miss_veto",
                    "scoring": f"{primary_cam}+side_veto",
                    "vetoed_from": reason,
                    "veto_votes": miss_reasons,
                    "primary_reason": reason,
                }
                out.append(new_item)
                continue

        out.append(item)
    return out


def run_shot_outcome_session(
    session_id: str,
    clip_alignments: list[dict] | None = None,
) -> list[Path]:
    """
    Run ball/hoop detection on shot-outcome camera and emit make/miss outcomes
    aligned to action clips by timestamp.
    """
    cam_id = get_shot_outcome_camera()
    out_dir = data_path("sessions", session_id, "shot_outcomes")
    out_dir.mkdir(parents=True, exist_ok=True)
    track_path = out_dir / "ball_track.json"
    outcomes_path = out_dir / "outcomes.json"

    video = _find_video(session_id, cam_id)
    if video is None:
        raw_dir = data_path("sessions", session_id, "raw")
        candidates = list(raw_dir.glob("*.mp4")) if raw_dir.exists() else []
        video = candidates[0] if candidates else None

    if video is None or not video.exists():
        stub = ShotOutcomeRecord(
            session_id=session_id,
            camera_id=cam_id,
            status="pending_video",
            outcomes=[],
            metadata={"message": f"No video for {cam_id}; place {cam_id}.mp4 under sessions/{session_id}/raw/"},
        )
        outcomes_path.write_text(stub.model_dump_json(indent=2), encoding="utf-8")
        return [outcomes_path]

    try:
        if track_path.exists():
            track_doc = json.loads(track_path.read_text(encoding="utf-8"))
        else:
            track_doc = run_ball_tracking_on_video(video, out_json=track_path)
    except FileNotFoundError as e:
        stub = ShotOutcomeRecord(
            session_id=session_id,
            camera_id=cam_id,
            status="pending_model",
            outcomes=[],
            metadata={"message": str(e)},
        )
        outcomes_path.write_text(stub.model_dump_json(indent=2), encoding="utf-8")
        return [outcomes_path]
    except Exception as e:
        stub = ShotOutcomeRecord(
            session_id=session_id,
            camera_id=cam_id,
            status="error",
            outcomes=[],
            metadata={"message": str(e)},
        )
        outcomes_path.write_text(stub.model_dump_json(indent=2), encoding="utf-8")
        return [outcomes_path]

    clips = _load_action_clips(session_id)
    # Prefer clip-anchored offline scoring (rim occlusion + trajectory)
    scored = outcomes_from_clips_and_track(clips, track_doc, video_path=video)
    scored = _fuse_side_cam_outcomes(scored, clips, out_dir, cam_id)
    outcomes: list[ShotOutcome] = []
    for item in scored:
        clip = item.get("clip")
        made = item.get("made")
        outcomes.append(ShotOutcome(
            student_id=clip["student_id"] if clip else None,
            action_type=clip["action_type"] if clip else "free_throw",
            made=made if made is None else bool(made),
            confidence=float(item["confidence"]),
            anchor_timestamp_ms=float(item["timestamp_ms"]),
            clip_id=clip["clip_id"] if clip else None,
            metadata={
                "frame": item.get("frame"),
                "ball_points": len(item.get("ball_trajectory") or []),
                **(item.get("metadata") or {}),
            },
        ))

    # Keep all clip-aligned attempts (including undetermined make/miss)
    # so attempt count matches free_throw clips; only prefer scored when
    # everything has a decisive made flag.
    scored_outcomes = [o for o in outcomes if o.made is not None]
    if scored_outcomes and len(scored_outcomes) == len(outcomes):
        outcomes = scored_outcomes
    elif scored_outcomes and len(scored_outcomes) >= max(1, int(0.85 * len(outcomes))):
        # mostly scored — drop only the few undetermined
        outcomes = scored_outcomes
    # else: keep undetermined (made=None) so attempts ≈ clip count

    # Also keep raw online tracker events in metadata for debug
    online_events = track_doc.get("events") or []

    record = ShotOutcomeRecord(
        session_id=session_id,
        camera_id=cam_id,
        status="ok",
        outcomes=outcomes,
        metadata={
            "video": str(video),
            "ball_track": str(track_path),
            "stats": {
                "makes": sum(1 for o in outcomes if o.made is True),
                "attempts": len(outcomes),
                "misses": sum(1 for o in outcomes if o.made is False),
                "undetermined": sum(1 for o in outcomes if o.made is None),
                "online_events": len(online_events),
                "track_stats": track_doc.get("stats", {}),
            },
            "events_raw": len(online_events),
            "scoring": "rim_orange+trajectory",
        },
    )
    outcomes_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    # Persist offline events into ball_track for viz MADE/MISS overlays
    if scored:
        track_doc = dict(track_doc)
        track_doc["events"] = [
            {
                "frame": int(item["frame"]) if item.get("frame") is not None else 0,
                "timestamp_ms": float(item["timestamp_ms"]),
                "made": bool(item["made"]),
                "confidence": float(item["confidence"]),
                "ball_trajectory": item.get("ball_trajectory") or [],
                "metadata": item.get("metadata") or {},
            }
            for item in scored
            if item.get("made") is not None and item.get("frame") is not None
        ]
        track_doc["stats"] = {
            "makes": sum(1 for o in outcomes if o.made is True),
            "attempts": sum(1 for o in outcomes if o.made is not None),
        }
        track_path.write_text(json.dumps(track_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    return [outcomes_path, track_path]
