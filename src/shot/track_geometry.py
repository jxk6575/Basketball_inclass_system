"""Public ball/hoop track geometry helpers shared by action fusion and outcome."""

from __future__ import annotations


def hoop_geometry(track_doc: dict) -> tuple[float, float, float, float]:
    """Return (cx, cy, w, h) for fixed hoop or last hoop detection."""
    fixed = track_doc.get("fixed_hoop")
    if fixed and fixed.get("center"):
        cx, cy = fixed["center"][:2]
        bb = fixed.get("bbox") or [0, 0, 160, 120]
        if len(bb) >= 4:
            return float(cx), float(cy), float(bb[2]), float(bb[3])
        return float(cx), float(cy), 160.0, 120.0
    for fr in reversed(track_doc.get("frames") or []):
        h = fr.get("hoop")
        if h and h.get("center"):
            cx, cy = h["center"][:2]
            bb = h.get("bbox") or [0, 0, 160, 120]
            w = float(bb[2]) if len(bb) >= 4 else 160.0
            hh = float(bb[3]) if len(bb) >= 4 else 120.0
            return float(cx), float(cy), w, hh
    return 960.0, 540.0, 160.0, 120.0


def ball_samples(track_doc: dict) -> list[dict]:
    """Flatten per-frame ball detections into a chronological sample list.

    Prefers ``balls`` (multi-ball) when present; falls back to single ``ball``.
    """
    out: list[dict] = []
    for fr in track_doc.get("frames") or []:
        multi = fr.get("balls")
        if multi:
            for ball in multi:
                if not ball or not ball.get("center"):
                    continue
                out.append({
                    "frame": int(fr["frame"]),
                    "timestamp_ms": float(fr.get("timestamp_ms", 0)),
                    "center": list(ball["center"]),
                    "confidence": float(ball.get("confidence", 0)),
                    "bbox": ball.get("bbox"),
                })
            continue
        ball = fr.get("ball")
        if not ball or not ball.get("center"):
            continue
        out.append({
            "frame": int(fr["frame"]),
            "timestamp_ms": float(fr.get("timestamp_ms", 0)),
            "center": list(ball["center"]),
            "confidence": float(ball.get("confidence", 0)),
            "bbox": ball.get("bbox"),
        })
    return out


def segment_ball_trajectories(balls: list[dict], gap_frames: int = 60) -> list[list[dict]]:
    """Split ball samples into contiguous trajectory segments by frame gaps."""
    if not balls:
        return []
    segs: list[list[dict]] = [[balls[0]]]
    for b in balls[1:]:
        if b["frame"] - segs[-1][-1]["frame"] > gap_frames:
            segs.append([b])
        else:
            segs[-1].append(b)
    return segs


def _bbox_area(bb) -> float:
    if not bb or len(bb) < 4:
        return 0.0
    # [x,y,w,h] or [x1,y1,x2,y2]
    if float(bb[2]) > float(bb[0]) and float(bb[3]) > float(bb[1]) and float(bb[2]) > 20:
        return abs(float(bb[2]) - float(bb[0])) * abs(float(bb[3]) - float(bb[1]))
    return max(0.0, float(bb[2]) * float(bb[3]))


def multi_peak_above_hoop(
    points: list[dict],
    hoop_cy: float,
    hoop_area: float,
    *,
    min_peak_gap_ms: float = 2800.0,
    max_peaks: int = 24,
) -> list[dict]:
    """Greedy local minima of image-y among above-hoop samples (temporal NMS)."""
    above = [p for p in points if float(p["center"][1]) < float(hoop_cy)]
    if not above:
        return []
    ranked = sorted(above, key=lambda p: float(p["center"][1]))
    accepted: list[dict] = []
    for p in ranked:
        if len(accepted) >= max_peaks:
            break
        ba = _bbox_area(p.get("bbox"))
        if ba > 0.0 and ba >= hoop_area:
            continue
        t = float(p["timestamp_ms"])
        if any(abs(t - float(a["timestamp_ms"])) < min_peak_gap_ms for a in accepted):
            continue
        near = [q for q in above if abs(float(q["timestamp_ms"]) - t) <= 400.0]
        if near and float(p["center"][1]) > min(float(q["center"][1]) for q in near) + 8.0:
            continue
        accepted.append(p)
    accepted.sort(key=lambda p: float(p["timestamp_ms"]))
    return accepted


def shot_peak_segments(
    track_doc: dict,
    *,
    min_peak_gap_ms: float = 2800.0,
    half_window_ms: float | None = None,
    max_min_dist: float = 900.0,
    min_points: int = 3,
) -> list[list[dict]]:
    """Split glued cam_04 trajectories into one segment per free-throw peak.

    Long continuous ball tracks often merge several shots; outcome alignment
    and release fusion both need one window per attempt.
    """
    hoop_cx, hoop_cy, hoop_w, hoop_h = hoop_geometry(track_doc)
    hoop_area = max(1.0, float(hoop_w) * float(hoop_h))
    hw = float(half_window_ms if half_window_ms is not None else min_peak_gap_ms * 0.45)
    out: list[list[dict]] = []
    for seg in shot_like_segments(
        track_doc, max_min_dist=max_min_dist, min_points=min_points,
    ):
        peaks = multi_peak_above_hoop(
            seg, float(hoop_cy), hoop_area, min_peak_gap_ms=min_peak_gap_ms,
        )
        if not peaks:
            continue
        if len(peaks) == 1:
            out.append(seg)
            continue
        times = [float(p["timestamp_ms"]) for p in peaks]
        for i, peak in enumerate(peaks):
            t = float(peak["timestamp_ms"])
            lo = t - hw
            hi = t + hw
            if i > 0:
                lo = max(lo, 0.5 * (times[i - 1] + t))
            if i + 1 < len(times):
                hi = min(hi, 0.5 * (t + times[i + 1]))
            window = [
                p for p in seg
                if lo <= float(p["timestamp_ms"]) <= hi
            ]
            if len(window) < min_points:
                # ensure peak itself is kept
                window = [peak]
            # Drop junk peaks far from rim (false "above hoop" clutter)
            min_d = min(
                ((p["center"][0] - hoop_cx) ** 2 + (p["center"][1] - hoop_cy) ** 2) ** 0.5
                for p in window
            )
            if min_d > max(max_min_dist * 0.75, 2.2 * max(hoop_w, 80.0)):
                continue
            out.append(window)
    out.sort(key=lambda s: float(s[len(s) // 2]["timestamp_ms"]))
    return out


def shot_like_segments(
    track_doc: dict,
    max_min_dist: float = 900.0,
    min_points: int = 3,
    require_ball_smaller_than_hoop: bool = True,
) -> list[list[dict]]:
    """Keep trajectory segments that approach the rim (above / near hoop).

    ``require_ball_smaller_than_hoop``: reject frames where ball bbox area ≥ hoop
    area (filters giant false balls / near-camera clutter as shot events).
    """
    hoop_cx, hoop_cy, hoop_w, hoop_h = hoop_geometry(track_doc)
    hoop_area = max(1.0, float(hoop_w) * float(hoop_h))
    balls = ball_samples(track_doc)
    if require_ball_smaller_than_hoop:
        filtered = []
        for b in balls:
            ba = _bbox_area(b.get("bbox"))
            if ba <= 0.0 or ba < hoop_area:
                filtered.append(b)
        balls = filtered
    segs = segment_ball_trajectories(balls, gap_frames=55)
    kept: list[list[dict]] = []
    for seg in segs:
        if len(seg) < min_points:
            continue
        if not any(p["center"][1] < hoop_cy + 0.25 * hoop_h for p in seg):
            continue
        min_d = min(
            ((p["center"][0] - hoop_cx) ** 2 + (p["center"][1] - hoop_cy) ** 2) ** 0.5
            for p in seg
        )
        if min_d > max_min_dist:
            continue
        kept.append(seg)
    return kept
