"""Event-based multi-camera temporal sync (release peaks × optional rim events)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.cameras.registry import (
    get_action_segment_camera,
    get_camera_ids,
    get_shot_outcome_camera,
    get_sync_config,
)
from src.cameras.temporal import frame_to_timestamp_ms
from src.config import data_path


@dataclass
class SyncEvent:
    camera_id: str
    timestamp_ms: float
    frame: int | None = None
    kind: str = "release"  # release | rim_ball
    student_id: str | None = None
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _event_sync_cfg() -> dict[str, Any]:
    sync = get_sync_config()
    return dict(sync.get("event_sync") or {})


def list_session_student_ids(session_id: str, camera_ids: list[str] | None = None) -> list[str]:
    """Collect student_id labels present in pose2d outputs."""
    from src.action.detect import load_pose2d_for_camera

    camera_ids = camera_ids or [c for c in get_camera_ids() if c != get_shot_outcome_camera()]
    found: set[str] = set()
    for cam in camera_ids:
        doc = load_pose2d_for_camera(session_id, cam)
        for fr in doc.get("frames") or []:
            for person in fr.get("persons") or []:
                sid = person.get("student_id")
                if sid:
                    found.add(str(sid))
    return sorted(found)


def extract_release_events(
    session_id: str,
    camera_id: str,
    student_id: str,
) -> list[SyncEvent]:
    """Wrist-release peaks for one student on one camera (local clock)."""
    from src.action.multicam_release import _raw_peaks_for_camera

    peaks = _raw_peaks_for_camera(session_id, camera_id, student_id)
    return [
        SyncEvent(
            camera_id=camera_id,
            timestamp_ms=float(p.timestamp_ms),
            frame=int(p.frame),
            kind="release",
            student_id=student_id,
            score=1.0 / max(float(p.wrist_y), 1.0),
        )
        for p in peaks
    ]


def extract_rim_events(session_id: str) -> list[SyncEvent]:
    """cam_04 shot-like ball segment midpoints (local cam_04 clock)."""
    from src.cameras.registry import get_camera

    cam = get_shot_outcome_camera()
    times: list[float] = []
    try:
        from src.action.multicam_release import _cam04_segment_times
        times = list(_cam04_segment_times(session_id) or [])
    except Exception:
        times = []
    if not times:
        try:
            from src.action.multicam_release import _cam04_ball_above_hoop_events
            times = [float(e["timestamp_ms"]) for e in _cam04_ball_above_hoop_events(session_id)]
        except Exception:
            times = []

    fps = float(get_camera(cam).get("fps") or get_sync_config().get("default_fps") or 30.0)
    out: list[SyncEvent] = []
    for t in times:
        fr = int(round(float(t) * fps / 1000.0))
        out.append(SyncEvent(
            camera_id=cam,
            timestamp_ms=float(t),
            frame=fr,
            kind="rim_ball",
            student_id=None,
            score=1.0,
        ))
    return out


def match_event_series(
    anchor: list[SyncEvent],
    other: list[SyncEvent],
    *,
    max_match_ms: float = 2500.0,
) -> list[dict[str, Any]]:
    """
    Greedy one-to-one nearest matching in time.

    Returns matches with dt_ms = other.timestamp_ms - anchor.timestamp_ms
    (i.e. offset to subtract from other to land on anchor clock).
    """
    if not anchor or not other:
        return []
    used_o: set[int] = set()
    matches: list[dict[str, Any]] = []
    for ai, a in enumerate(sorted(anchor, key=lambda e: e.timestamp_ms)):
        best_j = None
        best_abs = None
        for oj, o in enumerate(other):
            if oj in used_o:
                continue
            d = abs(float(o.timestamp_ms) - float(a.timestamp_ms))
            if d > max_match_ms:
                continue
            if best_abs is None or d < best_abs:
                best_abs = d
                best_j = oj
        if best_j is None:
            continue
        used_o.add(best_j)
        o = other[best_j]
        matches.append({
            "anchor_ms": float(a.timestamp_ms),
            "anchor_frame": a.frame,
            "anchor_kind": a.kind,
            "other_ms": float(o.timestamp_ms),
            "other_frame": o.frame,
            "other_kind": o.kind,
            "dt_ms": float(o.timestamp_ms) - float(a.timestamp_ms),
            "abs_dt_ms": float(best_abs),
            "student_id": a.student_id or o.student_id,
        })
    return matches


def estimate_offset_from_matches(matches: list[dict[str, Any]]) -> float | None:
    """Robust offset = median(dt_ms)."""
    if not matches:
        return None
    dts = np.asarray([m["dt_ms"] for m in matches], dtype=np.float64)
    return float(np.median(dts))


def collect_events_for_camera(
    session_id: str,
    camera_id: str,
    student_ids: list[str],
    *,
    include_rim: bool = True,
) -> list[SyncEvent]:
    events: list[SyncEvent] = []
    rim_cam = get_shot_outcome_camera()
    if camera_id == rim_cam:
        if include_rim:
            events.extend(extract_rim_events(session_id))
        return events
    for sid in student_ids:
        events.extend(extract_release_events(session_id, camera_id, sid))
    return events


def estimate_camera_offsets(
    session_id: str,
    *,
    anchor_camera: str | None = None,
    student_ids: list[str] | None = None,
    camera_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Estimate constant per-camera clock offsets vs anchor via event matching.

    Convention:
      common_ms = local_ms - camera_time_offsets_ms[cam]
      local_ms   = common_ms + camera_time_offsets_ms[cam]
    Anchor offset is always 0.
    """
    cfg = _event_sync_cfg()
    sync = get_sync_config()
    anchor = anchor_camera or sync.get("event_anchor_camera") or get_action_segment_camera()
    camera_ids = camera_ids or get_camera_ids()
    max_match = float(cfg.get("max_match_ms", 2500.0))
    min_matches = int(cfg.get("min_matches", 2))
    include_rim = bool(cfg.get("use_rim_events", True))

    if student_ids is None:
        student_ids = list_session_student_ids(session_id, camera_ids)
    if not student_ids:
        # Fall back: still allow rim-only pairing later; releases need an id
        student_ids = []

    per_cam_events: dict[str, list[SyncEvent]] = {}
    for cam in camera_ids:
        per_cam_events[cam] = collect_events_for_camera(
            session_id, cam, student_ids, include_rim=include_rim,
        )

    anchor_events = per_cam_events.get(anchor) or []
    # If anchor has no releases but has students on other cams, keep going with empty → zero offsets
    offsets: dict[str, float] = {anchor: 0.0}
    match_docs: dict[str, Any] = {}
    quality: dict[str, Any] = {}

    for cam in camera_ids:
        if cam == anchor:
            match_docs[cam] = {"matches": [], "n_matches": 0, "offset_ms": 0.0}
            quality[cam] = {"status": "anchor", "n_events": len(anchor_events)}
            continue

        other = per_cam_events.get(cam) or []
        # Prefer same-kind matching: release↔release; rim only vs anchor release if needed
        a_rel = [e for e in anchor_events if e.kind == "release"]
        o_rel = [e for e in other if e.kind == "release"]
        matches = match_event_series(a_rel, o_rel, max_match_ms=max_match)

        if cam == get_shot_outcome_camera() and include_rim:
            a_for_rim = a_rel or anchor_events
            o_rim = [e for e in other if e.kind == "rim_ball"]
            rim_matches = match_event_series(a_for_rim, o_rim, max_match_ms=max_match)
            # Prefer rim matches when available (often cleaner than side-view misses)
            if len(rim_matches) >= min_matches or (not matches and rim_matches):
                matches = rim_matches

        off = estimate_offset_from_matches(matches)
        n = len(matches)
        if off is None or n < min_matches:
            # Weak evidence: still use median if ≥1 match, else 0
            if off is not None and n >= 1:
                offsets[cam] = float(off)
                status = "weak"
            else:
                offsets[cam] = 0.0
                status = "no_match"
        else:
            offsets[cam] = float(off)
            status = "ok"

        residuals = [m["dt_ms"] - offsets[cam] for m in matches] if matches else []
        match_docs[cam] = {
            "matches": matches,
            "n_matches": n,
            "offset_ms": offsets[cam],
            "residual_mad_ms": float(np.median(np.abs(residuals))) if residuals else None,
        }
        quality[cam] = {
            "status": status,
            "n_events": len(other),
            "n_matches": n,
            "offset_ms": offsets[cam],
        }

    return {
        "anchor_camera": anchor,
        "student_ids": student_ids,
        "camera_time_offsets_ms": offsets,
        "per_camera_matches": match_docs,
        "quality": quality,
        "events": {
            cam: [e.to_dict() for e in evs] for cam, evs in per_cam_events.items()
        },
        "config": {
            "max_match_ms": max_match,
            "min_matches": min_matches,
            "use_rim_events": include_rim,
        },
    }


def apply_offset(local_ms: float, offset_ms: float) -> float:
    """local → common (anchor) clock."""
    return float(local_ms) - float(offset_ms)


def invert_offset(common_ms: float, offset_ms: float) -> float:
    """common (anchor) → local clock."""
    return float(common_ms) + float(offset_ms)


def load_alignment(session_id: str) -> dict[str, Any] | None:
    path = data_path("sessions", session_id, "sync", "alignment.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_camera_offsets_ms(session_id: str) -> dict[str, float]:
    doc = load_alignment(session_id) or {}
    raw = doc.get("camera_time_offsets_ms") or {}
    return {str(k): float(v) for k, v in raw.items()}
