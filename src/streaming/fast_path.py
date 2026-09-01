"""Streaming / near-realtime classroom path.

Design (matches 10s teacher feedback goal)
-----------------------------------------
* Always-on (target): cam_03 pose + cam_04 ball into ``TimestampRingBuffer``
* On event (ball above hoop / release peak): finalize a short window
* Emit compact JSON immediately; 3D / viz / dashboard are async

**Current status:** production validation uses ``finalize_action_from_session``
on already-perceived sessions (pose/ball on disk). ``TimestampRingBuffer`` is
the online buffer type for a future always-on capture worker; it is not yet
fed by live cameras.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.action.registry import normalize_action_type
from src.config import data_path
from src.types import ActionClip


@dataclass
class RingFrame:
    timestamp_ms: float
    frame_idx: int
    payload: dict[str, Any]


@dataclass
class TimestampRingBuffer:
    """Fixed-duration ring of timestamped payloads (pose / ball samples)."""

    capacity_ms: float = 60_000.0
    _items: deque[RingFrame] = field(default_factory=deque)

    def push(self, timestamp_ms: float, frame_idx: int, payload: dict[str, Any]) -> None:
        self._items.append(RingFrame(timestamp_ms, frame_idx, payload))
        self._trim(timestamp_ms)

    def _trim(self, now_ms: float) -> None:
        lo = now_ms - self.capacity_ms
        while self._items and self._items[0].timestamp_ms < lo:
            self._items.popleft()

    def window(self, t0_ms: float, t1_ms: float) -> list[RingFrame]:
        return [x for x in self._items if t0_ms <= x.timestamp_ms <= t1_ms]

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class FastPathResult:
    """Compact teacher-facing result for one action (≤10s path)."""

    student_id: str
    action_type: str
    start_ms: float | None
    end_ms: float | None
    release_ms: float | None
    confidence: float
    phases: list[dict[str, Any]]
    made: bool | None = None
    outcome_confidence: float | None = None
    latency_ms: float = 0.0
    source: str = "fast_path"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "action_type": self.action_type,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "release_ms": self.release_ms,
            "confidence": self.confidence,
            "phases": self.phases,
            "made": self.made,
            "outcome_confidence": self.outcome_confidence,
            "latency_ms": round(self.latency_ms, 1),
            "source": self.source,
            "metadata": self.metadata,
        }


def _clip_to_fast_result(
    clip: ActionClip,
    student_id: str,
    *,
    made: bool | None = None,
    outcome_confidence: float | None = None,
    latency_ms: float = 0.0,
    extra_meta: dict | None = None,
) -> FastPathResult:
    atype = normalize_action_type(clip.action_type)
    phases = []
    for ph in clip.phases:
        phases.append({
            "name": ph.name,
            "start_ms": ph.start_ms,
            "end_ms": ph.end_ms,
        })
    release_ms = None
    for ph in clip.phases:
        if ph.name == "release" and ph.start_ms is not None:
            release_ms = float(ph.start_ms)
            break
    meta = dict(clip.metadata or {})
    if extra_meta:
        meta.update(extra_meta)
    return FastPathResult(
        student_id=student_id,
        action_type=atype,
        start_ms=clip.start_ms,
        end_ms=clip.end_ms,
        release_ms=release_ms,
        confidence=float(clip.confidence),
        phases=phases,
        made=made,
        outcome_confidence=outcome_confidence,
        latency_ms=latency_ms,
        metadata=meta,
    )


def finalize_action_from_session(
    session_id: str,
    student_id: str,
    *,
    include_outcome: bool = True,
) -> tuple[list[FastPathResult], dict[str, float]]:
    """
    Fast-path finalize for a session: detect actions + optional make/miss.

    Measures wall time of the *finalize* stage only (pose/ball already exist),
    which models the post-action budget when always-on inference is warm.
    """
    from src.action.clip_io import action_clip_to_dict
    from src.action.pipeline import detect_actions_auto
    from src.shot.outcome import outcomes_from_clips_and_track

    t0 = time.perf_counter()
    actions = detect_actions_auto(session_id, student_id)
    t_detect = time.perf_counter() - t0

    outcomes_by_clip: dict[str, dict] = {}
    t_out = 0.0
    if include_outcome:
        track_path = data_path("sessions", session_id, "shot_outcomes", "ball_track.json")
        if track_path.exists():
            t1 = time.perf_counter()
            track = json.loads(track_path.read_text(encoding="utf-8"))
            clip_dicts = [
                action_clip_to_dict(c, student_id, i)
                for i, c in enumerate(actions.clips)
            ]
            outs = outcomes_from_clips_and_track(clip_dicts, track)
            for o in outs:
                clip_meta = o.get("clip") if isinstance(o.get("clip"), dict) else {}
                cid = o.get("clip_id") or clip_meta.get("clip_id")
                if cid:
                    outcomes_by_clip[cid] = o
            t_out = time.perf_counter() - t1

    results: list[FastPathResult] = []
    total_ms = (t_detect + t_out) * 1000.0
    per = total_ms / max(len(actions.clips), 1)
    for i, c in enumerate(actions.clips):
        cid = f"{student_id}:{i}"
        o = outcomes_by_clip.get(cid) or {}
        results.append(_clip_to_fast_result(
            c, student_id,
            made=o.get("made"),
            outcome_confidence=o.get("confidence"),
            latency_ms=per,
            extra_meta={"fast_path_mode": "session_finalize"},
        ))

    timings = {
        "detect_s": round(t_detect, 3),
        "outcome_s": round(t_out, 3),
        "total_s": round(t_detect + t_out, 3),
        "per_action_s": round((t_detect + t_out) / max(len(actions.clips), 1), 3),
        "n_clips": float(len(actions.clips)),
    }
    return results, timings


def simulate_per_action_latency(
    session_id: str,
    student_id: str,
    *,
    pre_ms: float = 8000.0,
    post_ms: float = 5000.0,
) -> list[dict[str, Any]]:
    """
    For each detected action, estimate always-on finalize budget vs 10s target.

    Teacher-facing latency ≈ post-event wait (already buffered by always-on)
    + window slice + detect/outcome. Always-on pose/ball cost is amortized and
    excluded; we only charge a small finalize overhead on the event window.
    """
    results, timings = finalize_action_from_session(session_id, student_id)
    # Always-on already filled the ring; finalize only re-slices + fuses.
    # Use measured detect/outcome share, plus a small buffer for I/O.
    measured = float(timings["per_action_s"])
    overhead_s = 0.3
    rows = []
    for r in results:
        rel = r.release_ms if r.release_ms is not None else r.start_ms
        if rel is None:
            continue
        # After release, wait until post window is complete, then finalize.
        wait_after_release_s = post_ms / 1000.0
        est_teacher_s = wait_after_release_s + measured + overhead_s
        rows.append({
            **r.to_dict(),
            "window_s": round((pre_ms + post_ms) / 1000.0, 2),
            "wait_after_release_s": round(wait_after_release_s, 2),
            "est_finalize_s": round(measured + overhead_s, 3),
            "est_teacher_latency_s": round(est_teacher_s, 2),
            "meets_10s": est_teacher_s <= 10.0,
            "measured_batch_per_action_s": timings["per_action_s"],
        })
    return rows
