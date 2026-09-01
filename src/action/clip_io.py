"""Serialize ActionClip → flat dicts for outcome / fast-path consumers."""

from __future__ import annotations

import json
from typing import Any

from src.config import data_path
from src.types import ActionClip


def release_ms_from_clip_dict(clip: dict[str, Any]) -> float | None:
    r = clip.get("release_ms")
    if r is not None:
        return float(r)
    start, end = clip.get("start_ms"), clip.get("end_ms")
    if start is not None and end is not None:
        return 0.5 * (float(start) + float(end))
    return None


def action_clip_to_dict(
    clip: ActionClip,
    student_id: str,
    index: int,
) -> dict[str, Any]:
    """Flat dict used by shot outcome alignment and streaming finalize."""
    release_ms = next(
        (float(p.start_ms) for p in clip.phases if p.name == "release" and p.start_ms is not None),
        clip.start_ms,
    )
    return {
        "student_id": clip.student_id or student_id,
        "participant_ids": list(clip.participant_ids or [clip.student_id or student_id]),
        "clip_id": f"{student_id}:{index}",
        "action_type": clip.action_type,
        "start_ms": clip.start_ms,
        "end_ms": clip.end_ms,
        "release_ms": release_ms,
    }


def load_session_action_clip_dicts(session_id: str) -> list[dict[str, Any]]:
    """Flatten all student action JSONs under a session for timestamp association."""
    act_dir = data_path("sessions", session_id, "actions")
    if not act_dir.exists():
        return []
    clips: list[dict[str, Any]] = []
    for path in sorted(act_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        sid = doc.get("student_id") or path.stem
        for i, clip in enumerate(doc.get("clips", [])):
            # Prefer cam_04 rim clock when present — leave_after windows skew
            # phase release late and break make/miss segment alignment.
            meta = clip.get("metadata") or {}
            mc = meta.get("multicam") or {}
            release = None
            if mc.get("rim_timestamp_ms") is not None:
                release = float(mc["rim_timestamp_ms"])
            elif clip.get("release_ms") is not None:
                release = float(clip["release_ms"])
            elif meta.get("release_ms") is not None:
                release = float(meta["release_ms"])
            else:
                release = next(
                    (
                        p.get("start_ms")
                        for p in clip.get("phases", [])
                        if p.get("name") == "release"
                    ),
                    None,
                )
            clip_sid = clip.get("student_id") or sid
            parts = list(clip.get("participant_ids") or [])
            if not parts:
                parts = [clip_sid]
            clips.append({
                "student_id": clip_sid,
                "participant_ids": parts,
                "clip_id": f"{sid}:{i}",
                "action_type": clip.get("action_type", "unknown"),
                "start_ms": clip.get("start_ms"),
                "end_ms": clip.get("end_ms"),
                "release_ms": release,
                "metadata": meta,
            })
    return clips
