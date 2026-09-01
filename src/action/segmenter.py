"""Backward-compatible aliases for action session entry points.

Prefer ``src.action.pipeline.run_action_session_auto`` in new code.
"""

from __future__ import annotations

import json
import warnings

from src.action.pipeline import (
    _enrich_clip_timestamps,
    detect_actions_auto,
    run_action_session_auto,
)
from src.config import data_path
from src.types import StudentActions

__all__ = [
    "_enrich_clip_timestamps",
    "detect_actions_auto",
    "run_action_for_zone",
    "run_action_session",
    "run_action_session_auto",
]


def run_action_for_zone(
    session_id: str,
    zone_id: str,
    student_id: str,
    action_type: str | None = None,
) -> StudentActions | None:
    """Detect actions; ``action_type`` is ignored (always auto-classified)."""
    if action_type is not None:
        warnings.warn(
            "run_action_for_zone(action_type=...) is deprecated; "
            "action types are always auto-classified.",
            DeprecationWarning,
            stacklevel=2,
        )
    result = detect_actions_auto(session_id, student_id)
    if not result.clips:
        return None
    clips = [c.model_copy(update={"zone_id": zone_id}) for c in result.clips]
    return StudentActions(student_id=student_id, clips=clips)


def run_action_session(
    session_id: str,
    student_ids: list[str] | None = None,
    zone_id: str | None = None,
    action_type: str | None = None,
) -> list[str]:
    """Session action detection (auto-classify). Zone path stamps ``zone_id``."""
    if action_type is not None:
        warnings.warn(
            "run_action_session(action_type=...) is deprecated and ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    done = run_action_session_auto(session_id, student_ids)
    if zone_id is None or not done:
        return done

    out_dir = data_path("sessions", session_id, "actions")
    stamped: list[str] = []
    for sid in done:
        path = out_dir / f"{sid}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for c in data.get("clips") or []:
            c["zone_id"] = zone_id
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        stamped.append(sid)
    return stamped
