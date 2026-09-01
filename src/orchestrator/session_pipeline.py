"""Session lifecycle and pipeline orchestration — v1 per-camera."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.acquisition.recorder import mark_recording_start, mark_recording_stop
from src.action.pipeline import run_action_session_auto
from src.cameras.registry import get_camera_ids, load_camera_layout
from src.cameras.temporal import run_temporal_alignment
from src.config import data_path, load_yaml
from src.identity.perception import run_perception_session
from src.privacy.audit import log_audit
from src.privacy.db import get_conn
from src.pose.pose2sim_wrapper import run_pose3d_session
from src.scoring.fusion import run_scoring_session
from src.shot.outcome import run_shot_outcome_session
from src.types import SessionStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


STAGE_ORDER = [
    SessionStatus.CREATED,
    SessionStatus.CONSENT_OK,
    SessionStatus.ENROLLED,
    SessionStatus.RECORDED,
    SessionStatus.PERCEPTION_DONE,
    SessionStatus.SYNC_DONE,
    SessionStatus.POSE3D_DONE,
    SessionStatus.ACTION_DONE,
    SessionStatus.SHOT_OUTCOME_DONE,
    SessionStatus.SCORED,
    SessionStatus.REPORT_READY,
]


def create_session(class_id: str, metadata: dict | None = None) -> str:
    session_id = str(uuid.uuid4())
    layout_id = load_camera_layout().get("layout_id", "v2_4cam_zoned")
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, class_id, status, camera_layout, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                class_id,
                SessionStatus.CREATED.value,
                layout_id,
                _now(),
                _now(),
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
    log_audit("session_create", session_id=session_id, detail={"class_id": class_id})
    return session_id


def update_session_status(session_id: str, status: SessionStatus) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
            (status.value, _now(), session_id),
        )


def get_session(session_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def register_student(student_id: str, display_name: str, class_id: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO students (student_id, display_name, class_id, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET display_name = excluded.display_name
            """,
            (student_id, display_name, class_id, _now()),
        )


STAGE_ORDER = ("perception", "sync", "pose3d", "action", "shot", "scoring")


def _stage_at_or_after(stage: str, from_stage: str) -> bool:
    """True if `stage` should run when resuming at `from_stage`."""
    if from_stage not in STAGE_ORDER:
        raise ValueError(f"Unknown from_stage={from_stage!r}; expected one of {STAGE_ORDER}")
    return STAGE_ORDER.index(stage) >= STAGE_ORDER.index(from_stage)


def run_pipeline(
    session_id: str,
    from_stage: str = "perception",
    student_ids: list[str] | None = None,
    camera_ids: list[str] | None = None,
) -> dict:
    """
    Offline pipeline v2 — per-camera isolated processing + event/timestamp fusion.

    from_stage: start at this stage and run through scoring (skip earlier stages).
    Requires upstream artifacts to already exist when skipping.
    Stages: perception → sync → pose3d → action → shot → scoring
    """
    log_audit("pipeline_run", session_id=session_id, detail={"from_stage": from_stage})
    camera_ids = camera_ids or get_camera_ids()
    results: dict = {
        "session_id": session_id,
        "stages": {},
        "cameras": camera_ids,
        "from_stage": from_stage,
    }

    try:
        if _stage_at_or_after("perception", from_stage):
            outs = run_perception_session(session_id, camera_ids)
            results["stages"]["perception"] = {
                "mode": "per_camera_isolated",
                "outputs": [str(o) for o in outs],
            }
            update_session_status(session_id, SessionStatus.PERCEPTION_DONE)

        if _stage_at_or_after("sync", from_stage):
            align_path = run_temporal_alignment(
                session_id, camera_ids, student_ids=student_ids, use_events=True,
            )
            results["stages"]["sync"] = str(align_path)
            update_session_status(session_id, SessionStatus.SYNC_DONE)

        if _stage_at_or_after("pose3d", from_stage):
            sids = run_pose3d_session(session_id, camera_ids)
            results["stages"]["pose3d"] = {"students": sids, "fusion": "timestamp_aligned"}
            update_session_status(session_id, SessionStatus.POSE3D_DONE)

        if _stage_at_or_after("action", from_stage):
            sids = run_action_session_auto(session_id, student_ids)
            results["stages"]["action"] = sids
            update_session_status(session_id, SessionStatus.ACTION_DONE)

        if _stage_at_or_after("shot", from_stage):
            shot_paths = run_shot_outcome_session(session_id)
            results["stages"]["shot_outcome"] = [str(p) for p in shot_paths]
            update_session_status(session_id, SessionStatus.SHOT_OUTCOME_DONE)

        if _stage_at_or_after("scoring", from_stage):
            paths = run_scoring_session(session_id, student_ids)
            results["stages"]["scoring"] = [str(p) for p in paths]
            update_session_status(session_id, SessionStatus.REPORT_READY)

        results["status"] = "ok"
    except Exception as e:
        update_session_status(session_id, SessionStatus.FAILED)
        results["status"] = "failed"
        results["error"] = str(e)
        raise

    return results


def checkpoint_path(session_id: str) -> Path:
    return data_path("sessions", session_id, "checkpoint.json")


def save_checkpoint(session_id: str, data: dict) -> None:
    p = checkpoint_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
