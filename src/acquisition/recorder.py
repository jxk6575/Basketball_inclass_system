"""Multi-camera recording utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import data_path, load_yaml
from src.privacy.audit import log_audit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_sync_meta(
    session_id: str,
    camera_ids: list[str],
    duration_hint_sec: float = 0,
    camera_time_offsets_ms: dict[str, float] | None = None,
) -> Path:
    raw_dir = data_path("sessions", session_id, "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    cam_cfg = load_yaml("cameras.yaml")
    meta = {
        "session_id": session_id,
        "recorded_at": _now(),
        "layout_id": cam_cfg.get("layout_id"),
        "cameras": camera_ids,
        "sync_method": cam_cfg.get("sync", {}).get("mode", "independent"),
        "align_method": cam_cfg.get("sync", {}).get("align_method", "event_anchor"),
        "processing_mode": cam_cfg.get("processing", {}).get("mode", "per_camera_isolated"),
        "duration_hint_sec": duration_hint_sec,
        "camera_time_offsets_ms": camera_time_offsets_ms or {},
        "note": "Per-camera videos may differ in frame count; fusion uses timestamp_ms",
    }
    path = raw_dir / "sync_meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def mark_recording_start(session_id: str, actor: str | None = None) -> None:
    log_audit("recording_start", actor=actor, session_id=session_id)


def mark_recording_stop(session_id: str, actor: str | None = None) -> None:
    log_audit("recording_stop", actor=actor, session_id=session_id)


def list_raw_videos(session_id: str) -> dict[str, Path]:
    raw_dir = data_path("sessions", session_id, "raw")
    return {p.stem: p for p in raw_dir.glob("*.mp4")}
