"""Per-camera isolated perception pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from src.cameras.registry import get_camera, get_camera_ids
from src.config import data_path
from src.identity.perception import run_perception_on_video


def run_single_camera_perception(
    session_id: str,
    camera_id: str,
    video_path: Path | None = None,
    student_ids_filter: list[str] | None = None,
    stride: int = 1,
    force_student_id: str | None = None,
) -> Path:
    """
    Run detection + identity + pose for ONE camera only.
    Output uses local frame_idx and timestamp_ms — never assumes sync with other cams.
    """
    if video_path is None:
        video_path = data_path("sessions", session_id, "raw", f"{camera_id}.mp4")
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video for {camera_id}: {video_path}")

    out = run_perception_on_video(
        session_id=session_id,
        camera_id=camera_id,
        video_path=video_path,
        student_ids_filter=student_ids_filter,
        stride=stride,
        force_student_id=force_student_id,
    )

    meta_path = out / "camera_meta.json"
    cam = get_camera(camera_id)
    meta_path.write_text(
        json.dumps({
            "camera_id": camera_id,
            "index": cam.get("index"),
            "pipeline": cam.get("pipeline", "isolated"),
            "roles": cam.get("roles", []),
            "coverage": cam.get("coverage"),
            "processing": "per_camera_isolated",
            "stride": stride,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def run_perception_all_cameras(
    session_id: str,
    camera_ids: list[str] | None = None,
    student_ids_filter: list[str] | None = None,
    stride: int = 1,
    force_student_id: str | None = None,
) -> dict[str, str | None]:
    """
    Run each camera independently. Returns {cam_id: output_path or None if skipped}.
    """
    camera_ids = camera_ids or get_camera_ids()
    results: dict[str, str | None] = {}

    for cam_id in camera_ids:
        video = data_path("sessions", session_id, "raw", f"{cam_id}.mp4")
        if not video.exists():
            # also accept .mkv
            mkv = data_path("sessions", session_id, "raw", f"{cam_id}.mkv")
            video = mkv if mkv.exists() else video
        if not video.exists():
            results[cam_id] = None
            continue
        out = run_single_camera_perception(
            session_id,
            cam_id,
            video,
            student_ids_filter=student_ids_filter,
            stride=stride,
            force_student_id=force_student_id,
        )
        results[cam_id] = str(out)

    return results