"""Camera registry and per-camera processing utilities."""

from src.cameras.registry import (
    get_action_segment_camera,
    get_camera,
    get_camera_ids,
    get_enrollment_camera,
    get_shot_outcome_camera,
    get_zone,
    list_zones,
    load_camera_layout,
)
from src.cameras.temporal import (
    align_clips_across_cameras,
    build_per_camera_timelines,
    collect_student_kpts_at_time,
    frame_to_timestamp_ms,
    run_temporal_alignment,
)

__all__ = [
    "load_camera_layout",
    "get_camera_ids",
    "get_camera",
    "get_enrollment_camera",
    "get_action_segment_camera",
    "get_shot_outcome_camera",
    "get_zone",
    "list_zones",
    "frame_to_timestamp_ms",
    "build_per_camera_timelines",
    "align_clips_across_cameras",
    "collect_student_kpts_at_time",
    "run_temporal_alignment",
]
