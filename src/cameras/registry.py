"""Camera layout registry — v2 per-camera isolated processing."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.config import load_yaml


@lru_cache(maxsize=1)
def load_camera_layout() -> dict[str, Any]:
    return load_yaml("cameras.yaml")


@lru_cache(maxsize=1)
def load_zones_config() -> dict[str, Any]:
    return load_yaml("zones.yaml")


def get_camera_ids() -> list[str]:
    cams = load_camera_layout().get("cameras", {})
    return sorted(cams.keys(), key=lambda k: cams[k].get("index", 99))


def get_camera(camera_id: str) -> dict[str, Any]:
    cams = load_camera_layout().get("cameras", {})
    if camera_id not in cams:
        raise KeyError(f"Unknown camera: {camera_id}")
    return cams[camera_id]


def get_enrollment_camera() -> str:
    return load_camera_layout().get("enrollment_camera", "cam_03")


def get_action_segment_camera() -> str:
    return load_camera_layout().get("action_segment_camera", "cam_03")


def get_shot_outcome_camera() -> str:
    return load_camera_layout().get("shot_outcome_camera", "cam_04")


def get_sync_config() -> dict[str, Any]:
    return load_camera_layout().get("sync", {})


def get_processing_config() -> dict[str, Any]:
    return load_camera_layout().get("processing", {})


def get_perception_config() -> dict[str, Any]:
    return load_camera_layout().get("perception", {})


def camera_has_role(camera_id: str, role: str) -> bool:
    return role in get_camera(camera_id).get("roles", [])


def camera_runs_pose2d(camera_id: str) -> bool:
    """True if this camera should run person detection + pose2d (cam_04 is ball-only)."""
    return camera_has_role(camera_id, "pose2d")


def get_cameras_by_role(role: str) -> list[str]:
    return [cid for cid in get_camera_ids() if camera_has_role(cid, role)]


def get_zone(zone_id: str) -> dict[str, Any]:
    zones = load_zones_config().get("zones", {})
    if zone_id not in zones:
        raise KeyError(f"Unknown zone: {zone_id}")
    return zones[zone_id]


def list_zones() -> list[str]:
    return list(load_zones_config().get("zones", {}).keys())


def primary_camera_for_zone(zone_id: str) -> str:
    return get_zone(zone_id)["primary_camera"]


def assist_cameras_for_zone(zone_id: str) -> list[str]:
    return get_zone(zone_id).get("assist_cameras", [])
