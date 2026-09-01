"""Group-level multi-camera time offsets (manual or auto).

Convention (same as session sync):
  common_ms = local_ms - offset_ms
  local_ms  = common_ms + offset_ms
  Anchor camera offset is always 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import load_yaml


def default_sync_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "sync"


def group_sync_path(data_dir: Path, group_id: int) -> Path:
    return default_sync_dir(data_dir) / f"group_{int(group_id):02d}.json"


def discover_group_videos(data_dir: Path, group_id: int) -> dict[str, Path]:
    """Map cam_01..04 → {g}-{c}.mkv|mp4 under data_dir."""
    g = int(group_id)
    out: dict[str, Path] = {}
    for c in range(1, 5):
        cam = f"cam_{c:02d}"
        for ext in (".mkv", ".mp4", ".avi"):
            p = Path(data_dir) / f"{g}-{c}{ext}"
            if p.exists():
                out[cam] = p
                break
    return out


def empty_group_sync(
    group_id: int,
    *,
    dataset: str = "",
    anchor_camera: str | None = None,
) -> dict[str, Any]:
    sync = load_yaml("cameras.yaml").get("sync") or {}
    anchor = anchor_camera or sync.get("event_anchor_camera") or "cam_03"
    return {
        "version": 1,
        "group_id": int(group_id),
        "dataset": dataset,
        "anchor_camera": anchor,
        "offset_convention": (
            "common_ms = local_ms - offset_ms; "
            "local_ms = common_ms + offset_ms; "
            "anchor offset is 0"
        ),
        "camera_time_offsets_ms": {
            "cam_01": 0.0,
            "cam_02": 0.0,
            "cam_03": 0.0,
            "cam_04": 0.0,
            anchor: 0.0,
        },
        "source": "manual_gui",
        "notes": "",
    }


def load_group_sync(data_dir: Path, group_id: int) -> dict[str, Any] | None:
    path = group_sync_path(data_dir, group_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_group_sync(data_dir: Path, doc: dict[str, Any]) -> Path:
    path = group_sync_path(data_dir, int(doc["group_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Force anchor to 0
    anchor = str(doc.get("anchor_camera") or "cam_03")
    offs = dict(doc.get("camera_time_offsets_ms") or {})
    offs[anchor] = 0.0
    doc["camera_time_offsets_ms"] = {str(k): float(v) for k, v in offs.items()}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def apply_group_sync_to_session(
    session_id: str,
    group_doc: dict[str, Any],
    *,
    data_dir: Path | None = None,
) -> Path:
    """Write group offsets into sessions/<id>/raw/sync_meta.json (manual preferred)."""
    import json as _json

    from src.cameras.temporal import write_manual_offsets
    from src.config import data_path

    offs = {
        str(k): float(v)
        for k, v in (group_doc.get("camera_time_offsets_ms") or {}).items()
    }
    path = write_manual_offsets(session_id, offs, merge=True)
    meta = _json.loads(path.read_text(encoding="utf-8"))
    meta["offset_source"] = "group_gui"
    meta["group_id"] = int(group_doc.get("group_id") or 0)
    if data_dir is not None:
        meta["group_sync_data_dir"] = str(Path(data_dir).resolve())
    path.write_text(_json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # Ensure raw dir exists even if write_manual already did
    data_path("sessions", session_id, "raw").mkdir(parents=True, exist_ok=True)
    return path


def list_groups_with_videos(data_dir: Path) -> list[int]:
    found: set[int] = set()
    for p in Path(data_dir).glob("[0-9]-1.*"):
        try:
            found.add(int(p.name.split("-", 1)[0]))
        except ValueError:
            continue
    return sorted(found)
