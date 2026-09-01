"""Temporal alignment across independently processed cameras."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.cameras.registry import get_camera, get_sync_config
from src.config import data_path, load_yaml


def frame_to_timestamp_ms(frame_idx: int, fps: float) -> float:
    return float(frame_idx) / max(fps, 1e-6) * 1000.0


def _load_pose2d_doc(session_id: str, camera_id: str) -> dict:
    p = data_path("sessions", session_id, "perception", camera_id, "pose2d.json")
    if not p.exists():
        return {"camera_id": camera_id, "frames": [], "fps": 30.0}
    return json.loads(p.read_text(encoding="utf-8"))


def build_per_camera_timelines(session_id: str, camera_ids: list[str] | None = None) -> dict[str, dict]:
    """
    Build independent timelines per camera.
    Each frame carries local frame_idx + timestamp_ms (no cross-cam frame equality).
    """
    from src.cameras.registry import get_camera_ids

    camera_ids = camera_ids or get_camera_ids()
    timelines: dict[str, dict] = {}

    for cam_id in camera_ids:
        doc = _load_pose2d_doc(session_id, cam_id)
        fps = float(doc.get("fps") or get_camera(cam_id).get("fps") or 30.0)
        frames = []
        for fr in doc.get("frames", []):
            fidx = int(fr["frame"])
            ts = fr.get("timestamp_ms")
            if ts is None:
                ts = frame_to_timestamp_ms(fidx, fps)
            frames.append({**fr, "timestamp_ms": float(ts)})
        timelines[cam_id] = {
            "camera_id": cam_id,
            "fps": fps,
            "frame_count": len(frames),
            "duration_ms": frames[-1]["timestamp_ms"] if frames else 0.0,
            "frames": frames,
        }
    return timelines


def _nearest_frame_by_time(frames: list[dict], target_ms: float) -> dict | None:
    if not frames:
        return None
    best = min(frames, key=lambda f: abs(f["timestamp_ms"] - target_ms))
    return best


def _group_offsets_for_session(session_id: str, raw_meta: dict | None) -> dict[str, float]:
    """Load offsets from data/<dataset>/sync/group_XX.json when discoverable."""
    from src.cameras.group_sync import load_group_sync
    from src.config import ROOT

    sync = get_sync_config()
    group_id = None
    data_dir = None
    if raw_meta:
        group_id = raw_meta.get("group_id")
        if raw_meta.get("group_sync_data_dir"):
            data_dir = Path(raw_meta["group_sync_data_dir"])
    try:
        from src.orchestrator.session_pipeline import get_session

        row = get_session(session_id)
        if row and row.get("metadata"):
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            if group_id is None and meta.get("group_id") is not None:
                group_id = meta.get("group_id")
            if data_dir is None and meta.get("data_dir"):
                data_dir = Path(meta["data_dir"])
    except Exception:
        pass

    if group_id is None:
        return {}
    if data_dir is None:
        rel = sync.get("group_sync_data_dir") or "data/test_data_v3"
        data_dir = Path(rel)
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
    doc = load_group_sync(data_dir, int(group_id))
    if not doc:
        return {}
    return {
        str(k): float(v)
        for k, v in (doc.get("camera_time_offsets_ms") or {}).items()
    }


def _merge_offsets(
    session_id: str,
    event_offsets: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Merge offset sources. Priority (high → low) when prefer_manual_offsets:

      1. configs/cameras.yaml sync.manual_offsets_ms   (shared baseline)
      2. <data-dir>/sync/group_XX.json                   (GUI / group-level)
      3. sessions/<id>/raw/sync_meta.json               (per-session manual)
      4. event-estimated offsets                        (fill gaps only)

    When prefer_manual_offsets is false (legacy): event overwrites manual.
    """
    sync = get_sync_config()
    prefer_manual = bool(sync.get("prefer_manual_offsets", True))

    yaml_manual = {
        str(k): float(v)
        for k, v in (sync.get("manual_offsets_ms") or {}).items()
    }
    meta_manual: dict[str, float] = {}
    raw_meta: dict = {}
    raw_meta_path = data_path("sessions", session_id, "raw", "sync_meta.json")
    if raw_meta_path.exists():
        raw_meta = json.loads(raw_meta_path.read_text(encoding="utf-8"))
        for k, v in (raw_meta.get("camera_time_offsets_ms") or {}).items():
            meta_manual[str(k)] = float(v)

    group_manual = _group_offsets_for_session(session_id, raw_meta)
    event = {str(k): float(v) for k, v in (event_offsets or {}).items()}

    if prefer_manual:
        offsets = dict(event)
        offsets.update(yaml_manual)
        offsets.update(group_manual)
        offsets.update(meta_manual)  # session manual wins
    else:
        offsets = dict(yaml_manual)
        offsets.update(group_manual)
        offsets.update(meta_manual)
        offsets.update(event)
    return offsets


def write_manual_offsets(
    session_id: str,
    offsets: dict[str, float],
    *,
    merge: bool = True,
) -> Path:
    """Persist per-session manual offsets into raw/sync_meta.json."""
    path = data_path("sessions", session_id, "raw", "sync_meta.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {}
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
    existing = dict(doc.get("camera_time_offsets_ms") or {}) if merge else {}
    existing.update({str(k): float(v) for k, v in offsets.items()})
    doc["camera_time_offsets_ms"] = existing
    doc["offset_source"] = "manual"
    doc["offset_convention"] = (
        "common_ms = local_ms - offset_ms; "
        "local_ms = common_ms + offset_ms; "
        "anchor offset is 0"
    )
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def align_clips_across_cameras(
    session_id: str,
    anchor_camera: str,
    anchor_start_ms: float,
    anchor_end_ms: float,
    camera_ids: list[str] | None = None,
    *,
    apply_offsets: bool = True,
) -> dict[str, dict]:
    """
    Map an event window on the *anchor / common* clock to per-camera local frames.

    If apply_offsets=True, uses camera_time_offsets_ms from alignment.json:
      local_ms = common_ms + offset_ms[cam]
    """
    from src.cameras.event_sync import get_camera_offsets_ms
    from src.cameras.registry import get_camera_ids

    sync = get_sync_config()
    half = float(sync.get("max_drift_ms", 200))
    camera_ids = camera_ids or get_camera_ids()
    timelines = build_per_camera_timelines(session_id, camera_ids)
    offsets = get_camera_offsets_ms(session_id) if apply_offsets else {}

    anchor_frames = timelines.get(anchor_camera, {}).get("frames", [])
    anchor_start_frame = _nearest_frame_by_time(anchor_frames, anchor_start_ms)
    anchor_end_frame = _nearest_frame_by_time(anchor_frames, anchor_end_ms)

    result = {
        "anchor_camera": anchor_camera,
        "anchor_start_ms": anchor_start_ms,
        "anchor_end_ms": anchor_end_ms,
        "camera_time_offsets_ms": offsets,
        "per_camera": {},
    }

    for cam_id, tl in timelines.items():
        frames = tl.get("frames", [])
        off = float(offsets.get(cam_id, 0.0))
        start_ms = anchor_start_ms + off
        end_ms = anchor_end_ms + off
        if cam_id != anchor_camera:
            start_ms -= half
            end_ms += half
        start_fr = _nearest_frame_by_time(frames, start_ms)
        end_fr = _nearest_frame_by_time(frames, end_ms)
        result["per_camera"][cam_id] = {
            "offset_ms": off,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_frame": start_fr["frame"] if start_fr else None,
            "end_frame": end_fr["frame"] if end_fr else None,
            "start_timestamp_ms": start_fr["timestamp_ms"] if start_fr else None,
            "end_timestamp_ms": end_fr["timestamp_ms"] if end_fr else None,
        }

    if anchor_start_frame and anchor_end_frame:
        result["anchor_start_frame"] = anchor_start_frame["frame"]
        result["anchor_end_frame"] = anchor_end_frame["frame"]

    return result


def run_temporal_alignment(
    session_id: str,
    camera_ids: list[str] | None = None,
    *,
    student_ids: list[str] | None = None,
    use_events: bool | None = None,
) -> Path:
    """
    Persist sync/alignment.json for Pose2Sim / multi-view fusion.

    Default align_method=event_anchor: estimate constant Δt from release/rim events.
    Falls back to sync_meta / zero offsets when events are insufficient.
    """
    from src.cameras.event_sync import estimate_camera_offsets
    from src.cameras.registry import get_camera_ids

    camera_ids = camera_ids or get_camera_ids()
    sync_cfg = get_sync_config()
    method = str(sync_cfg.get("align_method") or "event_anchor")
    if use_events is None:
        use_events = method in ("event_anchor", "event", "release_event")

    timelines = build_per_camera_timelines(session_id, camera_ids)

    event_doc: dict | None = None
    event_offsets: dict[str, float] = {}
    if use_events:
        try:
            event_doc = estimate_camera_offsets(
                session_id,
                anchor_camera=sync_cfg.get("event_anchor_camera"),
                student_ids=student_ids,
                camera_ids=camera_ids,
            )
            event_offsets = {
                str(k): float(v)
                for k, v in (event_doc.get("camera_time_offsets_ms") or {}).items()
            }
        except Exception as exc:  # noqa: BLE001 — never block pipeline on sync
            event_doc = {"error": str(exc)}
            event_offsets = {}

    offsets = _merge_offsets(session_id, event_offsets)

    out_dir = data_path("sessions", session_id, "sync")
    out_dir.mkdir(parents=True, exist_ok=True)

    alignment_doc = {
        "session_id": session_id,
        "layout_id": load_yaml("cameras.yaml").get("layout_id"),
        "mode": sync_cfg.get("mode", "independent"),
        "align_method": "event_anchor" if use_events else method,
        "anchor_camera": (event_doc or {}).get("anchor_camera")
            or sync_cfg.get("event_anchor_camera", "cam_03"),
        "camera_time_offsets_ms": offsets,
        "offset_convention": (
            "common_ms = local_ms - offset_ms; "
            "local_ms = common_ms + offset_ms; "
            "anchor offset is 0"
        ),
        "timelines": {
            cam: {
                "fps": tl["fps"],
                "frame_count": tl["frame_count"],
                "duration_ms": tl["duration_ms"],
            }
            for cam, tl in timelines.items()
        },
        "event_sync": {
            "student_ids": (event_doc or {}).get("student_ids"),
            "quality": (event_doc or {}).get("quality"),
            "per_camera_matches": (event_doc or {}).get("per_camera_matches"),
            "config": (event_doc or {}).get("config"),
            "error": (event_doc or {}).get("error"),
        } if use_events else None,
    }

    out_path = out_dir / "alignment.json"
    out_path.write_text(json.dumps(alignment_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    if use_events and event_doc and event_doc.get("events"):
        (out_dir / "events.json").write_text(
            json.dumps({
                "session_id": session_id,
                "events": event_doc["events"],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Cross-camera ID prior (cam_01↔cam_02 L/R reverse)
    try:
        from src.identity.cross_cam_spatial import repair_cross_camera_identities
        spatial = repair_cross_camera_identities(session_id, camera_ids)
        alignment_doc["spatial_identity_repair"] = {
            "status": spatial.get("status"),
            "mode": spatial.get("mode"),
            "n_reassigned": spatial.get("n_reassigned"),
            "n_pairs": spatial.get("n_pairs"),
            "clusters": spatial.get("clusters"),
            "output": spatial.get("output"),
        }
        out_path.write_text(json.dumps(alignment_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  [sync] ID prior mode={spatial.get('mode')} status={spatial.get('status')} "
            f"reassigned={spatial.get('n_reassigned')}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [sync] spatial ID repair skipped: {exc}", flush=True)

    # Track-level ID hysteresis (suppress gallery flicker on continuous tracks)
    try:
        from src.identity.track_id_smooth import smooth_session_identities
        smooth = smooth_session_identities(session_id, camera_ids=camera_ids, write=True)
        alignment_doc["track_id_smooth"] = {
            "enabled": smooth.get("enabled"),
            "cameras": {
                cam: {"n_changed": st.get("n_changed"), "n_tracks": st.get("n_tracks")}
                for cam, st in (smooth.get("cameras") or {}).items()
            },
        }
        out_path.write_text(json.dumps(alignment_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        n_ch = sum(int(st.get("n_changed") or 0) for st in (smooth.get("cameras") or {}).values())
        print(f"  [sync] track_id_smooth changed_slots={n_ch}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [sync] track_id_smooth skipped: {exc}", flush=True)

    return out_path


def collect_student_kpts_at_time(
    session_id: str,
    student_id: str,
    target_ms: float,
    camera_ids: list[str] | None = None,
    *,
    target_on_anchor_clock: bool = True,
) -> dict[str, np.ndarray]:
    """
    Fetch per-camera 133-point pose nearest to target_ms for 3D triangulation.

    target_ms is on the *anchor/common* clock by default; each camera query uses
    local_ms = target_ms + offset_ms[cam].
    """
    from src.cameras.event_sync import get_camera_offsets_ms
    from src.cameras.registry import get_camera_ids

    camera_ids = camera_ids or get_camera_ids()
    timelines = build_per_camera_timelines(session_id, camera_ids)
    offsets = get_camera_offsets_ms(session_id) if target_on_anchor_clock else {}
    out: dict[str, np.ndarray] = {}

    for cam_id, tl in timelines.items():
        frames = tl.get("frames", [])
        local_ms = float(target_ms) + float(offsets.get(cam_id, 0.0))
        fr = _nearest_frame_by_time(frames, local_ms)
        if not fr:
            continue
        for person in fr.get("persons", []):
            if person.get("student_id") != student_id:
                continue
            kpts = np.array(person["keypoints"], dtype=np.float32)
            if kpts.shape[0] < 133:
                pad = np.zeros((133 - kpts.shape[0], kpts.shape[1]), dtype=np.float32)
                kpts = np.vstack([kpts, pad])
            out[cam_id] = kpts
            break
    return out
