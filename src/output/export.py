"""Export session perception/action/shot data as SessionOutput JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.action.halpe2h36m import wholebody133_to_h36m
from src.cameras.registry import get_action_segment_camera, get_shot_outcome_camera
from src.config import data_path
from src.output.schema import (
    ActionClipRef,
    ActionType,
    Ball3D,
    CoordinateSystem,
    MotionRecord,
    SessionOutput,
    Skeleton3D,
)
from src.pose.reference_template import H36M_JOINT_NAMES, kpts133_to_pseudo3d, normalize_h36m


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_at_ms(clips: list[dict], timestamp_ms: float) -> tuple[str, str, str | None]:
    """Return (action_type, action_phase, clip_id) for a timestamp."""
    for clip in clips:
        start = clip.get("start_ms")
        end = clip.get("end_ms")
        if start is None or end is None:
            continue
        if not (float(start) <= timestamp_ms <= float(end)):
            continue
        atype = clip.get("action_type", "unknown")
        clip_id = clip.get("clip_id")
        for ph in clip.get("phases", []):
            ps, pe = ph.get("start_ms"), ph.get("end_ms")
            if ps is not None and pe is not None and float(ps) <= timestamp_ms <= float(pe):
                return atype, ph.get("name", "full"), clip_id
        return atype, "full", clip_id
    return "unknown", "none", None


def _ball_at_ms(ball_frames: list[dict], timestamp_ms: float, max_drift_ms: float = 50.0) -> dict | None:
    if not ball_frames:
        return None
    best = None
    best_d = max_drift_ms
    for fr in ball_frames:
        d = abs(float(fr.get("timestamp_ms", 0)) - timestamp_ms)
        if d <= best_d and fr.get("ball"):
            best_d = d
            best = fr["ball"]
    return best


def _skeleton_from_kpts133(kpts: np.ndarray) -> Skeleton3D:
    k3 = kpts133_to_pseudo3d(np.asarray(kpts, dtype=np.float32))
    h36m = wholebody133_to_h36m(k3)
    # Keep approximate image-plane root so court preview shows body travel
    # (templates still use normalize_h36m without keep_root).
    if h36m.shape[1] >= 3:
        joints = normalize_h36m(h36m[:, :3], keep_root=True)
    else:
        joints = normalize_h36m(np.hstack([h36m, np.zeros((17, 1))]), keep_root=True)
    scores = None
    if k3.shape[1] >= 3:
        # map body scores roughly via first 17 coco
        scores = [float(k3[i, 2]) if i < len(k3) else 0.0 for i in range(17)]
    return Skeleton3D(
        format="h36m_17",
        joints=np.round(joints, 6).tolist(),
        joint_scores=scores,
        joint_names=list(H36M_JOINT_NAMES),
    )


def _normalize_action_type(raw: str) -> ActionType:
    from src.action.registry import normalize_action_type

    return normalize_action_type(raw)  # type: ignore[return-value]


def _collect_clips(session_id: str) -> list[dict]:
    act_dir = data_path("sessions", session_id, "actions")
    clips: list[dict] = []
    if not act_dir.exists():
        return clips
    for path in sorted(act_dir.glob("*.json")):
        doc = _load_json(path)
        if not isinstance(doc, dict):
            continue
        sid = doc.get("student_id") or path.stem
        for i, clip in enumerate(doc.get("clips", [])):
            release_ms = None
            phases = []
            for ph in clip.get("phases", []):
                phases.append({
                    "name": ph.get("name"),
                    "start_ms": ph.get("start_ms"),
                    "end_ms": ph.get("end_ms"),
                    "start": ph.get("start"),
                    "end": ph.get("end"),
                })
                if ph.get("name") == "release":
                    release_ms = ph.get("start_ms")
            meta = clip.get("metadata") or {}
            mc = meta.get("multicam") or {}
            # Prefer cam_03 pose / phase release for timeline (matches GT cam_03 clock).
            # Keep rim_timestamp_ms in metadata for attempt identity / outcome align.
            if mc.get("pose_timestamp_ms") is not None:
                release_ms = float(mc["pose_timestamp_ms"])
            elif meta.get("release_ms") is not None and release_ms is None:
                release_ms = float(meta["release_ms"])
            elif release_ms is None and mc.get("rim_timestamp_ms") is not None:
                release_ms = float(mc["rim_timestamp_ms"])
            clip_sid = clip.get("student_id") or sid
            parts = clip.get("participant_ids") or []
            if not parts:
                parts = [clip_sid]
            clips.append({
                "clip_id": f"{sid}:{i}",
                "student_id": clip_sid,
                "participant_ids": list(parts),
                "action_type": clip.get("action_type", "unknown"),
                "start_ms": clip.get("start_ms"),
                "end_ms": clip.get("end_ms"),
                "release_ms": release_ms,
                "anchor_camera": clip.get("anchor_camera"),
                "zone_id": clip.get("zone_id"),
                "phases": phases,
                "confidence": clip.get("confidence", 0.0),
                "metadata": {
                    k: meta[k]
                    for k in (
                        "multicam",
                        "action_classify",
                        "reason",
                        "relabel_reason",
                        "shooting_hand",
                        "shooting_hand_meta",
                    )
                    if k in meta
                } or None,
            })
    return _dedupe_overlapping_clips(clips)


def _dedupe_overlapping_clips(clips: list[dict]) -> list[dict]:
    """Merge near-duplicate clips across students (esp. pass with 2 actors)."""
    if len(clips) <= 1:
        return clips
    ordered = sorted(
        clips,
        key=lambda c: (float(c.get("start_ms") or 0), -(float(c.get("confidence") or 0))),
    )
    kept: list[dict] = []
    for c in ordered:
        s0, e0 = float(c.get("start_ms") or 0), float(c.get("end_ms") or 0)
        rim_c = None
        mc_c = ((c.get("metadata") or {}).get("multicam") or {})
        if mc_c.get("rim_timestamp_ms") is not None:
            rim_c = float(mc_c["rim_timestamp_ms"])
        elif c.get("release_ms") is not None:
            rim_c = float(c["release_ms"])
        merged = False
        for k in kept:
            if k.get("action_type") != c.get("action_type"):
                continue
            rim_k = None
            mc_k = ((k.get("metadata") or {}).get("multicam") or {})
            if mc_k.get("rim_timestamp_ms") is not None:
                rim_k = float(mc_k["rim_timestamp_ms"])
            elif k.get("release_ms") is not None:
                rim_k = float(k["release_ms"])
            # Distinct cam_04 rim attempts must never merge via window IoU
            if rim_c is not None and rim_k is not None and abs(rim_c - rim_k) >= 900.0:
                continue
            s1, e1 = float(k.get("start_ms") or 0), float(k.get("end_ms") or 0)
            inter = max(0.0, min(e0, e1) - max(s0, s1))
            union = max(e0, e1) - min(s0, s1) + 1e-6
            if inter / union < 0.45:
                continue
            # Union participants; keep higher-confidence / earlier clip shell
            parts = list(dict.fromkeys(
                list(k.get("participant_ids") or []) + list(c.get("participant_ids") or [])
            ))
            k["participant_ids"] = parts
            if c.get("action_type") == "pass" and len(parts) >= 1:
                k["student_id"] = parts[0]
            if float(c.get("confidence") or 0) > float(k.get("confidence") or 0) + 0.05:
                for key in ("clip_id", "start_ms", "end_ms", "release_ms", "phases", "confidence", "metadata"):
                    if c.get(key) is not None:
                        k[key] = c[key]
                k["student_id"] = c.get("student_id") or k.get("student_id")
            merged = True
            break
        if not merged:
            kept.append(dict(c))
    return kept


def build_session_output(
    session_id: str,
    anchor_camera: str | None = None,
    sample_stride: int = 1,
    group_id: str | None = None,
) -> SessionOutput:
    """
    Fuse pose2d (anchor cam) + ball_track + actions into SessionOutput.
    Skeleton uses single-camera pseudo3d until Pose2Sim is wired.
    """
    anchor = anchor_camera or get_action_segment_camera()
    shot_cam = get_shot_outcome_camera()
    pose_doc = _load_json(data_path("sessions", session_id, "perception", anchor, "pose2d.json")) or {}
    ball_doc = _load_json(data_path("sessions", session_id, "shot_outcomes", "ball_track.json")) or {}
    outcomes_doc = _load_json(data_path("sessions", session_id, "shot_outcomes", "outcomes.json")) or {}

    clips_raw = _collect_clips(session_id)
    ball_frames = ball_doc.get("frames", []) if isinstance(ball_doc, dict) else []

    clip_refs: list[ActionClipRef] = []
    for c in clips_raw:
        if c.get("start_ms") is None or c.get("end_ms") is None:
            continue
        parts = list(c.get("participant_ids") or [])
        if not parts and c.get("student_id"):
            parts = [c["student_id"]]
        meta = c.get("metadata") or {}
        clip_refs.append(ActionClipRef(
            clip_id=c["clip_id"],
            student_id=c["student_id"],
            action_type=_normalize_action_type(c["action_type"]),
            start_ms=float(c["start_ms"]),
            end_ms=float(c["end_ms"]),
            release_ms=float(c["release_ms"]) if c.get("release_ms") is not None else None,
            anchor_camera=c.get("anchor_camera"),
            zone_id=c.get("zone_id"),
            phases=c.get("phases") or [],
            participant_ids=parts,
            shooting_hand=meta.get("shooting_hand") if meta.get("shooting_hand") in ("left", "right") else None,
            metadata=c.get("metadata"),
        ))

    records: list[MotionRecord] = []
    frames = pose_doc.get("frames", []) if isinstance(pose_doc, dict) else []
    for i, fr in enumerate(frames):
        if sample_stride > 1 and (i % sample_stride) != 0:
            continue
        ts = float(fr.get("timestamp_ms", 0))
        persons = fr.get("persons") or []
        if not persons:
            continue
        # Prefer person with student_id; else first
        person = next((p for p in persons if p.get("student_id")), persons[0])
        sid = person.get("student_id") or "unknown"
        kpts = np.asarray(person.get("keypoints"), dtype=np.float32)
        if kpts.size == 0:
            continue
        if kpts.ndim == 1:
            continue
        atype, aphase, clip_id = _phase_at_ms(clips_raw, ts)
        ball_det = _ball_at_ms(ball_frames, ts)
        if ball_det and ball_det.get("center"):
            # image-plane xy as placeholder 3D (z=0) until triangulation
            cx, cy = ball_det["center"]
            ball = Ball3D(
                position=[float(cx), float(cy), 0.0],
                confidence=float(ball_det.get("confidence", 0.0)),
                status="tracked",
                source_camera=shot_cam,
            )
        else:
            ball = Ball3D(position=None, confidence=0.0, status="not_visible", source_camera=shot_cam)

        records.append(MotionRecord(
            student_id=sid,
            timestamp_ms=ts,
            action_type=_normalize_action_type(atype),
            action_phase=aphase if aphase in ("load", "set", "release", "follow_through", "full", "none") else "none",  # type: ignore[arg-type]
            clip_id=clip_id,
            skeleton_3d=_skeleton_from_kpts133(kpts),
            ball=ball,
            identity_confidence="high",
            pose_source="pseudo3d",
            metadata={
                "frame": fr.get("frame"),
                "anchor_camera": anchor,
                "track_id": person.get("track_id"),
            },
        ))

    outcomes = []
    if isinstance(outcomes_doc, dict):
        for o in outcomes_doc.get("outcomes", []):
            outcomes.append(o)

    return SessionOutput(
        session_id=session_id,
        generated_at=_iso_now(),
        coordinate_system=CoordinateSystem(
            space="pseudo3d",
            unit="normalized",
            origin="pelvis",
            axes="x_right, y_down_image_then_normalized, z_pseudo_height",
            note="v1 single-camera pseudo3d; ball xy in image pixels until multi-view fusion",
        ),
        records=records,
        clips=clip_refs,
        metadata={
            "group_id": group_id,
            "anchor_camera": anchor,
            "shot_camera": shot_cam,
            "record_count": len(records),
            "clip_count": len(clip_refs),
            "shot_outcomes": outcomes,
            "ball_stats": ball_doc.get("stats") if isinstance(ball_doc, dict) else {},
            "export_stage": "v1_testset",
        },
    )


def write_session_output(session_id: str, out_path: Path, **kwargs: Any) -> SessionOutput:
    doc = build_session_output(session_id, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return doc


def build_group_report(session_id: str, motion: SessionOutput, group_id: str) -> dict:
    outcomes = motion.metadata.get("shot_outcomes") or []
    return {
        "group_id": group_id,
        "session_id": session_id,
        "generated_at": motion.generated_at,
        "clip_count": len(motion.clips),
        "record_count": len(motion.records),
        "clips": [c.model_dump() for c in motion.clips],
        "shot_outcomes": outcomes,
        "shot_stats": {
            "attempts": len(outcomes),
            "makes": sum(1 for o in outcomes if o.get("made") is True),
            "misses": sum(1 for o in outcomes if o.get("made") is False),
            "undetermined": sum(1 for o in outcomes if o.get("made") is None),
        },
        "ball_stats": motion.metadata.get("ball_stats") or {},
    }
