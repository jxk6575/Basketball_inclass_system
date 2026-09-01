"""Canonical session output schema — student motion + ball in world 3D."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Coordinate & skeleton
# ---------------------------------------------------------------------------

JointFormat = Literal["h36m_17", "coco_133"]
ActionType = Literal[
    "free_throw",
    "jump_shot",
    "layup",
    "triple_threat",
    "pass",
    "unknown",
]
ActionPhase = Literal["load", "set", "release", "follow_through", "full", "none"]
BallStatus = Literal["tracked", "predicted", "not_visible", "not_available"]


class CoordinateSystem(BaseModel):
    """World / court coordinate convention for all 3D points in this file."""

    space: Literal["court_world", "camera_normalized", "pseudo3d"] = "court_world"
    unit: Literal["meter", "normalized"] = "meter"
    origin: str = "near_baseline_midpoint"
    axes: str = "x_right, y_toward_center_line, z_up"
    note: str = "Matches configs/calibration/court_landmarks_fiba.yaml"


class Skeleton3D(BaseModel):
    """Single-frame 3D skeleton."""

    format: JointFormat = "h36m_17"
    joints: list[list[float]] = Field(
        ...,
        description="(J, 3) world coordinates; order matches joint_names or format spec",
    )
    joint_scores: list[float] | None = Field(
        default=None,
        description="Per-joint confidence in [0, 1]",
    )
    joint_names: list[str] | None = None


class Ball3D(BaseModel):
    """Basketball position; cam_04 / ball detector when available."""

    position: list[float] | None = Field(
        default=None,
        description="[x, y, z] in same coordinate_system as skeleton",
    )
    confidence: float = 0.0
    status: BallStatus = "not_available"
    source_camera: str | None = None


# ---------------------------------------------------------------------------
# Atomic record — one student at one timestamp
# ---------------------------------------------------------------------------

class MotionRecord(BaseModel):
    """
    One fused sample: who, when, what action context, body + ball in 3D.
    This is the primary unit downstream consumers should read.
    """

    student_id: str = Field(..., description="Student UUID from enrollment")
    timestamp_ms: float = Field(..., description="Session timeline in milliseconds")

    action_type: ActionType = "unknown"
    action_phase: ActionPhase = "none"
    clip_id: str | None = Field(
        default=None,
        description="Links to action clip, e.g. {student_id}:{clip_index}",
    )

    skeleton_3d: Skeleton3D
    ball: Ball3D = Field(default_factory=Ball3D)

    identity_confidence: Literal["high", "medium", "low"] = "high"
    pose_source: Literal["triangulated", "single_camera", "pseudo3d", "stub"] = "stub"

    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action clip summary (optional, for phase boundaries)
# ---------------------------------------------------------------------------

class ActionClipRef(BaseModel):
    """Lightweight clip index; detailed samples live in records[]."""

    clip_id: str
    student_id: str
    action_type: ActionType
    start_ms: float
    end_ms: float
    release_ms: float | None = None
    anchor_camera: str | None = None
    zone_id: str | None = None
    phases: list[dict[str, Any]] = Field(default_factory=list)
    # Pass: [passer_id, receiver_id]; others usually [student_id]
    participant_ids: list[str] = Field(default_factory=list)
    shooting_hand: Literal["left", "right"] | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Session-level envelope
# ---------------------------------------------------------------------------

class SessionOutput(BaseModel):
    """
    Top-level JSON written per session after pose3d + action (+ shot) stages.

    Suggested path: data/sessions/{session_id}/export/motion.jsonl
    or motion.json for bundled export.
    """

    session_id: str
    generated_at: str = Field(..., description="ISO-8601 UTC")
    coordinate_system: CoordinateSystem = Field(default_factory=CoordinateSystem)

    records: list[MotionRecord] = Field(default_factory=list)
    clips: list[ActionClipRef] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)
