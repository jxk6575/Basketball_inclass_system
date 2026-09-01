"""Shared types and constants."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    CREATED = "created"
    CONSENT_OK = "consent_ok"
    ENROLLED = "enrolled"
    RECORDED = "recorded"
    PERCEPTION_DONE = "perception_done"
    SYNC_DONE = "sync_done"
    POSE3D_DONE = "pose3d_done"
    ACTION_DONE = "action_done"
    SHOT_OUTCOME_DONE = "shot_outcome_done"
    SCORED = "scored"
    REPORT_READY = "report_ready"
    FAILED = "failed"


class CameraRole(str, Enum):
    THREE_POINT_ZONE = "three_point_zone"
    ENROLLMENT = "enrollment"
    FACE_PRIMARY = "face_primary"
    ACTION_SEGMENTATION = "action_segmentation"
    RIM_VIEW = "rim_view"
    SHOT_OUTCOME = "shot_outcome"
    BALL_TRACKING = "ball_tracking"


class ConsentScope(str, Enum):
    VIDEO = "video"
    FACE = "face"
    REPORT = "report"


class DetectionRecord(BaseModel):
    frame: int
    bbox: list[float]
    track_id: int | None = None
    student_id: str | None = None
    face_sim: float | None = None
    body_sim: float | None = None
    alpha: float = 0.0
    identity_confidence: str = "high"


class ActionPhase(BaseModel):
    name: str
    start: int
    end: int
    start_ms: float | None = None
    end_ms: float | None = None
    anchor_camera: str | None = None


class ActionClip(BaseModel):
    action_type: str
    start_frame: int
    end_frame: int
    phases: list[ActionPhase] = Field(default_factory=list)
    confidence: float = 0.0
    zone_id: str | None = None
    anchor_camera: str | None = None
    start_ms: float | None = None
    end_ms: float | None = None
    # Primary actor; for pass, passer. participant_ids: pass → [passer, receiver]
    student_id: str | None = None
    participant_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudentActions(BaseModel):
    student_id: str
    clips: list[ActionClip] = Field(default_factory=list)


class PhaseScore(BaseModel):
    phase: str
    metric: str
    value: float
    score: float
    weight: float
    feedback: str = ""


class StudentReport(BaseModel):
    student_id: str
    session_id: str
    action_type: str
    total_score: float
    phase_scores: list[PhaseScore] = Field(default_factory=list)
    summary: str = ""
    identity_confidence: str = "high"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ShotOutcome(BaseModel):
    student_id: str | None = None
    action_type: str = "unknown"
    made: bool | None = None
    confidence: float = 0.0
    anchor_timestamp_ms: float | None = None
    clip_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ShotOutcomeRecord(BaseModel):
    session_id: str
    camera_id: str
    status: str = "stub"
    outcomes: list[ShotOutcome] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
