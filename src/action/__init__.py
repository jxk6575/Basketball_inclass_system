"""Action detection: auto-classify pipeline + detectors."""

from src.action.pipeline import detect_actions_auto, run_action_session_auto
from src.action.registry import (
    SHOOTING_ACTION_TYPES,
    is_shooting_action,
    uses_rim_aux,
)

__all__ = [
    "detect_actions_auto",
    "run_action_session_auto",
    "SHOOTING_ACTION_TYPES",
    "is_shooting_action",
    "uses_rim_aux",
]
