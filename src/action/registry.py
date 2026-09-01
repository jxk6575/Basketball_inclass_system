"""Action-type registry — allowed labels only.

Canonical set (v2 production):
  pass | triple_threat | free_throw | layup | jump_shot

- ``triple_threat`` covers 三威胁 + 突破 (no separate dribble label);
  requires crouch evidence (person bbox aspect h/w drop / lowered CoG)
- shooting family: free_throw | jump_shot | layup
  (jump_shot ≈ free_throw form + vertical jump; layup = run-up toward hoop)
"""

from __future__ import annotations

# Labels the system may emit after auto-classification
KNOWN_ACTION_TYPES = (
    "free_throw",
    "jump_shot",
    "layup",
    "triple_threat",
    "pass",
    "unknown",
)

# Shooting family: release peak + required cam_04 ball-above-hoop gate / make-miss
SHOOTING_ACTION_TYPES = frozenset({"free_throw", "jump_shot", "layup"})

# cam_04 ball/hoop is **required** to gate shooting events (ball above hoop)
# and for make/miss. It must NEVER be used to decide free_throw vs jump_shot vs layup.
USES_RIM_CAMERA_AUX = SHOOTING_ACTION_TYPES

# Pose-only family: pass needs cam_01–03 ball; triple_threat is pose-first (ball aux)
POSE_ONLY_ACTION_TYPES = frozenset({"triple_threat", "pass"})


def is_shooting_action(action_type: str | None) -> bool:
    return (action_type or "") in SHOOTING_ACTION_TYPES


def uses_rim_aux(action_type: str | None) -> bool:
    return (action_type or "") in USES_RIM_CAMERA_AUX


def normalize_action_type(action_type: str | None) -> str:
    """Map legacy / internal labels onto the canonical five (+ unknown)."""
    t = (action_type or "").strip()
    if t in {"free_throw", "jump_shot", "layup", "triple_threat", "pass"}:
        return t
    if t in {"dribble", "drive", "breakthrough"}:
        return "triple_threat"
    if t in {"shot", "jumper"}:
        return "jump_shot"
    return "unknown"
