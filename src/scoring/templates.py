"""Load action scoring templates from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.config import CONFIGS


def load_action_template(action_type: str) -> dict:
    mapping = {
        "free_throw": "free_throw.yaml",
        "jump_shot": "jump_shot.yaml",
        "layup": "layup.yaml",
        "triple_threat": "triple_threat.yaml",
    }
    name = mapping.get(action_type, f"{action_type}.yaml")
    path = CONFIGS / "actions" / name
    if not path.exists():
        raise FileNotFoundError(f"Action template not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def score_metric(value: float, min_v: float, max_v: float) -> tuple[float, str]:
    if value != value:  # nan
        return 0.0, "数据缺失，无法评估该指标"
    if min_v <= value <= max_v:
        return 100.0, "符合标准"
    if value < min_v:
        dist = (min_v - value) / max(abs(min_v), 1e-6)
        return max(0.0, 100.0 - dist * 80), "低于标准下限"
    dist = (value - max_v) / max(abs(max_v), 1e-6)
    return max(0.0, 100.0 - dist * 80), "高于标准上限"
