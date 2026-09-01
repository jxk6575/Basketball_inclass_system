"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
MODELS = ROOT / "models"


def load_models_config() -> dict[str, Any]:
    path = CONFIGS / "models.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_path(relative: str) -> Path:
    return ROOT / relative


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIGS / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_path(*parts: str) -> Path:
    p = DATA.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
