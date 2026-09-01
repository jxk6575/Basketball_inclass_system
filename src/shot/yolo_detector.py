"""YOLO ball / hoop detectors — adapted from ref_code ball_detector / basket_detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_models_config, model_path

CLASS_BALL = 0
CLASS_HOOP = 1


def _yolo_cfg() -> dict[str, Any]:
    cfg = load_models_config().get("ball_hoop", {})
    return {
        "path": cfg.get("path", "models/detection/yolo_ball/Basketball_v1.pt"),
        "imgsz": int(cfg.get("imgsz", 640)),
        "ball_conf": float(cfg.get("ball_confidence", 0.3)),
        "hoop_conf": float(cfg.get("hoop_confidence", 0.5)),
        "half": bool(cfg.get("half", False)),
        # Basketball.pt uses inverted ids (0=hoop, 1=ball) vs Basketball_v1
        "swap_classes": bool(cfg.get("swap_ball_hoop_classes", False)),
        "class_ball": cfg.get("class_ball"),
        "class_hoop": cfg.get("class_hoop"),
    }


def _numeric_ball_hoop_ids(model_file: Path, cfg: dict[str, Any]) -> tuple[int, int]:
    """Return (ball_cls_id, hoop_cls_id) for models with numeric/empty class names."""
    if cfg.get("class_ball") is not None and cfg.get("class_hoop") is not None:
        return int(cfg["class_ball"]), int(cfg["class_hoop"])
    swap = bool(cfg.get("swap_classes"))
    # Plain Basketball.pt (not v1 / gym) is known-inverted
    stem = model_file.stem.lower()
    if stem == "basketball":
        swap = True
    if swap:
        return CLASS_HOOP, CLASS_BALL  # ball←1, hoop←0
    return CLASS_BALL, CLASS_HOOP


def _class_is_ball(cls_id: int, name: str | None, ball_id: int) -> bool:
    n = (name or "").strip().lower()
    if n in {"0", "1", ""} or n.isdigit():
        return int(cls_id) == ball_id
    if "hoop" in n or "rim" in n or n == "basket":
        return False
    return "ball" in n


def _class_is_hoop(cls_id: int, name: str | None, hoop_id: int) -> bool:
    n = (name or "").strip().lower()
    if n in {"0", "1", ""} or n.isdigit():
        return int(cls_id) == hoop_id
    return "hoop" in n or "rim" in n or n in {"basket", "backboard"}


def _resolve_device() -> str:
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class YoloBallHoopDetector:
    """Single YOLO model shared for basketball and hoop."""

    def __init__(
        self,
        model_file: str | Path | None = None,
        imgsz: int | None = None,
        ball_conf: float | None = None,
        hoop_conf: float | None = None,
        half: bool | None = None,
        device: str | None = None,
    ):
        cfg = _yolo_cfg()
        self.model_file = Path(model_file) if model_file else model_path(cfg["path"])
        self.imgsz = imgsz if imgsz is not None else cfg["imgsz"]
        self.ball_conf = ball_conf if ball_conf is not None else cfg["ball_conf"]
        self.hoop_conf = hoop_conf if hoop_conf is not None else cfg["hoop_conf"]
        self.half = half if half is not None else cfg["half"]
        self.device = device or _resolve_device()
        self._cfg = cfg
        self._model = None
        self._ball_id, self._hoop_id = _numeric_ball_hoop_ids(self.model_file, cfg)

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.model_file.exists():
            raise FileNotFoundError(f"Ball/hoop YOLO model not found: {self.model_file}")
        from ultralytics import YOLO
        self._model = YOLO(str(self.model_file))

    def detect(
        self,
        frame: np.ndarray,
        *,
        hoop_upper_half_only: bool = False,
    ) -> dict[str, list[dict]]:
        """
        Returns:
          {"ball": [...], "hoop": [...]}
          each item: {class_id, class_name, confidence, bbox(xywh), center, area}

        ``hoop_upper_half_only``: for side/baseline cams (cam_01–03), reject hoop
        detections whose center is in the lower half of the image.
        """
        self._ensure_model()
        if frame is None or frame.size == 0:
            return {"ball": [], "hoop": []}

        kwargs = dict(
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if self.half:
            kwargs["half"] = True
        results = self._model(frame, **kwargs)
        names = getattr(self._model, "names", None) or {}
        balls: list[dict] = []
        hoops: list[dict] = []
        frame_h = int(frame.shape[0])
        y_mid = 0.5 * frame_h
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = str(names.get(cls_id, cls_id))
                cy = y1 + h / 2
                det = {
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": (x1, y1, w, h),
                    "center": (int(x1 + w / 2), int(cy)),
                    "area": w * h,
                }
                if _class_is_ball(cls_id, cls_name, self._ball_id) and conf >= self.ball_conf:
                    det["class_name"] = "basketball"
                    balls.append(det)
                elif _class_is_hoop(cls_id, cls_name, self._hoop_id) and conf >= self.hoop_conf:
                    if hoop_upper_half_only and cy >= y_mid:
                        continue
                    det["class_name"] = "hoop"
                    hoops.append(det)

        # Prefer highest-confidence detections (ball list kept sorted by conf)
        if balls:
            balls = sorted(balls, key=lambda d: d["confidence"], reverse=True)
        if hoops:
            hoops = [max(hoops, key=lambda d: d["confidence"])]
        return {"ball": balls, "hoop": hoops}

    def detect_ball(self, frame: np.ndarray) -> list[dict]:
        return self.detect(frame)["ball"]

    def detect_hoop(
        self,
        frame: np.ndarray,
        *,
        hoop_upper_half_only: bool = False,
    ) -> list[dict]:
        return self.detect(frame, hoop_upper_half_only=hoop_upper_half_only)["hoop"]
