"""YOLO-Pose person detector — bbox + COCO-17 keypoints for human validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import load_models_config, model_path

# COCO-17 indices used to validate a real human (head + torso + limbs)
_CORE_KPT_INDICES = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

_yolo_pose_backend = None


def _yolo_pose_cfg() -> dict:
    cfg = load_models_config().get("yolo_pose", {})
    return {
        "path": cfg.get("path", "models/detection/yolo_pose/yolo11m-pose.pt"),
        "imgsz": int(cfg.get("imgsz", 640)),
        "conf": float(cfg.get("confidence", 0.55)),
        "kpt_conf": float(cfg.get("keypoint_confidence", 0.5)),
        "min_core_keypoints": int(cfg.get("min_core_keypoints", 5)),
        "min_valid_keypoints": int(cfg.get("min_valid_keypoints", 8)),
        "min_aspect_ratio": float(cfg.get("min_aspect_ratio", 1.0)),
        "half": bool(cfg.get("half", True)),
    }


def _resolve_device() -> str:
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class YoloPosePersonDetector:
    """Ultralytics YOLO pose model — person bbox gated by keypoint structure."""

    def __init__(
        self,
        model_file: str | Path | None = None,
        imgsz: int | None = None,
        conf: float | None = None,
        kpt_conf: float | None = None,
        half: bool | None = None,
        device: str | None = None,
    ):
        cfg = _yolo_pose_cfg()
        self.model_file = Path(model_file) if model_file else model_path(cfg["path"])
        self.imgsz = imgsz if imgsz is not None else cfg["imgsz"]
        self.conf = conf if conf is not None else cfg["conf"]
        self.kpt_conf = kpt_conf if kpt_conf is not None else cfg["kpt_conf"]
        self.min_core_kpts = cfg["min_core_keypoints"]
        self.min_valid_kpts = cfg["min_valid_keypoints"]
        self.min_aspect = cfg["min_aspect_ratio"]
        self.half = half if half is not None else cfg["half"]
        self.device = device or _resolve_device()
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.model_file.exists():
            raise FileNotFoundError(f"YOLO pose model not found: {self.model_file}")
        from ultralytics import YOLO
        self._model = YOLO(str(self.model_file))

    @staticmethod
    def _count_valid_kpts(kpts: np.ndarray, kpt_thr: float, indices: tuple[int, ...] | None = None) -> int:
        if kpts is None or kpts.size == 0:
            return 0
        arr = np.asarray(kpts, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        idxs = range(len(arr)) if indices is None else indices
        n = 0
        for i in idxs:
            if i >= len(arr):
                continue
            if arr.shape[1] >= 3 and float(arr[i, 2]) >= kpt_thr:
                n += 1
            elif arr.shape[1] == 2:
                n += 1
        return n

    def _is_valid_person(
        self,
        bbox_xyxy: list[float],
        kpts: np.ndarray | None,
        box_conf: float,
        score_thr: float,
    ) -> bool:
        if box_conf < score_thr:
            return False
        x1, y1, x2, y2 = bbox_xyxy
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if w <= 1 or h <= 1:
            return False
        if h / w < self.min_aspect:
            return False
        if kpts is None:
            return False
        total = self._count_valid_kpts(kpts, self.kpt_conf)
        core = self._count_valid_kpts(kpts, self.kpt_conf, _CORE_KPT_INDICES)
        return core >= self.min_core_kpts and total >= self.min_valid_kpts

    def detect_persons(
        self,
        frame_bgr: np.ndarray,
        score_thr: float | None = None,
        best_only: bool = False,
    ) -> list[dict]:
        """
        Returns list of {bbox: [x1,y1,x2,y2], score, keypoints: (17,3)}.
        Only detections with plausible human keypoints are kept.
        """
        self._ensure_model()
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        thr = float(score_thr if score_thr is not None else self.conf)
        kwargs = dict(
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            conf=max(0.25, thr - 0.15),
        )
        if self.half:
            kwargs["half"] = True

        results = self._model(frame_bgr, **kwargs)
        candidates: list[dict] = []

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue
            boxes = r.boxes
            keypoints = r.keypoints.data
            n = min(len(boxes), len(keypoints))
            for i in range(n):
                try:
                    conf = float(boxes.conf[i])
                    x1, y1, x2, y2 = map(float, boxes.xyxy[i])
                    kpts = np.asarray(keypoints[i].cpu().numpy(), dtype=np.float32)
                    if not self._is_valid_person([x1, y1, x2, y2], kpts, conf, thr):
                        continue
                    candidates.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": conf,
                        "keypoints": kpts,
                    })
                except Exception:
                    continue

        if not candidates:
            return []
        if best_only:
            candidates = [max(candidates, key=lambda d: d["score"])]
        candidates.sort(key=lambda d: d["score"], reverse=True)
        return candidates


def create_yolo_pose_detector(
    model_file: str | Path | None = None,
    *,
    force_reload: bool = False,
) -> YoloPosePersonDetector | None:
    """Create / cache YOLO-Pose detector. Pass ``model_file`` to override config."""
    global _yolo_pose_backend
    if force_reload:
        _yolo_pose_backend = None
    if model_file is not None:
        try:
            return YoloPosePersonDetector(model_file=model_file)
        except Exception:
            return None
    if _yolo_pose_backend is not None:
        return _yolo_pose_backend if _yolo_pose_backend is not False else None
    try:
        cfg = _yolo_pose_cfg()
        path = model_path(cfg["path"])
        if not path.exists():
            _yolo_pose_backend = False
            return None
        _yolo_pose_backend = YoloPosePersonDetector(model_file=path)
        return _yolo_pose_backend
    except Exception:
        _yolo_pose_backend = False
        return None
