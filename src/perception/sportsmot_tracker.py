"""SportsMOT-style multi-object tracker (BoT-SORT / ByteTrack via Ultralytics).

SportsMOT is a sports MOT *benchmark*; there is no single official detector
checkpoint. BoT-SORT / ByteTrack are the standard trackers used on that
benchmark and ship with Ultralytics — we use them on top of a YOLO-Pose
detector for association quality comparison vs FaceBodyTracker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_models_config, model_path


def _device() -> str:
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class SportsMotStyleTracker:
    """
    Persistent Ultralytics tracker (botsort | bytetrack) over a YOLO-Pose model.

    Each ``update`` call returns detections with ``track_id`` assigned by
    the SportsMOT-style association algorithm.
    """

    def __init__(
        self,
        model_file: str | Path | None = None,
        tracker: str = "botsort.yaml",
        imgsz: int = 640,
        conf: float = 0.35,
        device: str | None = None,
    ):
        cfg = load_models_config().get("yolo_pose", {})
        default = cfg.get("path", "models/detection/yolo_pose/yolo11m-pose.pt")
        self.model_file = Path(model_file) if model_file else model_path(default)
        self.tracker = tracker
        self.imgsz = imgsz
        self.conf = conf
        self.device = device or _device()
        self._model = None
        self._persist = True

    def _ensure(self):
        if self._model is not None:
            return
        if not self.model_file.exists():
            raise FileNotFoundError(f"YOLO pose model not found: {self.model_file}")
        from ultralytics import YOLO
        self._model = YOLO(str(self.model_file))

    def reset(self) -> None:
        """Drop tracker state (new video / camera)."""
        self._model = None

    def update(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        """
        Returns list of {bbox, score, keypoints, track_id}.
        """
        self._ensure()
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        results = self._model.track(
            frame_bgr,
            persist=self._persist,
            tracker=self.tracker,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        out: list[dict[str, Any]] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes
            kps = r.keypoints.data if r.keypoints is not None else None
            ids = boxes.id
            n = len(boxes)
            for i in range(n):
                try:
                    conf = float(boxes.conf[i])
                    x1, y1, x2, y2 = map(float, boxes.xyxy[i])
                    tid = int(ids[i]) if ids is not None else -1
                    kpt = None
                    if kps is not None and i < len(kps):
                        kpt = np.asarray(kps[i].cpu().numpy(), dtype=np.float32)
                    out.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": conf,
                        "keypoints": kpt,
                        "track_id": tid,
                    })
                except Exception:
                    continue
        return out


def create_sportsmot_tracker(
    model_file: str | Path | None = None,
    tracker: str = "botsort.yaml",
) -> SportsMotStyleTracker:
    cfg = load_models_config().get("yolo_pose", {})
    imgsz = int(cfg.get("imgsz", 640))
    conf = float(cfg.get("confidence", 0.45))
    return SportsMotStyleTracker(
        model_file=model_file,
        tracker=tracker,
        imgsz=imgsz,
        conf=max(0.25, conf - 0.1),
    )
