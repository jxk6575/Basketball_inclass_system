"""RTMLib-based person detection and wholebody 133 pose."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class RTMLibPerception:
    def __init__(
        self,
        det_model: str | None = None,
        pose_model: str | None = None,
        device: str = "cuda",
        backend: str = "onnxruntime",
        mode: str = "performance",
    ):
        from rtmlib import RTMPose, Wholebody
        from rtmlib.tools.object_detection.yolox import YOLOX

        self.device = device
        self.backend = backend

        if det_model and Path(det_model).exists():
            self.detector = YOLOX(
                onnx_model=det_model,
                model_input_size=(640, 640),
                backend=backend,
                device=device,
            )
        else:
            self.detector = None

        if pose_model and Path(pose_model).exists():
            self.pose = RTMPose(
                onnx_model=pose_model,
                model_input_size=(288, 384),  # rtmlib: (width, height) for 384x288 model
                backend=backend,
                device=device,
            )
            self._use_custom_pose = True
        else:
            self.wholebody = Wholebody(
                mode=mode,
                backend=backend,
                device=device,
                to_openpose=False,
            )
            self._use_custom_pose = False

    @staticmethod
    def _parse_det_output(det_out) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Normalize YOLOX outputs: tuple, (N,4) bbox, or (N,5) bbox+score."""
        if det_out is None:
            return None, None
        if isinstance(det_out, tuple) and len(det_out) == 2:
            return det_out[0], det_out[1]
        arr = np.asarray(det_out)
        if arr.size == 0:
            return None, None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] >= 5:
            return arr[:, :4], arr[:, 4]
        if arr.shape[1] == 4:
            return arr, np.ones(arr.shape[0], dtype=np.float32)
        return None, None

    def detect_persons(self, frame_bgr: np.ndarray, score_thr: float = 0.4) -> list[dict]:
        if self.detector is None:
            from src.identity.perception import _stub_person_detections
            return [{"bbox": b, "score": 1.0} for b in _stub_person_detections(frame_bgr)]

        rgb = frame_bgr[:, :, ::-1].copy()
        det_out = self.detector(rgb)
        bboxes, scores = self._parse_det_output(det_out)
        if bboxes is None or getattr(bboxes, "size", len(bboxes)) == 0:
            return []
        out = []
        for bbox, score in zip(bboxes, scores):
            if float(score) < score_thr:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            out.append({"bbox": [x1, y1, x2, y2], "score": float(score)})
        return out

    def estimate_pose133(self, frame_bgr: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, np.ndarray]:
        rgb = frame_bgr[:, :, ::-1].copy()
        if self._use_custom_pose:
            kpts, scores = self.pose(rgb, bboxes=[bbox])
            if len(kpts) == 0:
                kpts = np.zeros((1, 133, 2))
                scores = np.zeros((1, 133))
            k3 = np.zeros((kpts.shape[1], 3), dtype=np.float32)
            k3[:, :2] = kpts[0]
            k3[:, 2] = scores[0]
            return k3, scores[0]

        kpts, scores = self.wholebody(rgb)
        if len(kpts) == 0:
            return np.zeros((133, 3), dtype=np.float32), np.zeros(133, dtype=np.float32)
        # pick closest to bbox center
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        best_i = 0
        if len(kpts) > 1:
            dists = []
            for i in range(len(kpts)):
                bx = kpts[i][:, 0].mean()
                by = kpts[i][:, 1].mean()
                dists.append((bx - cx) ** 2 + (by - cy) ** 2)
            best_i = int(np.argmin(dists))
        k3 = np.zeros((kpts.shape[1], 3), dtype=np.float32)
        k3[:, :2] = kpts[best_i]
        k3[:, 2] = scores[best_i]
        return k3, scores[best_i]


def create_rtmlib_perception() -> RTMLibPerception | None:
    try:
        from src.identity.backends import get_runtime_config
        rt = get_runtime_config()
        if not rt.get("models_ready"):
            return None
        device = "cuda" if rt["use_cuda"] else "cpu"
        return RTMLibPerception(
            det_model=rt["detector_path"],
            pose_model=rt["pose_path"],
            device=device,
            mode=rt.get("pose_mode", "performance"),
        )
    except Exception:
        return None
