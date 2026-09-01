"""OSNet body ReID embedder via torchreid."""

from __future__ import annotations

import cv2
import numpy as np

from src.identity.embedders import BodyEmbedder, cosine_sim


class OSNetBodyEmbedder(BodyEmbedder):
    dim = 512

    def __init__(self, weight_path: str, device: str = "cuda"):
        import torch
        import torchreid

        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        state = torch.load(weight_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        # Drop classifier head — we only need the 512-d embedding
        state = {k: v for k, v in state.items() if not k.startswith("classifier")}
        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1,
            pretrained=False,
        )
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)
        self._torch = torch

    def _preprocess(self, image_bgr: np.ndarray, bbox: list[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image_bgr
        crop = cv2.resize(crop, (128, 256))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crop = (crop - mean) / std
        return crop.transpose(2, 0, 1)

    def embed(self, image_bgr: np.ndarray, bbox: list[float]) -> np.ndarray:
        arr = self._preprocess(image_bgr, bbox)
        t = self._torch.from_numpy(arr).unsqueeze(0).float().to(self.device)
        with self._torch.no_grad():
            feat = self.model(t)
        feat = feat.cpu().numpy().reshape(-1).astype(np.float32)
        feat /= np.linalg.norm(feat) + 1e-8
        return feat


__all__ = ["OSNetBodyEmbedder", "cosine_sim"]
