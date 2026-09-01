"""Enrollment gallery management."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import data_path
from src.identity.embedders import BodyEmbedder, FaceEmbedder, create_body_embedder, create_face_embedder


class EnrollmentGallery:
    def __init__(
        self,
        session_id: str,
        face_embedder: FaceEmbedder | None = None,
        body_embedder: BodyEmbedder | None = None,
    ):
        self.session_id = session_id
        self.root = data_path("enrollment", session_id)
        self.face_embedder = face_embedder or create_face_embedder()
        self.body_embedder = body_embedder or create_body_embedder()
        self._cache: dict[str, dict] = {}

    def student_dir(self, student_id: str) -> Path:
        d = self.root / student_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add_face_sample(self, student_id: str, image_bgr, face_bbox: list[float]) -> bool:
        emb = self.face_embedder.embed(image_bgr, face_bbox)
        if emb is None:
            return False
        d = self.student_dir(student_id)
        idx = len(list(d.glob("face_*.npy")))
        np.save(d / f"face_{idx:03d}.npy", emb)
        self._cache.pop(student_id, None)
        return True

    def add_body_sample(self, student_id: str, image_bgr, body_bbox: list[float]) -> bool:
        emb = self.body_embedder.embed(image_bgr, body_bbox)
        d = self.student_dir(student_id)
        idx = len(list(d.glob("body_*.npy")))
        np.save(d / f"body_{idx:03d}.npy", emb)
        self._cache.pop(student_id, None)
        return True

    def add_clothing_color_sample(
        self,
        student_id: str,
        image_bgr,
        body_bbox: list[float],
        keypoints=None,
    ) -> bool:
        from src.identity.clothing_color import extract_clothing_color

        desc = extract_clothing_color(image_bgr, body_bbox, keypoints=keypoints)
        d = self.student_dir(student_id)
        idx = len(list(d.glob("color_*.npy")))
        np.save(d / f"color_{idx:03d}.npy", desc)
        self._cache.pop(student_id, None)
        return True

    def load_student(self, student_id: str) -> dict:
        if student_id in self._cache:
            return self._cache[student_id]
        d = self.root / student_id
        faces = [np.load(p) for p in sorted(d.glob("face_*.npy"))]
        bodies = [np.load(p) for p in sorted(d.glob("body_*.npy"))]
        colors = [np.load(p) for p in sorted(d.glob("color_*.npy"))]
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        data = {"face": faces, "body": bodies, "color": colors, "meta": meta}
        self._cache[student_id] = data
        return data

    def list_students(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def save_meta(self, student_id: str, display_name: str, extra: dict | None = None) -> None:
        d = self.student_dir(student_id)
        meta = {"display_name": display_name, **(extra or {})}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
