"""Frame-by-frame shot tracker — adapted from ref_code/shot_detector.py (no UI)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shot.geometry import (
    TrackPoint,
    clean_ball_pos,
    clean_hoop_pos,
    detect_down,
    detect_up,
    track_point_to_dict,
)
from src.shot.yolo_detector import YoloBallHoopDetector


@dataclass
class ShotEvent:
    frame: int
    made: bool
    confidence: float
    ball_trajectory: list[dict]
    hoop: dict | None
    metadata: dict[str, Any] = field(default_factory=dict)


class ShotTracker:
    """Detect ball/hoop, track trajectories, emit make/miss events."""

    def __init__(
        self,
        detector: YoloBallHoopDetector | None = None,
        calibrate_hoop_frames: int = 2,
        keep_shot_frames: bool = True,
        hoop_upper_half_only: bool = False,
        keep_all_balls: bool = True,
        max_balls: int = 8,
    ):
        self.detector = detector or YoloBallHoopDetector()
        self.calibrate_hoop_frames = max(1, int(calibrate_hoop_frames))
        self.keep_shot_frames = keep_shot_frames
        self.hoop_upper_half_only = bool(hoop_upper_half_only)
        self.keep_all_balls = bool(keep_all_balls)
        self.max_balls = max(1, int(max_balls))

        self.ball_pos: list[TrackPoint] = []
        self.hoop_pos: list[TrackPoint] = []
        self.pending_ball_pos: list[TrackPoint] = []
        self.frame_count = 0

        self.fixed_hoop: dict | None = None
        self._hoop_calib: list[dict] = []
        self._frame_shape: tuple[int, int] | None = None  # (h, w)

        self.shot_in_progress = False
        self.shot_ball_positions: list[TrackPoint] = []
        self.shot_frames: list[np.ndarray] = []
        self.events: list[ShotEvent] = []
        self.makes = 0
        self.attempts = 0

    def _plausible_hoop(self, det: dict, frame_shape: tuple[int, int] | None = None) -> bool:
        """Reject giant / edge-stuck false hoops before freeze."""
        bb = det.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) < 4:
            return False
        x, y, w, h = [float(v) for v in bb[:4]]
        if w < 12 or h < 12:
            return False
        shape = frame_shape or self._frame_shape
        if shape is not None:
            fh, fw = float(shape[0]), float(shape[1])
            # Reject boxes covering most of the frame or glued to (0,0)
            if w * h > 0.28 * fw * fh:
                return False
            if w > 0.55 * fw or h > 0.55 * fh:
                return False
            if x <= 2 and y <= 2 and (w > 0.25 * fw or h > 0.25 * fh):
                return False
            if self.hoop_upper_half_only and (y + 0.5 * h) >= 0.55 * fh:
                return False
        aspect = w / max(h, 1.0)
        if aspect > 4.0 or aspect < 0.25:
            return False
        return True

    def calibrate_hoop(self, frame: np.ndarray | None = None) -> bool:
        """Average first N plausible hoop detections, then freeze permanently."""
        if self.fixed_hoop is not None:
            return True
        if frame is not None and len(self._hoop_calib) < self.calibrate_hoop_frames:
            self._frame_shape = (int(frame.shape[0]), int(frame.shape[1]))
            dets = self.detector.detect(
                frame, hoop_upper_half_only=self.hoop_upper_half_only,
            ).get("hoop") or []
            for d in dets:
                if self._plausible_hoop(d):
                    self._hoop_calib.append(d)
                    break

        if len(self._hoop_calib) < self.calibrate_hoop_frames:
            return False

        samples = self._hoop_calib[: self.calibrate_hoop_frames]
        # Require samples to agree (avoid averaging a good + bad box)
        if len(samples) >= 2:
            c0 = samples[0]["center"]
            c1 = samples[1]["center"]
            drift = math.hypot(float(c0[0]) - float(c1[0]), float(c0[1]) - float(c1[1]))
            w0 = float(samples[0]["bbox"][2])
            if drift > max(80.0, 1.5 * w0):
                # Keep the higher-confidence sample and wait for another close one
                best = max(samples, key=lambda d: float(d.get("confidence") or 0))
                self._hoop_calib = [best]
                return False

        cx = int(round(sum(float(d["center"][0]) for d in samples) / len(samples)))
        cy = int(round(sum(float(d["center"][1]) for d in samples) / len(samples)))
        bw = int(round(sum(float(d["bbox"][2]) for d in samples) / len(samples)))
        bh = int(round(sum(float(d["bbox"][3]) for d in samples) / len(samples)))
        bw, bh = max(8, bw), max(8, bh)
        self.fixed_hoop = {
            "center": (cx, cy),
            "bbox": (cx - bw // 2, cy - bh // 2, bw, bh),
            "confidence": 1.0,
            "n_calib_frames": len(samples),
            "frozen": True,
        }
        return True

    def _should_skip_middle(self, point_b: TrackPoint, point_c: TrackPoint) -> bool:
        if not self.ball_pos:
            return False
        a = self.ball_pos[-1][0]
        b, c = point_b[0], point_c[0]
        dist_ac = math.hypot(c[0] - a[0], c[1] - a[1])
        dist_ab = math.hypot(b[0] - a[0], b[1] - a[1])
        dist_bc = math.hypot(c[0] - b[0], c[1] - b[1])
        avg = (self.ball_pos[-1][2] + self.ball_pos[-1][3]) / 2
        return dist_ac < avg and dist_ab > avg * 3 and dist_bc > avg * 3

    def _append_ball(self, det: dict, head_center: tuple[float, float] | None = None) -> None:
        center = det["center"]
        x1, y1, w, h = det["bbox"]
        if head_center is not None:
            dist = math.hypot(center[0] - head_center[0], center[1] - head_center[1])
            if dist < math.hypot(w, h) * 0.8:
                return
        new_pt: TrackPoint = (center, self.frame_count, w, h, float(det["confidence"]))
        self.pending_ball_pos.append(new_pt)
        if len(self.pending_ball_pos) >= 2:
            b, c = self.pending_ball_pos[0], self.pending_ball_pos[1]
            if self.ball_pos and self._should_skip_middle(b, c):
                self.pending_ball_pos.pop(0)
            else:
                self.ball_pos.append(b)
                self.pending_ball_pos.pop(0)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        head_center: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        self.frame_count = frame_idx
        if self.shot_in_progress and self.keep_shot_frames:
            self.shot_frames.append(frame.copy())

        dets = self.detector.detect(
            frame, hoop_upper_half_only=self.hoop_upper_half_only,
        )
        balls = sorted(
            dets["ball"] or [],
            key=lambda d: float(d.get("confidence", 0)),
            reverse=True,
        )
        # cam_01–03 often have multiple basketballs; keep all for export/pass.
        # Primary trajectory still follows the highest-confidence ball.
        if self.keep_all_balls:
            dets["ball"] = balls[: self.max_balls]
        else:
            dets["ball"] = balls[:1]
        if dets["ball"]:
            self._append_ball(dets["ball"][0], head_center=head_center)

        self._frame_shape = (int(frame.shape[0]), int(frame.shape[1]))

        # Once frozen, never unfreeze (avoids later false detections jumping the rim)
        if self.fixed_hoop is not None:
            center = self.fixed_hoop["center"]
            bb = self.fixed_hoop["bbox"]
            w, h = int(bb[2]), int(bb[3])
            self.hoop_pos.append((center, frame_idx, w, h, 1.0))
        else:
            # Calibrate from first N plausible detections, then freeze
            hoop_dets = [d for d in (dets.get("hoop") or []) if self._plausible_hoop(d)]
            if hoop_dets:
                d = hoop_dets[0]
                _x1, _y1, w, h = d["bbox"]
                self.hoop_pos.append((d["center"], frame_idx, w, h, float(d["confidence"])))
                self._hoop_calib.append(d)
                if len(self._hoop_calib) >= self.calibrate_hoop_frames:
                    self.calibrate_hoop()
            dets["hoop"] = hoop_dets

        self.ball_pos = clean_ball_pos(self.ball_pos, frame_idx)
        # Do not clean/jump a frozen hoop trajectory
        if self.fixed_hoop is None and len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)

        event = self.check_shot()
        return {
            "ball": dets["ball"],
            "hoop": dets["hoop"] or ([self.fixed_hoop] if self.fixed_hoop else []),
            "event": event,
        }

    def check_shot(self) -> ShotEvent | None:
        if not self.hoop_pos or not self.ball_pos:
            return None

        if not self.shot_in_progress and detect_up(self.ball_pos, self.hoop_pos):
            # Start attempt when ball is above hoop (size check is advisory only)
            self.shot_in_progress = True
            self.shot_ball_positions = []
            self.shot_frames = []

        if not self.shot_in_progress:
            return None

        # Flush latest ball into attempt trajectory (pending may lag one detection)
        if self.ball_pos:
            last = self.ball_pos[-1]
            if not self.shot_ball_positions or self.shot_ball_positions[-1][1] != last[1]:
                if last[1] >= self.frame_count - 4:
                    self.shot_ball_positions.append(last)

        if len(self.shot_ball_positions) < 2:
            return None

        if not detect_down(self.ball_pos, self.hoop_pos):
            return None

        # Require the ball to have been above hoop during this attempt
        hoop_cy = self.hoop_pos[-1][0][1]
        if not any(p[0][1] < hoop_cy for p in self.shot_ball_positions):
            self.shot_in_progress = False
            self.shot_ball_positions = []
            self.shot_frames = []
            return None

        self.shot_in_progress = False
        self.attempts += 1
        frames = self.shot_frames if self.keep_shot_frames else None

        # Combined scoring: rim occlusion veto + trajectory make/miss
        from src.shot.outcome import evaluate_make_miss

        seg = []
        for p in self.shot_ball_positions:
            (x, y), fr, w, h, conf = p
            seg.append({
                "center": [float(x), float(y)],
                "frame": int(fr),
                "bbox": [float(x - w / 2), float(y - h / 2), float(w), float(h)],
                "confidence": float(conf),
                "timestamp_ms": float(fr),
            })
        hx, hy = self.hoop_pos[-1][0]
        hw, hh = float(self.hoop_pos[-1][2]), float(self.hoop_pos[-1][3])
        made, conf, meta = evaluate_make_miss(
            seg, float(hx), float(hy), hw, hh, shot_frames=frames,
        )
        if made:
            self.makes += 1

        event = ShotEvent(
            frame=self.frame_count,
            made=made,
            confidence=conf,
            ball_trajectory=[track_point_to_dict(p) for p in self.shot_ball_positions],
            hoop=track_point_to_dict(self.hoop_pos[-1]) if self.hoop_pos else None,
            metadata={"attempt": self.attempts, "makes": self.makes, **meta},
        )
        self.events.append(event)
        self.shot_ball_positions = []
        self.shot_frames = []
        return event

    def trajectory_log(self) -> list[dict]:
        """Full ball track for action enhancement / export."""
        return [track_point_to_dict(p) for p in self.ball_pos]
