"""Stable student identity colors locked to ``student_id`` (presentation).

Same ``student_id`` → same color within/across cams. Overlay / dashboard labels
show ``stu_XX`` only (no A/B/C presentation letters).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# Distinct BGR palette (OpenCV). Index locks to stu_XX numeric suffix when possible.
STUDENT_PALETTE_BGR: list[tuple[int, int, int]] = [
    (40, 140, 255),   # stu_00 — amber/orange
    (80, 200, 90),    # stu_01 — green
    (255, 170, 60),   # stu_02 — sky blue
    (220, 90, 200),   # stu_03 — magenta
    (60, 220, 220),   # stu_04 — yellow
    (180, 120, 40),   # stu_05 — steel blue
    (100, 100, 255),  # stu_06 — coral
    (200, 200, 80),   # stu_07 — cyan-ish
]

_UNLABELED_BGR = (160, 160, 160)
_STU_NUM_RE = re.compile(r"stu_(\d+)$", re.IGNORECASE)


def _clip_time_ms(clip: dict[str, Any]) -> float | None:
    if clip.get("release_ms") is not None:
        return float(clip["release_ms"])
    meta = clip.get("metadata") or {}
    mc = meta.get("multicam") or {}
    for key in ("rim_timestamp_ms", "release_ms", "release_common_ms", "anchor_ms"):
        if mc.get(key) is not None:
            return float(mc[key])
        if meta.get(key) is not None:
            return float(meta[key])
    if clip.get("start_ms") is not None:
        return float(clip["start_ms"])
    return None


def compute_chrono_display_order(
    clips: Iterable[dict[str, Any]],
    enrolled_ids: list[str] | None = None,
) -> list[str]:
    """Order students by first action time for dashboard legend / tables.

    Students with no clips are appended in enrolled / lexical order.
    """
    first_t: dict[str, float] = {}
    for c in clips:
        sid = c.get("student_id")
        if not sid:
            continue
        t = _clip_time_ms(c)
        if t is None:
            continue
        sid = str(sid)
        if sid not in first_t or t < first_t[sid]:
            first_t[sid] = t

    ordered = sorted(first_t.keys(), key=lambda s: (first_t[s], s))
    seen = set(ordered)
    extras: list[str] = []
    for sid in enrolled_ids or []:
        s = str(sid)
        if s not in seen:
            extras.append(s)
            seen.add(s)
    extras.sort()
    return ordered + extras


# Back-compat alias (letters removed; callers should use compute_chrono_display_order)
def compute_chrono_letter_map(
    clips: Iterable[dict[str, Any]],
    enrolled_ids: list[str] | None = None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    display_order = compute_chrono_display_order(clips, enrolled_ids=enrolled_ids)
    return display_order, {}, {}


def student_color_index(student_id: str | None) -> int | None:
    """Palette index locked to student_id (stu_XX → X; else stable hash)."""
    if not student_id:
        return None
    m = _STU_NUM_RE.match(str(student_id).strip())
    if m:
        return int(m.group(1)) % len(STUDENT_PALETTE_BGR)
    h = 0
    for ch in str(student_id):
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return int(h % len(STUDENT_PALETTE_BGR))


def student_color_bgr(
    student_id: str | None,
    *,
    sid_to_letter: dict[str, str] | None = None,  # unused; kept for call-site compat
    track_id: int | None = None,
) -> tuple[int, int, int]:
    """BGR color locked to student_id; unlabeled falls back to track_id tint."""
    del sid_to_letter  # presentation letters removed
    idx = student_color_index(student_id)
    if idx is not None:
        return STUDENT_PALETTE_BGR[idx % len(STUDENT_PALETTE_BGR)]
    if track_id is not None:
        # Deterministic gray-blue tint for unlabeled tracks (not shared with students)
        rng = (int(track_id) * 9973) % 120
        return (140 + rng // 3, 140 + rng // 2, 150 + rng // 4)
    return _UNLABELED_BGR


def student_color_hex(
    student_id: str | None,
    *,
    sid_to_letter: dict[str, str] | None = None,
) -> str:
    b, g, r = student_color_bgr(student_id, sid_to_letter=sid_to_letter)
    return f"#{r:02x}{g:02x}{b:02x}"


def format_student_label(
    student_id: str | None,
    sid_to_letter: dict[str, str] | None = None,
    *,
    include_sid: bool = True,
) -> str:
    """Display label: ``stu_XX`` only (letters removed)."""
    del sid_to_letter, include_sid
    if not student_id:
        return "?"
    return str(student_id)


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-8)


class VizIdentitySticky:
    """Frame-to-frame IoU sticky for viz display IDs (does not rewrite perception).

    When ReID briefly flips student_id on a spatially continuous box, keep the
    previous display_id so box color / label stay locked to one person.
    High-confidence new IDs still override.
    """

    def __init__(self, iou_thr: float = 0.45, sticky_frames: int = 12):
        self.iou_thr = iou_thr
        self.sticky_frames = sticky_frames
        self._prev: list[dict] = []  # {bbox, display_id, ttl}

    def resolve(self, recs: list[dict]) -> list[dict]:
        """Return shallow copies of recs with ``display_student_id`` set."""
        used_prev: set[int] = set()
        out: list[dict] = []
        new_prev: list[dict] = []

        for rec in recs:
            bbox = rec.get("bbox")
            sid = rec.get("student_id")
            conf = str(rec.get("identity_confidence") or "")
            display = sid
            matched_i = None
            best_iou = 0.0
            if bbox is not None:
                for i, prev in enumerate(self._prev):
                    if i in used_prev:
                        continue
                    iou = bbox_iou(bbox, prev["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        matched_i = i

            if matched_i is not None and best_iou >= self.iou_thr:
                used_prev.add(matched_i)
                prev = self._prev[matched_i]
                prev_sid = prev.get("display_id")
                # Keep sticky unless unlabeled→labeled or high-conf explicit change
                if prev_sid and sid and prev_sid != sid:
                    if conf in ("high",) and best_iou < 0.75:
                        display = sid
                    elif conf in ("high", "medium") and prev.get("ttl", 0) <= 0:
                        display = sid
                    else:
                        display = prev_sid
                elif prev_sid and not sid:
                    display = prev_sid
                elif sid:
                    display = sid
                else:
                    display = prev_sid
                ttl = self.sticky_frames if display == sid else max(int(prev.get("ttl", 0)) - 1, 0)
            else:
                ttl = self.sticky_frames if display else 0

            row = dict(rec)
            row["display_student_id"] = display
            out.append(row)
            if bbox is not None:
                new_prev.append({"bbox": list(bbox), "display_id": display, "ttl": ttl})

        self._prev = new_prev
        return out
