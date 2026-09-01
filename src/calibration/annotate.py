"""Interactive + semi-auto court landmark annotation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.calibration.auto_lines import (
    detect_court_lines,
    intersection_candidates,
    project_and_snap,
    snap_to_candidates,
)
from src.calibration.court_model import (
    annotation_order_for_camera,
    landmark_name,
    landmark_xyz,
    load_court_model,
)
from src.calibration.solve import default_intrinsics, solve_camera_pnp
from src.calibration.text_cn import draw_hud_banner, paste_chip


def empty_annotation_doc(camera_ids: list[str]) -> dict[str, Any]:
    return {
        "version": 2,
        "frames": {},
        "image_size": {},
        "points": {c: {} for c in camera_ids},
        "notes": (
            "按各相机 annotation_order_by_camera 标注；"
            "左键精确落点，Shift+左键吸附线交点；"
            "cam_01/02 点≥6 可估内参+畸变。"
        ),
    }


def _map_mouse_to_image(
    x: int, y: int, win: str, img_w: int, img_h: int,
) -> tuple[int, int]:
    """
    Map HighGUI mouse coords → image pixels when WINDOW_NORMAL is resized.
    Without this, clicks and drawn markers diverge (common after window resize).
    """
    try:
        _wx, _wy, ww, wh = cv2.getWindowImageRect(win)
    except cv2.error:
        return int(x), int(y)
    if ww <= 1 or wh <= 1:
        return int(x), int(y)
    if abs(ww - img_w) <= 2 and abs(wh - img_h) <= 2:
        return max(0, min(img_w - 1, int(x))), max(0, min(img_h - 1, int(y)))
    # Some backends already report image-space coords (can exceed window size)
    if x >= ww or y >= wh:
        return max(0, min(img_w - 1, int(x))), max(0, min(img_h - 1, int(y)))
    u = int(round(x * img_w / float(ww)))
    v = int(round(y * img_h / float(wh)))
    return max(0, min(img_w - 1, u)), max(0, min(img_h - 1, v))


def load_annotations(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_annotations(doc: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def auto_fill_from_seeds(
    image: np.ndarray,
    seeds: dict[str, list[float]],
    model: dict[str, Any] | None = None,
    intrinsics: dict[str, Any] | None = None,
    snap_px: float = 45.0,
    target_ids: list[str] | None = None,
) -> dict[str, list[float]]:
    """
    Use ≥4 manual seeds → PnP → project remaining landmarks → snap to line intersections.

    If target_ids is set, only fill those (recommended per-camera list).
    """
    model = model or load_court_model()
    h, w = image.shape[:2]
    intr = intrinsics or default_intrinsics(w, h)
    if len(seeds) < 4:
        return dict(seeds)

    solved = solve_camera_pnp(
        model, seeds, intr, min_points=4, image_height=h, enforce_above_ground=True,
    )
    if solved.get("status") != "ok":
        return dict(seeds)

    segs = detect_court_lines(image)
    cands = intersection_candidates(segs, image.shape)
    rvec = np.asarray(solved["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(solved["tvec"], dtype=np.float64).reshape(3, 1)
    # Prefer solver-exported K (may have fy < 0 after chirality fix)
    intr_use = solved.get("intrinsics") or intr
    K = np.asarray(intr_use["camera_matrix"], dtype=np.float64)
    D = np.asarray(intr_use.get("dist_coeffs") or [0, 0, 0, 0, 0], dtype=np.float64)

    out = dict(seeds)
    for pid in target_ids or []:
        if pid in out:
            continue
        xyz = landmark_xyz(model, pid)
        snapped = project_and_snap(xyz, rvec, tvec, K, D, cands, max_dist=snap_px)
        if snapped is None:
            imgpts, _ = cv2.projectPoints(xyz.reshape(1, 1, 3), rvec, tvec, K, D)
            u, v = float(imgpts[0, 0, 0]), float(imgpts[0, 0, 1])
            if 0 <= u < w and 0 <= v < h:
                out[pid] = [round(u, 1), round(v, 1)]
            continue
        out[pid] = [round(snapped[0], 1), round(snapped[1], 1)]
    return out


def annotate_camera_gui(
    image_path: Path,
    camera_id: str,
    annotations_path: Path,
    model: dict[str, Any] | None = None,
    point_order: list[str] | None = None,
) -> dict[str, Any]:
    """
    OpenCV GUI (per-camera prescribed landmarks):
      left-click      = place current landmark at exact click (image pixels)
      Shift+left-click = same, but snap to nearby line intersection
      n / space       = next landmark
      p               = previous
      d / backspace   = delete current
      a               = auto-fill remaining *in this camera's list* from seeds (≥4)
      s               = save
      q / esc / enter = finish this camera
    """
    model = model or load_court_model()
    order = point_order or annotation_order_for_camera(model, camera_id)
    if not order:
        raise ValueError(f"No annotation order for {camera_id}")

    img0 = cv2.imread(str(image_path))
    if img0 is None:
        raise FileNotFoundError(image_path)
    h, w = img0.shape[:2]

    if annotations_path.exists():
        doc = load_annotations(annotations_path)
    else:
        doc = empty_annotation_doc([camera_id])
    doc.setdefault("frames", {})[camera_id] = str(image_path)
    doc.setdefault("image_size", {})[camera_id] = [w, h]
    doc.setdefault("points", {}).setdefault(camera_id, {})
    pts: dict[str, list[float]] = {
        k: v for k, v in dict(doc["points"][camera_id]).items() if k in order
    }

    win = f"annotate {camera_id}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Keep aspect; start near native size so mouse↔pixel stay 1:1 until user resizes
    try:
        cv2.resizeWindow(win, min(w, 1600), min(h, 900))
    except cv2.error:
        pass
    # Show loading tip while Hough runs (~1–2s) so UI never looks dead
    boot = img0.copy()
    banner = draw_hud_banner(w, [f"{camera_id} 正在检测球场线，请稍候…"], font_size=24)
    boot[: banner.shape[0]] = cv2.addWeighted(
        boot[: banner.shape[0]], 0.35, banner, 0.65, 0,
    )
    cv2.imshow(win, boot)
    cv2.waitKey(1)

    segs = detect_court_lines(img0)
    cands = intersection_candidates(segs, img0.shape)

    # Cache static overlay (lines + candidates) — never redraw in the hot loop
    base = img0.copy()
    for s in segs:
        cv2.line(base, (int(s.x1), int(s.y1)), (int(s.x2), int(s.y2)), (40, 40, 40), 1)
    for c in cands:
        cv2.circle(base, (int(c[0]), int(c[1])), 3, (0, 180, 255), -1)

    idx = 0
    for i, pid in enumerate(order):
        if pid not in pts:
            idx = i
            break

    # Mouse only queues clicks; all logic runs in the main loop (avoids HighGUI stalls)
    pending_click: tuple[int, int, bool] | None = None  # x, y, shift_snap

    def on_mouse(event, x, y, flags, param):
        nonlocal pending_click
        if event == cv2.EVENT_LBUTTONDOWN:
            u, v = _map_mouse_to_image(int(x), int(y), win, w, h)
            shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
            pending_click = (u, v, shift)

    cv2.setMouseCallback(win, on_mouse)

    def current_id() -> str:
        return order[idx % len(order)]

    def persist() -> None:
        doc["points"][camera_id] = pts
        save_annotations(doc, annotations_path)

    def advance_to_missing() -> bool:
        """Move idx to next missing point. Returns False if all done."""
        nonlocal idx
        for step in range(0, len(order)):
            j = (idx + step) % len(order)
            if order[j] not in pts:
                idx = j
                return True
        return False

    status = ""
    while True:
        # --- handle queued click in main thread ---
        if pending_click is not None:
            x, y, do_snap = pending_click
            pending_click = None
            if all(p in pts for p in order):
                break
            pid = current_id()
            # Default: exact click. Old auto-snap (35px) pulled corners to wrong
            # Hough intersections — looked like click vs marker mismatch on cam_02.
            if do_snap:
                snapped = snap_to_candidates((float(x), float(y)), cands, max_dist=18.0)
                placed = snapped if snapped is not None else (float(x), float(y))
                tag = "吸附" if snapped is not None else "精确(无交点)"
            else:
                placed = (float(x), float(y))
                tag = "精确"
            pts[pid] = [round(placed[0], 1), round(placed[1], 1)]
            persist()
            status = f"已标 {landmark_name(model, pid)}（{tag}）"
            print(f"[{camera_id}] + {pid} @ {pts[pid]} [{tag}]", flush=True)
            if not advance_to_missing():
                status = f"{camera_id} 已完成，按 q/回车 进入下一相机"
                print(f"[{camera_id}] 全部完成，等待确认", flush=True)

        done_n = sum(1 for p in order if p in pts)
        all_done = done_n >= len(order)
        pid = current_id()
        name = landmark_name(model, pid)

        vis = base.copy()
        for p_id, uv in pts.items():
            cv2.circle(vis, (int(uv[0]), int(uv[1])), 7, (0, 220, 0), 2)
            paste_chip(
                vis, landmark_name(model, p_id),
                (int(uv[0]) + 8, int(uv[1]) - 4),
                font_size=18, color_bgr=(0, 220, 0),
            )
        if pid in pts:
            uv = pts[pid]
            cv2.circle(vis, (int(uv[0]), int(uv[1])), 12, (0, 255, 255), 2)

        # Short HUD lines (avoid one giant string)
        if all_done:
            lines = [
                f"{camera_id} 已完成 {done_n}/{len(order)} — 按 q 或回车进入下一相机",
                status or "全部点位已标注",
            ]
        else:
            lines = [
                f"{camera_id} [{idx + 1}/{len(order)}] 请标注: {name}",
                f"进度 {done_n}/{len(order)} | 左键精确 | Shift+左键吸附 | d删除 | n/p | s保存 | q下一相机",
                status,
            ]
        # Compact checklist on its own lines (max 2)
        marks = [f"{'✓' if p in pts else '○'}{landmark_name(model, p)}" for p in order]
        mid = (len(marks) + 1) // 2
        lines.append(" ".join(marks[:mid]))
        if mid < len(marks):
            lines.append(" ".join(marks[mid:]))

        banner = draw_hud_banner(w, [ln for ln in lines if ln], font_size=20, line_h=28)
        bh = banner.shape[0]
        vis[:bh] = cv2.addWeighted(vis[:bh], 0.30, banner, 0.70, 0)

        cv2.setWindowTitle(win, f"{camera_id} {done_n}/{len(order)} — {name}")
        cv2.imshow(win, vis)

        key = cv2.waitKeyEx(15)
        if key < 0:
            continue
        # Normalize: waitKeyEx may return 0x100000*mod + ascii
        key8 = key & 0xFF
        if key in (ord("q"), ord("Q"), 27, 13) or key8 in (ord("q"), ord("Q"), 27, 13):
            break
        if key8 in (ord("n"), ord("N"), ord(" ")):
            idx = (idx + 1) % len(order)
            status = f"切换到 {landmark_name(model, current_id())}"
        elif key8 in (ord("p"), ord("P")):
            idx = (idx - 1) % len(order)
            status = f"切换到 {landmark_name(model, current_id())}"
        elif key8 in (ord("d"), ord("D"), 8, 127) or key in (8, 127, 65535):
            cur = current_id()
            if cur in pts:
                pts.pop(cur, None)
                persist()
                status = f"已删除 {landmark_name(model, cur)}"
                print(f"[{camera_id}] - {cur}", flush=True)
            else:
                status = "当前点尚未标注，无需删除"
        elif key8 in (ord("a"), ord("A")):
            status = "自动补全中…"
            cv2.imshow(win, vis)
            cv2.waitKey(1)
            pts = auto_fill_from_seeds(img0, pts, model=model, target_ids=order)
            persist()
            advance_to_missing()
            status = f"补全后 {sum(1 for p in order if p in pts)}/{len(order)}"
            print(f"[{camera_id}] auto-fill → {list(pts)}", flush=True)
        elif key8 in (ord("s"), ord("S")):
            persist()
            status = "已保存"
            print(f"saved → {annotations_path}", flush=True)

    try:
        cv2.destroyWindow(win)
    except cv2.error:
        pass
    persist()
    missing = [landmark_name(model, p) for p in order if p not in pts]
    if missing:
        print(f"[{camera_id}] 尚未完成: {', '.join(missing)}", flush=True)
    else:
        print(f"[{camera_id}] 全部 {len(order)} 点已标注 ✓", flush=True)
    return doc
