#!/usr/bin/env python3
"""Visual multi-camera time sync for one test-data group.

Automated options (also in repo)
--------------------------------
- Event sync after perception: release peaks × cam_04 rim events
  (`scripts/sync_cameras.py --session <uuid>`). Needs pose/ball first and is
  often weak when detections are sparse.
- No shared audio on v3 mkv → audio cross-correlation not available.

This GUI
--------
Scrub a shared timeline and nudge each camera's constant offset until
actions line up. Saves offsets under ``<data-dir>/sync/group_XX.json``.

  PYTHONPATH=. python scripts/sync_group_gui.py \\
      --data-dir data/test_data_v3 --group 1

Controls
--------
  Space          play / pause
  ← / →          step −/+ 1 frame (anchor fps)
  Shift+←/→      step −/+ 10 frames
  [ / ]          nudge selected camera −/+ 1 frame
  ; / '          nudge selected camera −/+ 5 frames
  1–4            select camera for nudge
  S              save
  Q / Esc        quit (prompts if dirty)
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cameras.group_sync import (  # noqa: E402
    discover_group_videos,
    empty_group_sync,
    load_group_sync,
    save_group_sync,
)
from src.cameras.registry import get_sync_config  # noqa: E402


CAMS = ("cam_01", "cam_02", "cam_03", "cam_04")
PANEL_W, PANEL_H = 640, 360


class VideoSource:
    def __init__(self, path: Path, cam_id: str):
        self.path = path
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_ms = (self.n_frames / max(self.fps, 1e-6)) * 1000.0
        self._cached_idx = -1
        self._cached_bgr: np.ndarray | None = None

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        if self.n_frames <= 0:
            return None
        fi = int(np.clip(frame_idx, 0, max(0, self.n_frames - 1)))
        if fi == self._cached_idx and self._cached_bgr is not None:
            return self._cached_bgr
        # Prefer sequential grab when close
        if self._cached_idx >= 0 and 0 < fi - self._cached_idx <= 8:
            for _ in range(fi - self._cached_idx):
                self.cap.grab()
            ok, fr = self.cap.retrieve()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = self.cap.read()
        if not ok or fr is None:
            return self._cached_bgr
        self._cached_idx = fi
        self._cached_bgr = fr
        return fr

    def close(self) -> None:
        self.cap.release()


class SyncGroupGUI:
    def __init__(self, data_dir: Path, group_id: int, anchor: str):
        self.data_dir = Path(data_dir)
        self.group_id = int(group_id)
        self.anchor = anchor
        videos = discover_group_videos(self.data_dir, self.group_id)
        missing = [c for c in CAMS if c not in videos]
        if missing:
            raise SystemExit(f"missing videos for group {group_id}: {missing} in {data_dir}")

        self.sources = {c: VideoSource(videos[c], c) for c in CAMS}
        existing = load_group_sync(self.data_dir, self.group_id)
        if existing:
            self.doc = existing
            self.doc["anchor_camera"] = self.anchor
        else:
            self.doc = empty_group_sync(
                self.group_id,
                dataset=str(self.data_dir.name),
                anchor_camera=self.anchor,
            )
        offs = dict(self.doc.get("camera_time_offsets_ms") or {})
        for c in CAMS:
            offs.setdefault(c, 0.0)
        offs[self.anchor] = 0.0
        self.offsets_ms = {c: float(offs[c]) for c in CAMS}
        self.dirty = False

        # Common timeline length: shortest (local_duration - offset) coverage
        self.common_max_ms = min(
            max(0.0, src.duration_ms - self.offsets_ms[c])
            for c, src in self.sources.items()
        )
        self.common_ms = 0.0
        self.playing = False
        self.selected = self.anchor
        self._photo: dict[str, ImageTk.PhotoImage] = {}
        self._tick_job = None

        self.root = tk.Tk()
        self.root.title(
            f"Camera sync — {self.data_dir.name} / group_{self.group_id:02d} "
            f"(anchor {self.anchor})"
        )
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        self._build_ui()
        self._bind_keys()
        self._refresh_all()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.BOTH, expand=True)

        grid = ttk.Frame(top)
        grid.pack(fill=tk.BOTH, expand=True)
        self.labels: dict[str, tk.Label] = {}
        self.offset_vars: dict[str, tk.DoubleVar] = {}
        self.offset_labels: dict[str, ttk.Label] = {}

        for i, cam in enumerate(CAMS):
            cell = ttk.LabelFrame(grid, text=cam)
            cell.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4)
            lab = tk.Label(cell, bg="#222")
            lab.pack()
            self.labels[cam] = lab
            var = tk.DoubleVar(value=self.offsets_ms[cam])
            self.offset_vars[cam] = var
            row = ttk.Frame(cell)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text="offset ms").pack(side=tk.LEFT)
            state = "disabled" if cam == self.anchor else "normal"
            scale = ttk.Scale(
                row, from_=-5000, to=5000, variable=var, orient=tk.HORIZONTAL,
                command=lambda _v, c=cam: self._on_offset_scale(c),
                state=state,
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            ol = ttk.Label(row, width=8)
            ol.pack(side=tk.RIGHT)
            self.offset_labels[cam] = ol
            self._update_offset_label(cam)

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        # Master timeline
        bot = ttk.Frame(top, padding=(0, 8, 0, 0))
        bot.pack(fill=tk.X)
        self.time_var = tk.DoubleVar(value=0.0)
        ttk.Label(bot, text="common t").pack(side=tk.LEFT)
        self.time_scale = ttk.Scale(
            bot, from_=0, to=max(1.0, self.common_max_ms),
            variable=self.time_var, orient=tk.HORIZONTAL,
            command=self._on_time_scale,
        )
        self.time_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.time_label = ttk.Label(bot, width=18)
        self.time_label.pack(side=tk.RIGHT)

        ctrl = ttk.Frame(top)
        ctrl.pack(fill=tk.X, pady=4)
        ttk.Button(ctrl, text="▶/❚❚ Space", command=self._toggle_play).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="−1f", command=lambda: self._step_common(-1)).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="+1f", command=lambda: self._step_common(1)).pack(side=tk.LEFT)
        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(ctrl, text="nudge cam:").pack(side=tk.LEFT)
        self.sel_var = tk.StringVar(value=self.selected)
        for cam in CAMS:
            ttk.Radiobutton(
                ctrl, text=cam[-1], value=cam, variable=self.sel_var,
                command=self._on_select_cam,
            ).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="offset −1f", command=lambda: self._nudge_selected(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="offset +1f", command=lambda: self._nudge_selected(1)).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Save (S)", command=self._save).pack(side=tk.RIGHT, padx=2)
        ttk.Button(ctrl, text="Quit", command=self._on_quit).pack(side=tk.RIGHT)

        self.status = ttk.Label(
            top,
            text=(
                "对齐同一瞬间（出手/落地/拍手）。绿框=锚点。 "
                "offset>0 → 该机画面相对锚点更「晚」的内容对齐到当前 common 时刻。"
            ),
            wraplength=1200,
        )
        self.status.pack(fill=tk.X, pady=4)

    def _bind_keys(self) -> None:
        r = self.root
        r.bind("<space>", lambda _e: self._toggle_play())
        r.bind("<Left>", lambda _e: self._step_common(-1))
        r.bind("<Right>", lambda _e: self._step_common(1))
        r.bind("<Shift-Left>", lambda _e: self._step_common(-10))
        r.bind("<Shift-Right>", lambda _e: self._step_common(10))
        r.bind("<bracketleft>", lambda _e: self._nudge_selected(-1))
        r.bind("<bracketright>", lambda _e: self._nudge_selected(1))
        r.bind("<semicolon>", lambda _e: self._nudge_selected(-5))
        r.bind("<apostrophe>", lambda _e: self._nudge_selected(5))
        for i, cam in enumerate(CAMS, 1):
            r.bind(str(i), lambda _e, c=cam: self._select_cam(c))
        r.bind("<s>", lambda _e: self._save())
        r.bind("<S>", lambda _e: self._save())
        r.bind("<q>", lambda _e: self._on_quit())
        r.bind("<Escape>", lambda _e: self._on_quit())

    def _on_select_cam(self) -> None:
        self.selected = self.sel_var.get()
        self._refresh_all()

    def _select_cam(self, cam: str) -> None:
        self.sel_var.set(cam)
        self.selected = cam
        self._refresh_all()

    def _update_offset_label(self, cam: str) -> None:
        self.offset_labels[cam].configure(text=f"{self.offsets_ms[cam]:+.0f}")

    def _on_offset_scale(self, cam: str) -> None:
        if cam == self.anchor:
            self.offset_vars[cam].set(0.0)
            return
        self.offsets_ms[cam] = float(self.offset_vars[cam].get())
        self._update_offset_label(cam)
        self.dirty = True
        self._recompute_common_max()
        self._refresh_all()

    def _recompute_common_max(self) -> None:
        self.common_max_ms = min(
            max(0.0, src.duration_ms - self.offsets_ms[c])
            for c, src in self.sources.items()
        )
        self.time_scale.configure(to=max(1.0, self.common_max_ms))
        if self.common_ms > self.common_max_ms:
            self.common_ms = self.common_max_ms
            self.time_var.set(self.common_ms)

    def _on_time_scale(self, _v: str | None = None) -> None:
        self.common_ms = float(self.time_var.get())
        if self.playing:
            return
        self._refresh_all()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self._tick()
        elif self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None

    def _tick(self) -> None:
        if not self.playing:
            return
        fps = self.sources[self.anchor].fps
        self.common_ms = min(self.common_max_ms, self.common_ms + 1000.0 / max(fps, 1e-6))
        self.time_var.set(self.common_ms)
        self._refresh_all()
        if self.common_ms >= self.common_max_ms - 1e-3:
            self.playing = False
            return
        delay = max(8, int(1000 / max(fps, 1e-6)))
        self._tick_job = self.root.after(delay, self._tick)

    def _step_common(self, frames: int) -> None:
        fps = self.sources[self.anchor].fps
        self.common_ms = float(np.clip(
            self.common_ms + frames * (1000.0 / max(fps, 1e-6)),
            0.0, self.common_max_ms,
        ))
        self.time_var.set(self.common_ms)
        self._refresh_all()

    def _nudge_selected(self, frames: int) -> None:
        cam = self.selected
        if cam == self.anchor:
            self.status.configure(text="锚点机位 offset 固定为 0，请选择其他相机微调。")
            return
        fps = self.sources[cam].fps
        delta = frames * (1000.0 / max(fps, 1e-6))
        self.offsets_ms[cam] = float(self.offsets_ms[cam] + delta)
        self.offset_vars[cam].set(self.offsets_ms[cam])
        self._update_offset_label(cam)
        self.dirty = True
        self._recompute_common_max()
        self._refresh_all()

    def _bgr_to_photo(self, bgr: np.ndarray, cam: str) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (PANEL_W, PANEL_H))
        # overlays
        if cam == self.anchor:
            cv2.rectangle(rgb, (2, 2), (PANEL_W - 3, PANEL_H - 3), (40, 200, 80), 3)
        if cam == self.selected:
            cv2.rectangle(rgb, (8, 8), (PANEL_W - 9, PANEL_H - 9), (255, 200, 40), 2)
        text = f"{cam}  off={self.offsets_ms[cam]:+.0f}ms"
        cv2.putText(rgb, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        img = Image.fromarray(rgb)
        return ImageTk.PhotoImage(image=img)

    def _refresh_all(self) -> None:
        for cam, src in self.sources.items():
            local_ms = self.common_ms + self.offsets_ms[cam]
            fi = int(round(local_ms * src.fps / 1000.0))
            fr = src.read_frame(fi)
            if fr is None:
                blank = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                photo = self._bgr_to_photo(blank, cam)
            else:
                photo = self._bgr_to_photo(fr, cam)
            self._photo[cam] = photo
            self.labels[cam].configure(image=photo)
        tsec = self.common_ms / 1000.0
        self.time_label.configure(text=f"{tsec:7.2f}s / {self.common_max_ms/1000:.1f}s")
        dirty = " *未保存*" if self.dirty else ""
        self.status.configure(
            text=(
                f"common={self.common_ms:.0f}ms  selected={self.selected}  "
                f"offsets={{{', '.join(f'{c[-1]}:{self.offsets_ms[c]:+.0f}' for c in CAMS)}}}{dirty}"
            )
        )

    def _save(self) -> None:
        self.doc["camera_time_offsets_ms"] = {c: float(self.offsets_ms[c]) for c in CAMS}
        self.doc["anchor_camera"] = self.anchor
        self.doc["dataset"] = str(self.data_dir.name)
        self.doc["source"] = "manual_gui"
        self.doc["fps"] = {c: self.sources[c].fps for c in CAMS}
        self.doc["duration_ms"] = {c: self.sources[c].duration_ms for c in CAMS}
        path = save_group_sync(self.data_dir, self.doc)
        self.dirty = False
        self.status.configure(text=f"已保存 → {path}")
        messagebox.showinfo("Saved", f"Offsets written to:\n{path}")

    def _on_quit(self) -> None:
        if self.dirty:
            if messagebox.askyesno("Unsaved", "有未保存的偏移，仍要退出吗？"):
                pass
            else:
                return
        if self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
        for src in self.sources.values():
            src.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visual 4-cam group time sync")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "test_data_v3")
    ap.add_argument("--group", type=int, required=True, help="group id, e.g. 1")
    ap.add_argument(
        "--anchor",
        default=None,
        help="anchor camera (default: sync.event_anchor_camera / cam_03)",
    )
    args = ap.parse_args()
    anchor = args.anchor or (get_sync_config().get("event_anchor_camera") or "cam_03")
    SyncGroupGUI(args.data_dir, args.group, anchor).run()


if __name__ == "__main__":
    main()
