#!/usr/bin/env python3
"""
Build interactive 4D (3D + time) skeleton viewer HTML for a group.

- With data/calibration/v2_4cam_zoned → multi-view triangulation (metric court)
- Without calibration → cam_03 pseudo-3D preview (clearly labeled)

Usage:
  python scripts/build_pose3d_viewer.py --group-dir data/outputs/v1/group_01
  python scripts/build_pose3d_viewer.py --all-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.viz.pose3d_scene import build_group_pose3d_scene  # noqa: E402


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>4D Skeleton — __TITLE__</title>
<style>
  :root { --bg:#0b1220; --panel:#151d2c; --text:#e8eef6; --muted:#8b9bb0; --accent:#f5a623; --ok:#3ecf8e; --warn:#f0a020; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: "Segoe UI","PingFang SC","Noto Sans SC",sans-serif; background:var(--bg); color:var(--text); height:100vh; display:flex; flex-direction:column; }
  header { padding:0.75rem 1rem; border-bottom:1px solid #243044; display:flex; flex-wrap:wrap; gap:0.75rem; align-items:center; justify-content:space-between; }
  header h1 { margin:0; font-size:1.05rem; }
  .badge { font-size:0.75rem; padding:0.2rem 0.5rem; border-radius:4px; }
  .badge.warn { background:rgba(240,160,32,0.2); color:var(--warn); }
  .badge.ok { background:rgba(62,207,142,0.2); color:var(--ok); }
  #view { flex:1; position:relative; min-height:320px; }
  #c { width:100%; height:100%; display:block; }
  .bar { padding:0.65rem 1rem 1rem; background:var(--panel); border-top:1px solid #243044; }
  .row { display:flex; flex-wrap:wrap; gap:0.75rem; align-items:center; margin-bottom:0.5rem; }
  input[type=range] { flex:1; min-width:180px; }
  button { background:#243044; color:var(--text); border:1px solid #33455e; border-radius:6px; padding:0.35rem 0.7rem; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  .meta { color:var(--muted); font-size:0.8rem; }
  select { background:#243044; color:var(--text); border:1px solid #33455e; border-radius:6px; padding:0.3rem 0.5rem; }
</style>
</head>
<body>
<header>
  <div>
    <h1 id="title">4D Skeleton</h1>
    <div class="meta" id="subtitle"></div>
  </div>
  <span class="badge" id="mode-badge"></span>
</header>
<div id="view"><canvas id="c"></canvas></div>
<div class="bar">
  <div class="row">
    <button id="play">播放</button>
    <button id="prev">-1</button>
    <button id="next">+1</button>
    <input type="range" id="scrub" min="0" max="0" value="0"/>
    <span class="meta" id="time-label">0.0s</span>
    <label class="meta">跳到出手
      <select id="clip-sel"><option value="">—</option></select>
    </label>
  </div>
  <div class="meta" id="hint"></div>
</div>
<script type="importmap">
{ "imports": { "three": "https://unpkg.com/three@0.160.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" } }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DATA = __DATA__;

const title = document.getElementById("title");
const subtitle = document.getElementById("subtitle");
const badge = document.getElementById("mode-badge");
const hint = document.getElementById("hint");
title.textContent = `4D Skeleton — ${DATA.group_id || ""}`;
subtitle.textContent = `${DATA.student_id || ""} · ${DATA.n_frames || 0} frames · ${DATA.mode_note || ""}`;
badge.textContent = DATA.mode === "triangulated" ? "triangulated" : (DATA.mode === "pseudo3d_video" ? "pseudo3d+root" : "pseudo3d");
badge.className = "badge " + (DATA.mode === "triangulated" ? "ok" : "warn");
    hint.textContent = DATA.mode === "triangulated"
  ? "拖拽旋转/滚轮缩放。若骨架与球场对齐合理、出手时肘膝弯曲连贯，说明标定与三角化基本可信。"
  : (DATA.mode === "pseudo3d_video"
    ? "由 cam_03 重提姿态：可看球场上的近似位移；已抑制左右翻转跳变。仍非标定三角化。"
    : "当前为 motion.json 伪3D 预览。建议用 --from-video 重提以查看位移。同目录 2D: viz/phases.mp4");

const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1220);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
camera.position.set(8, 6, 14);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 1.2, 6);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dir = new THREE.DirectionalLight(0xffffff, 0.85);
dir.position.set(5, 10, 3);
scene.add(dir);

// Court floor + lines (court X,Z_up→Y, Y_floor→Z)
function courtToThree(p) { return new THREE.Vector3(p[0], p[2] ?? 0, p[1]); }

const court = DATA.court || { points: {}, segments: [] };
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(16, 16),
  new THREE.MeshStandardMaterial({ color: 0x1a3040, roughness: 0.95 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.set(0, 0, 7);
scene.add(floor);

const lineMat = new THREE.LineBasicMaterial({ color: 0xf5a623 });
for (const seg of (court.segments || [])) {
  const g = new THREE.BufferGeometry().setFromPoints([courtToThree(seg.a), courtToThree(seg.b)]);
  scene.add(new THREE.Line(g, lineMat));
}
for (const [id, p] of Object.entries(court.points || {})) {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(0.08, 10, 10),
    new THREE.MeshStandardMaterial({ color: 0x3ecf8e })
  );
  m.position.copy(courtToThree(p));
  scene.add(m);
}

// Skeleton
const jointMeshes = [];
const boneLines = [];
const jointMat = new THREE.MeshStandardMaterial({ color: 0x5ec8ff });
const boneMat = new THREE.LineBasicMaterial({ color: 0xd0e8ff });
for (let i = 0; i < 17; i++) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 12), jointMat);
  m.visible = false;
  scene.add(m);
  jointMeshes.push(m);
}
for (const _ of (DATA.edges || [])) {
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const line = new THREE.Line(g, boneMat);
  scene.add(line);
  boneLines.push(line);
}

let idx = 0;
let playing = false;
const scrub = document.getElementById("scrub");
const timeLabel = document.getElementById("time-label");
scrub.max = Math.max(0, (DATA.frames || []).length - 1);

function setFrame(i) {
  const frames = DATA.frames || [];
  if (!frames.length) return;
  idx = Math.max(0, Math.min(frames.length - 1, i));
  scrub.value = String(idx);
  const fr = frames[idx];
  timeLabel.textContent = `${(fr.t_ms / 1000).toFixed(2)}s  (#${idx+1}/${frames.length})`;
  const joints = fr.joints || [];
  for (let j = 0; j < 17; j++) {
    const p = joints[j];
    if (!p) { jointMeshes[j].visible = false; continue; }
    jointMeshes[j].visible = true;
    jointMeshes[j].position.set(p[0], p[1], p[2]);
  }
  (DATA.edges || []).forEach((e, bi) => {
    const a = joints[e[0]], b = joints[e[1]];
    const line = boneLines[bi];
    if (!a || !b) { line.visible = false; return; }
    line.visible = true;
    const pos = line.geometry.attributes.position;
    pos.setXYZ(0, a[0], a[1], a[2]);
    pos.setXYZ(1, b[0], b[1], b[2]);
    pos.needsUpdate = true;
  });
}

scrub.oninput = () => setFrame(Number(scrub.value));
document.getElementById("prev").onclick = () => setFrame(idx - 1);
document.getElementById("next").onclick = () => setFrame(idx + 1);
document.getElementById("play").onclick = () => {
  playing = !playing;
  document.getElementById("play").textContent = playing ? "暂停" : "播放";
};

const clipSel = document.getElementById("clip-sel");
for (const c of (DATA.clips || [])) {
  const opt = document.createElement("option");
  opt.value = String(c.release_ms ?? "");
  opt.textContent = `#${c.i} ${c.action_type || ""} @ ${((c.release_ms||0)/1000).toFixed(1)}s`;
  clipSel.appendChild(opt);
}
clipSel.onchange = () => {
  const t = Number(clipSel.value);
  if (!Number.isFinite(t)) return;
  const frames = DATA.frames || [];
  let best = 0, bestD = 1e18;
  frames.forEach((f, i) => {
    const d = Math.abs(f.t_ms - t);
    if (d < bestD) { bestD = d; best = i; }
  });
  setFrame(best);
};

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  }
}

let last = 0;
function loop(t) {
  resize();
  if (playing && t - last > 1000 / Math.max(1, (DATA.fps || 15) / 2)) {
    last = t;
    setFrame(idx + 1);
    if (idx >= (DATA.frames || []).length - 1) playing = false;
    document.getElementById("play").textContent = playing ? "暂停" : "播放";
  }
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
setFrame(0);
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def resolve_cam03_video(group_dir: Path, data_dir: Path) -> Path | None:
    name = group_dir.name  # group_01
    if not name.startswith("group_"):
        return None
    try:
        gid = int(name.split("_", 1)[1])
    except ValueError:
        return None
    cand = data_dir / f"{gid}-3.mkv"
    return cand if cand.exists() else None


def write_viewer(
    group_dir: Path,
    calib_dir: Path | None = None,
    *,
    from_video: bool = False,
    video_path: Path | None = None,
    video_stride: int = 2,
) -> Path:
    scene = build_group_pose3d_scene(
        group_dir,
        calib_dir=calib_dir,
        video_path=video_path if from_video else None,
        video_stride=video_stride,
    )
    (group_dir / "pose3d_scene.json").write_text(
        json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    html = (
        HTML_TEMPLATE
        .replace("__TITLE__", scene.get("group_id") or group_dir.name)
        .replace("__DATA__", json.dumps(scene, ensure_ascii=False))
    )
    out = group_dir / "pose3d_viewer.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-dir", type=Path, default=None)
    ap.add_argument("--all-v1", action="store_true")
    ap.add_argument("--calib-dir", type=Path, default=ROOT / "data/calibration/v2_4cam_zoned")
    ap.add_argument("--from-video", action="store_true",
                    help="Re-extract pose from cam_03 so body travels on court")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/test_data_v1")
    ap.add_argument("--video-stride", type=int, default=2)
    args = ap.parse_args()

    dirs = []
    if args.all_v1:
        dirs = sorted((ROOT / "data/outputs/v1").glob("group_*"))
    elif args.group_dir:
        dirs = [args.group_dir]
    else:
        raise SystemExit("pass --group-dir or --all-v1")

    calib = args.calib_dir if args.calib_dir.exists() else None
    for gdir in dirs:
        if not (gdir / "summary.json").exists():
            print(f"skip {gdir.name}")
            continue
        video = resolve_cam03_video(gdir, args.data_dir) if args.from_video else None
        if args.from_video and video is None:
            print(f"warn {gdir.name}: no cam_03 video, falling back to motion.json")
        out = write_viewer(
            gdir,
            calib_dir=calib,
            from_video=args.from_video and video is not None,
            video_path=video,
            video_stride=args.video_stride,
        )
        scene = json.loads((gdir / "pose3d_scene.json").read_text(encoding="utf-8"))
        pelvis = []
        for fr in scene.get("frames") or []:
            j = fr.get("joints") or []
            if j and j[0]:
                pelvis.append(j[0])
        prange = [0, 0, 0]
        if len(pelvis) >= 2:
            import numpy as np
            p = np.asarray(pelvis, dtype=float)
            prange = (p.max(0) - p.min(0)).round(2).tolist()
        print(json.dumps({
            "group": scene["group_id"],
            "mode": scene["mode"],
            "n_frames": scene["n_frames"],
            "pelvis_range_xyz": prange,
            "html": str(out),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
