#!/usr/bin/env python3
"""
Extract multi-view triangulated skeletons for v1 groups' action clips,
ground feet to the court plane, and embed in a court+camera 3D viewer.

Usage:
  PYTHONPATH=. python scripts/extract_action_skeletons_3d.py --groups 1,2,3,4
  PYTHONPATH=. python scripts/extract_action_skeletons_3d.py --groups 1 --stride 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose.action_skeleton3d import process_group_action_skeletons  # noqa: E402

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Court + Skeleton — __TITLE__</title>
<style>
  :root { --bg:#0b1220; --panel:#151d2c; --text:#e8eef6; --muted:#8b9bb0; --accent:#f5a623; --ok:#3ecf8e; --warn:#f0a020; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Segoe UI","PingFang SC","Noto Sans SC",sans-serif; background:var(--bg); color:var(--text); height:100vh; display:flex; flex-direction:column; }
  header { padding:0.7rem 1rem; border-bottom:1px solid #243044; }
  header h1 { margin:0; font-size:1.05rem; }
  .meta { color:var(--muted); font-size:0.82rem; margin-top:0.25rem; }
  #view { flex:1; min-height:360px; position:relative; }
  #c { width:100%; height:100%; display:block; }
  .bar { padding:0.65rem 1rem; background:var(--panel); border-top:1px solid #243044; }
  .row { display:flex; flex-wrap:wrap; gap:0.6rem; align-items:center; margin-bottom:0.4rem; }
  input[type=range] { flex:1; min-width:160px; }
  button, select { background:#243044; color:var(--text); border:1px solid #33455e; border-radius:6px; padding:0.35rem 0.65rem; cursor:pointer; }
  .ok { color:var(--ok); } .warn { color:var(--warn); }
</style>
</head>
<body>
<header>
  <h1 id="title">球场 · 相机 · 三角化骨架</h1>
  <div class="meta" id="subtitle"></div>
</header>
<div id="view"><canvas id="c"></canvas></div>
<div class="bar">
  <div class="row">
    <button id="play">播放</button>
    <button id="prev">-1</button>
    <button id="next">+1</button>
    <input type="range" id="scrub" min="0" max="0" value="0"/>
    <span class="meta" id="time-label">0s</span>
    <select id="clip-sel"><option value="">clip…</option></select>
  </div>
  <div class="meta" id="hint">绿点=踝关节（脚）；脚底应贴近橙色球场线。若脚悬空/钻地，说明三角化或同步仍有误差。</div>
</div>
<script type="importmap">
{ "imports": { "three": "https://unpkg.com/three@0.160.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" } }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DATA = __DATA__;
document.getElementById("title").textContent = `Group ${DATA.group_id} · triangulated`;
document.getElementById("subtitle").textContent =
  `${DATA.student_id||""} · ${DATA.n_frames||0} frames · floorΔz=${(DATA.floor_z_subtracted_m||0).toFixed(3)}m · offsets=${JSON.stringify(DATA.offsets_ms||{})}`;

const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1220);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
camera.position.set(10, 8, 16);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 1.0, 6);
controls.update();
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(5, 12, 3); scene.add(dir);

// court (X,Y_floor,Z_up) → three (X, Z_up, Y_floor)
function c2t(p) { return new THREE.Vector3(p[0], p[2] ?? 0, p[1]); }

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(16, 16),
  new THREE.MeshStandardMaterial({ color: 0x1a3040, roughness: 0.95, transparent:true, opacity:0.9 })
);
floor.rotation.x = -Math.PI/2; floor.position.set(0,0,7); scene.add(floor);
scene.add(new THREE.AxesHelper(2.0));

const lineMat = new THREE.LineBasicMaterial({ color: 0xf5a623 });
for (const seg of (DATA.court?.segments || [])) {
  const g = new THREE.BufferGeometry().setFromPoints([c2t(seg.a), c2t(seg.b)]);
  scene.add(new THREE.Line(g, lineMat));
}
for (const p of Object.values(DATA.court?.points || {})) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 10), new THREE.MeshStandardMaterial({ color: 0x3ecf8e }));
  m.position.copy(c2t(p)); scene.add(m);
}

// cameras
const COLORS = { cam_01: 0x5ec8ff, cam_02: 0xff6b8a, cam_03: 0xf5a623 };
for (const cam of (DATA.cameras || [])) {
  const col = cam.z_below_ground ? 0xff3333 : (COLORS[cam.id] || 0xffffff);
  const C = c2t(cam.center);
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.18, 14, 14), new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.2 }));
  body.position.copy(C); scene.add(body);
  const fwd = new THREE.Vector3(cam.forward[0], cam.forward[2], cam.forward[1]).normalize();
  const rayG = new THREE.BufferGeometry().setFromPoints([C, C.clone().add(fwd.clone().multiplyScalar(3.5))]);
  scene.add(new THREE.Line(rayG, new THREE.LineBasicMaterial({ color: col })));
}

// skeleton
const jointMeshes = [], boneLines = [];
const boneMat = new THREE.LineBasicMaterial({ color: 0xd0e8ff });
const bodyMat = new THREE.MeshStandardMaterial({ color: 0x5ec8ff });
const footMat = new THREE.MeshStandardMaterial({ color: 0x3ecf8e, emissive: 0x3ecf8e, emissiveIntensity: 0.35 });
for (let i = 0; i < 17; i++) {
  const mat = (i === 3 || i === 6) ? footMat : bodyMat;
  const m = new THREE.Mesh(new THREE.SphereGeometry(i===3||i===6 ? 0.09 : 0.055, 12, 12), mat);
  m.visible = false; scene.add(m); jointMeshes.push(m);
}
for (const _ of (DATA.edges || [])) {
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const line = new THREE.Line(g, boneMat); scene.add(line); boneLines.push(line);
}

let idx = 0, playing = false;
const scrub = document.getElementById("scrub");
const timeLabel = document.getElementById("time-label");
scrub.max = Math.max(0, (DATA.frames || []).length - 1);

function setFrame(i) {
  const frames = DATA.frames || [];
  if (!frames.length) return;
  idx = Math.max(0, Math.min(frames.length - 1, i));
  scrub.value = String(idx);
  const fr = frames[idx];
  const fz = fr.foot_z_m;
  const fzCls = (fz!=null && Math.abs(fz) < 0.12) ? "ok" : "warn";
  timeLabel.innerHTML = `${(fr.t_ms/1000).toFixed(2)}s (#${idx+1}/${frames.length}) · views=${fr.n_views||"?"} · reproj=${fr.reproj_px??"?"}px · foot_z=<span class="${fzCls}">${fz??"?"}</span>m`;
  const joints = fr.joints || [];
  for (let j = 0; j < 17; j++) {
    const p = joints[j];
    if (!p || Number.isNaN(p[0])) { jointMeshes[j].visible = false; continue; }
    jointMeshes[j].visible = true;
    jointMeshes[j].position.copy(c2t(p));
  }
  (DATA.edges || []).forEach((e, bi) => {
    const a = joints[e[0]], b = joints[e[1]];
    const line = boneLines[bi];
    if (!a || !b || Number.isNaN(a[0]) || Number.isNaN(b[0])) { line.visible = false; return; }
    line.visible = true;
    const A = c2t(a), B = c2t(b);
    const pos = line.geometry.attributes.position;
    pos.setXYZ(0, A.x, A.y, A.z); pos.setXYZ(1, B.x, B.y, B.z); pos.needsUpdate = true;
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
(DATA.clips || []).forEach((c, i) => {
  const opt = document.createElement("option");
  opt.value = String(c.release_ms ?? c.start_ms ?? "");
  opt.textContent = `#${i} ${c.action_type||""} @ ${((c.release_ms||0)/1000).toFixed(1)}s`;
  clipSel.appendChild(opt);
});
clipSel.onchange = () => {
  const t = Number(clipSel.value);
  if (!Number.isFinite(t)) return;
  let best=0, bestD=1e18;
  (DATA.frames||[]).forEach((f,i)=>{ const d=Math.abs(f.t_ms-t); if(d<bestD){bestD=d;best=i;} });
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
  if (playing && t - last > 1000 / Math.max(1, DATA.fps || 20)) {
    last = t; setFrame(idx + 1);
    if (idx >= (DATA.frames||[]).length - 1) {
      playing = false; document.getElementById("play").textContent = "播放";
    }
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


def write_viewer(scene: dict, out_html: Path) -> None:
    payload = json.dumps(scene, ensure_ascii=False)
    html = HTML.replace("__TITLE__", f"group_{int(scene.get('group_id', 0)):02d}")
    html = html.replace("__DATA__", payload)
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="1,2,3,4", help="Comma-separated group ids")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out-root", type=Path, default=ROOT / "data/outputs/v1")
    ap.add_argument("--calib", type=Path, default=ROOT / "data/calibration/v2_4cam_zoned")
    ap.add_argument("--videos", type=Path, default=ROOT / "data/test_data_v1")
    args = ap.parse_args()

    groups = [int(x.strip()) for x in args.groups.split(",") if x.strip()]
    summary = []
    for g in groups:
        gdir = args.out_root / f"group_{g:02d}"
        print(f"\n=== group_{g:02d} ===", flush=True)
        scene = process_group_action_skeletons(
            gdir,
            calib_dir=args.calib,
            videos_dir=args.videos,
            group_id=g,
            stride=args.stride,
        )
        json_path = gdir / "skeleton3d_triangulated.json"
        html_path = gdir / "skeleton3d_court_viewer.html"
        json_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
        if scene.get("frames"):
            write_viewer(scene, html_path)
        foot_zs = [f.get("foot_z_m") for f in scene.get("frames") or [] if f.get("foot_z_m") is not None]
        med_foot = float(sorted(foot_zs)[len(foot_zs) // 2]) if foot_zs else None
        info = {
            "group": g,
            "n_frames": scene.get("n_frames"),
            "status": scene.get("status"),
            "offsets_ms": scene.get("offsets_ms"),
            "floor_z_subtracted_m": scene.get("floor_z_subtracted_m"),
            "median_foot_z_after_m": med_foot,
            "json": str(json_path),
            "html": str(html_path) if scene.get("frames") else None,
            "clip_stats": scene.get("clip_stats"),
        }
        summary.append(info)
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)

    out_sum = args.out_root / "skeleton3d_groups_summary.json"
    # Merge with existing summary so partial --groups runs don't wipe others
    if out_sum.exists():
        try:
            prev = json.loads(out_sum.read_text(encoding="utf-8"))
            by_g = {int(x["group"]): x for x in prev if isinstance(x, dict) and "group" in x}
            for item in summary:
                by_g[int(item["group"])] = item
            summary = [by_g[k] for k in sorted(by_g)]
        except Exception:
            pass
    out_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary → {out_sum}", flush=True)


if __name__ == "__main__":
    main()
