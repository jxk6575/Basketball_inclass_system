#!/usr/bin/env python3
"""Build interactive 3D HTML viewer of calibrated camera poses on the court."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.court_model import load_court_model  # noqa: E402
from src.viz.pose3d_scene import _court_mesh  # noqa: E402

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Camera Scene — Calibration</title>
<style>
  :root { --bg:#0b1220; --panel:#151d2c; --text:#e8eef6; --muted:#8b9bb0; --warn:#f0a020; --ok:#3ecf8e; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Segoe UI","PingFang SC","Noto Sans SC",sans-serif; background:var(--bg); color:var(--text); height:100vh; display:flex; flex-direction:column; }
  header { padding:0.75rem 1rem; border-bottom:1px solid #243044; }
  header h1 { margin:0; font-size:1.05rem; }
  .meta { color:var(--muted); font-size:0.82rem; margin-top:0.25rem; }
  .warn { color:var(--warn); }
  #view { flex:1; min-height:360px; }
  #c { width:100%; height:100%; display:block; }
  .bar { padding:0.65rem 1rem; background:var(--panel); border-top:1px solid #243044; font-size:0.85rem; color:var(--muted); }
  table { border-collapse:collapse; margin-top:0.4rem; width:100%; max-width:720px; }
  td, th { text-align:left; padding:0.2rem 0.5rem; border-bottom:1px solid #243044; }
  .bad { color:var(--warn); }
  .good { color:var(--ok); }
</style>
</head>
<body>
<header>
  <h1>标定相机位姿 · 球场坐标系</h1>
  <div class="meta" id="note"></div>
  <div class="meta warn" id="alert"></div>
</header>
<div id="view"><canvas id="c"></canvas></div>
<div class="bar">
  <div>拖拽旋转 · 滚轮缩放 · 右键平移。锥体尖端=光心，开口方向=相机朝向。</div>
  <table id="tbl"><thead><tr><th>相机</th><th>X (m)</th><th>Y (m)</th><th>Z↑ (m)</th><th>重投影</th><th>备注</th></tr></thead><tbody></tbody></table>
</div>
<script type="importmap">
{ "imports": { "three": "https://unpkg.com/three@0.160.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" } }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const DATA = __DATA__;
document.getElementById("note").textContent = DATA.note || "";
const below = (DATA.cameras || []).filter(c => c.z_below_ground);
document.getElementById("alert").textContent = below.length
  ? `警告: ${below.map(c=>c.id).join(", ")} 的光心 Z<0（在地面以下），外参很可能取错了平面 PnP 镜像解，请勿直接用于三角化。`
  : "所有相机光心在地面上方。";

const tbody = document.querySelector("#tbl tbody");
for (const c of (DATA.cameras || [])) {
  const tr = document.createElement("tr");
  const z = c.center[2];
  tr.innerHTML = `<td>${c.id}</td><td>${c.center[0].toFixed(2)}</td><td>${c.center[1].toFixed(2)}</td><td class="${z<0?"bad":"good"}">${z.toFixed(2)}</td><td>${(c.reproj_mean??0).toFixed(1)} px</td><td>${c.z_below_ground?"地下·不可信":"OK"}</td>`;
  tbody.appendChild(tr);
}

const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.position = "absolute";
labelRenderer.domElement.style.top = "0";
labelRenderer.domElement.style.pointerEvents = "none";
document.getElementById("view").appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1220);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
camera.position.set(12, 10, 18);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.5, 7);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(5, 12, 3);
scene.add(dir);

// court (X, Y_floor, Z_up) → three (X, Z_up, Y_floor)
function c2t(p) { return new THREE.Vector3(p[0], p[2] ?? 0, p[1]); }

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(16, 16),
  new THREE.MeshStandardMaterial({ color: 0x1a3040, roughness: 0.95, transparent: true, opacity: 0.85 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.set(0, 0, 7);
scene.add(floor);

// axes helper at origin
scene.add(new THREE.AxesHelper(2.5));
const originLbl = document.createElement("div");
originLbl.textContent = "原点(底线中点)";
originLbl.style.cssText = "color:#8b9bb0;font-size:11px;white-space:nowrap;";
const oObj = new CSS2DObject(originLbl);
oObj.position.set(0, 0.15, 0);
scene.add(oObj);

const lineMat = new THREE.LineBasicMaterial({ color: 0xf5a623 });
for (const seg of (DATA.court?.segments || [])) {
  const g = new THREE.BufferGeometry().setFromPoints([c2t(seg.a), c2t(seg.b)]);
  scene.add(new THREE.Line(g, lineMat));
}
for (const [id, p] of Object.entries(DATA.court?.points || {})) {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(0.07, 10, 10),
    new THREE.MeshStandardMaterial({ color: 0x3ecf8e })
  );
  m.position.copy(c2t(p));
  scene.add(m);
}

const COLORS = { cam_01: 0x5ec8ff, cam_02: 0xff6b8a, cam_03: 0xf5a623 };
for (const cam of (DATA.cameras || [])) {
  const col = COLORS[cam.id] || 0xffffff;
  const C = c2t(cam.center);
  const fwd = new THREE.Vector3(cam.forward[0], cam.forward[2], cam.forward[1]).normalize();
  const color = cam.z_below_ground ? 0xff3333 : col;

  // camera body
  const body = new THREE.Mesh(
    new THREE.SphereGeometry(0.22, 16, 16),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.25 })
  );
  body.position.copy(C);
  scene.add(body);

  // look ray
  const rayLen = 4.0;
  const rayG = new THREE.BufferGeometry().setFromPoints([C, C.clone().add(fwd.clone().multiplyScalar(rayLen))]);
  scene.add(new THREE.Line(rayG, new THREE.LineBasicMaterial({ color })));

  // simple frustum pyramid
  const depth = 2.2, hw = 1.1, hh = 0.7;
  const tip = C.clone();
  const centerFar = C.clone().add(fwd.clone().multiplyScalar(depth));
  // build orthonormal basis from forward
  let up = new THREE.Vector3(cam.up[0], cam.up[2], cam.up[1]).normalize();
  let right = new THREE.Vector3().crossVectors(fwd, up).normalize();
  if (right.lengthSq() < 1e-6) {
    right = new THREE.Vector3(1, 0, 0);
    up = new THREE.Vector3().crossVectors(right, fwd).normalize();
    right = new THREE.Vector3().crossVectors(fwd, up).normalize();
  } else {
    up = new THREE.Vector3().crossVectors(right, fwd).normalize();
  }
  const corners = [
    centerFar.clone().add(right.clone().multiplyScalar(hw)).add(up.clone().multiplyScalar(hh)),
    centerFar.clone().add(right.clone().multiplyScalar(-hw)).add(up.clone().multiplyScalar(hh)),
    centerFar.clone().add(right.clone().multiplyScalar(-hw)).add(up.clone().multiplyScalar(-hh)),
    centerFar.clone().add(right.clone().multiplyScalar(hw)).add(up.clone().multiplyScalar(-hh)),
  ];
  const frMat = new THREE.LineBasicMaterial({ color });
  const edgePairs = [
    [corners[0], corners[1]], [corners[1], corners[2]],
    [corners[2], corners[3]], [corners[3], corners[0]],
    [tip, corners[0]], [tip, corners[1]], [tip, corners[2]], [tip, corners[3]],
  ];
  for (const [pa, pb] of edgePairs) {
    const g = new THREE.BufferGeometry().setFromPoints([pa, pb]);
    scene.add(new THREE.Line(g, frMat));
  }

  // drop line to floor plane (three Y = court Z)
  const drop = new THREE.BufferGeometry().setFromPoints([C, new THREE.Vector3(C.x, 0, C.z)]);
  scene.add(new THREE.Line(drop, new THREE.LineBasicMaterial({ color: 0x445566 })));

  const div = document.createElement("div");
  div.textContent = cam.id + (cam.z_below_ground ? " ⚠地下" : "");
  div.style.cssText = `color:#fff;background:rgba(0,0,0,0.55);padding:2px 6px;border-radius:4px;font-size:12px;border-left:3px solid #${color.toString(16).padStart(6,"0")}`;
  const lab = new CSS2DObject(div);
  lab.position.copy(C.clone().add(new THREE.Vector3(0, 0.45, 0)));
  scene.add(lab);
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    labelRenderer.setSize(w, h);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  }
}
function loop() {
  resize();
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  requestAnimationFrame(loop);
}
loop();
</script>
</body>
</html>
"""


def build_scene(calib_dir: Path) -> dict:
    bundle = json.loads((calib_dir / "cameras.json").read_text(encoding="utf-8"))
    solved = bundle["solved"]["cameras"]
    model = load_court_model()
    court = _court_mesh()
    cams = []
    for cid in ("cam_01", "cam_02", "cam_03"):
        if cid not in solved or solved[cid].get("status") != "ok":
            continue
        r = solved[cid]
        R = np.asarray(r["rotation_matrix"], dtype=np.float64)
        C = np.asarray(r["camera_center_world"], dtype=np.float64)
        forward = (R.T @ np.array([0.0, 0.0, 1.0])).tolist()
        up = (R.T @ np.array([0.0, -1.0, 0.0])).tolist()
        right = (R.T @ np.array([1.0, 0.0, 0.0])).tolist()
        cams.append({
            "id": cid,
            "center": C.tolist(),
            "forward": forward,
            "up": up,
            "right": right,
            "reproj_mean": (r.get("reproj_error_px") or {}).get("mean"),
            "n_points": r.get("n_points"),
            "z_below_ground": bool(C[2] < 0),
        })
    return {
        "axes": model.get("axes"),
        "unit": "meter",
        "court": court,
        "cameras": cams,
        "note": (
            "坐标系: 原点=进攻端底线中点, +X右, +Y向中线, +Z上。"
            "画面中竖直=Z(高度), 进深=Y(向中线)。"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=Path, default=ROOT / "data/calibration/v2_4cam_zoned")
    args = ap.parse_args()
    scene = build_scene(args.calib)
    out_json = args.calib / "camera_scene.json"
    out_html = args.calib / "camera_scene_viewer.html"
    out_json.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    html = HTML.replace("__DATA__", json.dumps(scene, ensure_ascii=False))
    out_html.write_text(html, encoding="utf-8")
    print(json.dumps({
        "html": str(out_html),
        "json": str(out_json),
        "cameras": {
            c["id"]: {
                "center_m": [round(x, 2) for x in c["center"]],
                "z_below_ground": c["z_below_ground"],
                "reproj_px": None if c["reproj_mean"] is None else round(c["reproj_mean"], 1),
            }
            for c in scene["cameras"]
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
