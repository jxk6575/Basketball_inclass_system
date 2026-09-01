#!/usr/bin/env python3
"""Build group dashboard JSON: actions, shot FG%, joint angles at phases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import data_path  # noqa: E402
from src.pose.angles import compute_frame_angles, compute_h36m_angles  # noqa: E402
from src.pose.reference_template import kpts133_to_pseudo3d  # noqa: E402
from src.viz.identity_style import (  # noqa: E402
    compute_chrono_display_order,
    format_student_label,
    student_color_hex,
)

ANGLE_KEYS = [
    "right_elbow",
    "left_elbow",
    "right_knee",
    "left_knee",
    "right_wrist",
    "shooting_elbow",
    "shooting_wrist",
]


def _clip_shooting_hand(clip: dict) -> str | None:
    hand = clip.get("shooting_hand")
    if hand in ("left", "right"):
        return hand
    meta = clip.get("metadata") or {}
    hand = meta.get("shooting_hand")
    return hand if hand in ("left", "right") else None


def _load_skeleton3d(group_dir: Path) -> dict | None:
    path = group_dir / "skeleton3d_triangulated.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _angles_from_skeleton3d(
    skel_doc: dict,
    target_ms: float,
    clip_id: str | None = None,
    max_gap_ms: float = 250.0,
    shooting_hand: str | None = None,
) -> dict[str, float] | None:
    """Nearest triangulated frame (prefer same clip_id) → court-world joint angles."""
    frames = skel_doc.get("frames") or []
    if not frames:
        return None
    candidates = frames
    if clip_id:
        same = [f for f in frames if f.get("clip_id") == clip_id]
        if same:
            candidates = same
    best = None
    best_dt = 1e18
    for fr in candidates:
        t = fr.get("t_ms")
        if t is None:
            continue
        dt = abs(float(t) - float(target_ms))
        if dt < best_dt:
            best_dt = dt
            best = fr
    if best is None or best_dt > max_gap_ms:
        return None
    joints = np.asarray(best.get("joints"), dtype=np.float64)
    ang = compute_h36m_angles(joints, shooting_hand=shooting_hand)
    out = {k: round(float(v), 1) for k, v in ang.items() if k in ANGLE_KEYS and v == v}
    return out or None


def _angles_at_frame(
    pose_doc: dict,
    frame_idx: int,
    student_id: str,
    shooting_hand: str | None = None,
) -> dict[str, float] | None:
    """Fallback only: 2D pose → pseudo-3D (marked by caller)."""
    for fr in pose_doc.get("frames", []):
        if int(fr.get("frame", -1)) != int(frame_idx):
            continue
        for p in fr.get("persons") or []:
            if p.get("student_id") != student_id:
                continue
            k = np.asarray(p["keypoints"], dtype=np.float32)
            if k.ndim != 2 or k.shape[0] < 17:
                return None
            k3 = kpts133_to_pseudo3d(k)
            ang = compute_frame_angles(k3, shooting_hand=shooting_hand)
            return {k: round(float(v), 1) for k, v in ang.items() if v == v}
    return None


def _nearest_pose_frame(pose_doc: dict, target: int, student_id: str, max_gap: int = 8) -> int | None:
    frames = []
    for fr in pose_doc.get("frames", []):
        for p in fr.get("persons") or []:
            if p.get("student_id") == student_id:
                frames.append(int(fr["frame"]))
                break
    if not frames:
        return None
    best = min(frames, key=lambda f: abs(f - target))
    return best if abs(best - target) <= max_gap else None


def build_dashboard(group_dir: Path) -> dict:
    summary = json.loads((group_dir / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((group_dir / "report.json").read_text(encoding="utf-8"))
    session_id = summary["session_id"]
    student_id = summary["student_id"]

    pose_path = data_path("sessions", session_id, "perception", "cam_03", "pose2d.json")
    pose_doc = json.loads(pose_path.read_text(encoding="utf-8")) if pose_path.exists() else {"frames": []}
    skel_doc = _load_skeleton3d(group_dir)
    angle_source = "triangulated_3d" if skel_doc and (skel_doc.get("frames")) else "pseudo3d_fallback"

    outcomes = report.get("shot_outcomes") or []
    by_clip = {o.get("clip_id"): o for o in outcomes if o.get("clip_id")}

    shots = []
    angle_series = {k: [] for k in ANGLE_KEYS}  # per-shot release angles for charts

    for i, clip in enumerate(report.get("clips") or [], start=1):
        cid = clip["clip_id"]
        outcome = by_clip.get(cid) or {}
        shooting_hand = _clip_shooting_hand(clip)
        phases = clip.get("phases") or []
        phase_angles: dict = {}
        used_3d = False

        for ph in phases:
            fname = ph.get("name")
            ang = None
            # Prefer court-world 3D
            t_ms = None
            if ph.get("start_ms") is not None and ph.get("end_ms") is not None:
                t_ms = 0.5 * (float(ph["start_ms"]) + float(ph["end_ms"]))
            elif fname == "release" and clip.get("release_ms") is not None:
                t_ms = float(clip["release_ms"])
            elif clip.get("start_ms") is not None and clip.get("end_ms") is not None:
                t_ms = 0.5 * (float(clip["start_ms"]) + float(clip["end_ms"]))
            if skel_doc is not None and t_ms is not None:
                ang = _angles_from_skeleton3d(
                    skel_doc, t_ms, clip_id=cid, max_gap_ms=400.0, shooting_hand=shooting_hand,
                )
                if ang:
                    used_3d = True
            if ang is None:
                fs, fe = ph.get("start"), ph.get("end")
                if fs is None or fe is None:
                    continue
                mid = int((int(fs) + int(fe)) / 2)
                nf = _nearest_pose_frame(pose_doc, mid, student_id)
                if nf is None:
                    continue
                ang = _angles_at_frame(pose_doc, nf, student_id, shooting_hand=shooting_hand)
            if ang:
                phase_angles[fname] = ang

        # release angles (primary)
        release_ang = None
        rel = next((p for p in phases if p.get("name") == "release"), None)
        rel_ms = clip.get("release_ms")
        if rel_ms is None and rel is not None and rel.get("start_ms") is not None:
            rel_ms = 0.5 * (float(rel["start_ms"]) + float(rel.get("end_ms") or rel["start_ms"]))
        if skel_doc is not None and rel_ms is not None:
            release_ang = _angles_from_skeleton3d(
                skel_doc, float(rel_ms), clip_id=cid, max_gap_ms=400.0, shooting_hand=shooting_hand,
            )
            if release_ang:
                used_3d = True
        if release_ang is None and rel is not None:
            mid = int((int(rel["start"]) + int(rel["end"])) / 2)
            nf = _nearest_pose_frame(pose_doc, mid, student_id)
            if nf is not None:
                release_ang = _angles_at_frame(pose_doc, nf, student_id)

        made = outcome.get("made")
        parts = list(clip.get("participant_ids") or [])
        clip_sid = clip.get("student_id") or student_id
        atype = clip.get("action_type")
        # Non-pass: shooter is clip.student_id (participant_ids can disagree / be stale)
        if atype == "pass":
            if not parts:
                parts = [clip_sid]
        else:
            parts = [clip_sid]
        # Display time: release for shots; otherwise clip start (or start–end)
        t_display_ms = clip.get("release_ms")
        if t_display_ms is None:
            t_display_ms = clip.get("start_ms")
        shots.append({
            "shot_index": i,
            "clip_id": cid,
            "student_id": clip_sid,
            "participant_ids": parts,
            "action_type": atype,
            "shooting_hand": shooting_hand,
            "start_ms": round(float(clip["start_ms"]), 0) if clip.get("start_ms") is not None else None,
            "end_ms": round(float(clip["end_ms"]), 0) if clip.get("end_ms") is not None else None,
            "release_ms": round(float(clip["release_ms"]), 0) if clip.get("release_ms") is not None else None,
            "time_ms": round(float(t_display_ms), 0) if t_display_ms is not None else None,
            "made": made,
            "result": "MAKE" if made is True else ("MISS" if made is False else "—"),
            "confidence": round(float(outcome.get("confidence", 0)), 2) if outcome else None,
            "reason": (outcome.get("metadata") or {}).get("reason"),
            "phases": [p.get("name") for p in phases],
            "phase_angles": phase_angles,
            "release_angles": release_ang,
            "angles_source": "triangulated_3d" if used_3d else "pseudo3d_fallback",
        })
        if release_ang:
            for k in ANGLE_KEYS:
                if k in release_ang:
                    angle_series[k].append({"shot": f"#{i}", "deg": release_ang[k]})

    stats = report.get("shot_stats") or {}
    attempts = int(stats.get("attempts", len(shots)))
    makes = int(stats.get("makes", sum(1 for s in shots if s["made"] is True)))
    misses = int(stats.get("misses", sum(1 for s in shots if s["made"] is False)))
    fg = round(100.0 * makes / attempts, 1) if attempts else 0.0

    enrolled = list(summary.get("student_ids") or [student_id])
    display_order = compute_chrono_display_order(
        report.get("clips") or shots, enrolled_ids=enrolled,
    )
    for sh in shots:
        sid = sh.get("student_id")
        sh["display_label"] = format_student_label(sid)
        if sh.get("action_type") == "pass" and len(sh.get("participant_ids") or []) >= 2:
            a, b = sh["participant_ids"][0], sh["participant_ids"][1]
            sh["display_label"] = (
                f"{format_student_label(a)} → {format_student_label(b)}"
            )
        sh["color_hex"] = student_color_hex(sid)

    student_legend = [
        {
            "student_id": sid,
            "label": format_student_label(sid),
            "color_hex": student_color_hex(sid),
        }
        for sid in display_order
    ]

    # mean release angles across makes vs misses
    def mean_angles(subset):
        acc: dict[str, list[float]] = {k: [] for k in ANGLE_KEYS}
        for s in subset:
            ang = s.get("release_angles") or {}
            for k in ANGLE_KEYS:
                if k in ang:
                    acc[k].append(ang[k])
        return {k: round(sum(v) / len(v), 1) for k, v in acc.items() if v}

    dashboard = {
        "group_id": report.get("group_id") or summary.get("group_id"),
        "session_id": session_id,
        "student_id": student_id,
        "student_ids": display_order or enrolled,
        "student_ids_enrolled": enrolled,
        "display_order": display_order,
        "student_legend": student_legend,
        "generated_at": report.get("generated_at"),
        "angle_source": angle_source,
        "angle_source_note": (
            "court-world triangulated H36M-17"
            if angle_source == "triangulated_3d"
            else "cam_03 2D pseudo-3D fallback (no skeleton3d_triangulated.json)"
        ),
        "summary": {
            "clip_count": report.get("clip_count", len(shots)),
            "attempts": attempts,
            "makes": makes,
            "misses": misses,
            "fg_pct": fg,
            "record_count": report.get("record_count"),
        },
        "shots": shots,
        "release_angle_series": angle_series,
        "mean_release_angles_all": mean_angles(shots),
        "mean_release_angles_make": mean_angles([s for s in shots if s["made"] is True]),
        "mean_release_angles_miss": mean_angles([s for s in shots if s["made"] is False]),
        "viz": summary.get("outputs", {}).get("viz", {}),
        "ball_model": summary.get("ball_model") or "Basketball_v1.pt @960",
    }
    out = group_dir / "dashboard.json"
    out.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = write_dashboard_html(group_dir, dashboard)
    dashboard["_html"] = str(html_path)
    return dashboard


def _rel_media(group_dir: Path, path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        return str(p.relative_to(group_dir.resolve())).replace("\\", "/")
    except ValueError:
        # keep basename under viz/ if absolute path is outside
        name = p.name
        if (group_dir / "viz" / name).exists():
            return f"viz/{name}"
        return p.as_uri() if p.exists() else None


def write_dashboard_html(group_dir: Path, dashboard: dict) -> Path:
    """Self-contained HTML page (embedded JSON) — open in any browser."""
    viz = dashboard.get("viz") or {}
    media = {
        "phases": _rel_media(group_dir, viz.get("phases")),
        "cam_03": _rel_media(group_dir, viz.get("cam_03")),
        "cam_04": _rel_media(group_dir, viz.get("cam_04")),
        "cam_01": _rel_media(group_dir, viz.get("cam_01")),
        "cam_02": _rel_media(group_dir, viz.get("cam_02")),
    }
    payload = {**dashboard, "media": media}
    payload.pop("_html", None)
    data_js = json.dumps(payload, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Basketball Dashboard — {dashboard.get("group_id", "")}</title>
<style>
  :root {{
    --bg: #0f1419;
    --panel: #1a222c;
    --text: #e8eef4;
    --muted: #8b9aab;
    --make: #3ecf8e;
    --miss: #f07178;
    --accent: #f5a623;
    --line: #2a3542;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1e2a38 0%, var(--bg) 55%);
    color: var(--text); min-height: 100vh;
  }}
  header {{
    padding: 1.25rem 1.5rem 0.75rem; border-bottom: 1px solid var(--line);
    display: flex; flex-wrap: wrap; gap: 1rem; align-items: baseline; justify-content: space-between;
  }}
  header h1 {{ margin: 0; font-size: 1.35rem; font-weight: 650; letter-spacing: 0.02em; }}
  header .meta {{ color: var(--muted); font-size: 0.85rem; }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.75rem; padding: 1rem 1.5rem;
  }}
  .stat {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 0.85rem 1rem;
  }}
  .stat .label {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }}
  .stat .value {{ font-size: 1.55rem; font-weight: 700; margin-top: 0.2rem; }}
  .stat .value.fg {{ color: var(--accent); }}
  main {{ padding: 0 1.5rem 2rem; display: grid; gap: 1.25rem;
    grid-template-columns: 1.2fr 1fr; }}
  @media (max-width: 960px) {{ main {{ grid-template-columns: 1fr; }} }}
  section {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 1rem 1.1rem;
  }}
  section h2 {{ margin: 0 0 0.75rem; font-size: 1rem; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.45rem 0.4rem; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--muted); font-weight: 500; font-size: 0.75rem; text-transform: uppercase; }}
  tr:hover td {{ background: rgba(255,255,255,0.03); }}
  .badge {{
    display: inline-block; padding: 0.12rem 0.45rem; border-radius: 4px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.04em;
  }}
  .badge.make {{ background: rgba(62,207,142,0.18); color: var(--make); }}
  .badge.miss {{ background: rgba(240,113,120,0.18); color: var(--miss); }}
  .badge.unk {{ background: rgba(139,154,171,0.18); color: var(--muted); }}
  video {{ width: 100%; border-radius: 8px; background: #000; max-height: 360px; }}
  .media-tabs {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem; }}
  .media-tabs button {{
    background: transparent; border: 1px solid var(--line); color: var(--muted);
    border-radius: 6px; padding: 0.3rem 0.65rem; cursor: pointer; font-size: 0.8rem;
  }}
  .media-tabs button.active {{ border-color: var(--accent); color: var(--text); }}
  .angles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.5rem; }}
  .angle-card {{
    border: 1px solid var(--line); border-radius: 8px; padding: 0.55rem 0.7rem;
  }}
  .angle-card .k {{ color: var(--muted); font-size: 0.72rem; }}
  .angle-card .v {{ font-size: 1.1rem; font-weight: 600; }}
  .bars {{ display: flex; flex-direction: column; gap: 0.55rem; margin-top: 0.5rem; }}
  .bar-row {{ display: grid; grid-template-columns: 70px 1fr 40px; gap: 0.5rem; align-items: center; font-size: 0.8rem; }}
  .bar-track {{ height: 8px; background: #243040; border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: linear-gradient(90deg, #3a7bd5, var(--accent)); border-radius: 4px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 0.55rem; margin: 0.25rem 0 0.85rem; }}
  .legend-item {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.03); border: 1px solid var(--line);
    border-radius: 6px; padding: 0.25rem 0.55rem; font-size: 0.82rem;
  }}
  .swatch {{ width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }}
  .sid-chip {{
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-weight: 600; font-size: 0.85rem;
  }}
  footer {{ color: var(--muted); font-size: 0.75rem; padding: 0 1.5rem 1.5rem; }}
</style>
</head>
<body>
<header>
  <div>
    <h1 id="title">Basketball Dashboard</h1>
    <div class="meta" id="meta"></div>
  </div>
</header>
<div class="stats" id="stats"></div>
<main>
  <section>
    <h2>动作序列</h2>
    <div class="legend" id="student-legend"></div>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>#</th><th>时间</th><th>学生</th><th>动作</th><th>手</th><th>结果</th>
            <th>肘角°</th><th>膝角°</th><th>置信度</th>
          </tr>
        </thead>
        <tbody id="shots-body"></tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>可视化</h2>
    <div class="media-tabs" id="media-tabs"></div>
    <video id="player" controls playsinline></video>
    <p class="meta" id="media-hint" style="margin:0.5rem 0 0"></p>
  </section>
  <section>
    <h2>出手关节角（均值 · <span id="angle-src-label">三维</span>）</h2>
    <p class="muted" id="angle-src-note" style="margin-top:-0.4rem;font-size:0.85rem"></p>
    <div class="angles" id="mean-angles"></div>
  </section>
  <section>
    <h2>各次出手 — 右肘角</h2>
    <div class="bars" id="elbow-bars"></div>
  </section>
</main>
<footer id="footer"></footer>
<script>
const DATA = {data_js};

function fmtSec(ms) {{
  if (ms == null) return "—";
  return (ms / 1000).toFixed(1) + "s";
}}

function fmtTime(sh) {{
  // Prefer release for shots; else start; show range when start+end and no release
  if (sh.release_ms != null) return fmtSec(sh.release_ms);
  if (sh.start_ms != null && sh.end_ms != null && sh.action_type !== "free_throw" && sh.action_type !== "layup") {{
    return fmtSec(sh.start_ms) + "–" + fmtSec(sh.end_ms);
  }}
  if (sh.time_ms != null) return fmtSec(sh.time_ms);
  if (sh.start_ms != null) return fmtSec(sh.start_ms);
  return "—";
}}

function fmtStudents(sh) {{
  if (sh.display_label) return sh.display_label;
  const parts = sh.participant_ids || (sh.student_id ? [sh.student_id] : []);
  if (!parts.length) return "—";
  if (sh.action_type === "pass" && parts.length >= 2) {{
    return parts[0] + " → " + parts[1];
  }}
  return parts.join(", ");
}}

function studentCell(sh) {{
  const label = fmtStudents(sh);
  const color = sh.color_hex || "#8b9aab";
  return `<span class="sid-chip"><span class="swatch" style="background:${{color}}"></span>${{label}}</span>`;
}}

function badge(result) {{
  const cls = result === "MAKE" ? "make" : (result === "MISS" ? "miss" : "unk");
  return `<span class="badge ${{cls}}">${{result}}</span>`;
}}

(function render() {{
  const s = DATA.summary || {{}};
  document.getElementById("title").textContent =
    `Basketball Dashboard — ${{DATA.group_id || ""}}`;
  const legend = DATA.student_legend || [];
  const displayOrder = DATA.display_order || DATA.student_ids || [DATA.student_id];
  const stus = (legend.length
    ? legend.map(x => x.label || x.student_id)
    : (displayOrder || []).filter(Boolean)
  ).join(", ");
  document.getElementById("meta").textContent =
    `students ${{stus}} · session ${{(DATA.session_id || "").slice(0, 8)}}… · ${{DATA.ball_model || ""}}`;

  const legendEl = document.getElementById("student-legend");
  if (legendEl) {{
    legendEl.innerHTML = (legend.length ? legend : (displayOrder || []).map(sid => ({{
      student_id: sid, label: sid, color_hex: "#8b9aab"
    }}))).map(item =>
      `<span class="legend-item"><span class="swatch" style="background:${{item.color_hex || "#8b9aab"}}"></span>${{item.label || item.student_id}}</span>`
    ).join("");
  }}

  const stats = [
    ["Attempts", s.attempts],
    ["Makes", s.makes],
    ["Misses", s.misses],
    ["FG%", (s.fg_pct != null ? s.fg_pct + "%" : "—"), "fg"],
  ];
  document.getElementById("stats").innerHTML = stats.map(([label, val, cls]) =>
    `<div class="stat"><div class="label">${{label}}</div><div class="value ${{cls || ""}}">${{val ?? "—"}}</div></div>`
  ).join("");

  const tbody = document.getElementById("shots-body");
  tbody.innerHTML = (DATA.shots || []).map(sh => {{
    const elbow = (sh.release_angles || {{}}).shooting_elbow ?? (sh.release_angles || {{}}).right_elbow;
    const knee = (sh.release_angles || {{}}).right_knee;
    const hand = sh.shooting_hand === "left" ? "左" : (sh.shooting_hand === "right" ? "右" : "—");
    return `<tr>
      <td>${{sh.shot_index}}</td>
      <td>${{fmtTime(sh)}}</td>
      <td>${{studentCell(sh)}}</td>
      <td>${{sh.action_type || "—"}}</td>
      <td>${{hand}}</td>
      <td>${{badge(sh.result)}}</td>
      <td>${{elbow != null ? elbow.toFixed(0) : "—"}}</td>
      <td>${{knee != null ? knee.toFixed(0) : "—"}}</td>
      <td>${{sh.confidence != null ? sh.confidence.toFixed(2) : "—"}}</td>
    </tr>`;
  }}).join("");

  const mean = DATA.mean_release_angles_all || {{}};
  const labels = {{
    right_elbow: "右肘", left_elbow: "左肘",
    right_knee: "右膝", left_knee: "左膝", right_wrist: "右腕",
  }};
  const srcLabel = document.getElementById("angle-src-label");
  const srcNote = document.getElementById("angle-src-note");
  if (srcLabel) {{
    srcLabel.textContent = (DATA.angle_source === "triangulated_3d") ? "球场三维真值" : "2D伪三维回退";
  }}
  if (srcNote) {{
    srcNote.textContent = DATA.angle_source_note || "";
  }}
  document.getElementById("mean-angles").innerHTML = Object.keys(labels).map(k =>
    `<div class="angle-card"><div class="k">${{labels[k]}}</div><div class="v">${{mean[k] != null ? mean[k] + "°" : "—"}}</div></div>`
  ).join("");

  const series = (DATA.release_angle_series || {{}}).shooting_elbow
    || (DATA.release_angle_series || {{}}).right_elbow || [];
  const maxDeg = Math.max(180, ...series.map(x => x.deg || 0));
  document.getElementById("elbow-bars").innerHTML = series.map(x => {{
    const pct = Math.round(100 * (x.deg || 0) / maxDeg);
    return `<div class="bar-row"><span>${{x.shot}}</span><div class="bar-track"><div class="bar-fill" style="width:${{pct}}%"></div></div><span>${{Math.round(x.deg)}}°</span></div>`;
  }}).join("") || "<div class='meta'>无角度数据</div>";

  const media = DATA.media || {{}};
  const mediaOrder = ["phases", "cam_03", "cam_04", "cam_01", "cam_02"];
  const names = {{ phases: "四宫格 phases", cam_03: "cam_03 姿态", cam_04: "cam_04 球/筐", cam_01: "cam_01", cam_02: "cam_02" }};
  const tabs = document.getElementById("media-tabs");
  const player = document.getElementById("player");
  const hint = document.getElementById("media-hint");
  let first = null;
  mediaOrder.forEach(key => {{
    const src = media[key];
    if (!src) return;
    if (!first) first = {{ key, src }};
    const btn = document.createElement("button");
    btn.textContent = names[key] || key;
    btn.onclick = () => {{
      [...tabs.children].forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      player.src = src;
      hint.textContent = src + " （相对本 HTML 路径；请从 group 目录打开或起本地服务）";
    }};
    tabs.appendChild(btn);
  }});
  if (first) {{
    tabs.querySelector("button")?.classList.add("active");
    player.src = first.src;
    hint.textContent = first.src;
  }} else {{
    hint.textContent = "无可播放视频（viz 路径缺失）";
  }}

  document.getElementById("footer").textContent =
    `生成时间 ${{DATA.generated_at || "—"}} · 数据已内嵌，可直接双击打开本文件`;
}})();
</script>
</body>
</html>
"""
    out = group_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-dir", type=Path, default=ROOT / "data/outputs/v1/group_01")
    parser.add_argument("--all-v1", action="store_true", help="Build for all group_* under outputs/v1")
    args = parser.parse_args()

    dirs = sorted((ROOT / "data/outputs/v1").glob("group_*")) if args.all_v1 else [args.group_dir]
    for gdir in dirs:
        if not (gdir / "report.json").exists():
            print(f"skip {gdir.name}: no report.json")
            continue
        d = build_dashboard(gdir)
        print(json.dumps({
            "group": d["group_id"],
            "fg_pct": d["summary"]["fg_pct"],
            "attempts": d["summary"]["attempts"],
            "makes": d["summary"]["makes"],
            "shots": [
                {
                    "i": s["shot_index"],
                    "type": s["action_type"],
                    "result": s["result"],
                    "t_s": round((s["release_ms"] or 0) / 1000, 1) if s.get("release_ms") else None,
                    "elbow": (s.get("release_angles") or {}).get("right_elbow"),
                    "knee": (s.get("release_angles") or {}).get("right_knee"),
                }
                for s in d["shots"]
            ],
            "json": str(gdir / "dashboard.json"),
            "html": str(gdir / "dashboard.html"),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
