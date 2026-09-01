#!/usr/bin/env python3
"""Formal v2 batch runner for data/test_data_v2.

Highlights vs v1
----------------
- group0 = sequential frontal enrollment from cam_01 (no action pipeline)
- group1+ reuse the shared gallery (multi-student ReID)
- action labels include ``jump_shot`` (jump vs planted free_throw)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_conda = os.environ.get("CONDA_PREFIX")
if _conda:
    os.environ["LD_LIBRARY_PATH"] = f"{_conda}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Reuse v1 helpers (prepare / viz / mode resolution)
from scripts.run_v1_testset import (  # noqa: E402
    discover_groups,
    remux_to_mp4,
    render_group_visualizations,
    resolve_run_mode,
    _ball_track_path,
)

from src.cameras.registry import (  # noqa: E402
    camera_runs_pose2d,
    get_action_segment_camera,
    get_enrollment_camera,
    get_perception_config,
    get_shot_outcome_camera,
)
from src.cameras.temporal import run_temporal_alignment  # noqa: E402
from src.config import data_path  # noqa: E402
from src.identity.enrollment import EnrollmentGallery  # noqa: E402
from src.identity.lineup_enroll import enroll_lineup_from_video  # noqa: E402
from src.identity.sequential_enroll import enroll_sequential_from_video  # noqa: E402
from src.orchestrator.session_pipeline import create_session, register_student  # noqa: E402
from src.output.export import build_group_report, write_session_output  # noqa: E402
from src.perception.camera_pipeline import run_single_camera_perception  # noqa: E402
from src.privacy.consent import grant_consent  # noqa: E402
from src.privacy.db import init_db  # noqa: E402
from src.shot.outcome import run_ball_tracking_on_video, run_shot_outcome_session  # noqa: E402
from src.types import ConsentScope  # noqa: E402


GALLERY_MANIFEST = "gallery_manifest.json"


def _copy_gallery(src_session: str, dst_session: str) -> list[str]:
    """Copy enrollment gallery directory between sessions."""
    src = data_path("enrollment", src_session)
    dst = data_path("enrollment", dst_session)
    if not src.exists():
        raise FileNotFoundError(f"Missing enrollment gallery: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return EnrollmentGallery(dst_session).list_students()


def run_enroll_group(
    group_id: int,
    videos: dict[str, Path],
    out_root: Path,
    *,
    id_prefix: str = "stu",
    expected_persons: int = 6,
    enroll_mode: str = "auto",
) -> dict:
    """group0: build multi-student gallery from enrollment camera.

    enroll_mode:
      - sequential: one-by-one walk-ups (v2)
      - lineup: everyone faces camera together (v3 group0)
      - auto: use lineup when expected_persons<=4 else sequential
    """
    t0 = time.perf_counter()
    group_name = f"group_{group_id:02d}"
    group_dir = out_root / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    init_db()
    session_id = create_session("v2_testset", metadata={"group_id": group_id, "role": "enrollment"})
    enroll_cam = get_enrollment_camera()
    if enroll_cam not in videos:
        enroll_cam = "cam_01" if "cam_01" in videos else next(iter(videos))

    raw_dir = data_path("sessions", session_id, "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Path] = {}
    for cam_id, src in videos.items():
        dst = raw_dir / f"{cam_id}.mp4"
        print(f"  [{group_name}] prepare {cam_id} <- {src.name}")
        prepared[cam_id] = remux_to_mp4(src, dst)

    preview = group_dir / "enroll_preview"
    mode = enroll_mode
    if mode == "auto":
        mode = "lineup" if int(expected_persons) <= 4 else "sequential"
    print(f"  [{group_name}] enroll mode={mode} from {enroll_cam} (expected={expected_persons})")
    if mode == "lineup":
        student_ids = enroll_lineup_from_video(
            session_id,
            prepared[enroll_cam],
            id_prefix=id_prefix,
            expected_persons=expected_persons,
            preview_dir=preview,
        )
    else:
        student_ids = enroll_sequential_from_video(
            session_id,
            prepared[enroll_cam],
            id_prefix=id_prefix,
            preview_dir=preview,
            max_persons=16,
            expected_persons=expected_persons,
        )
    if not student_ids:
        raise RuntimeError(f"No students enrolled from {prepared[enroll_cam]}")

    for sid in student_ids:
        register_student(sid, f"Student {sid}", class_id="v2_testset")
        grant_consent(sid, session_id, [ConsentScope.VIDEO, ConsentScope.FACE, ConsentScope.REPORT])

    # Shared gallery pointer for later groups
    shared = out_root / GALLERY_MANIFEST
    manifest = {
        "enroll_group": group_name,
        "session_id": session_id,
        "enroll_camera": enroll_cam,
        "student_ids": student_ids,
        "n_students": len(student_ids),
        "preview_dir": str(preview),
    }
    shared.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (group_dir / "summary.json").write_text(
        json.dumps(
            {
                **manifest,
                "group_id": group_name,
                "role": "enrollment",
                "timings_sec": {"total": round(time.perf_counter() - t0, 2)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  [{group_name}] enrolled {len(student_ids)}: {student_ids}")
    print(f"  [{group_name}] gallery manifest → {shared}")
    return manifest


def process_action_group(
    group_id: int,
    videos: dict[str, Path],
    out_root: Path,
    *,
    data_dir: Path | None = None,
    stride: int = 2,
    skip_viz: bool = False,
    fast: bool = False,
    shot_ball_only: bool = False,
    gallery_session_id: str | None = None,
    student_ids: list[str] | None = None,
) -> dict:
    """group1+: perception + action + shot with shared multi-student gallery."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    group_name = f"group_{group_id:02d}"
    group_dir = out_root / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    sync_data_dir = Path(data_dir) if data_dir is not None else next(iter(videos.values())).parent

    perc = get_perception_config()
    anchor = get_action_segment_camera()
    assist_stride = int(perc.get("assist_camera_stride", 4)) if fast else stride
    shot_scale = float(perc.get("shot_camera_process_scale", 0.5))

    init_db()
    session_id = create_session(
        "v2_testset",
        metadata={"group_id": group_id, "data_dir": str(sync_data_dir.resolve())},
    )

    # Resolve gallery
    if not gallery_session_id or not student_ids:
        man_path = out_root / GALLERY_MANIFEST
        if not man_path.exists():
            raise SystemExit(
                f"Missing {man_path}. Run group0 enrollment first "
                "(--groups 0) before action groups."
            )
        man = json.loads(man_path.read_text(encoding="utf-8"))
        gallery_session_id = man["session_id"]
        student_ids = list(man["student_ids"])

    student_ids = _copy_gallery(gallery_session_id, session_id)
    primary = student_ids[0]
    for sid in student_ids:
        register_student(sid, f"Student {sid}", class_id="v2_testset")
        grant_consent(sid, session_id, [ConsentScope.VIDEO, ConsentScope.FACE, ConsentScope.REPORT])
    print(f"  [{group_name}] gallery students={student_ids} (from {gallery_session_id})")

    raw_dir = data_path("sessions", session_id, "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Path] = {}
    for cam_id, src in videos.items():
        dst = raw_dir / f"{cam_id}.mp4"
        print(f"  [{group_name}] prepare {cam_id} <- {src.name}")
        prepared[cam_id] = remux_to_mp4(src, dst)

    # Inject GUI / group-level time offsets when present
    from src.cameras.group_sync import apply_group_sync_to_session, load_group_sync

    gdoc = load_group_sync(sync_data_dir, group_id)
    if gdoc:
        sync_path = apply_group_sync_to_session(session_id, gdoc, data_dir=sync_data_dir)
        print(
            f"  [{group_name}] applied group sync offsets "
            f"{gdoc.get('camera_time_offsets_ms')} → {sync_path}"
        )
    else:
        print(f"  [{group_name}] no group sync file (optional: scripts/sync_group_gui.py)")
    timings["prepare"] = time.perf_counter() - t0

    t_pose = time.perf_counter()
    for cam_id, path in prepared.items():
        if not camera_runs_pose2d(cam_id):
            print(f"  [{group_name}] skip pose perception {cam_id} (ball-only)")
            continue
        cam_stride = stride if cam_id == anchor else assist_stride
        print(f"  [{group_name}] perception {cam_id} (stride={cam_stride})")
        run_single_camera_perception(
            session_id, cam_id, path,
            student_ids_filter=student_ids,
            stride=cam_stride,
        )
    timings["perception"] = time.perf_counter() - t_pose

    shot_cam = get_shot_outcome_camera()
    t_ball = time.perf_counter()
    for cam_id, path in prepared.items():
        track_out = _ball_track_path(session_id, cam_id, shot_cam)
        if cam_id == shot_cam:
            ball_stride, ball_scale = 1, 1.0
        elif fast or shot_ball_only:
            ball_stride = max(stride, assist_stride)
            ball_scale = shot_scale
        else:
            ball_stride, ball_scale = stride, 1.0
        print(f"  [{group_name}] ball track {cam_id} (scale={ball_scale}, stride={ball_stride})")
        run_ball_tracking_on_video(
            path, out_json=track_out, stride=ball_stride, process_scale=ball_scale,
            hoop_upper_half_only=(cam_id != shot_cam),
        )
    timings["ball_track"] = time.perf_counter() - t_ball

    t_align = time.perf_counter()
    print(f"  [{group_name}] temporal align (event_anchor)")
    run_temporal_alignment(
        session_id, list(prepared.keys()), student_ids=student_ids, use_events=True,
    )
    timings["align"] = time.perf_counter() - t_align

    t_action = time.perf_counter()
    print(f"  [{group_name}] action (auto-classify) students={student_ids}")
    from src.action.pipeline import run_action_session_auto
    done = run_action_session_auto(session_id, student_ids)
    print(f"  [{group_name}] action students done={done}")
    timings["action"] = time.perf_counter() - t_action

    t_shot = time.perf_counter()
    print(f"  [{group_name}] shot outcome")
    run_shot_outcome_session(session_id)
    timings["shot_outcome"] = time.perf_counter() - t_shot

    t_export = time.perf_counter()
    print(f"  [{group_name}] export JSON")
    motion_path = group_dir / "motion.json"
    motion = write_session_output(
        session_id, motion_path, group_id=group_name, sample_stride=1,
    )
    report = build_group_report(session_id, motion, group_name)
    report_path = group_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    timings["export"] = time.perf_counter() - t_export

    t_skel = time.perf_counter()
    print(f"  [{group_name}] skeleton3d triangulate")
    try:
        from src.pose.action_skeleton3d import process_group_action_skeletons
        from scripts.extract_action_skeletons_3d import write_viewer
        scene = process_group_action_skeletons(
            group_dir, group_id=group_id, stride=max(2, stride),
        )
        skel_path = group_dir / "skeleton3d_triangulated.json"
        skel_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
        if scene.get("frames"):
            write_viewer(scene, group_dir / "skeleton3d_court_viewer.html")
            print(f"  [{group_name}] skeleton3d frames={scene.get('n_frames')} status={scene.get('status')}")
        else:
            print(f"  [{group_name}] skeleton3d empty status={scene.get('status')}")
    except Exception as e:
        print(f"  [{group_name}] skeleton3d failed: {e}")
    timings["skeleton3d"] = time.perf_counter() - t_skel

    viz_paths: dict = {}
    if not skip_viz:
        t_viz = time.perf_counter()
        print(f"  [{group_name}] visualize")
        viz_paths = render_group_visualizations(
            group_dir, session_id, prepared, primary, stride=stride,
            student_ids=student_ids,
        )
        timings["viz"] = time.perf_counter() - t_viz

    act_types = [c.get("action_type") for c in (report.get("clips") or []) if c.get("action_type")]
    dominant = max(set(act_types), key=act_types.count) if act_types else "unknown"
    type_hist: dict[str, int] = {}
    for t in act_types:
        type_hist[t] = type_hist.get(t, 0) + 1

    # Write summary before dashboard (build_dashboard reads summary.json).
    summary = {
        "group_id": group_name,
        "session_id": session_id,
        "student_id": primary,
        "student_ids": student_ids,
        "gallery_session_id": gallery_session_id,
        "action_type": dominant,
        "action_type_hist": type_hist,
        "action_type_source": "auto_classify",
        "clip_count": report["clip_count"],
        "shot_stats": report["shot_stats"],
        "record_count": report["record_count"],
        "timings_sec": {},
        "fast_mode": fast,
        "shot_ball_only": shot_ball_only,
        "mode": "realtime" if (fast and shot_ball_only) else ("full" if not fast else "custom"),
        "outputs": {
            "motion": str(motion_path),
            "report": str(report_path),
            "viz": viz_paths,
        },
    }
    (group_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    t_dash = time.perf_counter()
    try:
        from scripts.build_group_dashboard import build_dashboard
        build_dashboard(group_dir)
        print(f"  [{group_name}] dashboard refreshed")
    except Exception as e:
        print(f"  [{group_name}] dashboard failed: {e}")
    timings["dashboard"] = time.perf_counter() - t_dash

    timings["total"] = time.perf_counter() - t0
    print(f"  [{group_name}] timing(s): " + ", ".join(f"{k}={v:.1f}" for k, v in timings.items()))
    summary["timings_sec"] = {k: round(v, 2) for k, v in timings.items()}
    (group_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Formal v2 test_data_v2 batch runner "
        "(group0=enroll, group1+=actions; modes: realtime/full)",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "test_data_v2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "outputs" / "v2")
    parser.add_argument("--groups", type=str, default="0,1", help="e.g. 0,1 or all")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--mode", choices=["realtime", "full"], default="realtime")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--shot-ball-only", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--with-viz", action="store_true")
    parser.add_argument(
        "--rerender-viz",
        action="store_true",
        help="仅用已有 session 重渲 viz + dashboard（不重跑感知/动作，不删 JSON）",
    )
    parser.add_argument(
        "--enroll-only",
        action="store_true",
        help="Only run enrollment groups (group id 0 by default)",
    )
    parser.add_argument(
        "--expected-persons",
        type=int,
        default=None,
        help="Enrollment target headcount (default: 6; use 4 for v3 A–D)",
    )
    args = parser.parse_args()

    groups = discover_groups(args.data_dir)
    if not groups:
        raise SystemExit(f"No groups found in {args.data_dir}")

    if args.groups != "all":
        wanted = {int(x.strip()) for x in args.groups.split(",") if x.strip()}
        groups = {g: v for g, v in groups.items() if g in wanted}

    expected_persons = args.expected_persons
    if expected_persons is None:
        # Heuristic: v3 dataset → 4 persons (A–D); else v2 default 6
        expected_persons = 4 if "v3" in str(args.data_dir) else 6

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fast, shot_ball_only, skip_viz, mode_label = resolve_run_mode(args)
    print(f"Groups: {sorted(groups.keys())}")
    print(f"Output: {args.out_dir}")
    print(f"Mode: {mode_label} (fast={fast}, shot_ball_only={shot_ball_only}, skip_viz={skip_viz})")
    print(f"Enrollment expected_persons={expected_persons}")

    if args.rerender_viz:
        from scripts.build_group_dashboard import build_dashboard

        n_ok = 0
        for gid in sorted(groups.keys()):
            if gid == 0:
                print(f"SKIP group {gid}: enrollment (enroll_preview already produced at enroll time)")
                continue
            gdir = args.out_dir / f"group_{gid:02d}"
            sp = gdir / "summary.json"
            if not sp.exists():
                print(f"SKIP group {gid}: no summary.json")
                continue
            summary = json.loads(sp.read_text(encoding="utf-8"))
            if summary.get("role") == "enrollment":
                print(f"SKIP group {gid}: enrollment role")
                continue
            sid = summary.get("session_id")
            if not sid:
                print(f"SKIP group {gid}: no session_id")
                continue
            stu = summary.get("student_id") or (summary.get("student_ids") or ["stu_00"])[0]
            stus = summary.get("student_ids") or [stu]
            prepared = {
                cam: data_path("sessions", sid, "raw", f"{cam}.mp4")
                for cam in ("cam_01", "cam_02", "cam_03", "cam_04")
            }
            missing = [c for c, p in prepared.items() if not p.exists()]
            if missing:
                print(f"SKIP group {gid}: missing raw {missing}")
                continue
            print(f"\n=== Re-render viz+dashboard group {gid} (session={sid}) ===")
            viz_paths = render_group_visualizations(
                gdir, sid, prepared, stu, stride=args.stride, student_ids=stus,
            )
            try:
                build_dashboard(gdir)
                print(f"  [{gdir.name}] dashboard refreshed")
            except Exception as e:
                print(f"  [{gdir.name}] dashboard failed: {e}")
            summary.setdefault("outputs", {})["viz"] = viz_paths
            summary["mode"] = summary.get("mode") or mode_label
            sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            n_ok += 1
            print(f"OK group {gid}: viz={list(viz_paths.keys())}")
        print(f"\nRe-rendered viz+dashboard for {n_ok} groups")
        return

    results = []
    gallery_session_id = None
    student_ids = None
    man_path = args.out_dir / GALLERY_MANIFEST
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        gallery_session_id = man.get("session_id")
        student_ids = man.get("student_ids")

    for gid in sorted(groups.keys()):
        print(f"\n=== Group {gid} ===")
        if gid == 0 or args.enroll_only:
            man = run_enroll_group(
                gid, groups[gid], args.out_dir,
                expected_persons=expected_persons,
            )
            gallery_session_id = man["session_id"]
            student_ids = man["student_ids"]
            results.append(man)
            if args.enroll_only and gid == 0:
                continue
            if gid == 0:
                continue
        summary = process_action_group(
            gid, groups[gid], args.out_dir,
            data_dir=args.data_dir,
            stride=args.stride,
            skip_viz=skip_viz,
            fast=fast,
            shot_ball_only=shot_ball_only,
            gallery_session_id=gallery_session_id,
            student_ids=student_ids,
        )
        results.append(summary)
        print(
            f"OK {summary['group_id']}: clips={summary.get('clip_count')} "
            f"types={summary.get('action_type_hist')} students={summary.get('student_ids')}"
        )

    manifest = {
        "version": "2.0.0",
        "mode": mode_label,
        "n_groups": len(results),
        "gallery": {
            "session_id": gallery_session_id,
            "student_ids": student_ids,
        },
        "groups": [
            {
                "group_id": r.get("group_id") or r.get("enroll_group"),
                "session_id": r.get("session_id"),
                "clip_count": r.get("clip_count"),
                "action_type": r.get("action_type"),
                "action_type_hist": r.get("action_type_hist"),
                "n_students": r.get("n_students") or len(r.get("student_ids") or []),
            }
            for r in results
        ],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\nDone. manifest → {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
