#!/usr/bin/env python3
"""Event-based / manual multi-camera temporal sync CLI.

Examples:
  # Auto event sync
  PYTHONPATH=. python scripts/sync_cameras.py --session <uuid>

  # Manual offsets (ms). Convention: common = local - offset; anchor cam_03 = 0
  PYTHONPATH=. python scripts/sync_cameras.py --session <uuid> \\
      --set-offset cam_01=120 --set-offset cam_02=-80 --set-offset cam_04=250

  # Apply manual only (skip event matching)
  PYTHONPATH=. python scripts/sync_cameras.py --session <uuid> --no-events

  # Dry-run estimate
  PYTHONPATH=. python scripts/sync_cameras.py --session <uuid> --dry-run

  # Inject group GUI offsets into a session
  PYTHONPATH=. python scripts/sync_cameras.py --session <uuid> \\
      --from-group 1 --data-dir data/test_data_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cameras.event_sync import estimate_camera_offsets, get_camera_offsets_ms  # noqa: E402
from src.cameras.group_sync import apply_group_sync_to_session, load_group_sync  # noqa: E402
from src.cameras.temporal import run_temporal_alignment, write_manual_offsets  # noqa: E402


def _parse_offset(s: str) -> tuple[str, float]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected cam_id=ms, got {s!r}")
    cam, raw = s.split("=", 1)
    cam = cam.strip()
    try:
        ms = float(raw.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad ms in {s!r}") from exc
    return cam, ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Event-anchor / manual camera time sync")
    ap.add_argument("--session", required=True, help="Session id under data/sessions/")
    ap.add_argument("--student", action="append", default=None,
                    help="Student id(s); repeatable. Default: auto-detect from pose2d")
    ap.add_argument("--no-events", action="store_true",
                    help="Skip event matching; only write timelines / sync_meta offsets")
    ap.add_argument(
        "--set-offset",
        action="append",
        default=[],
        metavar="CAM=MS",
        help="Manual offset for one camera (repeatable). Writes raw/sync_meta.json",
    )
    ap.add_argument("--replace-manual", action="store_true",
                    help="Replace (not merge) existing manual offsets when using --set-offset")
    ap.add_argument(
        "--from-group",
        type=int,
        default=None,
        help="Load offsets from <data-dir>/sync/group_XX.json into session sync_meta",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "test_data_v3",
        help="Dataset dir for --from-group (default: data/test_data_v3)",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Estimate offsets and print, do not write alignment.json")
    args = ap.parse_args()

    students = args.student

    if args.from_group is not None:
        doc = load_group_sync(args.data_dir, args.from_group)
        if not doc:
            raise SystemExit(
                f"no group sync at {args.data_dir}/sync/group_{args.from_group:02d}.json "
                f"(run scripts/sync_group_gui.py first)"
            )
        path = apply_group_sync_to_session(args.session, doc, data_dir=args.data_dir)
        print(json.dumps({
            "wrote_from_group": str(path),
            "group_id": args.from_group,
            "camera_time_offsets_ms": doc.get("camera_time_offsets_ms"),
        }, ensure_ascii=False, indent=2))

    if args.set_offset:
        manual = dict(_parse_offset(s) for s in args.set_offset)
        path = write_manual_offsets(
            args.session, manual, merge=not args.replace_manual,
        )
        print(json.dumps({
            "wrote_manual": str(path),
            "camera_time_offsets_ms": manual,
            "merged": not args.replace_manual,
        }, ensure_ascii=False, indent=2))

    if args.dry_run:
        doc = estimate_camera_offsets(args.session, student_ids=students)
        print(json.dumps({
            "anchor_camera": doc["anchor_camera"],
            "camera_time_offsets_ms": doc["camera_time_offsets_ms"],
            "quality": doc["quality"],
            "student_ids": doc["student_ids"],
        }, ensure_ascii=False, indent=2))
        return

    out = run_temporal_alignment(
        args.session,
        student_ids=students,
        use_events=not args.no_events,
    )
    offsets = get_camera_offsets_ms(args.session)
    print(json.dumps({
        "alignment": str(out),
        "camera_time_offsets_ms": offsets,
        "events_file": str(out.parent / "events.json"),
        "hint": (
            "手动对齐: --set-offset cam_01=Δt --set-offset cam_04=Δt "
            "(Δt 毫秒; common=local-offset; 锚点 cam_03=0)"
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
