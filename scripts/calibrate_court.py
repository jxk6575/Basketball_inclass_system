#!/usr/bin/env python3
"""
Court-landmark calibration for cam_01–03.

各相机标注点（见 configs/calibration/court_landmarks_fiba.yaml）:
  cam_01: 底线右角、限制区底线右、限制区罚球线右、罚球圈左右交点
  cam_02: 底线左角、限制区底线左右、限制区罚球线左右、罚球圈左右交点
  cam_03: 中圈-中线右、限制区罚球线右、罚球圈右交点、罚球线中点

Workflow
--------
  python scripts/calibrate_court.py extract-frames --videos data/test_data_v1 --group 1
  python scripts/calibrate_court.py annotate-all --frames data/calibration/frames
  python scripts/calibrate_court.py solve --ann data/calibration/annotations.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.annotate import (  # noqa: E402
    annotate_camera_gui,
    auto_fill_from_seeds,
    empty_annotation_doc,
    load_annotations,
    save_annotations,
)
from src.calibration.court_model import (  # noqa: E402
    annotation_order_for_camera,
    landmark_name,
    load_court_model,
)
from src.calibration.solve import export_calibration, solve_all_cameras  # noqa: E402


def cmd_list_landmarks(_: argparse.Namespace) -> None:
    model = load_court_model()
    print(f"standard={model.get('standard')}  axes={model.get('axes')}")
    print()
    for cam in ("cam_01", "cam_02", "cam_03"):
        order = annotation_order_for_camera(model, cam)
        print(f"=== {cam} 标注顺序 ({len(order)} 点) ===")
        for i, pid in enumerate(order, 1):
            xyz = model["landmarks"][pid]["xyz"]
            print(f"  {i}. {landmark_name(model, pid):20s}  {pid:16s}  xyz={xyz}")
        print()


def cmd_extract_frames(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.videos)
    cams = ("cam_01", "cam_02", "cam_03", "cam_04") if args.include_cam04 else ("cam_01", "cam_02", "cam_03")
    mapping = {
        "cam_01": ["cam_01.mp4", "1-1.mkv", "4-1.mkv"],
        "cam_02": ["cam_02.mp4", "1-2.mkv", "4-2.mkv"],
        "cam_03": ["cam_03.mp4", "1-3.mkv", "4-3.mkv"],
        "cam_04": ["cam_04.mp4", "1-4.mkv", "4-4.mkv"],
    }
    for cam_id in cams:
        candidates = [video_dir / n for n in mapping.get(cam_id, [])]
        candidates += list(video_dir.glob(f"{cam_id}.*"))
        if args.group is not None:
            g = int(args.group)
            c = int(cam_id.split("_")[1])
            candidates.append(video_dir / f"{g}-{c}.mkv")
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            print(f"  skip {cam_id}: no video in {video_dir}")
            continue
        cap = cv2.VideoCapture(str(src))
        start = int(args.frame)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        picked = None
        # v2 片头常有彩条：命中坏帧时向后扫
        for fi in range(start, (min(n, start + 400) if n else start + 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            small = cv2.resize(frame, (160, 90))
            stripe = float(small.mean(axis=0)[:, 0].std())
            if stripe > 55 and fi < start + 80:
                continue
            picked = (fi, frame)
            break
        cap.release()
        if picked is None:
            print(f"  fail read {src}")
            continue
        fi, frame = picked
        dst = out / f"{cam_id}.jpg"
        cv2.imwrite(str(dst), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"  {cam_id} ← {src.name} frame={fi} → {dst}")


def cmd_annotate(args: argparse.Namespace) -> None:
    annotate_camera_gui(
        Path(args.image),
        camera_id=args.camera,
        annotations_path=Path(args.ann),
    )


def cmd_annotate_all(args: argparse.Namespace) -> None:
    frames = Path(args.frames)
    ann = Path(args.ann)
    for cam_id in ("cam_01", "cam_02", "cam_03"):
        img = frames / f"{cam_id}.jpg"
        if not img.exists():
            print(f"skip {cam_id}: missing {img}")
            continue
        print(f"\n—— 开始标注 {cam_id} ——")
        annotate_camera_gui(img, camera_id=cam_id, annotations_path=ann)
    print(f"\n全部完成，标注文件: {ann}")
    print("下一步: python scripts/calibrate_court.py solve --ann", ann)


def cmd_auto_fill(args: argparse.Namespace) -> None:
    model = load_court_model()
    ann_path = Path(args.ann)
    doc = load_annotations(ann_path) if ann_path.exists() else empty_annotation_doc([args.camera])
    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    h, w = img.shape[:2]
    doc.setdefault("frames", {})[args.camera] = str(args.image)
    doc.setdefault("image_size", {})[args.camera] = [w, h]
    seeds = dict((doc.get("points") or {}).get(args.camera) or {})
    order = annotation_order_for_camera(model, args.camera)
    filled = auto_fill_from_seeds(img, seeds, model=model, target_ids=order)
    doc.setdefault("points", {})[args.camera] = filled
    save_annotations(doc, ann_path)
    print(json.dumps({
        "camera": args.camera,
        "seeds": len(seeds),
        "filled": len(filled),
        "points": filled,
        "out": str(ann_path),
    }, ensure_ascii=False, indent=2))


def cmd_solve(args: argparse.Namespace) -> None:
    doc = load_annotations(Path(args.ann))
    solved = solve_all_cameras(
        doc,
        estimate_distortion=not args.no_distortion,
        share_intrinsics_when_weak=not args.no_share_intrinsics,
    )
    out = export_calibration(solved, Path(args.out), annotations=doc)

    per = {}
    for cid, r in (solved.get("cameras") or {}).items():
        per[cid] = {
            "status": r.get("status"),
            "n_points": r.get("n_points"),
            "mean_reproj_px": (r.get("reproj_error_px") or {}).get("mean"),
            "intrinsic_source": (r.get("intrinsics") or {}).get("source"),
            "dist_coeffs": (r.get("intrinsics") or {}).get("dist_coeffs"),
            "camera_center_world_m": r.get("camera_center_world"),
            "point_ids": r.get("point_ids"),
        }
    print(json.dumps({
        "summary": solved.get("summary"),
        "per_camera": per,
        "export": str(out),
        "centers_file": str(Path(args.out) / "camera_centers_world.json"),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Court landmark calibration (cam_01–03)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list-landmarks", help="Show per-camera annotation checklist")
    s.set_defaults(func=cmd_list_landmarks)

    s = sub.add_parser("extract-frames", help="Grab stills from videos")
    s.add_argument("--videos", type=Path, required=True)
    s.add_argument("--out", type=Path, default=ROOT / "data/calibration/frames_v2")
    s.add_argument("--frame", type=int, default=60, help="start frame (skips color bars)")
    s.add_argument("--group", type=int, default=None, help="group id → {g}-{cam}.mkv")
    s.add_argument("--include-cam04", action="store_true", dest="include_cam04",
                   help="also extract cam_04 (usually hoop closeup; not for court PnP)")
    s.set_defaults(func=cmd_extract_frames)

    s = sub.add_parser("annotate", help="GUI annotate one camera")
    s.add_argument("--image", type=Path, required=True)
    s.add_argument("--camera", required=True, choices=["cam_01", "cam_02", "cam_03", "cam_04"])
    s.add_argument("--ann", type=Path, default=ROOT / "data/calibration/annotations_v2.json")
    s.set_defaults(func=cmd_annotate)

    s = sub.add_parser("annotate-all", help="Annotate cam_01→02→03 in sequence")
    s.add_argument("--frames", type=Path, default=ROOT / "data/calibration/frames_v2")
    s.add_argument("--ann", type=Path, default=ROOT / "data/calibration/annotations_v2.json")
    s.set_defaults(func=cmd_annotate_all)

    s = sub.add_parser("auto-fill", help="PnP+snap fill remaining points for one camera")
    s.add_argument("--image", type=Path, required=True)
    s.add_argument("--camera", required=True)
    s.add_argument("--ann", type=Path, default=ROOT / "data/calibration/annotations.json")
    s.set_defaults(func=cmd_auto_fill)

    s = sub.add_parser("solve", help="Estimate K/dist + extrinsics + camera centers")
    s.add_argument("--ann", type=Path, default=ROOT / "data/calibration/annotations_v2.json")
    s.add_argument("--out", type=Path, default=ROOT / "data/calibration/v2_4cam_zoned")
    s.add_argument("--no-distortion", action="store_true", help="Skip planar intrinsic/distortion fit")
    s.add_argument("--no-share-intrinsics", action="store_true",
                   help="Do not reuse cam01/02 K,D for weak cams (e.g. cam03)")
    s.set_defaults(func=cmd_solve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
