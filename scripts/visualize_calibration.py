#!/usr/bin/env python3
"""
Visualize court-landmark calibration quality: reproject 3D points onto each camera image.

  python scripts/visualize_calibration.py \
    --ann data/calibration/annotations.json \
    --calib data/calibration/v2_4cam_zoned \
    --out data/calibration/reproj_preview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.court_model import landmark_xyz, load_court_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--calib", type=Path, default=ROOT / "data/calibration/v2_4cam_zoned")
    ap.add_argument("--out", type=Path, default=ROOT / "data/calibration/reproj_preview")
    args = ap.parse_args()

    ann = json.loads(args.ann.read_text(encoding="utf-8"))
    calib_bundle = args.calib / "cameras.json"
    if not calib_bundle.exists():
        raise SystemExit(f"missing {calib_bundle} — run calibrate_court.py solve first")
    solved = json.loads(calib_bundle.read_text(encoding="utf-8"))["solved"]["cameras"]
    model = load_court_model()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {}
    for cam_id, res in solved.items():
        if res.get("status") != "ok":
            continue
        frame_path = (ann.get("frames") or {}).get(cam_id)
        if not frame_path or not Path(frame_path).exists():
            print(f"skip {cam_id}: no frame image")
            continue
        img = cv2.imread(frame_path)
        K = np.asarray(res["intrinsics"]["camera_matrix"], dtype=np.float64)
        D = np.asarray(res["intrinsics"].get("dist_coeffs") or [0, 0, 0, 0, 0], dtype=np.float64)
        rvec = np.asarray(res["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(res["tvec"], dtype=np.float64).reshape(3, 1)
        obs = (ann.get("points") or {}).get(cam_id) or {}

        # draw GT annotations (green) and reprojections (red)
        for pid, uv in obs.items():
            cv2.circle(img, (int(uv[0]), int(uv[1])), 6, (0, 220, 0), 2)
            cv2.putText(img, pid, (int(uv[0]) + 6, int(uv[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)

        ids = list(obs.keys())
        if ids:
            obj = np.stack([landmark_xyz(model, p) for p in ids]).astype(np.float64)
            proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, D)
            errs = []
            for pid, pr in zip(ids, proj.reshape(-1, 2)):
                cv2.circle(img, (int(pr[0]), int(pr[1])), 5, (0, 0, 255), 2)
                gt = np.asarray(obs[pid][:2], dtype=np.float64)
                e = float(np.linalg.norm(pr - gt))
                errs.append(e)
                cv2.line(img, (int(gt[0]), int(gt[1])), (int(pr[0]), int(pr[1])), (255, 180, 0), 1)
            mean_e = float(np.mean(errs)) if errs else None
        else:
            mean_e = None

        out_img = args.out / f"{cam_id}_reproj.jpg"
        cv2.imwrite(str(out_img), img)
        report[cam_id] = {"mean_reproj_px": mean_e, "image": str(out_img), "n": len(ids)}
        print(f"{cam_id}: mean_reproj={mean_e} → {out_img}")

    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
