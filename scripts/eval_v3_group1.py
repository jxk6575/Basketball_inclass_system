#!/usr/bin/env python3
"""Evaluate v3 group1 free-throw detections vs annotated ground truth.

GT (cam_03 local seconds, from docs/v3测试集动作真值.md):
  12 A, 17 B, 21 C, 24 D, 28 A, 32 B, 35 C, 39 D,
  43 A, 47 B, 50 C, 54 D, 58 A, 62 B
→ 14 × free_throw, persons A–D cycling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (t_sec, person_slot) — person_slot 0=A .. 3=D
GROUP1_FT_GT: list[tuple[float, int]] = [
    (12, 0), (17, 1), (21, 2), (24, 3),
    (28, 0), (32, 1), (35, 2), (39, 3),
    (43, 0), (47, 1), (50, 2), (54, 3),
    (58, 0), (62, 1),
]
PERSON_LABEL = "ABCD"


def _clip_time_s(c: dict) -> float | None:
    if c.get("release_ms") is not None:
        return float(c["release_ms"]) / 1000.0
    if c.get("start_ms") is not None and c.get("end_ms") is not None:
        return 0.5 * (float(c["start_ms"]) + float(c["end_ms"])) / 1000.0
    if c.get("start_ms") is not None:
        return float(c["start_ms"]) / 1000.0
    for key in ("release_common_ms", "t_center_ms", "t0_ms"):
        if c.get(key) is not None:
            return float(c[key]) / 1000.0
    meta = c.get("metadata") or {}
    for key in ("release_ms", "release_common_ms", "anchor_ms", "t_ms"):
        if meta.get(key) is not None:
            return float(meta[key]) / 1000.0
    return None


def _load_clips(group_dir: Path) -> list[dict]:
    for name in ("report.json", "motion.json", "summary.json"):
        p = group_dir / name
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        if name == "summary.json":
            continue
        clips = doc.get("clips") or (doc.get("motion") or {}).get("clips") or []
        if clips:
            return list(clips)
    # fallback: motion next to summary
    return []


def evaluate_group1(group_dir: Path, *, match_tol_s: float = 2.5) -> dict:
    summary_path = group_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    clips = _load_clips(group_dir)
    hist = dict(summary.get("action_type_hist") or {})
    if not hist and clips:
        for c in clips:
            t = str(c.get("action_type") or "unknown")
            hist[t] = hist.get(t, 0) + 1

    ft_clips = [c for c in clips if str(c.get("action_type") or "") == "free_throw"]
    other = [c for c in clips if str(c.get("action_type") or "") != "free_throw"]

    # Map student_id → most common person slot by nearest GT time
    timed = []
    for c in ft_clips:
        t = _clip_time_s(c)
        if t is None:
            continue
        timed.append((t, str(c.get("student_id") or "?"), c))

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    pairs: list[dict] = []
    # greedy nearest match
    candidates: list[tuple[float, int, int]] = []  # |dt|, gt_i, pred_i
    for gi, (gt_t, _slot) in enumerate(GROUP1_FT_GT):
        for pi, (pt, _sid, _c) in enumerate(timed):
            dt = abs(pt - gt_t)
            if dt <= match_tol_s:
                candidates.append((dt, gi, pi))
    candidates.sort()
    for dt, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        gt_t, slot = GROUP1_FT_GT[gi]
        pt, sid, _ = timed[pi]
        pairs.append({
            "gt_s": gt_t,
            "gt_person": PERSON_LABEL[slot],
            "pred_s": round(pt, 2),
            "pred_student": sid,
            "dt_s": round(pt - gt_t, 2),
        })

    # Consistency: same student should map to same person letter
    sid_to_letters: dict[str, set[str]] = {}
    for p in pairs:
        sid_to_letters.setdefault(p["pred_student"], set()).add(p["gt_person"])

    n_gt = len(GROUP1_FT_GT)
    n_hit = len(matched_gt)
    n_pred = len(ft_clips)
    false_types = {k: v for k, v in hist.items() if k != "free_throw" and v}
    ok_count = n_hit >= n_gt - 1 and n_pred <= n_gt + 2 and not false_types
    ok_id = all(len(v) == 1 for v in sid_to_letters.values()) if sid_to_letters else False

    return {
        "group_dir": str(group_dir),
        "n_gt": n_gt,
        "n_ft_pred": n_pred,
        "n_matched": n_hit,
        "n_false_alarm_ft": n_pred - len(matched_pred),
        "n_miss": n_gt - n_hit,
        "recall": round(n_hit / n_gt, 3),
        "precision": round(n_hit / max(n_pred, 1), 3),
        "action_type_hist": hist,
        "false_types": false_types,
        "other_clips": len(other),
        "student_ids": summary.get("student_ids") or [],
        "sid_to_gt_letters": {k: sorted(v) for k, v in sid_to_letters.items()},
        "id_consistent": ok_id,
        "pairs": pairs,
        "missed_gt_s": [GROUP1_FT_GT[i][0] for i in range(n_gt) if i not in matched_gt],
        "pass_almost_same": bool(ok_count and n_hit >= 13 and n_pred <= 15),
        "clip_count": summary.get("clip_count"),
        "shot_stats": summary.get("shot_stats"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-dir", type=Path, default=ROOT / "data/outputs/v3/group_01")
    ap.add_argument("--tol", type=float, default=2.5)
    args = ap.parse_args()
    if not args.group_dir.exists():
        raise SystemExit(f"missing {args.group_dir}")
    r = evaluate_group1(args.group_dir, match_tol_s=args.tol)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if r["pass_almost_same"]:
        print("\nPASS: nearly matches GT free_throw timeline", file=sys.stderr)
        sys.exit(0)
    print(
        f"\nFAIL: matched {r['n_matched']}/{r['n_gt']}  "
        f"pred_ft={r['n_ft_pred']}  false_types={r['false_types']}",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
