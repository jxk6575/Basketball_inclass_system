#!/usr/bin/env python3
"""Evaluate v1 group outputs against docs/v1测试集动作真值.md timelines.

Thin wrapper around eval_v3_gt with v1 defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_v3_gt import evaluate_group, parse_gt_md  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=ROOT / "docs/v1测试集动作真值.md")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/outputs/v1")
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--max-fa", type=int, default=2)
    ap.add_argument("--min-id-acc", type=float, default=0.75)
    args = ap.parse_args()

    gt_all = parse_gt_md(args.gt)
    if args.group not in gt_all or not gt_all[args.group]:
        raise SystemExit(f"no GT events for group {args.group}")
    group_dir = args.out_dir / f"group_{args.group:02d}"
    if not group_dir.exists():
        raise SystemExit(f"missing {group_dir}")

    r = evaluate_group(
        args.group, group_dir, gt_all[args.group],
        match_tol_s=args.tol,
        max_false_alarm=args.max_fa,
        min_id_acc=args.min_id_acc,
    )
    print(json.dumps(r, ensure_ascii=False, indent=2))
    status = "PASS" if r["pass"] else "FAIL"
    print(
        f"\n{status} group{args.group}: matched {r['n_matched']}/{r['n_gt']} "
        f"fa={r['n_false_alarm']} false_types={r['false_types']} "
        f"id_acc={r['id_acc']} outcome={r['n_outcome_ok']}/{r['n_outcome_gt']}",
        file=sys.stderr,
    )
    sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
