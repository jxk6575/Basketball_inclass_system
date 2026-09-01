#!/usr/bin/env python3
"""Validate streaming fast-path latency on existing / freshly run sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.streaming import finalize_action_from_session, simulate_per_action_latency  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="1,2,3,4")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data/outputs/v1")
    args = ap.parse_args()
    groups = [int(x) for x in args.groups.split(",") if x.strip()]

    report = {"groups": []}
    for g in groups:
        gdir = args.out_root / f"group_{g:02d}"
        sp = gdir / "summary.json"
        if not sp.exists():
            print(f"skip group_{g:02d}: no summary")
            continue
        summary = json.loads(sp.read_text(encoding="utf-8"))
        sid = summary["session_id"]
        stu = summary.get("student_id", f"stu_g{g:02d}")
        print(f"\n=== group_{g:02d} session={sid} ===", flush=True)
        results, timings = finalize_action_from_session(sid, stu)
        sim = simulate_per_action_latency(sid, stu)
        meets = sum(1 for r in sim if r.get("meets_10s"))
        shooting_made = [
            (r.action_type, r.made, r.outcome_confidence)
            for r in results
            if r.action_type in {"free_throw", "layup"}
        ]
        types = {}
        for r in results:
            types[r.action_type] = types.get(r.action_type, 0) + 1
        entry = {
            "group": g,
            "session_id": sid,
            "timings": timings,
            "action_types": types,
            "n_actions": len(results),
            "shooting_outcomes": [
                {"action_type": a, "made": m, "confidence": c}
                for a, m, c in shooting_made
            ],
            "est_finalize_meets_10s": meets,
            "est_finalize_total": len(sim),
            "sample": [r.to_dict() for r in results[:3]],
            "sim_sample": sim[:3],
        }
        report["groups"].append(entry)
        print(
            f"  finalize detect={timings['detect_s']}s outcome={timings['outcome_s']}s "
            f"per_action={timings['per_action_s']}s clips={timings['n_clips']} types={types}",
            flush=True,
        )
        print(f"  shooting made/miss: {shooting_made}", flush=True)
        print(f"  est teacher latency ≤10s: {meets}/{len(sim)}", flush=True)

    out = args.out_root / "fastpath_latency_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
