#!/usr/bin/env python3
"""Compare v2 batch outputs against docs/v2测试集动作真值.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Allowed action labels per group (group0 = enrollment only)
EXPECTED: dict[int, set[str]] = {
    0: set(),
    1: {"free_throw"},
    2: {"layup"},
    3: {"triple_threat", "layup"},
    4: {"triple_threat", "jump_shot", "free_throw", "layup"},
    5: {"triple_threat", "jump_shot", "free_throw"},
    6: {"pass"},
}

ENROLL_EXPECT = 6


def _load_summary(out_dir: Path, group_id: int) -> dict | None:
    path = out_dir / f"group_{group_id:02d}" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_group(group_id: int, summary: dict | None) -> dict:
    allowed = EXPECTED.get(group_id, set())
    row: dict = {
        "group": group_id,
        "status": "missing",
        "allowed": sorted(allowed),
        "detected": {},
        "illegal": [],
        "missing_types": [],
        "clip_count": 0,
        "enroll_count": None,
        "enroll_ok": None,
        "pass": False,
    }
    if summary is None:
        return row

    row["status"] = "ok"
    hist = summary.get("action_type_hist") or {}
    row["detected"] = dict(hist)
    row["clip_count"] = int(summary.get("clip_count") or 0)

    if group_id == 0:
        n = len(summary.get("student_ids") or [])
        row["enroll_count"] = n
        row["enroll_ok"] = n == ENROLL_EXPECT
        row["pass"] = row["enroll_ok"]
        return row

    illegal = sorted(t for t in hist if t not in allowed)
    missing = sorted(t for t in allowed if hist.get(t, 0) == 0)
    row["illegal"] = illegal
    row["missing_types"] = missing
    row["pass"] = not illegal and not missing and row["clip_count"] > 0
    return row


def print_report(rows: list[dict]) -> int:
    fails = 0
    print(f"{'Group':>5}  {'Pass':>4}  {'Clips':>5}  {'Detected':<40}  Issues")
    print("-" * 90)
    for r in rows:
        if not r["pass"]:
            fails += 1
        det = ", ".join(f"{k}:{v}" for k, v in sorted(r["detected"].items()))
        issues: list[str] = []
        if r["status"] == "missing":
            issues.append("no summary.json")
        if r.get("enroll_ok") is False:
            issues.append(f"enroll={r['enroll_count']}!={ENROLL_EXPECT}")
        if r["illegal"]:
            issues.append(f"illegal={r['illegal']}")
        if r["missing_types"]:
            issues.append(f"missing={r['missing_types']}")
        if r["group"] != 0 and r["clip_count"] == 0 and r["status"] == "ok":
            issues.append("clip_count=0")
        print(
            f"{r['group']:>5}  {'Y' if r['pass'] else 'N':>4}  "
            f"{r['clip_count']:>5}  {det:<40}  {', '.join(issues)}"
        )
    print("-" * 90)
    print(f"Pass: {len(rows) - fails}/{len(rows)}")
    return fails


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v2 outputs vs ground truth")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "outputs" / "v2",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()

    rows = []
    for gid in sorted(EXPECTED.keys()):
        summary = _load_summary(args.out_dir, gid)
        rows.append(analyze_group(gid, summary))

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        fails = print_report(rows)
        raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
