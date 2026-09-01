#!/usr/bin/env python3
"""Evaluate v3 group outputs against docs/v3测试集动作真值.md timelines.

Pass when:
  - every GT event matches a pred clip (same action_type, |Δt| ≤ tol)
  - no forbidden action types
  - false-alarm count ≤ max_fa
  - if GT has person letters: best stu↔letter bijection accuracy ≥ min_id_acc
    (groups with person "—" skip identity check)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ABBREV = {
    "ft": "free_throw",
    "js": "jump_shot",
    "layup": "layup",
    "tt": "triple_threat",
    "pass": "pass",
}

OUTCOME_TOKENS = {
    "make": True,
    "made": True,
    "miss": False,
    "missed": False,
}


def parse_gt_md(path: Path) -> dict[int, list[dict]]:
    text = path.read_text(encoding="utf-8")
    groups: dict[int, list[dict]] = {}
    cur: int | None = None
    in_table = False
    for line in text.splitlines():
        m = re.match(r"^## Group\s+(\d+)\b", line)
        if m:
            cur = int(m.group(1))
            groups.setdefault(cur, [])
            in_table = False
            continue
        if cur is None:
            continue
        if re.match(r"^\|\s*t\s*\(s\)\s*\|", line):
            in_table = True
            continue
        if in_table and re.match(r"^\|\s*[-: ]+\|", line):
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                in_table = False
                continue
            person = parts[1]
            abbr = parts[2].strip().lower()
            if abbr not in ABBREV:
                continue
            person_slot = None
            if person in "ABCD":
                person_slot = "ABCD".index(person)
            made = None
            if len(parts) >= 4:
                tok = parts[3].strip().lower()
                if tok in OUTCOME_TOKENS:
                    made = OUTCOME_TOKENS[tok]
            groups[cur].append({
                "t_s": t,
                "person": person if person != "—" else None,
                "person_slot": person_slot,
                "action": ABBREV[abbr],
                "abbr": abbr,
                "made": made,
            })
        elif in_table and not line.startswith("|"):
            in_table = False
    return groups


def _load_outcomes(group_dir: Path) -> list[dict]:
    for name in ("report.json", "motion.json"):
        p = group_dir / name
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        outs = doc.get("shot_outcomes") or []
        if outs:
            return list(outs)
    return []


def _clip_time_s(c: dict) -> float | None:
    meta = c.get("metadata") or {}
    mc = meta.get("multicam") or {}
    atype = str(c.get("action_type") or "")
    # Prefer cam_03 pose clock (GT is cam_03-local); rim is cam_04 and often late
    if mc.get("pose_timestamp_ms") is not None:
        return float(mc["pose_timestamp_ms"]) / 1000.0
    if c.get("release_ms") is not None and atype != "triple_threat":
        return float(c["release_ms"]) / 1000.0
    # Triple-threat GT marks the cut/drive moment near the end of the setup
    # window (often 1–2s after crouch onset), not the window open.
    if atype == "triple_threat":
        if c.get("end_ms") is not None and c.get("start_ms") is not None:
            s, e = float(c["start_ms"]), float(c["end_ms"])
            return (0.35 * s + 0.65 * e) / 1000.0
        if c.get("end_ms") is not None:
            return float(c["end_ms"]) / 1000.0
    if mc.get("rim_timestamp_ms") is not None:
        return float(mc["rim_timestamp_ms"]) / 1000.0
    for key in ("release_ms", "release_common_ms", "rim_timestamp_ms", "anchor_ms"):
        if meta.get(key) is not None:
            return float(meta[key]) / 1000.0
    if c.get("start_ms") is not None and c.get("end_ms") is not None:
        return 0.5 * (float(c["start_ms"]) + float(c["end_ms"])) / 1000.0
    if c.get("start_ms") is not None:
        return float(c["start_ms"]) / 1000.0
    return None


def _load_clips(group_dir: Path) -> list[dict]:
    for name in ("report.json", "motion.json"):
        p = group_dir / name
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        clips = doc.get("clips") or []
        if clips:
            return list(clips)
    return []


def _best_id_mapping(pairs: list[dict]) -> tuple[dict[str, str], float]:
    """Maximize correct assignments under bijection letter→sid."""
    letters = sorted({p["gt_person"] for p in pairs if p.get("gt_person")})
    sids = sorted({p["pred_student"] for p in pairs})
    if not letters or not sids:
        return {}, 0.0
    # pad sids if fewer than letters
    while len(sids) < len(letters):
        sids.append(f"__miss_{len(sids)}")
    best_map: dict[str, str] = {}
    best_n = -1
    for perm in permutations(sids, len(letters)):
        m = dict(zip(letters, perm))
        n = sum(1 for p in pairs if m.get(p["gt_person"]) == p["pred_student"])
        if n > best_n:
            best_n = n
            best_map = m
    return best_map, (best_n / max(len(pairs), 1))


def evaluate_group(
    group_id: int,
    group_dir: Path,
    gt_events: list[dict],
    *,
    match_tol_s: float = 2.5,
    max_false_alarm: int = 2,
    min_id_acc: float = 0.75,
) -> dict:
    summary_path = group_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    clips = _load_clips(group_dir)
    hist = Counter(str(c.get("action_type") or "unknown") for c in clips)
    allowed = {e["action"] for e in gt_events}
    false_types = {k: v for k, v in hist.items() if k not in allowed and v}

    preds = []
    for c in clips:
        t = _clip_time_s(c)
        if t is None:
            continue
        preds.append({
            "t_s": t,
            "action": str(c.get("action_type") or ""),
            "student_id": str(c.get("student_id") or "?"),
            "clip": c,
            "clip_id": c.get("clip_id"),
        })

    outcomes = _load_outcomes(group_dir)
    by_clip_outcome = {
        o.get("clip_id"): o for o in outcomes if o.get("clip_id")
    }

    # greedy match GT → pred by (action, |dt|)
    # Breakthrough / TT onsets are temporally soft vs rim-gated shots — allow
    # a slightly wider window without changing shooting tolerances.
    def _tol_for(action: str) -> float:
        if action == "triple_threat":
            return max(match_tol_s, 5.0)
        if action == "pass":
            return max(match_tol_s, 3.5)
        return match_tol_s

    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt_events):
        for pi, p in enumerate(preds):
            if p["action"] != g["action"]:
                continue
            dt = abs(p["t_s"] - g["t_s"])
            if dt <= _tol_for(g["action"]):
                candidates.append((dt, gi, pi))
    candidates.sort()
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    pairs: list[dict] = []
    outcome_pairs: list[dict] = []
    for dt, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        g, p = gt_events[gi], preds[pi]
        pair = {
            "gt_s": g["t_s"],
            "gt_person": g["person"],
            "gt_action": g["action"],
            "gt_made": g.get("made"),
            "pred_s": round(p["t_s"], 2),
            "pred_student": p["student_id"],
            "pred_action": p["action"],
            "dt_s": round(p["t_s"] - g["t_s"], 2),
        }
        if g.get("made") is not None:
            oc = by_clip_outcome.get(p.get("clip_id")) or {}
            pred_made = oc.get("made")
            # Also check clip-level made if present
            if pred_made is None:
                pred_made = (p.get("clip") or {}).get("made")
            ok = pred_made is not None and bool(pred_made) == bool(g["made"])
            pair["pred_made"] = pred_made
            pair["outcome_ok"] = ok
            outcome_pairs.append(pair)
        pairs.append(pair)

    need_id = any(e.get("person") for e in gt_events)
    id_pairs = [p for p in pairs if p.get("gt_person")]
    id_map, id_acc = _best_id_mapping(id_pairs) if need_id and id_pairs else ({}, 1.0)

    n_gt = len(gt_events)
    n_hit = len(matched_gt)
    n_pred = len(preds)
    n_fa = n_pred - len(matched_pred)
    missed = [gt_events[i] for i in range(n_gt) if i not in matched_gt]

    n_outcome_gt = len(outcome_pairs)
    n_outcome_ok = sum(1 for p in outcome_pairs if p.get("outcome_ok"))
    pass_outcome = (n_outcome_gt == 0) or (n_outcome_ok == n_outcome_gt)

    pass_events = (
        n_hit == n_gt
        and n_fa <= max_false_alarm
        and not false_types
    )
    pass_id = (not need_id) or (id_acc + 1e-9 >= min_id_acc)
    passed = bool(pass_events and pass_id and pass_outcome)

    return {
        "group_id": group_id,
        "group_dir": str(group_dir),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": n_hit,
        "n_miss": n_gt - n_hit,
        "n_false_alarm": n_fa,
        "recall": round(n_hit / max(n_gt, 1), 3),
        "precision": round(n_hit / max(n_pred, 1), 3),
        "action_type_hist": dict(hist),
        "false_types": false_types,
        "allowed": sorted(allowed),
        "need_id": need_id,
        "id_map": id_map,
        "id_acc": round(id_acc, 3),
        "id_consistent": pass_id,
        "pairs": pairs,
        "outcome_pairs": outcome_pairs,
        "n_outcome_gt": n_outcome_gt,
        "n_outcome_ok": n_outcome_ok,
        "pass_outcome": pass_outcome,
        "missed": [
            {"t_s": e["t_s"], "person": e["person"], "action": e["action"], "made": e.get("made")}
            for e in missed
        ],
        "pass_events": pass_events,
        "pass": passed,
        "shot_stats": summary.get("shot_stats"),
        "student_ids": summary.get("student_ids"),
        "tol_s": match_tol_s,
        "min_id_acc": min_id_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=ROOT / "docs/v3测试集动作真值.md")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/outputs/v3")
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
