"""Post-perception identity smoothing on continuous tracks.

One track_id = one physical person while the track lives. Gallery flicker
(stu_01↔stu_03 etc.) is suppressed with a conf-weighted sliding majority and
hysteresis — without collapsing the whole session to a single student_id
(no person-count prior).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.config import data_path, load_yaml

_CONF_W = {
    "high": 3.0,
    "medium": 2.0,
    "sticky": 1.5,
    "spatial_prior": 1.2,
    "lr_order": 0.8,
    "low": 0.15,
    "forced": 0.1,
    "displaced_exclusive": 0.05,
}


def _idcfg() -> dict:
    return dict((load_yaml("cameras.yaml").get("identity") or {}))


def _smooth_cfg() -> dict:
    return dict((_idcfg().get("track_id_smooth") or {}))


def _conf_w(conf: str | None) -> float:
    return float(_CONF_W.get(str(conf or "low"), 0.3))


def _track_global_majority(
    samples: list[tuple[str, float]],
) -> tuple[str | None, float, dict[str, float]]:
    mass: Counter[str] = Counter()
    for sid, w in samples:
        if sid:
            mass[str(sid)] += float(w)
    if not mass:
        return None, 0.0, {}
    ranked = mass.most_common(2)
    best, w1 = ranked[0]
    w2 = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    total = float(sum(mass.values()))
    share = float(w1) / max(total, 1e-6)
    margin = (float(w1) - w2) / max(float(w1), 1e-6)
    return best, share * (0.5 + 0.5 * margin), {k: float(v) for k, v in mass.items()}


def _iou(a: list[float], b: list[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(1e-6, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    bb = max(1e-6, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    return inter / (aa + bb - inter)


def _bbox_center(bb: list[float]) -> tuple[float, float] | None:
    if len(bb) < 4:
        return None
    return 0.5 * (float(bb[0]) + float(bb[2])), 0.5 * (float(bb[1]) + float(bb[3]))


def _center_dist(a: list[float], b: list[float]) -> float:
    ca, cb = _bbox_center(a), _bbox_center(b)
    if ca is None or cb is None:
        return 1e9
    return float(((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5)


def _link_track_chains(
    frames: list[dict],
    by_tid: dict[Any, list[tuple[int, int, str | None, float]]],
    global_dom: dict[Any, str | None],
    *,
    max_gap_frames: int = 45,
    min_iou: float = 0.15,
    max_center_dist: float = 220.0,
) -> dict[Any, str | None]:
    """
    Inherit identity across broken tracks when bbox temporally continues.

    Not a person-count prior: only links non-overlapping tracks with spatial
    continuity (re-ID after brief loss). Uses IoU or center proximity.
    """
    meta: dict[Any, dict[str, Any]] = {}
    for tid, samples in by_tid.items():
        if not samples:
            continue
        fi0, pi0 = samples[0][0], samples[0][1]
        fi1, pi1 = samples[-1][0], samples[-1][1]
        bb0 = (frames[fi0].get("persons") or [{}])[pi0].get("bbox") or []
        bb1 = (frames[fi1].get("persons") or [{}])[pi1].get("bbox") or []
        meta[tid] = {
            "start": fi0,
            "end": fi1,
            "bbox0": bb0,
            "bbox1": bb1,
            "n": len(samples),
        }

    chain_sid = dict(global_dom)
    order = sorted(meta.keys(), key=lambda t: (meta[t]["start"], -meta[t]["n"]))
    for tid in order:
        best_prev_tid = None
        best_prev_sid = None
        best_score = -1.0
        for prev, pm in meta.items():
            if prev == tid:
                continue
            if pm["end"] >= meta[tid]["start"]:
                continue
            gap = meta[tid]["start"] - pm["end"]
            if gap > max_gap_frames:
                continue
            iou = _iou(pm["bbox1"], meta[tid]["bbox0"])
            cdist = _center_dist(pm["bbox1"], meta[tid]["bbox0"])
            if iou < min_iou and cdist > max_center_dist:
                continue
            psid = chain_sid.get(prev) or global_dom.get(prev)
            if not psid:
                continue
            prox = iou if iou >= min_iou else max(0.0, 1.0 - cdist / max(max_center_dist, 1.0))
            score = prox * (1.0 + 0.02 * (max_gap_frames - gap)) * (1.0 + 0.001 * pm["n"])
            if score > best_score:
                best_score = score
                best_prev_tid = prev
                best_prev_sid = psid
        if best_prev_sid is None:
            continue
        prev_n = float(meta[best_prev_tid]["n"])
        cur_n = float(meta[tid]["n"])
        if cur_n <= prev_n * 0.85 or not chain_sid.get(tid):
            chain_sid[tid] = best_prev_sid
    return chain_sid

def smooth_pose2d_frames(
    frames: list[dict],
    *,
    window: int | None = None,
    min_mass: float | None = None,
    switch_margin: float | None = None,
    global_bias_share: float | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Return smoothed frames + stats. Does not mutate input frames in place."""
    cfg = _smooth_cfg()
    if window is None:
        window = int(cfg.get("window_frames", 60))
    if min_mass is None:
        min_mass = float(cfg.get("min_mass", 6.0))
    if switch_margin is None:
        switch_margin = float(cfg.get("switch_margin", 0.28))
    if global_bias_share is None:
        global_bias_share = float(cfg.get("global_bias_share", 0.55))
    max_gap = int(cfg.get("chain_max_gap_frames", 45))
    chain_iou = float(cfg.get("chain_min_iou", 0.15))

    # Gather per-track time-ordered samples
    by_tid: dict[Any, list[tuple[int, int, str | None, float]]] = defaultdict(list)
    # (frame_idx, person_idx, sid, weight)
    for fi, fr in enumerate(frames):
        for pi, p in enumerate(fr.get("persons") or []):
            tid = p.get("track_id")
            if tid is None:
                continue
            sid = p.get("student_id")
            w = _conf_w(p.get("identity_confidence"))
            by_tid[tid].append((fi, pi, str(sid) if sid else None, w))

    # Global majority per track (bias when local window is ambiguous)
    global_dom: dict[Any, str | None] = {}
    global_score: dict[Any, float] = {}
    for tid, samples in by_tid.items():
        g_sid, g_sc, _ = _track_global_majority(
            [(s, w) for _, _, s, w in samples if s]
        )
        global_dom[tid] = g_sid if g_sc >= global_bias_share else None
        global_score[tid] = g_sc

    chain_dom = _link_track_chains(
        frames, by_tid, global_dom,
        max_gap_frames=max_gap,
        min_iou=chain_iou,
        max_center_dist=float(cfg.get("chain_max_center_dist", 280.0)),
    )
    # Chain continuity wins over short-track gallery flicker
    for tid in by_tid:
        csid = chain_dom.get(tid)
        if not csid:
            continue
        if not global_dom.get(tid):
            global_dom[tid] = csid
            global_score[tid] = max(float(global_score.get(tid) or 0.0), 0.65)
        elif csid != global_dom.get(tid):
            # Prefer chained ID (same physical person after track break)
            global_dom[tid] = csid
            global_score[tid] = max(float(global_score.get(tid) or 0.0), 0.70)

    # Solo occupancy (video evidence): if this camera almost never sees >1
    # person, broken tracks are the same physical actor → one session ID.
    # Triggered by observed frame occupancy, not by group metadata / GT.
    solo_merge = False
    solo_sid: str | None = None
    if bool(cfg.get("solo_occupancy_merge", True)):
        occ = [len(fr.get("persons") or []) for fr in frames]
        if occ:
            max_occ = max(occ)
            # p95 via order statistic (avoid numpy dep here)
            sorted_occ = sorted(occ)
            p95 = sorted_occ[min(len(sorted_occ) - 1, int(0.95 * (len(sorted_occ) - 1)))]
            if max_occ <= 1 and p95 <= 1:
                mass: Counter[str] = Counter()
                for samples in by_tid.values():
                    for _, _, s, w in samples:
                        if s:
                            mass[str(s)] += float(w)
                if mass:
                    solo_sid = mass.most_common(1)[0][0]
                    solo_merge = True
                    for tid in by_tid:
                        global_dom[tid] = solo_sid
                        global_score[tid] = 1.0

    # Build assignment map (fi, pi) -> new_sid
    assign: dict[tuple[int, int], str | None] = {}
    n_changed = 0
    locked: dict[Any, str | None] = {
        tid: (global_dom.get(tid) or None) for tid in by_tid
    }

    for tid, samples in by_tid.items():
        n = len(samples)
        half = max(1, window // 2)
        for i, (fi, pi, sid, _w) in enumerate(samples):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            votes: Counter[str] = Counter()
            for j in range(lo, hi):
                s_j, w_j = samples[j][2], samples[j][3]
                if s_j:
                    votes[s_j] += w_j
            # Soft global / chain bias
            g = global_dom.get(tid)
            if solo_merge and solo_sid:
                # Occupancy-solo: do not let residual gallery flicker flip the lock
                new_sid = solo_sid
                locked[tid] = solo_sid
            else:
                if g and votes:
                    votes[g] += float(min_mass) * 0.45 * float(global_score.get(tid) or 0.0)
                elif g and not votes:
                    votes[g] += float(min_mass)

                new_sid = sid
                if votes:
                    ranked = votes.most_common(2)
                    best, w1 = ranked[0]
                    w2 = float(ranked[1][1]) if len(ranked) > 1 else 0.0
                    mass = float(sum(votes.values()))
                    margin = (float(w1) - w2) / max(float(w1), 1e-6)
                    prev = locked.get(tid)
                    if prev is None:
                        if mass >= min_mass:
                            new_sid = best
                            locked[tid] = best
                    elif best == prev:
                        new_sid = prev
                    elif margin >= switch_margin and mass >= min_mass and float(w1) >= 1.25 * max(w2, 1e-6):
                        new_sid = best
                        locked[tid] = best
                    else:
                        new_sid = prev
                elif locked.get(tid):
                    new_sid = locked[tid]

            assign[(fi, pi)] = new_sid
            if (new_sid or None) != (sid or None):
                n_changed += 1

    # Apply + exclusive cleanup per frame
    out_frames: list[dict] = []
    for fi, fr in enumerate(frames):
        persons = []
        for pi, p in enumerate(fr.get("persons") or []):
            p2 = dict(p)
            key = (fi, pi)
            if key in assign:
                ns = assign[key]
                if ns != p2.get("student_id"):
                    p2["student_id"] = ns
                    if ns:
                        p2["identity_confidence"] = "sticky"
                        p2["identity_smooth"] = "track_majority"
                    else:
                        p2["identity_confidence"] = "low"
            persons.append(p2)

        # Exclusive: one student_id per frame (keep highest conf weight / area)
        best_for: dict[str, tuple[float, int]] = {}
        for i, p in enumerate(persons):
            sid = p.get("student_id")
            if not sid:
                continue
            bb = p.get("bbox") or [0, 0, 0, 0]
            area = abs(float(bb[2] - bb[0]) * float(bb[3] - bb[1])) if len(bb) >= 4 else 0.0
            score = _conf_w(p.get("identity_confidence")) * (1.0 + area / 1e5)
            prev = best_for.get(str(sid))
            if prev is None or score > prev[0]:
                best_for[str(sid)] = (score, i)
        winners = {idx for _, idx in best_for.values()}
        for i, p in enumerate(persons):
            if p.get("student_id") and i not in winners:
                p["student_id"] = None
                p["identity_confidence"] = "displaced_exclusive"
                p["identity_smooth"] = "exclusive"
        fr2 = dict(fr)
        fr2["persons"] = persons
        out_frames.append(fr2)

    stats = {
        "n_tracks": len(by_tid),
        "n_person_slots": sum(len(v) for v in by_tid.values()),
        "n_changed": n_changed,
        "window": window,
        "global_dom": {str(k): v for k, v in global_dom.items() if v},
        "chain_dom": {str(k): v for k, v in chain_dom.items() if v},
        "solo_occupancy_merge": solo_merge,
        "solo_sid": solo_sid,
    }
    return out_frames, stats


def smooth_session_identities(
    session_id: str,
    *,
    camera_ids: list[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Smooth student_id flicker on each camera pose2d track."""
    cfg = _smooth_cfg()
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "cameras": {}}

    root = data_path("sessions", session_id, "perception")
    if not root.exists():
        return {"enabled": True, "cameras": {}, "error": "no_perception"}

    cams = camera_ids or sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "pose2d.json").exists()
    )
    report: dict[str, Any] = {"enabled": True, "cameras": {}}
    cam_docs: dict[str, dict] = {}
    for cam in cams:
        path = root / cam / "pose2d.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        frames = list(doc.get("frames") or [])
        new_frames, stats = smooth_pose2d_frames(frames)
        report["cameras"][cam] = stats
        doc = dict(doc)
        doc["frames"] = new_frames
        cam_docs[cam] = doc

    # Cross-camera solo consensus: when cams disagree on solo_sid, prefer the
    # action camera (or the longest solo occupancy mass).
    solo_votes = {
        cam: st.get("solo_sid")
        for cam, st in report["cameras"].items()
        if st.get("solo_occupancy_merge") and st.get("solo_sid")
    }
    if len(set(solo_votes.values())) > 1:
        prefer = None
        try:
            from src.cameras.registry import get_action_segment_camera
            prefer = get_action_segment_camera()
        except Exception:
            prefer = None
        consensus = solo_votes.get(prefer) if prefer in solo_votes else None
        if not consensus:
            # Fall back to majority of solo cams
            consensus = Counter(solo_votes.values()).most_common(1)[0][0]
        report["solo_consensus_sid"] = consensus
        for cam, doc in cam_docs.items():
            st = report["cameras"].get(cam) or {}
            if not st.get("solo_occupancy_merge"):
                continue
            if st.get("solo_sid") == consensus:
                continue
            for fr in doc.get("frames") or []:
                for p in fr.get("persons") or []:
                    if p.get("track_id") is None:
                        continue
                    p["student_id"] = consensus
                    p["identity_confidence"] = "sticky"
                    p["identity_smooth"] = "solo_consensus"
            st["solo_sid"] = consensus
            st["solo_consensus_applied"] = True

    if write:
        for cam, doc in cam_docs.items():
            path = root / cam / "pose2d.json"
            proc = str(doc.get("processing") or "")
            tag = "track_id_smooth"
            if tag not in proc:
                doc["processing"] = (proc + "+" + tag) if proc else tag
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        try:
            from src.action.spatial_shooter import _load_pose_frames
            _load_pose_frames.cache_clear()
        except Exception:
            pass
    return report
