"""Cross-camera identity repair via simple left/right image order.

cam_01 / cam_02 look at the paint from opposite sidelines, so the left→right
person order in one view is roughly the reverse of the other. We pair people
at synced times by that reversed rank and transfer student_id.

Court-XY mode is kept as an optional fallback (``mode: court_xy``).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.cameras.event_sync import get_camera_offsets_ms
from src.cameras.registry import get_camera_ids
from src.config import data_path, load_yaml


def _identity_cfg() -> dict:
    return load_yaml("cameras.yaml").get("identity", {})


def _spatial_cfg() -> dict:
    return dict((_identity_cfg().get("cross_cam_spatial") or {}))


def _conf_of(person: dict) -> str:
    return str(person.get("identity_confidence") or ("medium" if person.get("student_id") else "low"))


def _conf_rank(conf: str) -> float:
    # lr_order must stay *below* high/medium so exclusive-ID cleanup does not
    # displace a solid appearance match that we just used as the transfer source.
    return {
        "spatial_prior": 3.5,
        "high": 3.0,
        "medium": 2.0,
        "sticky": 1.5,
        "forced": 1.2,
        "lr_order": 1.1,
        "low": 0.5,
        "displaced_exclusive": 0.2,
    }.get(str(conf or "low"), 1.0)


def _person_cx(person: dict) -> float | None:
    bb = person.get("bbox")
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        return 0.5 * (float(bb[0]) + float(bb[2]))
    kps = person.get("keypoints") or []
    xs = [float(k[0]) for k in kps if isinstance(k, (list, tuple)) and len(k) >= 2]
    if xs:
        return float(np.median(xs))
    return None


def _load_pose_docs(session_id: str, camera_ids: list[str]) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for cam in camera_ids:
        path = data_path("sessions", session_id, "perception", cam, "pose2d.json")
        if path.exists():
            docs[cam] = json.loads(path.read_text(encoding="utf-8"))
    return docs


def _frame_common_ms(fr: dict, offset_ms: float) -> float:
    local = fr.get("timestamp_ms")
    if local is None:
        local = fr.get("frame", 0)
    return float(local) - float(offset_ms)


def _sorted_persons(persons: list[dict], *, reverse: bool) -> list[dict]:
    keyed: list[tuple[float, dict]] = []
    for p in persons:
        cx = _person_cx(p)
        if cx is None:
            continue
        keyed.append((cx, p))
    keyed.sort(key=lambda t: t[0], reverse=reverse)
    return [p for _, p in keyed]


def _never_overwrite_set(cfg: dict | None = None) -> set[str]:
    cfg = cfg if cfg is not None else _spatial_cfg()
    raw = cfg.get("never_overwrite")
    if raw is None:
        return {"high", "medium", "sticky", "spatial_prior"}
    return {str(x) for x in raw}


def _should_overwrite(
    person: dict,
    only_fix_low: bool,
    *,
    cfg: dict | None = None,
) -> bool:
    """Whether spatial repair may replace this person's student_id."""
    if not person.get("student_id"):
        return True
    conf = _conf_of(person)
    if conf in _never_overwrite_set(cfg):
        return False
    if not only_fix_low:
        # Legacy: may overwrite anything not in never_overwrite
        return True
    return conf in ("low", "forced", "displaced_exclusive", "lr_order")


def _assign(person: dict, sid: str, *, reason: str) -> bool:
    old = person.get("student_id")
    if old == sid:
        return False
    person["identity_prev"] = old
    person["student_id"] = sid
    person["identity_confidence"] = "lr_order"
    person["identity_repair_reason"] = reason
    return True


def _pair_lr_at_time(
    left_persons: list[dict],
    right_persons: list[dict],
    *,
    left_is_cam01: bool,
) -> list[tuple[dict, dict]]:
    """
    Pair by image-x order with reverse on one side.
    cam_01 ascending cx ↔ cam_02 descending cx (same physical left→right).
    """
    a = _sorted_persons(left_persons, reverse=False)   # cam_01: left→right
    b = _sorted_persons(right_persons, reverse=True)    # cam_02: right→left = phys left→right
    if not left_is_cam01:
        a, b = b, a
    n = min(len(a), len(b))
    return list(zip(a[:n], b[:n]))


def repair_lr_order(
    session_id: str,
    camera_ids: list[str] | None = None,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Match cam_01 ↔ cam_02 by reversed horizontal order at synced times."""
    cfg = _spatial_cfg()
    cam_a = str(cfg.get("pair_left") or "cam_01")
    cam_b = str(cfg.get("pair_right") or "cam_02")
    max_dt = float(cfg.get("max_dt_ms", 400.0))
    sample_stride = max(1, int(cfg.get("sample_stride", 3)))
    only_fix_low = bool(cfg.get("only_fix_low_or_missing", True))
    prefer_cam = str(cfg.get("prefer_camera") or cam_a)
    # Optional: also align front cam (cam_03) to sideline order (same x order as cam_01)
    front_cam = str(cfg.get("pair_front") or "cam_03")
    use_front = bool(cfg.get("align_front", False))

    report: dict[str, Any] = {
        "session_id": session_id,
        "mode": "image_lr_reverse",
        "pair": [cam_a, cam_b],
        "status": "skipped",
        "n_reassigned": 0,
        "n_pairs": 0,
        "repairs": [],
    }

    camera_ids = camera_ids or get_camera_ids()
    if cam_a not in camera_ids or cam_b not in camera_ids:
        report["status"] = "pair_missing"
        return report

    offsets = get_camera_offsets_ms(session_id)
    load_cams = [cam_a, cam_b]
    if use_front and front_cam in camera_ids:
        load_cams.append(front_cam)
    docs = _load_pose_docs(session_id, load_cams)
    if cam_a not in docs or cam_b not in docs:
        report["status"] = "insufficient_pose"
        return report

    def _index_frames(cam: str) -> list[tuple[float, dict]]:
        off = float(offsets.get(cam, 0.0))
        return [(_frame_common_ms(fr, off), fr) for fr in (docs[cam].get("frames") or [])]

    def _nearest(idx: list[tuple[float, dict]], t: float) -> tuple[dict | None, float | None]:
        best, best_abs = None, None
        for tb, fr in idx:
            d = abs(tb - t)
            if d > max_dt:
                continue
            if best_abs is None or d < best_abs:
                best_abs, best = d, fr
        return best, best_abs

    n_reassigned = 0
    n_pairs = 0
    repairs: list[dict] = []

    def _transfer_pairs(
        pairs: list[tuple[dict, dict]],
        *,
        cam_x: str,
        cam_y: str,
        fr_x: dict,
        fr_y: dict,
        dt: float | None,
    ) -> int:
        nonlocal n_reassigned
        for pa, pb in pairs:
            sa, sb = pa.get("student_id"), pb.get("student_id")
            if sa and sb and sa == sb:
                continue
            ra, rb = _conf_rank(_conf_of(pa)), _conf_rank(_conf_of(pb))
            cands: list[tuple[float, int, str, dict, str, str]] = []
            if sa:
                cands.append((ra, 1 if prefer_cam == cam_x else 0, sa, pb, cam_x, cam_y))
            if sb:
                cands.append((rb, 1 if prefer_cam == cam_y else 0, sb, pa, cam_y, cam_x))
            if not cands:
                continue
            cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
            _sc, _tb, src_sid, target, src_cam, tgt_cam = cands[0]
            if not _should_overwrite(target, only_fix_low, cfg=cfg):
                continue
            prev = target.get("student_id")
            if _assign(target, src_sid, reason=f"lr_order:{src_cam}->{tgt_cam}"):
                n_reassigned += 1
                repairs.append({
                    "from_cam": src_cam,
                    "to_cam": tgt_cam,
                    "frame_a": fr_x.get("frame"),
                    "frame_b": fr_y.get("frame"),
                    "dt_ms": round(float(dt or 0.0), 1),
                    "to": src_sid,
                    "prev": prev,
                })
        return len(pairs)

    frames_a = docs[cam_a].get("frames") or []
    b_idx = _index_frames(cam_b)
    f_idx = _index_frames(front_cam) if front_cam in docs else []
    off_a = float(offsets.get(cam_a, 0.0))

    for fi, fr_a in enumerate(frames_a):
        if sample_stride > 1 and (fi % sample_stride) != 0:
            continue
        t = _frame_common_ms(fr_a, off_a)
        fr_b, dt_b = _nearest(b_idx, t)
        if fr_b is not None:
            pairs = _pair_lr_at_time(
                fr_a.get("persons") or [],
                fr_b.get("persons") or [],
                left_is_cam01=True,
            )
            n_pairs += _transfer_pairs(
                pairs, cam_x=cam_a, cam_y=cam_b, fr_x=fr_a, fr_y=fr_b, dt=dt_b,
            )

        # Front cam: same left→right order as cam_01 (baseline view, not mirrored)
        if f_idx:
            fr_f, dt_f = _nearest(f_idx, t)
            if fr_f is not None:
                a_sorted = _sorted_persons(fr_a.get("persons") or [], reverse=False)
                f_sorted = _sorted_persons(fr_f.get("persons") or [], reverse=False)
                n = min(len(a_sorted), len(f_sorted))
                pairs_f = list(zip(a_sorted[:n], f_sorted[:n]))
                n_pairs += _transfer_pairs(
                    pairs_f, cam_x=cam_a, cam_y=front_cam,
                    fr_x=fr_a, fr_y=fr_f, dt=dt_f,
                )

    # Exclusive IDs within each frame (keep strongest conf; never let lr_order
    # displace high/medium after the rank fix above).
    protect = _never_overwrite_set(cfg)
    for doc in docs.values():
        for fr in doc.get("frames") or []:
            best_map: dict[str, tuple[float, dict]] = {}
            for person in fr.get("persons") or []:
                sid = person.get("student_id")
                if not sid:
                    continue
                score = _conf_rank(_conf_of(person))
                bb = person.get("bbox") or [0, 0, 0, 0]
                if len(bb) >= 4:
                    score += 1e-6 * abs((bb[2] - bb[0]) * (bb[3] - bb[1]))
                prev = best_map.get(sid)
                if prev is None or score > prev[0]:
                    if prev is not None:
                        # Prefer dropping the weaker duplicate
                        prev[1]["student_id"] = None
                        prev[1]["identity_confidence"] = "displaced_exclusive"
                    best_map[sid] = (score, person)
                else:
                    # If current is protected and incumbent is weaker spatial tag, swap
                    if (
                        _conf_of(person) in protect
                        and _conf_of(prev[1]) not in protect
                        and score + 0.05 >= prev[0]
                    ):
                        prev[1]["student_id"] = None
                        prev[1]["identity_confidence"] = "displaced_exclusive"
                        best_map[sid] = (score, person)
                    else:
                        person["student_id"] = None
                        person["identity_confidence"] = "displaced_exclusive"

    report["n_reassigned"] = n_reassigned
    report["n_pairs"] = n_pairs
    report["repairs"] = repairs[:200]
    report["n_repairs_total"] = len(repairs)
    report["status"] = "ok"
    report["front_cam"] = front_cam if (use_front and front_cam in docs) else None
    report["config"] = {
        "max_dt_ms": max_dt,
        "sample_stride": sample_stride,
        "only_fix_low_or_missing": only_fix_low,
        "prefer_camera": prefer_cam,
        "align_front": use_front,
        "never_overwrite": sorted(_never_overwrite_set(cfg)),
    }

    if write:
        for cam_id, doc in docs.items():
            doc["processing"] = "per_camera_isolated+lr_order_prior"
            path = data_path("sessions", session_id, "perception", cam_id, "pose2d.json")
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        out = data_path("sessions", session_id, "sync", "identity_spatial_repair.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["output"] = str(out)

    return report


def repair_cross_camera_identities(
    session_id: str,
    camera_ids: list[str] | None = None,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Entry point used by temporal sync. Default: image L/R reverse."""
    cfg = _spatial_cfg()
    if not bool(cfg.get("enabled", True)):
        return {
            "session_id": session_id,
            "enabled": False,
            "status": "skipped",
            "n_reassigned": 0,
        }

    mode = str(cfg.get("mode") or "image_lr_reverse").lower()
    if mode in ("image_lr_reverse", "lr", "lr_order", "sideline_reverse"):
        return repair_lr_order(session_id, camera_ids, write=write)

    # Optional legacy court-XY path
    return _repair_court_xy(session_id, camera_ids, write=write)


def _repair_court_xy(
    session_id: str,
    camera_ids: list[str] | None = None,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Legacy court-plane proximity clustering (requires calibration)."""
    from src.calibration.court_project import person_to_court_xy
    from src.pose.triangulate import load_camera_calibration

    cfg = _spatial_cfg()
    report: dict[str, Any] = {
        "session_id": session_id,
        "mode": "court_xy",
        "enabled": True,
        "status": "skipped",
        "repairs": [],
        "clusters": 0,
        "n_reassigned": 0,
    }
    camera_ids = camera_ids or get_camera_ids()
    cameras = load_camera_calibration()
    usable = [c for c in camera_ids if c in cameras]
    if len(usable) < 2:
        report["status"] = "no_calibration"
        return report

    # Minimal court path: fall back to L/R if calib weak
    # Keep previous behaviour lightly — delegate to lr if only cam_01/02
    if set(usable) <= {"cam_01", "cam_02", "cam_03"} and "cam_01" in usable and "cam_02" in usable:
        # Prefer simple path unless explicitly forced
        if not bool(cfg.get("force_court_xy", False)):
            return repair_lr_order(session_id, camera_ids, write=write)

    offsets = get_camera_offsets_ms(session_id)
    docs = _load_pose_docs(session_id, usable)
    max_dist = float(cfg.get("max_court_dist_m", 1.25))
    max_dt = float(cfg.get("max_dt_ms", 300.0))
    only_fix_low = bool(cfg.get("only_fix_low_or_missing", True))
    sample_stride = int(cfg.get("sample_stride", 2))

    # Build per-frame obs with court xy then nearest-neighbour across cams
    n_reassigned = 0
    repairs: list[dict] = []
    cams = list(docs.keys())
    for i, cam_i in enumerate(cams):
        for cam_j in cams[i + 1 :]:
            frames_i = docs[cam_i].get("frames") or []
            frames_j = docs[cam_j].get("frames") or []
            off_i = float(offsets.get(cam_i, 0.0))
            off_j = float(offsets.get(cam_j, 0.0))
            indexed_j = [(_frame_common_ms(fr, off_j), fr) for fr in frames_j]
            for fi, fr_i in enumerate(frames_i):
                if sample_stride > 1 and (fi % sample_stride) != 0:
                    continue
                t = _frame_common_ms(fr_i, off_i)
                best_fr = None
                best_abs = None
                for tj, fr_j in indexed_j:
                    d = abs(tj - t)
                    if d > max_dt:
                        continue
                    if best_abs is None or d < best_abs:
                        best_abs, best_fr = d, fr_j
                if best_fr is None:
                    continue
                for pi in fr_i.get("persons") or []:
                    xyi = person_to_court_xy(
                        cameras[cam_i], pi.get("bbox"), pi.get("keypoints"),
                    )
                    if xyi is None:
                        continue
                    best_p = None
                    best_d = float("inf")
                    for pj in best_fr.get("persons") or []:
                        xyj = person_to_court_xy(
                            cameras[cam_j], pj.get("bbox"), pj.get("keypoints"),
                        )
                        if xyj is None:
                            continue
                        d = float(np.linalg.norm(xyi - xyj))
                        if d < best_d:
                            best_d, best_p = d, pj
                    if best_p is None or best_d > max_dist:
                        continue
                    sa, sb = pi.get("student_id"), best_p.get("student_id")
                    if sa and (not sb or _conf_rank(_conf_of(pi)) >= _conf_rank(_conf_of(best_p))):
                        if _should_overwrite(best_p, only_fix_low, cfg=cfg) and _assign(
                            best_p, sa, reason=f"court_xy:{cam_i}->{cam_j}",
                        ):
                            n_reassigned += 1
                            repairs.append({"from": sa, "cam": cam_j, "dist_m": round(best_d, 3)})
                    elif sb and _should_overwrite(pi, only_fix_low, cfg=cfg) and _assign(
                        pi, sb, reason=f"court_xy:{cam_j}->{cam_i}",
                    ):
                        n_reassigned += 1
                        repairs.append({"from": sb, "cam": cam_i, "dist_m": round(best_d, 3)})

    report["n_reassigned"] = n_reassigned
    report["repairs"] = repairs[:200]
    report["status"] = "ok"
    if write:
        for cam_id, doc in docs.items():
            path = data_path("sessions", session_id, "perception", cam_id, "pose2d.json")
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        out = data_path("sessions", session_id, "sync", "identity_spatial_repair.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["output"] = str(out)
    return report
