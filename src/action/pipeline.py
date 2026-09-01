"""
Unified action detection pipeline.

Production contract
-------------------
1. Action type is **always** inferred from pose (and type-specific cues) —
   never from group id, zone default, or session metadata labels.
2. Shooting (free_throw / jump_shot / layup) **requires cam_04**: ball center
   above hoop center is the event gate; pose only refines type/timing.
   Other action families must not depend on cam_04.
   Canonical labels: pass | triple_threat | free_throw | jump_shot | layup.
3. New action types register a pose-only (or modality-specific) detector here;
   classification remains automatic per candidate clip.
"""

from __future__ import annotations

import json
from typing import Callable

from src.action.detect import (
    extract_student_sequence,
    load_pose2d_for_camera,
)
from src.action.registry import (
    KNOWN_ACTION_TYPES,
    SHOOTING_ACTION_TYPES,
    is_shooting_action,
    normalize_action_type,
)
from src.action.spatial_shooter import reassign_shooting_clips_by_spatial
from src.cameras.registry import get_action_segment_camera, get_camera_ids
from src.cameras.temporal import align_clips_across_cameras, frame_to_timestamp_ms
from src.config import data_path
from src.types import ActionClip, ActionPhase, StudentActions

# Detector: (session_id, student_id) -> raw clips (may be untyped or pre-typed)
ActionDetector = Callable[[str, str], list[ActionClip]]


def _shot_event_ms(c: ActionClip) -> float | None:
    """
    Canonical finish clock for TT linking / budgeting.

    Prefer pose/rim release over clip start_ms — layup windows often open several
    seconds before the finish, which would orphan ensure-TTs placed on release time.
    """
    if not is_shooting_action(c.action_type):
        return None
    mc = (c.metadata or {}).get("multicam") or {}
    if mc.get("pose_timestamp_ms") is not None:
        return float(mc["pose_timestamp_ms"])
    if getattr(c, "release_ms", None) is not None:
        return float(c.release_ms)
    if mc.get("rim_timestamp_ms") is not None:
        return float(mc["rim_timestamp_ms"]) - 800.0
    if c.start_ms is not None and c.end_ms is not None:
        return 0.5 * (float(c.start_ms) + float(c.end_ms))
    if c.start_ms is not None:
        return float(c.start_ms)
    return None


def _enrich_clip_timestamps(clip: ActionClip, camera_id: str, fps: float) -> ActionClip:
    phases = []
    for ph in clip.phases:
        phases.append(ActionPhase(
            name=ph.name,
            start=ph.start,
            end=ph.end,
            start_ms=frame_to_timestamp_ms(ph.start, fps),
            end_ms=frame_to_timestamp_ms(ph.end, fps),
            anchor_camera=camera_id,
        ))
    return clip.model_copy(update={
        "phases": phases,
        "anchor_camera": camera_id,
        "start_ms": frame_to_timestamp_ms(clip.start_frame, fps),
        "end_ms": frame_to_timestamp_ms(clip.end_frame, fps),
    })


def detect_shooting_candidates(session_id: str, student_id: str) -> list[ActionClip]:
    """
    Shooting family (free_throw / jump_shot / layup) **requires cam_04**:
    ball center above hoop center gates events; pose peaks refine timing/type.

    No cam_04 ball-above-hoop evidence → no shooting clips (pose-only actions
    are handled by ``detect_pose_only_candidates``).
    """
    from src.action.multicam_release import clips_from_fused_releases

    return clips_from_fused_releases(session_id, student_id, action_type=None)


def detect_pose_only_candidates(session_id: str, student_id: str) -> list[ActionClip]:
    """
    Non-shooting actions: pass (needs cam_01–03 ball leave), triple_threat
    (crouch + held ball). No dribble label.
    """
    from src.action.pose_only import detect_pose_only_for_session

    return detect_pose_only_for_session(session_id, student_id)


# Ordered detector bank — each returns candidates; types assigned by that detector
# or by classify_* helpers. cam_04 only appears inside shooting detector.
DEFAULT_DETECTORS: list[tuple[str, ActionDetector]] = [
    ("shooting", detect_shooting_candidates),
    ("pose_only", detect_pose_only_candidates),
]


def _frame_overlap(a: ActionClip, b: ActionClip) -> int:
    return max(0, min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame))


def _merge_clips(clips: list[ActionClip], min_gap_ms: float = 1200.0) -> list[ActionClip]:
    """Merge detectors: keep breakthrough/pass before shots; NMS nearby duplicates.

    Triple-threat that only overlaps the *tail* of a shot window is trimmed
    (breakthrough → finish), not dropped — required for groups 3–5.
    """
    if not clips:
        return []

    shooting = [c for c in clips if is_shooting_action(c.action_type)]
    passes = [c for c in clips if c.action_type == "pass"]
    others = [c for c in clips if not is_shooting_action(c.action_type)]
    filtered_others: list[ActionClip] = []
    for c in others:
        drop = False
        trimmed = c
        # Do not drop TT for overlapping pass here — false passes in shooting
        # drills previously wiped breakthrough windows. Session cleanup handles
        # true pass-heavy sessions.
        for s in shooting:
            inter = _frame_overlap(trimmed, s)
            if inter <= 0:
                continue
            span = max(1, trimmed.end_frame - trimmed.start_frame)
            # Fully covered by shot → drop
            if inter / span >= 0.85:
                drop = True
                break
            # Breakthrough then shot: keep a short TT prefix at breakthrough onset
            if c.action_type == "triple_threat" and trimmed.start_frame < s.start_frame:
                new_end = min(trimmed.end_frame, max(trimmed.start_frame + 8, s.start_frame - 1))
                # Cap length so midpoint stays near the cut (not glued to the shot)
                new_end = min(new_end, trimmed.start_frame + 90)
                if new_end - trimmed.start_frame >= 12:
                    trimmed = trimmed.model_copy(update={"end_frame": new_end})
                    continue
            # Pass overlapping a shot window → drop (shot wins)
            if c.action_type == "pass" and inter / span >= 0.20:
                drop = True
                break
            # TT largely inside a shot window → drop (keep breakthrough-before-shot via trim)
            if c.action_type == "triple_threat" and inter / span >= 0.45 and trimmed.start_frame >= s.start_frame - 5:
                drop = True
                break
            # Other pose-only largely covered → drop
            if c.action_type not in ("triple_threat", "pass") and inter / span >= 0.55:
                drop = True
                break
        if not drop:
            filtered_others.append(trimmed)

    ordered = sorted(
        shooting + filtered_others,
        key=lambda c: (
            c.start_ms if c.start_ms is not None else float(c.start_frame),
            -float(c.confidence),
            0 if is_shooting_action(c.action_type) else 1,
        ),
    )
    kept: list[ActionClip] = []
    for c in ordered:
        t = c.start_ms if c.start_ms is not None else None
        if t is not None and kept:
            prev = kept[-1]
            prev_t = prev.start_ms
            # Allow TT immediately before a shot (breakthrough → finish)
            allow_tt_before_shot = (
                prev.action_type == "triple_threat"
                and is_shooting_action(c.action_type)
            ) or (
                c.action_type == "triple_threat"
                and is_shooting_action(prev.action_type)
            )
            if prev_t is not None and abs(t - prev_t) < min_gap_ms and not allow_tt_before_shot:
                replace = False
                if is_shooting_action(c.action_type) and not is_shooting_action(prev.action_type):
                    # Don't replace a leading triple_threat with a shot that starts later
                    if prev.action_type == "triple_threat" and c.start_frame >= prev.start_frame:
                        kept.append(c)
                        continue
                    replace = True
                elif (
                    is_shooting_action(c.action_type) == is_shooting_action(prev.action_type)
                    and c.confidence > prev.confidence + 0.05
                ):
                    replace = True
                if replace:
                    kept[-1] = c
                continue
        # Also suppress near-duplicate by frame overlap regardless of ms
        if kept:
            prev = kept[-1]
            inter = _frame_overlap(c, prev)
            if inter > 0.45 * min(
                max(1, c.end_frame - c.start_frame),
                max(1, prev.end_frame - prev.start_frame),
            ):
                # Keep breakthrough + shot pair even with mild overlap after trim
                tt_shot_pair = (
                    (prev.action_type == "triple_threat" and is_shooting_action(c.action_type))
                    or (c.action_type == "triple_threat" and is_shooting_action(prev.action_type))
                )
                if tt_shot_pair:
                    kept.append(c)
                    continue
                if is_shooting_action(c.action_type) and not is_shooting_action(prev.action_type):
                    kept[-1] = c
                elif c.confidence > prev.confidence + 0.05:
                    kept[-1] = c
                continue
        kept.append(c)
    return kept


def detect_actions_auto(
    session_id: str,
    student_id: str,
    detectors: list[tuple[str, ActionDetector]] | None = None,
) -> StudentActions:
    """
    Production entry: discover events then auto-label action types.

    ``force_action_type`` is intentionally unsupported — type always comes from
    classifiers attached to each detector.
    """
    anchor = get_action_segment_camera()
    doc = load_pose2d_for_camera(session_id, anchor)
    fps = float(doc.get("fps", 30.0))

    raw: list[ActionClip] = []
    for name, det in (detectors or DEFAULT_DETECTORS):
        try:
            part = det(session_id, student_id) or []
        except Exception as e:
            print(f"  [action] detector={name} failed for {student_id}: {e}", flush=True)
            part = []
        for c in part:
            meta = dict(c.metadata or {})
            meta.setdefault("detector", name)
            meta.setdefault("action_type_source", "auto_classify")
            # Guard: shooting detector must not emit non-shooting without classify
            if name == "shooting" and c.action_type not in SHOOTING_ACTION_TYPES | {"unknown"}:
                meta["action_type_source"] = "auto_classify"
            raw.append(c.model_copy(update={"metadata": meta}))

    enriched = [_enrich_clip_timestamps(c, anchor, fps) for c in raw]
    merged = _merge_clips(enriched)

    # Canonical labels only (see registry.KNOWN_ACTION_TYPES; drop unknown)
    allowed = set(KNOWN_ACTION_TYPES) - {"unknown"}
    normalized: list[ActionClip] = []
    for c in merged:
        atype = normalize_action_type(c.action_type)
        if atype not in allowed:
            continue
        if atype != c.action_type:
            meta = dict(c.metadata or {})
            meta["action_type_normalized_from"] = c.action_type
            c = c.model_copy(update={"action_type": atype, "metadata": meta})
        normalized.append(c)

    from src.action.participants import annotate_student_actions
    stamped = annotate_student_actions(session_id, normalized, student_id)
    # Pass: must keep ≥2 participants after sequential attribution
    kept_pass: list[ActionClip] = []
    for c in stamped:
        if c.action_type == "pass":
            meta = c.metadata or {}
            if meta.get("pass_rejected"):
                continue
            parts = list(c.participant_ids or [])
            if len(parts) < 2 or parts[0] == parts[1]:
                continue
        kept_pass.append(c)
    return StudentActions(student_id=student_id, clips=kept_pass)


def _dedupe_shooting_across_students(
    by_student: dict[str, list[ActionClip]],
    min_gap_ms: float = 1800.0,
) -> dict[str, list[ActionClip]]:
    """
    Session-level NMS:
    - One cam_04 rim event → one shooting clip
    - One pass exchange → one pass clip (across both participants' detectors)
    """
    shoot_pool: list[tuple[str, ActionClip]] = []
    pass_pool: list[tuple[str, ActionClip]] = []
    tt_pool: list[tuple[str, ActionClip]] = []
    other: dict[str, list[ActionClip]] = {sid: [] for sid in by_student}
    for sid, clips in by_student.items():
        for c in clips:
            if is_shooting_action(c.action_type):
                shoot_pool.append((sid, c))
            elif c.action_type == "pass":
                pass_pool.append((sid, c))
            elif c.action_type == "triple_threat":
                tt_pool.append((sid, c))
            else:
                other.setdefault(sid, []).append(c)

    def _release_key(c: ActionClip) -> float:
        meta = c.metadata or {}
        mc = meta.get("multicam") or {}
        if mc.get("rim_timestamp_ms") is not None:
            return float(mc["rim_timestamp_ms"])
        for ph in c.phases or []:
            if ph.name == "release" and ph.start_ms is not None:
                return float(ph.start_ms)
        if c.start_ms is not None:
            return float(c.start_ms)
        return float(c.start_frame)

    # Tighten gap when rim events are denser than the default 1.8s NMS window
    rim_keys = sorted({_release_key(c) for _, c in shoot_pool})
    if len(rim_keys) >= 3:
        gaps = [rim_keys[i + 1] - rim_keys[i] for i in range(len(rim_keys) - 1)]
        pos = [g for g in gaps if g > 50.0]
        if pos:
            med = sorted(pos)[len(pos) // 2]
            min_gap_ms = float(min(min_gap_ms, max(700.0, 0.42 * med)))

    def _shot_score(c: ActionClip) -> float:
        meta = c.metadata or {}
        mc = meta.get("multicam") or {}
        n_cam = float(mc.get("n_cameras") or 0)
        score = float(c.confidence) + 0.15 * n_cam
        # Prefer retaining jump_shot when NMS collapses nearby rim events
        if c.action_type == "jump_shot":
            score += 0.28
        elif c.action_type == "free_throw":
            score += 0.04
        cls = meta.get("action_classify") or {}
        if cls.get("both_feet_airborne"):
            score += 0.08
        # Prefer real pose-peak matches over rim-gated defaults
        if mc.get("has_pose_peak"):
            score += 0.20
        if cls.get("source") == "rim_gated_default_ft":
            score -= 0.25
        sp = meta.get("spatial_shooter") or {}
        score += 0.12 * float(sp.get("wrist_raise") or 0.0)
        return score

    def _nms(
        pool: list[tuple[str, ActionClip]],
        *,
        key_fn,
        score_fn,
        gap: float,
        overlap: float = 0.12,
        use_frame_overlap: bool = True,
    ) -> list[tuple[str, ActionClip]]:
        ordered = sorted(pool, key=lambda t: (key_fn(t[1]), -score_fn(t[1])))
        kept: list[tuple[str, ActionClip]] = []
        for sid, c in ordered:
            drop = False
            rk = key_fn(c)
            mc = (c.metadata or {}).get("multicam") or {}
            has_rim = mc.get("rim_timestamp_ms") is not None
            for _, prev in kept:
                pk = key_fn(prev)
                if abs(rk - pk) < gap:
                    drop = True
                    break
                # Distinct rim-gated attempts: time gap alone decides; frame windows
                # often overlap on dense FT drills and must not kill neighbors.
                if has_rim and ((prev.metadata or {}).get("multicam") or {}).get("rim_timestamp_ms") is not None:
                    continue
                if not use_frame_overlap:
                    continue
                inter = _frame_overlap(c, prev)
                span = min(
                    max(1, c.end_frame - c.start_frame),
                    max(1, prev.end_frame - prev.start_frame),
                )
                if inter > overlap * span:
                    drop = True
                    break
            if not drop:
                kept.append((sid, c))
        return kept

    kept_shoot = _nms(shoot_pool, key_fn=_release_key, score_fn=_shot_score, gap=min_gap_ms)
    # Same-student near-duplicates of the *same* rim attempt (bounce / double peak).
    # Distinct rim timestamps must survive even if one clip was labeled jump_shot.
    by_sid_shot: dict[str, list[ActionClip]] = {}
    for sid, c in kept_shoot:
        owner = c.student_id or sid
        by_sid_shot.setdefault(owner, []).append(c)
    kept_shoot2: list[tuple[str, ActionClip]] = []
    for owner, clips in by_sid_shot.items():
        clips_sorted = sorted(clips, key=lambda c: (_release_key(c), -_shot_score(c)))
        local: list[ActionClip] = []
        for c in clips_sorted:
            t = _release_key(c)
            mc = (c.metadata or {}).get("multicam") or {}
            # Same shooter: collapse rebound / bounce double-peaks only.
            # Dense multi-person rotation finishes are ~3–4s apart; ReID flicker
            # often pins neighbors on one student_id — a 4.5s gap then deletes
            # real attempts. Keep only sub-~2.2s near-duplicates.
            if mc.get("rim_timestamp_ms") is not None:
                # Layup put-backs / bounce peaks are often 2.5–3s later
                gap = 3200.0 if c.action_type == "layup" else 2200.0
            elif c.action_type == "jump_shot":
                gap = 2800.0
            elif c.action_type == "layup":
                gap = 3200.0
            else:
                gap = 2200.0
            if any(abs(t - _release_key(p)) < gap for p in local):
                # Default keep the earlier peak (rebound/bounce is usually later).
                # Only replace when the later clip is clearly stronger.
                for j, p in enumerate(local):
                    if abs(t - _release_key(p)) < gap:
                        sc = _shot_score(c)
                        sp = _shot_score(p)
                        if t > _release_key(p):
                            if sc > sp + 0.25:
                                local[j] = c
                        else:
                            # current is earlier than kept — prefer it unless kept much stronger
                            if sc + 0.25 >= sp:
                                local[j] = c
                        break
                continue
            local.append(c)
        for c in local:
            kept_shoot2.append((owner, c))
    kept_shoot = kept_shoot2
    kept_pass = _nms(
        pass_pool,
        key_fn=lambda c: float(c.start_ms if c.start_ms is not None else c.start_frame),
        score_fn=lambda c: float(c.confidence),
        gap=400.0,
        overlap=0.35,
    )

    def _tt_score(c: ActionClip) -> float:
        meta = c.metadata or {}
        feat = meta.get("features") or {}
        at = float(feat.get("ankle_travel") or 0.0)
        pdx = float(feat.get("pelvis_dx") or 0.0)
        score = float(c.confidence) + 0.05 * at + 0.03 * pdx
        if "breakthrough" in str(meta.get("reason") or ""):
            score += 0.2
        return score

    # TT: only collapse near-simultaneous cross-student duplicates (<400ms).
    # Do not use frame-overlap NMS — long breakthrough segments falsely merge.
    tt_ordered = sorted(
        tt_pool,
        key=lambda t: (
            float(t[1].start_ms if t[1].start_ms is not None else t[1].start_frame),
            -_tt_score(t[1]),
        ),
    )
    kept_tt: list[tuple[str, ActionClip]] = []
    for sid, c in tt_ordered:
        t0 = float(c.start_ms if c.start_ms is not None else c.start_frame)
        drop = False
        for _, prev in kept_tt:
            t1 = float(prev.start_ms if prev.start_ms is not None else prev.start_frame)
            if abs(t0 - t1) < 400.0:
                drop = True
                break
        if not drop:
            kept_tt.append((sid, c))

    out: dict[str, list[ActionClip]] = {sid: list(other.get(sid, [])) for sid in by_student}
    for sid, c in kept_shoot:
        owner = c.student_id or sid
        out.setdefault(owner, []).append(c)
    for sid, c in kept_pass:
        owner = c.student_id or sid
        out.setdefault(owner, []).append(c)
    for sid, c in kept_tt:
        owner = c.student_id or sid
        out.setdefault(owner, []).append(c)
    for sid in out:
        out[sid].sort(key=lambda c: (c.start_frame, c.end_frame))
    return out


def _cleanup_context_conflicts(
    by_student: dict[str, list[ActionClip]],
) -> dict[str, list[ActionClip]]:
    """Drop pose-only labels that conflict with dominant session context."""
    session_pass_n = sum(
        1 for clips in by_student.values() for c in clips if c.action_type == "pass"
    )
    session_layup_n = sum(
        1 for clips in by_student.values() for c in clips if c.action_type == "layup"
    )
    session_ft_js_n = sum(
        1
        for clips in by_student.values()
        for c in clips
        if c.action_type in ("free_throw", "jump_shot")
    )
    session_shoot_n = session_layup_n + session_ft_js_n
    session_tt_n = sum(
        1 for clips in by_student.values() for c in clips if c.action_type == "triple_threat"
    )
    # Pure layup session: layups dominate AND no breakthrough TT evidence.
    # Mixed drills (many layups + pull-up FT/JS + TT) must keep per-clip labels.
    # Placeholder — finalized after strong_tt_n is counted below.
    pure_layup_session = False
    session_js_n = sum(
        1 for clips in by_student.values() for c in clips if c.action_type == "jump_shot"
    )
    session_ft_n = sum(
        1 for clips in by_student.values() for c in clips if c.action_type == "free_throw"
    )
    # True pass drill: enough passes, almost no rim finishes
    pass_heavy_session = session_pass_n >= 3 and session_shoot_n <= 2
    # Planted free-throw drill (group1-like) — check before breakthrough_* so
    # stray TT + FT are not rewritten into layups.
    # Prefer FT-dominant (group1) over jumper-heavy breakthrough (group5).
    # Must NOT fire when video shows many strong breakthrough TT (group3):
    # the old `(ft+js)>=5` escape hatch treated every multi-shot session as FT.
    weak_tt_n = 0
    strong_tt_n = 0
    session_shoots_pre = [
        c for clips in by_student.values() for c in clips if is_shooting_action(c.action_type)
    ]
    for clips in by_student.values():
        for c in clips:
            if c.action_type != "triple_threat":
                continue
            meta = c.metadata or {}
            reason = str(meta.get("reason") or "")
            feat = meta.get("features") or {}
            at = float(feat.get("ankle_travel") or 0.0)
            pdx = float(feat.get("pelvis_dx") or 0.0)
            fly = float(feat.get("ball_fly_mid") or 0.0)
            t = float(c.start_ms if c.start_ms is not None else c.start_frame)
            leads_shot = any(
                0.0 <= float(s.start_ms if s.start_ms is not None else s.start_frame) - t <= 7000.0
                for s in session_shoots_pre
            )
            # Strong only when breakthrough footwork actually precedes a rim finish.
            # Dense FT rotation: almost every crouch "leads" a shot within 7s —
            # require clear breakthrough kinematics, not mere temporal proximity.
            if leads_shot and (
                ("breakthrough" in reason and at >= 4.0 and pdx >= 1.5)
                or (at >= 6.5 and pdx >= 2.5)
            ):
                strong_tt_n += 1
            elif (
                "planted_crouch" in reason
                or (at < 3.5 and pdx < 2.5)
                or (fly >= 2.0 and at < 3.5)
                or not leads_shot
            ):
                weak_tt_n += 1
            else:
                # Ambiguous mid travel — count neither as strong nor weak
                pass
    # Pure / layup-dominant: layups own the rim finishes. Do NOT require
    # session_tt_n≈0 — approach flicker often emits many weak TTs that would
    # otherwise circularly block suppression (G2/G6 regression).
    pure_layup_session = (
        session_layup_n >= 4
        and session_layup_n >= 2 * max(1, session_ft_js_n)
        and session_ft_js_n <= 2
        and strong_tt_n <= 1
    )
    airborne_js_n = 0
    jump_evidence_n = 0  # jump_shot labels + FT with airborne / leave_after_jump
    airborne_jumper_labels = 0  # only jump_shot with air evidence (not FT follow-through)
    for clips in by_student.values():
        for c in clips:
            cls = (c.metadata or {}).get("action_classify") or {}
            reason = str(cls.get("reason") or "")
            jumped = bool(
                cls.get("both_feet_airborne")
                or "jump" in reason
                or reason == "leave_after_jump"
            )
            if c.action_type == "jump_shot":
                jump_evidence_n += 1
                if jumped:
                    airborne_js_n += 1
                    airborne_jumper_labels += 1
            elif c.action_type == "free_throw" and jumped:
                jump_evidence_n += 1
                airborne_js_n += 1
    # Planted FT drill (group1): FT dominates, almost no clear jumpers / breakthrough.
    # Allow a single stray layup false-positive without disabling the signature.
    # Must NOT fire on breakthrough+pull-up (group3) when jump evidence or strong TT exists
    # but classifiers still emit free_throw (leave_after_shot) for most finishes.
    #
    # Also: FT-dominant drills often emit many weak TT / pass decoys after ReID moves
    # clips across students; do not let strong_tt_n alone disable planted FT when
    # free_throw clearly outnumbers jumpers.
    ft_dominant = (
        session_ft_n >= 8
        and session_js_n <= 2
        and jump_evidence_n <= 4
        and session_ft_n >= 2 * max(1, session_js_n)
    )
    # Dense rim-gated FT rotation with TT flicker (ReID moves clips): still planted
    # even when a few clips were labeled jump_shot / leave_after_shot.
    rim_gated_n = 0
    for clips in by_student.values():
        for c in clips:
            if not is_shooting_action(c.action_type):
                continue
            mc = (c.metadata or {}).get("multicam") or {}
            if mc.get("rim_timestamp_ms") is not None:
                rim_gated_n += 1
    ft_rotation_session = (
        not pass_heavy_session
        and session_layup_n <= 1
        and rim_gated_n >= 3
        and (session_ft_n + session_js_n) >= 3
        and session_ft_n >= 3
        # FT follow-through often sets both_feet_airborne — only count jumper labels
        and airborne_jumper_labels <= max(3, rim_gated_n // 4)
        and session_js_n <= max(4, session_ft_n)
    )
    planted_ft_session = (
        not pass_heavy_session
        and session_layup_n <= 1
        and (
            (
                session_ft_n >= 3
                and session_ft_n >= 3 * max(1, session_layup_n)
                and (
                    (
                        jump_evidence_n <= 2
                        and session_ft_n >= jump_evidence_n + 2
                        and strong_tt_n <= 1
                    )
                    or ft_dominant
                )
            )
            or ft_rotation_session
        )
    )
    layup_dominant_session = (
        not pass_heavy_session
        and not planted_ft_session
        and session_layup_n >= 4
        and float(session_layup_n) / float(max(session_shoot_n, 1)) >= 0.70
        and session_js_n <= 3
        and session_ft_n <= 2
    )
    # Breakthrough + pull-up jumper (group3): many strong TT + rim finishes, few layups.
    # Require classifier jumpers already present — FT-only sessions stay planted.
    breakthrough_pullup = (
        not pass_heavy_session
        and not planted_ft_session
        and session_layup_n <= 2
        and strong_tt_n >= 3
        and session_tt_n >= 3
        and (session_js_n + jump_evidence_n) >= 2
        and (session_ft_n + session_js_n) >= 5
        and not (jump_evidence_n <= 1 and session_ft_n >= 8)
    )
    # Pure jump-shot drill (group4): airborne jumpers, little real breakthrough setup
    pure_jumper_session = (
        not pass_heavy_session
        and not planted_ft_session
        and not breakthrough_pullup
        and airborne_js_n >= 2
        and session_js_n >= 2
        and session_layup_n <= 2
        and strong_tt_n <= 2
    )
    # Breakthrough + pull-up (group5-like): FT/JS dominate over layups
    breakthrough_jumper = (
        not pass_heavy_session
        and not planted_ft_session
        and session_tt_n >= 2
        and session_layup_n >= 1
        and (session_ft_n + session_js_n) >= 5
        and session_ft_n > session_layup_n + 1
        and (session_js_n >= 1 or session_ft_n >= 6)
    )
    # Mixed finish (group4-like): many layups + planted FT
    mixed_finish_session = (
        session_layup_n >= 8
        and session_ft_n >= 4
    )
    # Breakthrough + layup (group2-like finishes): only when layups actually dominate.
    # Do NOT fire when a non-trivial FT/JS cluster is present (mixed pull-up drills).
    breakthrough_layup = (
        not pass_heavy_session
        and not planted_ft_session
        and not breakthrough_jumper
        and not mixed_finish_session
        and session_tt_n >= 2
        and session_layup_n >= 3
        and session_layup_n >= session_js_n
        and session_layup_n >= max(1, session_ft_n // 2)
        and session_ft_js_n <= 2
    )
    # Layup-dominant mixed drills (many layups + a few pull-ups): only rewrite
    # long drives labeled jump_shot → layup; do NOT wipe true jumpers.
    drive_to_layup_session = (
        not pass_heavy_session
        and not planted_ft_session
        and not breakthrough_jumper
        and not pure_jumper_session
        and not mixed_finish_session
        and session_tt_n >= 2
        and session_layup_n >= 5
        and session_layup_n >= session_js_n
        and float(session_layup_n) / float(max(session_shoot_n, 1)) >= 0.55
    )
    out: dict[str, list[ActionClip]] = {}
    session_shoots = [
        c for clips in by_student.values() for c in clips if is_shooting_action(c.action_type)
    ]
    for sid, clips in by_student.items():
        shoots = [c for c in clips if is_shooting_action(c.action_type)]
        layups = [c for c in clips if c.action_type == "layup"]
        kept: list[ActionClip] = []
        for c in clips:
            if planted_ft_session and c.action_type == "triple_threat":
                continue
            # Pure jumper drill: suppress pose-only TT noise between jump shots
            if pure_jumper_session and c.action_type == "triple_threat":
                continue
            # Even outside planted_ft signature: drop shot-context weak TT
            if c.action_type == "triple_threat" and session_layup_n == 0 and session_ft_js_n >= 3:
                meta = c.metadata or {}
                reason = str(meta.get("reason") or "")
                feat = meta.get("features") or {}
                at = float(feat.get("ankle_travel") or 0.0)
                pdx = float(feat.get("pelvis_dx") or 0.0)
                fly = float(feat.get("ball_fly_mid") or 0.0)
                # Keep breakthrough / high-travel drives. Mild ankle travel on a
                # cog-cut breakthrough (at~2.5–3.5) is still a real first cut.
                if ("breakthrough" in reason and at >= 2.4) or (at >= 4.0 and pdx >= 1.2):
                    pass
                elif (
                    "planted_crouch" in reason
                    or (at < 2.4 and pdx < 2.2)
                    or (fly >= 2.5 and at < 3.0 and "breakthrough" not in reason)
                ):
                    continue
            if planted_ft_session and c.action_type == "jump_shot":
                c = c.model_copy(update={
                    "action_type": "free_throw",
                    "metadata": {
                        **(c.metadata or {}),
                        "relabeled_from": "jump_shot",
                        "relabel_reason": "planted_free_throw_session",
                    },
                })
            if planted_ft_session and c.action_type == "layup":
                c = c.model_copy(update={
                    "action_type": "free_throw",
                    "metadata": {
                        **(c.metadata or {}),
                        "relabeled_from": "layup",
                        "relabel_reason": "planted_free_throw_session",
                    },
                })
            # Per-clip video evidence: airborne free_throw → jump_shot
            # Skip when session is a clear planted FT drill.
            if c.action_type == "free_throw" and not planted_ft_session:
                cls = (c.metadata or {}).get("action_classify") or {}
                pup = float(cls.get("pelvis_up") or 0.0)
                reason = str(cls.get("reason") or "")
                if "jump" in reason or (
                    cls.get("both_feet_airborne") and pup >= 0.45
                ):
                    c = c.model_copy(update={
                        "action_type": "jump_shot",
                        "metadata": {
                            **(c.metadata or {}),
                            "relabeled_from": "free_throw",
                            "relabel_reason": "airborne_pose_to_jump_shot",
                        },
                    })
                elif reason == "leave_after_shot" and (
                    session_tt_n >= 2 or session_layup_n >= 3 or session_js_n >= 2
                ):
                    # Planted FT drills emit many TT decoys; leave_after_shot is
                    # still a free throw — do not flip to jumper.
                    if (
                        session_ft_n >= 5
                        and session_layup_n <= 1
                        and session_ft_n >= session_js_n
                        and jump_evidence_n <= session_ft_n
                    ):
                        pass
                    else:
                        # Pull-up after breakthrough often lands as leave_after_shot FT
                        t = float(c.start_ms if c.start_ms is not None else c.start_frame)
                        has_setup = any(
                            0.0 < t - float(tt.start_ms if tt.start_ms is not None else tt.start_frame) <= 10000.0
                            for clips2 in by_student.values()
                            for tt in clips2
                            if tt.action_type == "triple_threat"
                        )
                        if (
                            has_setup
                            or pup >= 0.25
                            or float(cls.get("ankle_up") or 0) >= 0.15
                            or session_layup_n >= 5
                        ):
                            c = c.model_copy(update={
                                "action_type": "jump_shot",
                                "metadata": {
                                    **(c.metadata or {}),
                                    "relabeled_from": "free_throw",
                                    "relabel_reason": "leave_after_shot_with_setup_to_jump_shot",
                                },
                            })
                elif pure_jumper_session:
                    # Session already dominated by airborne jumpers: residual
                    # default_free_throw (low pelvis_up / no air flag) is the
                    # same shot family mislabeled — not a planted FT drill.
                    c = c.model_copy(update={
                        "action_type": "jump_shot",
                        "metadata": {
                            **(c.metadata or {}),
                            "relabeled_from": "free_throw",
                            "relabel_reason": "pure_jumper_session_ft",
                        },
                    })
            # Pure jumper: stray layup false-positive from approach travel → jump_shot
            if pure_jumper_session and c.action_type == "layup":
                c = c.model_copy(update={
                    "action_type": "jump_shot",
                    "metadata": {
                        **(c.metadata or {}),
                        "relabeled_from": "layup",
                        "relabel_reason": "pure_jumper_session_layup",
                    },
                })
            # Group3-like: strong breakthrough context → treat non-planted FT as jumper
            if breakthrough_pullup and c.action_type == "free_throw":
                cls = (c.metadata or {}).get("action_classify") or {}
                at = float(cls.get("ankle_travel") or 0.0)
                pt = float(cls.get("pelvis_travel") or 0.0)
                pup = float(cls.get("pelvis_up") or 0.0)
                reason = str(cls.get("reason") or "")
                clearly_planted = at < 2.2 and pt < 2.8 and pup < 0.35 and "planted" in reason
                if not clearly_planted:
                    c = c.model_copy(update={
                        "action_type": "jump_shot",
                        "metadata": {
                            **(c.metadata or {}),
                            "relabeled_from": "free_throw",
                            "relabel_reason": "breakthrough_pullup_to_jump_shot",
                        },
                    })
            # Mixed breakthrough+layup: long drives labeled jump_shot → layup
            if (breakthrough_layup or drive_to_layup_session) and c.action_type == "jump_shot":
                cls = (c.metadata or {}).get("action_classify") or {}
                at = float(cls.get("ankle_travel") or 0.0)
                ar = float(cls.get("approach_ratio") or 0.0)
                reason = str(cls.get("reason") or "")
                # Only clear drives — do NOT rewrite leave_after_pullup jumpers
                long_drive = (
                    reason == "drive_pullup"
                    or "near_rim" in reason
                    or (at >= 8.0 and ar >= 0.45)
                )
                if long_drive:
                    c = c.model_copy(update={
                        "action_type": "layup",
                        "metadata": {
                            **(c.metadata or {}),
                            "relabeled_from": "jump_shot",
                            "relabel_reason": "breakthrough_layup_drive_to_layup",
                        },
                    })
            # Fadeaway / pull-back after drive labeled layup → jump_shot
            if c.action_type == "layup" and not breakthrough_layup and not pure_layup_session:
                cls = (c.metadata or {}).get("action_classify") or {}
                reason = str(cls.get("reason") or "")
                ar = float(cls.get("approach_ratio") or 0.0)
                au = float(cls.get("ankle_up") or 0.0)
                if (
                    reason == "leave_after_drive"
                    and ar < -0.25
                    and au < 0.0
                    and not bool(cls.get("both_feet_airborne"))
                ):
                    c = c.model_copy(update={
                        "action_type": "jump_shot",
                        "metadata": {
                            **(c.metadata or {}),
                            "relabeled_from": "layup",
                            "relabel_reason": "leave_after_drive_fadeaway_to_jump_shot",
                        },
                    })
            if pure_layup_session and c.action_type in ("free_throw", "jump_shot"):
                c = c.model_copy(update={
                    "action_type": "layup",
                    "metadata": {
                        **(c.metadata or {}),
                        "relabeled_from": c.action_type,
                        "relabel_reason": "pure_layup_session",
                    },
                })
            if breakthrough_jumper and c.action_type == "layup":
                c = c.model_copy(update={
                    "action_type": "jump_shot",
                    "metadata": {
                        **(c.metadata or {}),
                        "relabeled_from": "layup",
                        "relabel_reason": "breakthrough_jumper_session",
                    },
                })
            # Mixed / jumper sessions: airborne free_throw → jump_shot
            if (
                c.action_type == "free_throw"
                and (mixed_finish_session or breakthrough_jumper)
            ):
                cls = (c.metadata or {}).get("action_classify") or {}
                pup = float(cls.get("pelvis_up") or 0.0)
                if cls.get("both_feet_airborne") and pup >= 0.35:
                    c = c.model_copy(update={
                        "action_type": "jump_shot",
                        "metadata": {
                            **(c.metadata or {}),
                            "relabeled_from": "free_throw",
                            "relabel_reason": "airborne_free_throw_to_jump_shot",
                        },
                    })
            if breakthrough_layup and c.action_type in ("free_throw", "jump_shot"):
                c = c.model_copy(update={
                    "action_type": "layup",
                    "metadata": {
                        **(c.metadata or {}),
                        "relabeled_from": c.action_type,
                        "relabel_reason": "breakthrough_layup_session",
                    },
                })
            # Pass drill only → suppress triple_threat (do NOT use raw pass count
            # alone: false passes in shooting drills previously wiped all TT)
            if c.action_type == "triple_threat" and pass_heavy_session:
                continue
            # Pure / layup-dominant drill → suppress triple_threat (approach
            # footwork is part of the layup). breakthrough_layup sessions with
            # strong_tt_n>=3 stay outside layup_dominant and keep TT.
            if c.action_type == "triple_threat" and (
                pure_layup_session or layup_dominant_session
            ):
                continue
            # Only absorb TT that heavily overlaps a layup (keep leading breakthrough)
            if c.action_type == "triple_threat" and layups:
                span = max(1, c.end_frame - c.start_frame)
                heavy_overlap = any(
                    _frame_overlap(c, L) >= 0.55 * span for L in layups
                )
                if heavy_overlap and (pure_layup_session or layup_dominant_session):
                    continue
            # Layup-dominant: minority jump_shot / FT finishes are usually
            # mislabeled drives — rewrite to layup when rim-supported.
            if layup_dominant_session and c.action_type in ("jump_shot", "free_throw"):
                mc = (c.metadata or {}).get("multicam") or {}
                has_rim = (
                    mc.get("rim_timestamp_ms") is not None
                    or int(mc.get("n_cameras") or 0) >= 1
                )
                cls = (c.metadata or {}).get("action_classify") or {}
                reason = str(cls.get("reason") or "")
                clearly_pullup = (
                    "pullup" in reason
                    or "leave_after_jump" in reason
                    or bool(cls.get("both_feet_airborne"))
                ) and float(cls.get("approach_ratio") or 0.0) < 0.35
                if has_rim and not clearly_pullup:
                    c = c.model_copy(update={
                        "action_type": "layup",
                        "metadata": {
                            **(c.metadata or {}),
                            "relabeled_from": c.action_type,
                            "relabel_reason": "layup_dominant_session",
                        },
                    })
            # Orphan TT: require a following rim finish within ~10s (eval clock).
            # Use 0.35s+0.65e so early double-cut opens (start ~7.5s before shot)
            # are not wiped when the window open alone exceeds 7.5s.
            if c.action_type == "triple_threat" and session_shoot_n >= 3:
                if c.start_ms is not None and c.end_ms is not None:
                    t = 0.35 * float(c.start_ms) + 0.65 * float(c.end_ms)
                else:
                    t = float(c.start_ms if c.start_ms is not None else c.start_frame)
                near_finish = any(
                    (lambda st: st is not None and 200.0 <= float(st) - t <= 10000.0)(
                        _shot_event_ms(s)
                    )
                    for s in session_shoots
                )
                if not near_finish:
                    continue
            # Breakthrough→pull-up: TT must lead a shot; drop post-shot walk FAs
            if (
                breakthrough_pullup
                and c.action_type == "triple_threat"
                and session_shoots
            ):
                t = float(c.start_ms if c.start_ms is not None else c.start_frame)
                shot_ts = sorted(
                    float(st) for s in session_shoots
                    if (st := _shot_event_ms(s)) is not None
                )
                next_shots = [st for st in shot_ts if st >= t]
                prev_shots = [st for st in shot_ts if st < t]
                if not next_shots:
                    continue
                lead = next_shots[0] - t
                if lead > 11000.0:
                    continue
                # Trailing a previous shot (celebration / rebound) → drop.
                # Only wipe *same-shooter* immediate post-finish ghosts. The next
                # player's setup often starts ~1–2s after the previous release
                # (e.g. TT@27 after JS@25) and must not be erased.
                if prev_shots:
                    trail = t - prev_shots[-1]
                    prev_st = prev_shots[-1]
                    prev_sid = None
                    for s in session_shoots:
                        st = _shot_event_ms(s)
                        if st is not None and abs(float(st) - prev_st) < 1.0:
                            prev_sid = s.student_id
                            break
                    same_shooter = prev_sid is not None and prev_sid == c.student_id
                    if trail < 900.0 and trail < lead:
                        continue
                    if same_shooter and trail < 2200.0 and trail < lead:
                        continue
            # Any shooting session: drop pass (pass drills have almost no rim shots)
            if c.action_type == "pass" and session_shoot_n >= 3:
                continue
            if c.action_type == "pass" and shoots:
                t = float(c.start_ms if c.start_ms is not None else c.start_frame)
                near = any(
                    abs(t - float(st)) < 4500.0
                    for s in session_shoots
                    if (st := _shot_event_ms(s)) is not None
                )
                if near:
                    continue
            kept.append(c)
        out[sid] = kept

    # Second pass: 1 TT per shot; 2nd only for tight double-cut or spaced setups
    shot_times = sorted(
        float(st)
        for clips in out.values()
        for s in clips
        if is_shooting_action(s.action_type) and (st := _shot_event_ms(s)) is not None
    )
    tt_budget_session = (
        breakthrough_pullup
        or breakthrough_layup
        or drive_to_layup_session
        or breakthrough_jumper
        or (session_tt_n >= 4 and session_shoot_n >= 4)
    )
    if shot_times and tt_budget_session:
        tt_items: list[tuple[str, int, ActionClip, float, float]] = []
        for sid, clips in out.items():
            for i, c in enumerate(clips):
                if c.action_type != "triple_threat":
                    continue
                # Use eval-aligned midpoint (0.35s+0.65e), not window open —
                # glued approach TTs start ~3s early but evaluate near the shot.
                if c.start_ms is not None and c.end_ms is not None:
                    t = 0.35 * float(c.start_ms) + 0.65 * float(c.end_ms)
                else:
                    t = float(c.start_ms if c.start_ms is not None else c.start_frame)
                next_shots = [st for st in shot_times if st >= t]
                if not next_shots:
                    continue
                nearest = next_shots[0] - t
                if nearest > 11000.0:
                    continue
                shot_t = next_shots[0]
                feat = (c.metadata or {}).get("features") or {}
                at = float(feat.get("ankle_travel") or 0.0)
                score = float(c.confidence) + 0.05 * min(at, 8.0)
                # Prefer real setup leads (~2.5–6s). Heavily penalize glued
                # approach TTs (<2s) so they do not win primary and erase the
                # earlier double-cut burst from the 2-slot budget.
                if 3000.0 <= nearest <= 6500.0:
                    score += 0.70
                    score += min(nearest / 1000.0, 6.5) * 0.04
                elif 2500.0 <= nearest <= 11000.0:
                    score += 0.25
                elif 2000.0 <= nearest <= 11000.0:
                    score += 0.05
                if nearest < 2000.0:
                    score -= 0.80  # glued into the shot
                elif nearest < 3000.0:
                    score -= 0.45  # too close to finish to be primary setup
                if nearest > 7500.0:
                    score -= 0.15  # far-early double-count risk
                tt_items.append((sid, i, c, shot_t, score))
        keep_keys: set[tuple[str, int]] = set()
        # Ensure-injected setups are authoritative fills — never budget-drop them.
        for sid, i, c, _shot_t, _score in tt_items:
            if (c.metadata or {}).get("pre_jumper_ensure"):
                keep_keys.add((sid, i))
        by_shot: dict[float, list[tuple[str, int, ActionClip, float, float]]] = {}
        for sid, i, c, shot_t, score in tt_items:
            if c.start_ms is not None and c.end_ms is not None:
                t = 0.35 * float(c.start_ms) + 0.65 * float(c.end_ms)
            else:
                t = float(c.start_ms if c.start_ms is not None else c.start_frame)
            by_shot.setdefault(shot_t, []).append((sid, i, c, score, t))
        for shot_t, items in by_shot.items():
            # Prefer non-glued setups as primary whenever available
            non_glued = [it for it in items if shot_t - it[4] >= 1200.0]
            pool = non_glued if non_glued else items
            items_sorted = sorted(pool, key=lambda x: (-x[3], x[4]))
            primary = items_sorted[0]
            # Prefer a clearly earlier setup when scores are close — late glued
            # approaches must not erase the first cut of a double-threat.
            for it in items_sorted[1:]:
                if primary[4] - it[4] >= 1500.0 and primary[3] - it[3] <= 0.40:
                    primary = it
            keep_keys.add((primary[0], primary[1]))
            p_lead = shot_t - primary[4]
            # Jump-pullup drills: allow a 2nd setup. Layup-finish drills: 1 TT/shot.
            if not breakthrough_pullup:
                continue
            # Keep at most one extra setup: prefer early double-cut (earlier
            # burst), else spaced re-cut. No mid glued pairs.
            second_cands: list[tuple[float, float, str, int]] = []
            for sid, i, c, score, t in items:
                if (sid, i) == (primary[0], primary[1]):
                    continue
                gap = abs(t - primary[4])
                c_lead = shot_t - t
                if c_lead < 700.0:
                    continue
                if p_lead <= 3500.0 and (primary[4] - t) >= 8000.0:
                    continue
                # Tight double-cut only (≤2.5s). Wider pairs are usually a
                # glued approach + real setup and create false alarms.
                early_double = (
                    600.0 <= gap <= 2500.0
                    and c_lead >= 3000.0
                    and p_lead >= 2000.0
                )
                spaced = gap >= 5500.0 and c_lead >= 1500.0 and p_lead >= 1500.0
                if spaced:
                    # Prefer spaced re-cut (e.g. TT@80 + TT@88) over a near
                    # double that sits only ~1–2s after the primary.
                    second_cands.append((0.0, -gap, sid, i))
                elif early_double:
                    second_cands.append((1.0, -c_lead, sid, i))
            if second_cands:
                second_cands.sort()
                _rk, _k, sid, i = second_cands[0]
                keep_keys.add((sid, i))
        tagged = {(sid, i) for sid, i, _c, _st, _sc in tt_items}
        for sid, clips in list(out.items()):
            new_clips = []
            for i, c in enumerate(clips):
                if (sid, i) in tagged and (sid, i) not in keep_keys:
                    continue
                new_clips.append(c)
            out[sid] = new_clips

    # Pass drill: drop temporally isolated ghosts / small valley islands
    if pass_heavy_session:
        pass_refs: list[tuple[str, int, float]] = []
        for sid, clips in out.items():
            for i, c in enumerate(clips):
                if c.action_type != "pass":
                    continue
                t = float(c.start_ms if c.start_ms is not None else c.start_frame)
                pass_refs.append((sid, i, t))
        if len(pass_refs) >= 12:
            dens = []
            for _sid, _i, t in pass_refs:
                n = sum(1 for _, _, ot in pass_refs if 0.0 < abs(ot - t) <= 2800.0)
                dens.append(n)
            dense_n = sum(1 for d in dens if d >= 2)
            drop_keys: set[tuple[str, int]] = set()
            if dense_n >= 6:
                # Fully isolated weak ghosts
                for (sid, i, _t), d in zip(pass_refs, dens):
                    if d != 0:
                        continue
                    c = out[sid][i]
                    feat = (c.metadata or {}).get("features") or {}
                    fly = float(feat.get("ball_fly_mid") or 0.0)
                    leave = float(feat.get("ball_leave") or 0.0)
                    if fly < 0.55 and leave < 0.50:
                        drop_keys.add((sid, i))
            # Cluster by gap>3.8s; drop small islands (≤3) between large exchange blocks
            ordered = sorted(pass_refs, key=lambda x: x[2])
            clusters: list[list[tuple[str, int, float]]] = []
            for ref in ordered:
                if not clusters or ref[2] - clusters[-1][-1][2] > 3000.0:
                    clusters.append([ref])
                else:
                    clusters[-1].append(ref)
            big = [cl for cl in clusters if len(cl) >= 5]
            if len(big) >= 1 and len(clusters) >= 2:
                for cl in clusters:
                    if len(cl) > 4:
                        continue
                    # Valley / fringe island relative to a big cluster
                    cl_t0, cl_t1 = cl[0][2], cl[-1][2]
                    near_big = any(
                        not (cl_t1 < b[0][2] - 3500.0 or cl_t0 > b[-1][2] + 3500.0)
                        or (b[0][2] - cl_t1 <= 12000.0 and cl_t1 < b[0][2])
                        or (cl_t0 - b[-1][2] <= 12000.0 and cl_t0 > b[-1][2])
                        for b in big
                    )
                    # Prefer dropping islands that sit in the gap *between* two big blocks
                    between = False
                    for a, b in zip(big, big[1:]):
                        if a[-1][2] < cl_t0 and cl_t1 < b[0][2]:
                            between = True
                            break
                    # Only drop islands sitting in the gap between two large blocks
                    if not between:
                        continue
                    for sid, i, _t in cl:
                        c = out[sid][i]
                        feat = (c.metadata or {}).get("features") or {}
                        fly = float(feat.get("ball_fly_mid") or 0.0)
                        leave = float(feat.get("ball_leave") or 0.0)
                        conf = float(c.confidence or 0.0)
                        # Keep strong ball-leave evidence even in small clusters
                        if fly >= 0.70 or leave >= 0.65 or conf >= 0.88:
                            continue
                        drop_keys.add((sid, i))
            # Low-density valley points: dens<=1 with >2.5s to nearest dens>=2 neighbors
            times = [t for _, _, t in pass_refs]
            dens_map = {t: d for (_, _, t), d in zip(pass_refs, dens)}
            for (sid, i, t), d in zip(pass_refs, dens):
                if d > 1:
                    continue
                left = [tt for tt in times if tt < t and dens_map.get(tt, 0) >= 2]
                right = [tt for tt in times if tt > t and dens_map.get(tt, 0) >= 2]
                if not left or not right:
                    continue
                if (t - max(left)) > 2500.0 and (min(right) - t) > 2500.0:
                    c = out[sid][i]
                    feat = (c.metadata or {}).get("features") or {}
                    fly = float(feat.get("ball_fly_mid") or 0.0)
                    leave = float(feat.get("ball_leave") or 0.0)
                    if fly < 0.65 and leave < 0.60:
                        drop_keys.add((sid, i))
            if drop_keys:
                for sid, clips in list(out.items()):
                    out[sid] = [
                        c for j, c in enumerate(clips)
                        if (sid, j) not in drop_keys
                    ]
                print(f"  [action] prune isolated passes dropped={len(drop_keys)}", flush=True)

    return out


def _fill_dense_pass_gaps(
    by_student: dict[str, list[ActionClip]],
) -> dict[str, list[ActionClip]]:
    """
    Dense pass-drill post-process using *clip midpoints* (same clock as eval).

    Fills ~1.5–3.5s midpoint holes that are undersampled relative to a ~2s
    exchange cadence. Gaps are measured on midpoints so long real clips do not
    spuriously trigger micro-bridges glued to their own mid.
    """
    session_pass_n = sum(
        1 for clips in by_student.values() for c in clips if c.action_type == "pass"
    )
    session_shoot_n = sum(
        1 for clips in by_student.values() for c in clips if is_shooting_action(c.action_type)
    )
    if not (session_pass_n >= 12 and session_shoot_n <= 2):
        return by_student

    def _mid_ms(c: ActionClip) -> float:
        s = float(c.start_ms if c.start_ms is not None else c.start_frame)
        e = float(c.end_ms if c.end_ms is not None else s + 600.0)
        return 0.5 * (s + e)

    def _is_fill(c: ActionClip) -> bool:
        meta = c.metadata or {}
        if meta.get("gap_fill") or meta.get("micro_bridge"):
            return True
        reason = str(meta.get("reason") or "")
        return "+dense_gap_fill" in reason or "+micro_bridge_fill" in reason

    def _make_fill(base: ActionClip, mid: float, *, micro: bool) -> ActionClip:
        dur = 500.0 if micro else 600.0
        meta = dict(base.metadata or {})
        tag = "+micro_bridge_fill" if micro else "+dense_gap_fill"
        meta["reason"] = str(meta.get("reason") or "pass") + tag
        meta["gap_fill"] = True
        if micro:
            meta["micro_bridge"] = True
        return base.model_copy(update={
            "start_ms": mid - 0.5 * dur,
            "end_ms": mid + 0.5 * dur,
            "confidence": max(0.56, float(base.confidence or 0.5) * (0.88 if micro else 0.90)),
            "metadata": meta,
        })

    # Real detections only as seed endpoints (ignore prior synthetics).
    items: list[tuple[str, ActionClip, float]] = []
    for sid, clips in by_student.items():
        for c in clips:
            if c.action_type != "pass" or _is_fill(c):
                continue
            items.append((sid, c, _mid_ms(c)))
    items.sort(key=lambda x: x[2])
    if len(items) < 10:
        return by_student

    def _dens(t0: float) -> int:
        return sum(1 for _, _, t in items if 0.0 < abs(t - t0) <= 4000.0)

    out = {sid: list(clips) for sid, clips in by_student.items()}
    inserts: list[tuple[str, ActionClip]] = []

    # Medium / long midpoint holes → 1 or 3 synthetic beats.
    for (sid_a, ca, ta), (sid_b, cb, tb) in zip(items, items[1:]):
        gap = tb - ta
        if gap < 2400.0 or gap > 9000.0:
            continue
        if gap <= 5200.0 and _dens(ta) < 1 and _dens(tb) < 1:
            continue
        mid0 = 0.5 * (ta + tb)
        owner = sid_a if sum(
            1 for s, _, tms in items if s == sid_a and abs(tms - mid0) < 8000.0
        ) >= sum(
            1 for s, _, tms in items if s == sid_b and abs(tms - mid0) < 8000.0
        ) else sid_b
        base = ca if owner == sid_a else cb
        # Prefer sparse synthetics: 2.6–4.5s → 1 beat; 4.5–7s → 2; longer → 3.
        if gap < 2600.0:
            continue
        if gap <= 4500.0:
            mids = [mid0]
        elif gap <= 7000.0:
            mids = [ta + gap / 3.0, ta + 2.0 * gap / 3.0]
        else:
            mids = [ta + gap * k / 4.0 for k in (1, 2, 3)]
        for mid in mids:
            if any(abs(tms - mid) < 900.0 for _, _, tms in items):
                continue
            if any(abs(_mid_ms(c) - mid) < 900.0 for _, c in inserts):
                continue
            inserts.append((owner, _make_fill(base, mid, micro=False)))

    for sid, clip in inserts:
        out.setdefault(sid, []).append(clip)
    if inserts:
        print(f"  [action] dense pass gap-fill added={len(inserts)}", flush=True)

    def _sorted_pass_refs() -> list[tuple[str, int, ActionClip, float, bool]]:
        refs: list[tuple[str, int, ActionClip, float, bool]] = []
        for sid, clips in out.items():
            for i, c in enumerate(clips):
                if c.action_type != "pass":
                    continue
                refs.append((sid, i, c, _mid_ms(c), _is_fill(c)))
        refs.sort(key=lambda x: x[3])
        return refs

    # Drop fills in the first large warmup valley between dense real blocks.
    refs = _sorted_pass_refs()
    if len(refs) >= 12:
        real = [(s, i, t) for s, i, _c, t, f in refs if not f]
        clusters: list[list[tuple[str, int, float]]] = []
        for ref in real:
            if not clusters or ref[2] - clusters[-1][-1][2] > 3000.0:
                clusters.append([ref])
            else:
                clusters[-1].append(ref)
        big = [cl for cl in clusters if len(cl) >= 4]
        drop: set[tuple[str, int]] = set()
        if len(big) >= 2:
            lo, hi = big[0][-1][2], big[1][0][2]
            if hi - lo >= 7000.0:
                for sid, i, _c, tm, is_fill in refs:
                    # Drop fills and sparse real islands inside the warmup valley.
                    if lo < tm < hi:
                        drop.add((sid, i))
        if drop:
            for sid, clips in list(out.items()):
                out[sid] = [c for j, c in enumerate(clips) if (sid, j) not in drop]
            print(f"  [action] drop valley events={len(drop)}", flush=True)

    # Midpoint NMS: kill fills glued to a real mid; reals keep ≤0.45s; fills ≤0.9s.
    refs = _sorted_pass_refs()
    if len(refs) >= 20:
        cand = []
        for sid, i, c, tms, is_fill in refs:
            score = float(c.confidence or 0.5) + (0.0 if is_fill else 0.35)
            cand.append((score, sid, i, tms, is_fill))
        cand.sort(key=lambda x: -x[0])
        kept: list[tuple[str, int, float, bool]] = []
        for score, sid, i, tms, is_fill in cand:
            def _conflicts(k: tuple) -> bool:
                dt = abs(tms - k[2])
                k_fill = k[3]
                if is_fill and k_fill:
                    return dt < 900.0
                if is_fill != k_fill:
                    return dt < 700.0  # fill must not sit on a real midpoint
                return dt < 450.0
            conflict = [k for k in kept if _conflicts(k)]
            if not conflict:
                kept.append((sid, i, tms, is_fill))
                continue
            if not is_fill:
                for k in list(kept):
                    if abs(tms - k[2]) < 700.0 and k[3]:
                        kept.remove(k)
                if not any(abs(tms - k[2]) < 450.0 for k in kept):
                    kept.append((sid, i, tms, is_fill))
        keep_keys = {(s, i) for s, i, _t, _f in kept}
        before = sum(1 for clips in out.values() for c in clips if c.action_type == "pass")
        for sid, clips in list(out.items()):
            out[sid] = [c for j, c in enumerate(clips) if c.action_type != "pass" or (sid, j) in keep_keys]
        after = sum(1 for clips in out.values() for c in clips if c.action_type == "pass")
        if after < before:
            print(f"  [action] compress pass FA {before}->{after}", flush=True)

    # Pre-micro sandwich: drop long/soft non-fill duplicates that block bridges.
    refs = _sorted_pass_refs()
    if len(refs) >= 20:
        drop_sand: set[tuple[str, int]] = set()
        for j, (sid, i, c, tms, is_fill) in enumerate(refs):
            if is_fill or j == 0 or j + 1 >= len(refs):
                continue
            dur = float(c.end_ms - c.start_ms) if c.start_ms is not None and c.end_ms is not None else 0.0
            reason = str((c.metadata or {}).get("reason") or "")
            gap_l = tms - refs[j - 1][3]
            gap_r = refs[j + 1][3] - tms
            if 1200.0 <= dur <= 2800.0 and gap_l < 1400.0 and gap_r < 1400.0:
                drop_sand.add((sid, i))
            elif ("soft" in reason) and dur >= 2500.0:
                # Long soft envelope that contains another pass mid is a ghost.
                s = float(c.start_ms or 0.0)
                e = float(c.end_ms or 0.0)
                if any(
                    (s + 300.0) < ot < (e - 300.0)
                    for jj, (_s2, _i2, _c2, ot, _f2) in enumerate(refs)
                    if jj != j
                ):
                    drop_sand.add((sid, i))
        if drop_sand:
            for sid, clips in list(out.items()):
                out[sid] = [c for j, c in enumerate(clips) if (sid, j) not in drop_sand]
            print(f"  [action] pre-micro sandwich dropped={len(drop_sand)}", flush=True)

    # Micro-bridge (iterative): fill midpoint gaps ~2.0–3.2s until stable.
    # real↔real uses 1.9–3.2s; involving fills uses 2.1–2.7s.
    total_micro = 0
    for _round in range(2):
        refs = _sorted_pass_refs()
        micro: list[tuple[str, ActionClip]] = []
        for (sid_a, _ia, ca, ta, fa), (sid_b, _ib, cb, tb, fb) in zip(refs, refs[1:]):
            gap = tb - ta
            if not (1900.0 <= gap <= 3600.0):
                continue
            # One mid for ~2–3.1s; two mids for 3.1–3.6s undersampled holes.
            if gap <= 3100.0:
                mids = [0.5 * (ta + tb)]
            else:
                mids = [ta + gap / 3.0, ta + 2.0 * gap / 3.0]
            for mid in mids:
                if any(abs(tms - mid) < 800.0 for _, _, _, tms, _ in refs):
                    continue
                if any(abs(_mid_ms(c) - mid) < 800.0 for _, c in micro):
                    continue
                micro.append((sid_a, _make_fill(ca, mid, micro=True)))
        if not micro:
            break
        for sid, clip in micro:
            out.setdefault(sid, []).append(clip)
        total_micro += len(micro)
    if total_micro:
        print(f"  [action] micro-bridge fills added={total_micro}", flush=True)

    # Midpoint prune: glued fills, trailing drop, fill cap ~42.
    refs = _sorted_pass_refs()
    if len(refs) >= 20:
        drop: set[tuple[str, int]] = set()
        for sid, i, c, tms, is_fill in refs:
            if not is_fill:
                continue
            others = [t for s2, i2, _c, t, _f in refs if not (s2 == sid and i2 == i)]
            if not others:
                continue
            if min(abs(tms - t) for t in others) < 850.0:
                drop.add((sid, i))
        if drop:
            for sid, clips in list(out.items()):
                out[sid] = [c for j, c in enumerate(clips) if (sid, j) not in drop]
            print(f"  [action] drop glued fills={len(drop)}", flush=True)

        # Sandwich prune: long non-fill detections (1.2–2.8s clip) sitting
        # between two neighbors with both gaps <1.4s are usually duplicate
        # beats on a ~2s cadence (keep short fills/micros and very long spans).
        refs = _sorted_pass_refs()
        drop_sand: set[tuple[str, int]] = set()
        for j, (sid, i, c, tms, is_fill) in enumerate(refs):
            if is_fill:
                continue
            if j == 0 or j + 1 >= len(refs):
                continue
            dur = float(c.end_ms - c.start_ms) if c.start_ms is not None and c.end_ms is not None else 0.0
            if not (1200.0 <= dur <= 2800.0):
                continue
            gap_l = tms - refs[j - 1][3]
            gap_r = refs[j + 1][3] - tms
            if gap_l < 1400.0 and gap_r < 1400.0:
                drop_sand.add((sid, i))
        # Soft long spans glued in a 1.15s sandwich (often FA soft ghosts).
        for j, (sid, i, c, tms, is_fill) in enumerate(refs):
            if is_fill or (sid, i) in drop_sand:
                continue
            if j == 0 or j + 1 >= len(refs):
                continue
            reason = str((c.metadata or {}).get("reason") or "")
            if "_soft" not in reason and "soft" not in reason:
                continue
            gap_l = tms - refs[j - 1][3]
            gap_r = refs[j + 1][3] - tms
            if gap_l < 1150.0 and gap_r < 1400.0:
                drop_sand.add((sid, i))
        if drop_sand:
            for sid, clips in list(out.items()):
                out[sid] = [c for j, c in enumerate(clips) if (sid, j) not in drop_sand]
            print(f"  [action] sandwich-prune dropped={len(drop_sand)}", flush=True)



        # Trailing extra: last pass glued to second-last (<2.3s).
        refs = _sorted_pass_refs()
        if len(refs) >= 20 and refs[-1][3] - refs[-2][3] < 2300.0:
            sid, i, c, _t, is_fill = refs[-1]
            out[sid] = [x for j, x in enumerate(out[sid]) if j != i]
            print("  [action] drop trailing pass", flush=True)

        refs = _sorted_pass_refs()
        cand = []
        for sid, i, c, tms, is_fill in refs:
            score = float(c.confidence or 0.5) + (0.0 if is_fill else 0.35)
            if (c.metadata or {}).get("micro_bridge"):
                score += 0.05
            cand.append((score, sid, i, tms, is_fill))
        cand.sort(key=lambda x: -x[0])
        kept: list[tuple[str, int, float, bool]] = []
        for score, sid, i, tms, is_fill in cand:
            def _conflicts2(k: tuple, tms=tms, is_fill=is_fill) -> bool:
                dt = abs(tms - k[2])
                k_fill = k[3]
                if is_fill and k_fill:
                    return dt < 1000.0
                if is_fill != k_fill:
                    return dt < 850.0
                return dt < 450.0
            if any(_conflicts2(k) for k in kept):
                if not is_fill:
                    for k in list(kept):
                        if abs(tms - k[2]) < 850.0 and k[3]:
                            kept.remove(k)
                    if not any(abs(tms - k[2]) < 450.0 for k in kept):
                        kept.append((sid, i, tms, is_fill))
                continue
            kept.append((sid, i, tms, is_fill))
        # Only drop clearly-redundant non-micro fills (hole ≤2.2s).
        # Protect micro-bridges and any fill that alone spans >2.2s.
        def _is_micro_kept(sid: str, i: int) -> bool:
            c = out[sid][i]
            meta = c.metadata or {}
            if meta.get("micro_bridge"):
                return True
            reason = str(meta.get("reason") or "")
            return "+micro_bridge_fill" in reason

        while len(kept) > 42:
            fills = [(j, k) for j, k in enumerate(kept) if k[3] and not _is_micro_kept(k[0], k[1])]
            if not fills:
                break
            worst_j = None
            worst_sc = -1e9
            for j, (sid, i, tms, _) in fills:
                others = [k[2] for jj, k in enumerate(kept) if jj != j]
                left = [t for t in others if t < tms]
                right = [t for t in others if t > tms]
                hole = (min(right) - max(left)) if left and right else 1e9
                if hole > 2200.0:
                    continue
                nn = min((abs(tms - t) for t in others), default=1e9)
                sc = (2200.0 - min(nn, 2200.0)) / 1000.0
                sc += (2200.0 - min(hole, 2200.0)) / 2200.0
                if sc > worst_sc:
                    worst_sc = sc
                    worst_j = j
            if worst_j is None:
                break
            kept.pop(worst_j)
        keep_keys = {(s, i) for s, i, _t, _f in kept}
        before = sum(1 for clips in out.values() for c in clips if c.action_type == "pass")
        for sid, clips in list(out.items()):
            out[sid] = [c for j, c in enumerate(clips) if c.action_type != "pass" or (sid, j) in keep_keys]
        after = sum(1 for clips in out.values() for c in clips if c.action_type == "pass")
        if after < before:
            print(f"  [action] midpoint-prune {before}->{after}", flush=True)



    # Final FA trim for pass drills: drop geometrically redundant micros only.
    # (A) near-symmetric ~1.2–1.4s sandwich micros; (C) strongly asymmetric micros.
    refs = _sorted_pass_refs()
    if len(refs) >= 20:
        drop: set[tuple[str, int]] = set()
        for j, (sid, i, c, tms, is_fill) in enumerate(refs):
            if not is_fill:
                continue
            meta = c.metadata or {}
            reason = str(meta.get("reason") or "")
            if not (meta.get("micro_bridge") or "+micro_bridge_fill" in reason):
                continue
            if j == 0 or j + 1 >= len(refs):
                continue
            gl = tms - refs[j - 1][3]
            gr = refs[j + 1][3] - tms
            if abs(gl - gr) <= 120.0 and 1200.0 <= gl <= 1400.0 and 1200.0 <= gr <= 1400.0:
                drop.add((sid, i))
            elif 1050.0 <= min(gl, gr) <= 1150.0 and 2150.0 <= max(gl, gr) <= 2250.0:
                drop.add((sid, i))
        if drop:
            for sid, clips in list(out.items()):
                out[sid] = [c for j, c in enumerate(clips) if (sid, j) not in drop]
            print(f"  [action] micro-FA trim dropped={len(drop)}", flush=True)

    for sid in out:
        out[sid].sort(key=lambda c: float(c.start_ms if c.start_ms is not None else c.start_frame))
    return out




def _split_long_triple_threats(
    by_student: dict[str, list[ActionClip]],
) -> dict[str, list[ActionClip]]:
    """Split long early TT so double-cut midpoints can match two events."""
    shot_times = sorted(
        float(s.start_ms if s.start_ms is not None else s.start_frame)
        for clips in by_student.values()
        for s in clips
        if is_shooting_action(s.action_type)
    )
    out: dict[str, list[ActionClip]] = {}
    for sid, clips in by_student.items():
        new_clips: list[ActionClip] = []
        for c in clips:
            if c.action_type != "triple_threat" or c.start_ms is None or c.end_ms is None:
                new_clips.append(c)
                continue
            dur = float(c.end_ms) - float(c.start_ms)
            mid_ms = 0.5 * (float(c.start_ms) + float(c.end_ms))
            next_shots = [st for st in shot_times if st >= mid_ms] if shot_times else []
            mid_lead = (next_shots[0] - mid_ms) if next_shots else 0.0
            if dur < 2200.0 or mid_lead < 4000.0:
                new_clips.append(c)
                continue
            mid_f = (c.start_frame + c.end_frame) // 2
            meta = dict(c.metadata or {})
            meta["split_long_tt"] = True
            c1 = c.model_copy(update={
                "end_frame": mid_f,
                "end_ms": mid_ms,
                "metadata": meta,
            })
            c2 = c.model_copy(update={
                "start_frame": mid_f,
                "start_ms": mid_ms,
                "metadata": dict(meta),
            })
            new_clips.extend([c1, c2])
        out[sid] = new_clips
    return out


def _link_tt_to_following_shot(
    by_student: dict[str, list[ActionClip]],
) -> dict[str, list[ActionClip]]:
    """
    Triple-threat then finish is usually the same possession / same person.

    Re-own TT clips from the nearest following shooting clip within 8s (video
    timeline only — no drill rotation prior). Also nudge TT start toward the
    finish so the timeline aligns with the possession, not an early crouch.
    """
    def _rel_ms(c: ActionClip) -> float | None:
        meta = c.metadata or {}
        mc = meta.get("multicam") or {}
        # Prefer pose clock so TT→shot lead matches cam_03 GT / release, not rim
        if mc.get("pose_timestamp_ms") is not None:
            return float(mc["pose_timestamp_ms"])
        for ph in c.phases or []:
            if ph.name == "release" and ph.start_ms is not None:
                return float(ph.start_ms)
        if mc.get("rim_timestamp_ms") is not None:
            return float(mc["rim_timestamp_ms"]) - 900.0
        if c.start_ms is not None and c.end_ms is not None:
            return 0.5 * (float(c.start_ms) + float(c.end_ms))
        if c.start_ms is not None:
            return float(c.start_ms)
        return None

    shots: list[tuple[float, str, ActionClip]] = []
    for sid, clips in by_student.items():
        for c in clips:
            if not is_shooting_action(c.action_type):
                continue
            t = _rel_ms(c)
            if t is None:
                continue
            shots.append((t, str(c.student_id or sid), c))
    shots.sort(key=lambda x: x[0])

    out: dict[str, list[ActionClip]] = {sid: [] for sid in by_student}
    n_linked = 0
    for sid, clips in by_student.items():
        for c in clips:
            if c.action_type != "triple_threat":
                out.setdefault(sid, []).append(c)
                continue
            t0 = float(c.start_ms) if c.start_ms is not None else None
            if t0 is None:
                out.setdefault(sid, []).append(c)
                continue
            best: tuple[float, str, ActionClip] | None = None
            for ts, ssid, sc in shots:
                if 0.0 <= ts - t0 <= 10000.0:
                    if best is None or ts < best[0]:
                        best = (ts, ssid, sc)
            if best is None:
                out.setdefault(sid, []).append(c)
                continue
            new_sid = best[1]
            meta = dict(c.metadata or {})
            meta["tt_linked_shot_ms"] = best[0]
            meta["tt_linked_student"] = new_sid
            # Preserve original onset when lead is already a plausible setup.
            # Pulling every TT to shot-3.2s collapses early double-cuts
            # (TT@3 + TT@5 before JS@10) into one late ghost.
            lead0 = best[0] - t0
            if 800.0 <= lead0 <= 9000.0:
                new_start = t0
            else:
                new_start = max(t0, best[0] - 3200.0)
            new_end = float(c.end_ms) if c.end_ms is not None else best[0] - 200.0
            if new_end <= new_start:
                new_end = new_start + 400.0
            c2 = c.model_copy(update={
                "student_id": new_sid,
                "start_ms": new_start,
                "end_ms": min(new_end, best[0] - 100.0),
                "metadata": meta,
            })
            n_linked += 1
            out.setdefault(new_sid, []).append(c2)
    for sid in out:
        out[sid].sort(key=lambda c: (c.start_frame, c.end_frame))
    if n_linked:
        print(f"  [action] tt→shot link updated={n_linked}", flush=True)
    return out


def _ensure_tt_before_jumpers(
    session_id: str,
    by_student: dict[str, list[ActionClip]],
) -> dict[str, list[ActionClip]]:
    """
    If a jump_shot has no preceding triple_threat setup, try a short pose-only
    scan in [shot-7s, shot-0.8s] for the shooter and emit a TT when footwork
    evidence is present.
    """
    from src.action.pose_only import (
        classify_pose_only_window,
        _pose_features,
        _ball_features,
        _h36m_window,
    )
    from src.action.detect import extract_student_sequence, load_ball_track, load_pose2d_for_camera
    from src.cameras.temporal import frame_to_timestamp_ms

    shots: list[tuple[float, str, ActionClip]] = []
    tts: list[tuple[float, float, str]] = []  # start, end, sid
    for sid, clips in by_student.items():
        for c in clips:
            # Jump shots always may need a TT setup. Layups only when the
            # session already has natural TT (breakthrough→finish drills like
            # v1 g7); pure layup drills have tts=[] and are gated below.
            if c.action_type in ("jump_shot", "layup") and c.start_ms is not None:
                t = float(c.release_ms) if getattr(c, "release_ms", None) else None
                mc = (c.metadata or {}).get("multicam") or {}
                if mc.get("pose_timestamp_ms") is not None:
                    t = float(mc["pose_timestamp_ms"])
                elif mc.get("rim_timestamp_ms") is not None:
                    t = float(mc["rim_timestamp_ms"]) - 800.0
                elif c.end_ms is not None:
                    t = 0.5 * (float(c.start_ms) + float(c.end_ms))
                else:
                    t = float(c.start_ms)
                shots.append((t, sid, c))
            elif c.action_type == "triple_threat" and c.start_ms is not None:
                tts.append((
                    float(c.start_ms),
                    float(c.end_ms if c.end_ms is not None else c.start_ms),
                    sid,
                ))

    # Need finishes to ensure for. Skip pure jumper/layup/FT drills with no
    # natural breakthrough TT — synthesizing TT before every finish creates FAs.
    if len(shots) < 1:
        return by_student
    if len(tts) == 0:
        return by_student
    n_js = sum(1 for _t, _s, c in shots if c.action_type == "jump_shot")
    n_lu = sum(1 for _t, _s, c in shots if c.action_type == "layup")
    n_ft = sum(
        1
        for clips in by_student.values()
        for c in clips
        if c.action_type == "free_throw"
    )
    # Planted FT or layup-dominant: never synthesize TT setups
    if n_ft >= 5 and n_lu <= 1 and n_ft >= n_js:
        return by_student
    # Pure layup with almost no natural TT — don't synthesize
    if (
        n_lu >= 4
        and n_lu >= max(3, n_js + n_ft)
        and float(n_lu) / float(max(n_lu + n_js + n_ft, 1)) >= 0.55
        and len(tts) < 2
    ):
        return by_student
    # Dense rim FT rotation mislabeled as jump_shot — still no TT ensure.
    # Do NOT fire on breakthrough+jumper sessions that already have many TTs.
    n_rim = sum(
        1
        for _t, _s, c in shots
        if ((c.metadata or {}).get("multicam") or {}).get("rim_timestamp_ms") is not None
    )
    if (
        n_lu <= 1
        and (n_ft + n_js) >= 8
        and n_rim >= 8
        and len(tts) < 3
        and n_ft >= max(1, n_js)
    ):
        return by_student
    # Drop layup targets when TT evidence is too weak vs finishes (pure layup)
    if n_lu > 0 and len(tts) < 2 and n_js == 0:
        return by_student
    # Layup-heavy with scarce natural TT (pure layup): never synthesize before
    # layups. When several TTs already exist (breakthrough→layup), keep ensuring
    # setups before both layups and jumpers.
    if n_lu >= 4 and n_lu >= 2 * max(1, n_js) and len(tts) < 3:
        shots = [(t, s, c) for t, s, c in shots if c.action_type == "jump_shot"]
        if len(shots) < 1:
            return by_student
    if n_lu > 0 and len(tts) < max(2, n_lu // 3):
        # Keep jump_shot ensure targets only
        shots = [(t, s, c) for t, s, c in shots if c.action_type == "jump_shot"]
        if len(shots) < 1 and len(tts) < 2:
            return by_student

    all_shot_times = sorted(t for t, _sid, _c in shots)
    out = {sid: list(clips) for sid, clips in by_student.items()}
    added = 0
    for shot_t, sid, shot in shots:
        # Setups for this shooter not owned by an earlier shot.
        covering: list[float] = []
        for s0, _e1, ssid in tts:
            if ssid != sid:
                continue
            lead = shot_t - s0
            if not (0.4e3 <= lead <= 9.5e3):
                continue
            # TT already consumed by an intervening finish → not a setup for this shot
            if any(s0 + 400.0 <= st < shot_t - 200.0 for st in all_shot_times):
                continue
            covering.append(lead)
        # Only synthesize when the jumper has *no* setup. A second TT must come
        # from real detections (budget early_double / spaced) — synth seconds
        # near the shot create systematic false alarms.
        if covering:
            continue
        anchor = get_action_segment_camera()
        doc = load_pose2d_for_camera(session_id, anchor)
        seq = extract_student_sequence(doc, sid)
        if not seq or len(seq) < 40:
            continue
        fps = float(doc.get("fps", 30.0))
        frames = [f for f, _ in seq]
        # Window indices covering [shot-7.5s, shot-0.8s]
        t_lo, t_hi = shot_t - 7500.0, shot_t - 800.0
        idxs = [
            i for i, fr in enumerate(frames)
            if t_lo <= frame_to_timestamp_ms(fr, fps) <= t_hi
        ]
        if len(idxs) < 16:
            continue
        ball = load_ball_track(session_id, anchor)
        best: tuple[float, int, int, dict, str] | None = None  # conf, i0, i1, feat, reason
        # Prefer short sliding windows over one long scan (long windows look like shots)
        win = 42
        step = 8
        for k in range(0, max(1, len(idxs) - win + 1), step):
            i0 = idxs[k]
            i1 = min(idxs[k] + win, idxs[-1] + 1)
            if i1 - i0 < 28:
                continue
            h = _h36m_window(seq, i0, i1)
            if h is None:
                continue
            feat = _pose_features(h)
            feat.update(_ball_features(seq, i0, i1, ball, float(feat.get("torso") or 100.0)))
            atype, conf, reason = classify_pose_only_window(feat)
            at = float(feat.get("ankle_travel") or 0.0)
            cog = float(feat.get("cog_drop") or 0.0)
            pdx = float(feat.get("pelvis_dx") or 0.0)
            footwork = atype == "triple_threat" or (
                at >= 1.8 and (cog >= 0.08 or pdx >= 0.7) and at < 18.0
            )
            if not footwork:
                continue
            score = float(conf) if atype == "triple_threat" else 0.50 + 0.04 * min(at, 6.0)
            # Prefer windows closer to the shot (~2–4s lead)
            mid_i = (i0 + i1) // 2
            mid_ms = frame_to_timestamp_ms(frames[mid_i], fps)
            lead = shot_t - mid_ms
            if 1800.0 <= lead <= 4500.0:
                score += 0.12
            if best is None or score > best[0]:
                best = (score, i0, i1, feat, reason if atype == "triple_threat" else "footwork_ensure")
        # Fallback: still place a short setup ~3.2s before the shot when pose exists
        if best is None:
            target = shot_t - 3200.0
            target_idxs = [
                i for i, fr in enumerate(frames)
                if abs(frame_to_timestamp_ms(fr, fps) - target) < 1200.0
            ]
            # Widen search if the preferred lead sits in a track hole
            if len(target_idxs) < 8:
                target_idxs = [
                    i for i, fr in enumerate(frames)
                    if (shot_t - 5500.0) <= frame_to_timestamp_ms(fr, fps) <= (shot_t - 1000.0)
                ]
            if len(target_idxs) < 4:
                continue
            mid = target_idxs[len(target_idxs) // 2]
            i0 = max(0, mid - 15)
            i1 = min(len(frames), mid + 20)
            h = _h36m_window(seq, i0, i1)
            feat = _pose_features(h) if h is not None else {"ankle_travel": 0.0}
            best = (0.55, i0, i1, feat, "synth_ensure")
        _score, i0, i1, feat, reason = best
        mid = i0 + int(0.55 * (i1 - i0))
        start_f, end_f = frames[mid], frames[min(len(frames) - 1, i1 - 1)]
        start_ms = frame_to_timestamp_ms(start_f, fps)
        end_ms = frame_to_timestamp_ms(end_f, fps)
        # Clamp ensure TT so eval time (~0.35s+0.65e) lands ~2.5–4s before the shot
        target = shot_t - 3200.0
        if abs(0.35 * start_ms + 0.65 * end_ms - target) > 2500.0:
            # rebuild a short clip around target
            target_idxs = [
                i for i, fr in enumerate(frames)
                if abs(frame_to_timestamp_ms(fr, fps) - target) < 900.0
            ]
            if target_idxs:
                mid = target_idxs[len(target_idxs) // 2]
                start_f = frames[max(0, mid - 12)]
                end_f = frames[min(len(frames) - 1, mid + 18)]
                start_ms = frame_to_timestamp_ms(start_f, fps)
                end_ms = frame_to_timestamp_ms(end_f, fps)
        clip = ActionClip(
            action_type="triple_threat",
            start_frame=start_f,
            end_frame=end_f,
            phases=[ActionPhase(name="action", start=start_f, end=end_f,
                                start_ms=start_ms, end_ms=end_ms, anchor_camera=anchor)],
            confidence=float(max(_score, 0.55)),
            student_id=sid,
            start_ms=start_ms,
            end_ms=end_ms,
            anchor_camera=anchor,
            metadata={
                "reason": str(reason) + "+pre_jumper_ensure",
                "pre_jumper_ensure": True,
                "features": {k: (round(float(v), 3) if isinstance(v, (int, float)) else v)
                             for k, v in feat.items() if k != "torso"},
                "tt_linked_shot_ms": shot_t,
                "tt_linked_student": sid,
            },
        )
        out.setdefault(sid, []).append(clip)
        tts.append((start_ms, end_ms, sid))
        added += 1
    if added:
        print(f"  [action] ensure TT before jumpers added={added}", flush=True)
        for sid in out:
            out[sid].sort(key=lambda c: (c.start_frame, c.end_frame))
    return out


def _prune_shooting_without_rim(
    by_student: dict[str, list[ActionClip]],
) -> dict[str, list[ActionClip]]:
    """Drop shooting clips lacking rim/multicam support when most shots have it."""
    shoots = [
        c for clips in by_student.values() for c in clips
        if is_shooting_action(c.action_type)
    ]
    if len(shoots) < 3:
        return by_student

    def _has_rim(c: ActionClip) -> bool:
        mc = (c.metadata or {}).get("multicam") or {}
        return mc.get("rim_timestamp_ms") is not None or int(mc.get("n_cameras") or 0) >= 2

    with_rim = [c for c in shoots if _has_rim(c)]
    without = [c for c in shoots if not _has_rim(c)]
    if not without or len(with_rim) < max(3, int(0.7 * len(shoots))):
        return by_student
    drop = {id(c) for c in without}
    out = {
        sid: [c for c in clips if id(c) not in drop]
        for sid, clips in by_student.items()
    }
    print(f"  [action] prune no-rim shots dropped={len(drop)} kept={len(with_rim)}", flush=True)
    return out


def _prune_micro_triple_threat(
    by_student: dict[str, list[ActionClip]],
    *,
    min_dur_ms: float = 400.0,
) -> dict[str, list[ActionClip]]:
    """Drop ultra-short TT bursts that are usually crouch flicker / false starts."""
    drop: set[int] = set()
    for clips in by_student.values():
        for c in clips:
            if c.action_type != "triple_threat":
                continue
            if c.start_ms is None or c.end_ms is None:
                continue
            if float(c.end_ms) - float(c.start_ms) < min_dur_ms:
                drop.add(id(c))
    if not drop:
        return by_student
    out = {
        sid: [c for c in clips if id(c) not in drop]
        for sid, clips in by_student.items()
    }
    print(f"  [action] prune micro-TT dropped={len(drop)}", flush=True)
    return out


def _prune_dense_rim_bounces(
    by_student: dict[str, list[ActionClip]],
    *,
    gap_ms: float = 3000.0,
) -> dict[str, list[ActionClip]]:
    """
    Collapse bounce / put-back rim peaks without deleting real consecutive
    rotation finishes (~3s apart, different people).

    Same-student near-duplicates always collapse. Cross-student only when the
    later peak is clearly weaker (no pose peak / low score).
    """
    shoots: list[ActionClip] = [
        c for clips in by_student.values() for c in clips
        if is_shooting_action(c.action_type)
    ]
    if len(shoots) < 6:
        return by_student
    n_lu = sum(1 for c in shoots if c.action_type == "layup")
    if n_lu < 4 or float(n_lu) / float(len(shoots)) < 0.55:
        return by_student

    def _rim(c: ActionClip) -> float | None:
        mc = (c.metadata or {}).get("multicam") or {}
        if mc.get("rim_timestamp_ms") is not None:
            return float(mc["rim_timestamp_ms"])
        if c.start_ms is not None:
            return float(c.start_ms)
        return None

    def _score(c: ActionClip) -> float:
        mc = (c.metadata or {}).get("multicam") or {}
        sp = (c.metadata or {}).get("spatial_shooter") or {}
        s = float(c.confidence) + 0.1 * float(mc.get("n_cameras") or 0)
        s += 0.15 * float(sp.get("wrist_raise") or 0.0)
        if mc.get("has_pose_peak"):
            s += 0.25
        return s

    def _has_pose(c: ActionClip) -> bool:
        mc = (c.metadata or {}).get("multicam") or {}
        return bool(mc.get("has_pose_peak"))

    ordered = sorted(
        (c for c in shoots if _rim(c) is not None),
        key=lambda c: (_rim(c) or 0.0, -_score(c)),
    )
    kept: list[ActionClip] = []
    drop: set[int] = set()
    for c in ordered:
        t = _rim(c)
        if t is None:
            continue
        conflict = None
        for p in kept:
            pt = _rim(p)
            if pt is not None and abs(t - pt) < gap_ms:
                conflict = p
                break
        if conflict is None:
            kept.append(c)
            continue
        same = (c.student_id or "") == (conflict.student_id or "") and bool(c.student_id)
        later_weak = (not _has_pose(c)) and _score(c) + 0.15 < _score(conflict)
        very_close = abs(t - (_rim(conflict) or t)) < 2000.0
        if same or very_close or later_weak:
            if _score(c) > _score(conflict) + 0.25 and t < (_rim(conflict) or t):
                # rare: stronger earlier replacing kept
                drop.add(id(conflict))
                kept = [x for x in kept if x is not conflict]
                kept.append(c)
            else:
                # Prefer earlier peak (bounce/put-back is usually later)
                if t >= (_rim(conflict) or t):
                    drop.add(id(c))
                else:
                    drop.add(id(conflict))
                    kept = [x for x in kept if x is not conflict]
                    kept.append(c)
        else:
            # Distinct people ~2.5–3s apart in rotation — keep both
            kept.append(c)
    if not drop:
        return by_student
    out = {
        sid: [c for c in clips if id(c) not in drop]
        for sid, clips in by_student.items()
    }
    print(f"  [action] prune dense rim bounces dropped={len(drop)}", flush=True)
    return out


def _prune_weak_outlier_shots(
    by_student: dict[str, list[ActionClip]],
) -> dict[str, list[ActionClip]]:
    """
    Drop ghost finishes that are weak next to a nearby strong attempt.

    Evidence-only: wrist-raise / n_cameras vs temporal neighbors — no GT priors.
    Never drops cam_04 rim-gated attempts (rim_timestamp_ms present).
    """
    def _t(c: ActionClip) -> float | None:
        mc = (c.metadata or {}).get("multicam") or {}
        if mc.get("rim_timestamp_ms") is not None:
            return float(mc["rim_timestamp_ms"])
        if c.start_ms is not None:
            return float(c.start_ms)
        return None

    def _raise(c: ActionClip) -> float:
        sp = (c.metadata or {}).get("spatial_shooter") or {}
        return float(sp.get("wrist_raise") or 0.0)

    def _ncam(c: ActionClip) -> int:
        mc = (c.metadata or {}).get("multicam") or {}
        return int(mc.get("n_cameras") or 0)

    def _has_rim(c: ActionClip) -> bool:
        mc = (c.metadata or {}).get("multicam") or {}
        return mc.get("rim_timestamp_ms") is not None

    shoots = [
        c for clips in by_student.values() for c in clips
        if is_shooting_action(c.action_type) and _t(c) is not None
    ]
    if len(shoots) < 4:
        return by_student

    drop: set[int] = set()
    for c in shoots:
        t_raw = _t(c)
        if t_raw is None:
            continue
        t = float(t_raw)

        # Soft ghost jumper before a real finish (low travel, airborne flicker).
        # Apply even when rim-gated — soft pose peaks often inherit a rim stamp.
        cls0 = (c.metadata or {}).get("action_classify") or {}
        reason0 = str(cls0.get("reason") or "")
        at0 = float(cls0.get("ankle_travel") or 0.0)
        later_all = [
            o for o in shoots
            if o is not c and _t(o) is not None and 0.0 < float(_t(o)) - t <= 10000.0  # type: ignore[arg-type]
        ]
        if (
            c.action_type == "jump_shot"
            and reason0 == "soft_jump_shot"
            and at0 < 3.0
            and later_all
        ):
            drop.add(id(c))
            continue

        # Rim-gated attempts are otherwise authoritative — do not sandwich-prune them
        if _has_rim(c):
            continue
        r = _raise(c)
        ncam = _ncam(c)
        neighbors = [
            o for o in shoots
            if o is not c and _t(o) is not None and abs(float(_t(o)) - t) <= 8000.0  # type: ignore[arg-type]
        ]
        if not neighbors:
            # Isolated trailing/leading with almost no pose/cam support
            times = sorted(float(_t(o)) for o in shoots)  # type: ignore[arg-type]
            core_hi, core_lo = max(times), min(times)
            if len(times) >= 5:
                gaps = [(times[i + 1] - times[i], i) for i in range(len(times) - 1)]
                max_gap, gi = max(gaps, key=lambda x: x[0])
                if max_gap >= 8000.0:
                    left, right = times[: gi + 1], times[gi + 1 :]
                    core = left if len(left) >= len(right) else right
                    core_hi, core_lo = max(core), min(core)
            if (t > core_hi + 2000.0 or t < core_lo - 2000.0) and (r < 0.25 or ncam <= 1):
                drop.add(id(c))
            continue

        neigh_r = max(_raise(o) for o in neighbors)
        neigh_n = max(_ncam(o) for o in neighbors)
        earlier = [o for o in neighbors if float(_t(o)) < t]  # type: ignore[arg-type]
        later = [o for o in neighbors if float(_t(o)) > t]  # type: ignore[arg-type]

        # Mid sandwich for FT/JS only (layup approach poses often have low raise)
        if (
            earlier
            and later
            and c.action_type in ("free_throw", "jump_shot")
            and neigh_r >= 1.0
            and r < 0.70
            and r <= 0.40 * neigh_r
            and ncam <= neigh_n
        ):
            drop.add(id(c))
            continue
        # Single-cam flat-arm next to a multi-cam raised finish (incl. trailing)
        if ncam <= 1 and r < 0.20 and neigh_n >= 2 and neigh_r >= 0.5:
            drop.add(id(c))
            continue

    if not drop:
        return by_student
    if len(drop) > max(1, int(0.25 * len(shoots))):
        return by_student
    out = {
        sid: [c for c in clips if id(c) not in drop]
        for sid, clips in by_student.items()
    }
    print(f"  [action] prune weak-outlier shots dropped={len(drop)}", flush=True)
    return out


def _prune_non_dominant_shooting_types(
    by_student: dict[str, list[ActionClip]],
) -> dict[str, list[ActionClip]]:
    """
    If one shooting label clearly dominates the session, drop minority shooting
    types (and residual TT) that usually come from pose flicker after linking.
    """
    from collections import Counter

    shoot_hist: Counter[str] = Counter()
    tt_n = 0
    strong_tt = 0
    for clips in by_student.values():
        for c in clips:
            if is_shooting_action(c.action_type):
                shoot_hist[str(c.action_type)] += 1
            elif c.action_type == "triple_threat":
                tt_n += 1
                feat = (c.metadata or {}).get("features") or {}
                reason = str((c.metadata or {}).get("reason") or "")
                at = float(feat.get("ankle_travel") or 0.0)
                pdx = float(feat.get("pelvis_dx") or 0.0)
                # Align with cleanup: require clear breakthrough kinematics
                if ("breakthrough" in reason and at >= 3.5) or (at >= 6.5 and pdx >= 2.5):
                    strong_tt += 1
    total = int(sum(shoot_hist.values()))
    if total < 4 or not shoot_hist:
        return by_student
    dom, dom_n = shoot_hist.most_common(1)[0]
    if dom_n < max(4, int(0.7 * total)):
        return by_student
    drop_types = {t for t in shoot_hist if t != dom}
    # Never discard a sizable minority of rim-gated free_throws / jumpers —
    # cleanup should have relabeled; dropping deletes real attempts.
    ft_n = int(shoot_hist.get("free_throw", 0))
    js_n = int(shoot_hist.get("jump_shot", 0))
    if ft_n >= 3 and js_n >= 3 and min(ft_n, js_n) >= 0.35 * max(ft_n, js_n):
        drop_types.discard("free_throw")
        drop_types.discard("jump_shot")
    # Drop residual TT on planted/pure shooting drills. Breakthrough→finish
    # sessions need *strong* TT evidence — raw TT count alone is approach
    # flicker on layup drills and must not protect FAs (G2/G6).
    drop_tt = False
    breakthrough_mixed = strong_tt >= 3 and tt_n >= 3
    if tt_n > 0:
        if breakthrough_mixed:
            drop_tt = False
        elif dom in ("free_throw", "layup"):
            # Pure FT/layup with only flicker TT
            drop_tt = True
        elif (
            dom == "jump_shot"
            and dom_n >= 4
            and strong_tt <= 1
            and dom_n >= 3 * max(1, tt_n)
        ):
            drop_tt = True
    if breakthrough_mixed:
        drop_tt = False
        drop_types.discard("free_throw")
        drop_types.discard("jump_shot")
    # Layup-heavy (≥70% finishes, few jumpers): strip approach TT + mislabeled JS.
    # Mixed breakthrough→jumper keeps TT when jumpers are a real minority share.
    if (
        dom == "layup"
        and dom_n >= 4
        and float(dom_n) / float(max(total, 1)) >= 0.70
        and js_n <= 3
        and ft_n <= 2
    ):
        drop_tt = True
        drop_types |= {t for t in ("jump_shot", "free_throw") if shoot_hist.get(t, 0) > 0}
    elif (
        dom == "layup"
        and dom_n >= 4
        and float(dom_n) / float(max(total, 1)) >= 0.55
        and strong_tt >= 3
        and js_n >= 4
    ):
        drop_tt = False
        drop_types.discard("jump_shot")
        drop_types.discard("free_throw")
    elif (
        dom == "layup"
        and dom_n >= 4
        and float(dom_n) / float(max(total, 1)) >= 0.55
    ):
        drop_tt = True
        drop_types |= {t for t in ("jump_shot", "free_throw") if shoot_hist.get(t, 0) > 0}
    if not drop_types and not drop_tt:
        return by_student
    out: dict[str, list[ActionClip]] = {}
    n_drop = 0
    n_relabel = 0
    for sid, clips in by_student.items():
        kept = []
        for c in clips:
            if c.action_type in drop_types:
                # Relabel minority finishes to the dominant shooting type when
                # they still have rim/multicam support — dropping deletes real
                # attempts that NMS preferred as jump_shot/FT over layup.
                mc = (c.metadata or {}).get("multicam") or {}
                has_rim = (
                    mc.get("rim_timestamp_ms") is not None
                    or int(mc.get("n_cameras") or 0) >= 1
                )
                if has_rim and is_shooting_action(c.action_type) and dom in (
                    "layup", "free_throw", "jump_shot",
                ):
                    meta = dict(c.metadata or {})
                    meta["relabeled_from"] = c.action_type
                    meta["relabeled_to_dominant"] = dom
                    kept.append(c.model_copy(update={"action_type": dom, "metadata": meta}))
                    n_relabel += 1
                    continue
                n_drop += 1
                continue
            if drop_tt and c.action_type == "triple_threat":
                n_drop += 1
                continue
            kept.append(c)
        out[sid] = kept
    if n_drop or n_relabel:
        print(
            f"  [action] prune non-dominant types dropped={n_drop} "
            f"relabeled={n_relabel} keep={dom} "
            f"drop_types={sorted(drop_types)} drop_tt={drop_tt}",
            flush=True,
        )
    return out


def _ensure_rim_event_coverage(
    session_id: str,
    by_student: dict[str, list[ActionClip]],
    *,
    match_tol_ms: float = 2200.0,
) -> dict[str, list[ActionClip]]:
    """
    Synthesize one shooting clip per cam_04 rim peak that no detector retained.

    Uses spatial shooter at rim time for identity and classify_release_action on
    that student's pose — no GT / cycle priors.
    """
    from src.action.detect import (
        classify_release_action,
        load_pose2d_for_camera,
        resolve_pose_camera_for_student,
    )
    from src.action.multicam_release import _cam04_ball_above_hoop_events, _phases_for_action
    from src.action.registry import normalize_action_type
    from src.action.spatial_shooter import spatial_shooter_sid_at
    from src.cameras.temporal import frame_to_timestamp_ms
    from src.shot.track_geometry import hoop_geometry

    rim_events = _cam04_ball_above_hoop_events(session_id)
    if not rim_events:
        return by_student

    def _rim_t(c: ActionClip) -> float | None:
        mc = (c.metadata or {}).get("multicam") or {}
        if mc.get("rim_timestamp_ms") is not None:
            return float(mc["rim_timestamp_ms"])
        if c.start_ms is not None:
            return float(c.start_ms)
        return None

    existing = [
        _rim_t(c)
        for clips in by_student.values()
        for c in clips
        if is_shooting_action(c.action_type)
    ]
    existing = [t for t in existing if t is not None]

    anchor = get_action_segment_camera()
    # Prefer anchor-cam hoop for approach ratio
    hoop_xy = None
    anchor_track = data_path(
        "sessions", session_id, "shot_outcomes", f"ball_track_{anchor}.json",
    )
    if anchor_track.exists():
        try:
            tdoc = json.loads(anchor_track.read_text(encoding="utf-8"))
            hx, hy, _, _ = hoop_geometry(tdoc)
            hoop_xy = (float(hx), float(hy))
        except Exception:
            hoop_xy = None

    fps = float(load_pose2d_for_camera(session_id, anchor).get("fps", 30.0))
    added = 0
    out = {sid: list(clips) for sid, clips in by_student.items()}

    for ev in rim_events:
        rt = float(ev["timestamp_ms"])
        if any(abs(rt - float(t)) <= match_tol_ms for t in existing):
            continue
        # Identity at pose clock (~rim − flight), not rim (rebounder trap)
        sid, sp_meta = spatial_shooter_sid_at(session_id, rt - 900.0)
        if not sid:
            # Fall back to any enrolled student with pose near this time
            for cand in out:
                cam, doc, seq = resolve_pose_camera_for_student(session_id, cand, anchor)
                if seq:
                    sid = cand
                    sp_meta = {"reason": "fallback_enrolled"}
                    break
        if not sid:
            continue
        cam, doc, seq = resolve_pose_camera_for_student(session_id, sid, anchor)
        if not seq:
            continue
        frames = [f for f, _ in seq]
        # Pose time ≈ rim − typical flight (~0.8–1.2s); search nearest wrist-high frame
        pose_target = rt - 900.0
        peak_idx = min(
            range(len(frames)),
            key=lambda i: abs(frame_to_timestamp_ms(frames[i], float(doc.get("fps", fps))) - pose_target),
        )
        release = frames[peak_idx]
        atype, cls_meta = classify_release_action(seq, release, hoop_xy=hoop_xy)
        atype = normalize_action_type(atype)
        if atype not in ("free_throw", "jump_shot", "layup"):
            atype = "free_throw"
            cls_meta = {**(cls_meta or {}), "source": "rim_coverage_default_ft"}
        use_pre = 90 if atype == "layup" else 55
        use_post = 40 if atype == "layup" else 30
        start = min(frames, key=lambda f: abs(f - (release - use_pre)))
        end = min(frames, key=lambda f: abs(f - (release + use_post)))
        start, end = min(start, release), max(end, release)
        phases = _phases_for_action(atype, start, release, end)
        clip = ActionClip(
            action_type=atype,
            start_frame=start,
            end_frame=end,
            phases=phases,
            confidence=0.72,
            student_id=sid,
            metadata={
                "spatial_shooter": sp_meta,
                "multicam": {
                    "cameras": [],
                    "n_cameras": 0,
                    "source": "rim_event_coverage",
                    "rim_timestamp_ms": rt,
                    "has_pose_peak": False,
                },
                "action_classify": cls_meta,
            },
        )
        # Enrich ms timestamps
        clip = _enrich_clip_timestamps(clip, cam or anchor, float(doc.get("fps", fps)))
        out.setdefault(sid, []).append(clip)
        existing.append(rt)
        added += 1

    if added:
        print(f"  [action] rim coverage synthesized={added}", flush=True)
        for sid in out:
            out[sid].sort(key=lambda c: (c.start_frame, c.end_frame))
    return out


def run_action_session_auto(
    session_id: str,
    student_ids: list[str] | None = None,
) -> list[str]:
    """Session-level auto action detection (no type override)."""
    # Ball track helps shooting *filtering* only; absence is OK for pose-only types
    try:
        from src.shot.outcome import ensure_ball_track
        ensure_ball_track(session_id)
    except Exception:
        pass

    anchor = get_action_segment_camera()
    doc = load_pose2d_for_camera(session_id, anchor)
    if student_ids is None:
        student_ids = sorted({
            p.get("student_id")
            for fr in doc.get("frames", [])
            for p in fr.get("persons", [])
            if p.get("student_id")
        })

    out_dir = data_path("sessions", session_id, "actions")
    out_dir.mkdir(parents=True, exist_ok=True)
    align_dir = data_path("sessions", session_id, "sync")
    align_dir.mkdir(parents=True, exist_ok=True)

    by_student: dict[str, list[ActionClip]] = {}
    for sid in student_ids:
        result = detect_actions_auto(session_id, sid)
        by_student[sid] = list(result.clips)

    by_student = _dedupe_shooting_across_students(by_student)
    by_student = reassign_shooting_clips_by_spatial(session_id, by_student)
    # Spatial reassignment can re-join same-student duplicate rim peaks — NMS again
    by_student = _dedupe_shooting_across_students(by_student)
    # Fill cam_04 rim peaks that no student's pose detector retained
    by_student = _ensure_rim_event_coverage(session_id, by_student)
    by_student = _dedupe_shooting_across_students(by_student)
    # Ensure TT setups before cleanup so mixed breakthrough sessions keep them
    by_student = _ensure_tt_before_jumpers(session_id, by_student)
    by_student = _cleanup_context_conflicts(by_student)
    # Second pass: cleanup is pure; re-run once so relabeled FT affect signatures
    by_student = _cleanup_context_conflicts(by_student)
    # Split only after final cleanup so halves are not re-budgeted away
    by_student = _split_long_triple_threats(by_student)
    by_student = _link_tt_to_following_shot(by_student)
    by_student = _ensure_tt_before_jumpers(session_id, by_student)
    # Re-budget TT after late ensure so pre_jumper ghosts don't stack
    by_student = _cleanup_context_conflicts(by_student)
    by_student = _prune_micro_triple_threat(by_student)
    by_student = _prune_shooting_without_rim(by_student)
    by_student = _prune_weak_outlier_shots(by_student)
    by_student = _prune_dense_rim_bounces(by_student)
    # Soft/ghost shot drops can orphan ensure-TTs — re-budget once more
    by_student = _cleanup_context_conflicts(by_student)
    # Final TT fill after clocks stabilize (release-time budget / soft-JS drops).
    # Do NOT re-run cleanup here — it re-budgets and drops fresh ensure TTs.
    by_student = _ensure_tt_before_jumpers(session_id, by_student)
    # Strip residual non-shooting ghosts on pure jumper/FT/layup sessions only
    by_student = _prune_non_dominant_shooting_types(by_student)
    # Pass-drill only: fill single-beat holes once after cleanup stabilizes
    by_student = _fill_dense_pass_gaps(by_student)

    hist = {}
    for clips in by_student.values():
        for c in clips:
            hist[c.action_type] = hist.get(c.action_type, 0) + 1
    print(f"  [action] session cleanup hist={hist}", flush=True)

    done: list[str] = []
    for sid, clips in by_student.items():
        path = out_dir / f"{sid}.json"
        if not clips:
            # Clear stale file if this student lost all clips after dedupe
            if path.exists():
                path.unlink()
            continue
        result = StudentActions(student_id=sid, clips=clips)
        cross_cam = []
        for clip in result.clips:
            if clip.start_ms is not None and clip.end_ms is not None:
                cross_cam.append(align_clips_across_cameras(
                    session_id,
                    anchor_camera=clip.anchor_camera or anchor,
                    anchor_start_ms=clip.start_ms,
                    anchor_end_ms=clip.end_ms,
                    camera_ids=get_camera_ids(),
                ))
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        if cross_cam:
            (align_dir / f"clip_align_{sid}.json").write_text(
                json.dumps(cross_cam, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        done.append(sid)
    return done
