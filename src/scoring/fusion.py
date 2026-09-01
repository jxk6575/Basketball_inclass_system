"""Merge action clips with Pose2Sim 3D angles into student reports."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import data_path
from src.scoring.templates import load_action_template, score_metric
from src.scoring.report_html import render_report_html
from src.types import ActionClip, PhaseScore, StudentActions, StudentReport


def _load_angles(session_id: str, student_id: str) -> tuple[dict[int, dict], dict[float, dict]]:
    p = data_path("sessions", session_id, "angles", f"{student_id}.json")
    if not p.exists():
        return {}, {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    by_frame = {}
    by_time: dict[float, dict] = {}
    for i, fr in enumerate(doc.get("frames", [])):
        by_frame[fr.get("frame", i)] = fr
        if "timestamp_ms" in fr:
            by_time[float(fr["timestamp_ms"])] = fr
    return by_frame, by_time


def _lookup_angles(
    phase,
    by_frame: dict[int, dict],
    by_time: dict[float, dict],
) -> dict:
    if phase.start_ms is not None and by_time:
        mid_ms = (float(phase.start_ms) + float(phase.end_ms)) / 2 if phase.end_ms else float(phase.start_ms)
        if by_time:
            nearest = min(by_time.keys(), key=lambda t: abs(t - mid_ms))
            return by_time[nearest]
    mid_frame = (phase.start + phase.end) // 2
    return by_frame.get(mid_frame, by_frame.get(phase.start, {}))


def _load_actions(session_id: str, student_id: str) -> StudentActions | None:
    p = data_path("sessions", session_id, "actions", f"{student_id}.json")
    if not p.exists():
        return None
    return StudentActions.model_validate_json(p.read_text(encoding="utf-8"))


def _metric_value(angles: dict, metric_cfg: dict) -> float:
    joint = metric_cfg.get("joint")
    metric = metric_cfg.get("metric")
    if joint and joint in angles:
        return float(angles[joint])
    if metric == "wrist_height_m" and "wrist_height_m" in angles:
        return float(angles["wrist_height_m"])
    return float("nan")


def score_clip(
    session_id: str,
    student_id: str,
    clip: ActionClip,
    identity_confidence: str = "high",
) -> StudentReport:
    template = load_action_template(clip.action_type)
    by_frame, by_time = _load_angles(session_id, student_id)
    phase_scores: list[PhaseScore] = []
    weighted_sum = 0.0
    weight_total = 0.0

    phase_map = {p.name: p for p in clip.phases}
    for phase_cfg in template.get("phases", []):
        pname = phase_cfg["name"]
        phase = phase_map.get(pname)
        if not phase:
            continue
        angles = _lookup_angles(phase, by_frame, by_time)
        for mname, mcfg in phase_cfg.get("metrics", {}).items():
            val = _metric_value(angles, mcfg)
            s, fb = score_metric(val, mcfg["min"], mcfg["max"])
            w = float(mcfg.get("weight", 0.1))
            if val < mcfg["min"] and mcfg.get("feedback_low"):
                fb = mcfg["feedback_low"]
            elif val > mcfg["max"] and mcfg.get("feedback_high"):
                fb = mcfg["feedback_high"]
            phase_scores.append(PhaseScore(
                phase=pname,
                metric=mname,
                value=val if val == val else -1,
                score=s,
                weight=w,
                feedback=fb,
            ))
            weighted_sum += s * w
            weight_total += w

    total = weighted_sum / weight_total if weight_total > 0 else 0.0
    thresholds = template.get("scoring", {})
    pass_th = thresholds.get("pass_threshold", 60)
    summary = "动作规范，继续保持。" if total >= pass_th else "部分环节需要调整，请查看分项反馈。"

    return StudentReport(
        student_id=student_id,
        session_id=session_id,
        action_type=clip.action_type,
        total_score=round(total, 1),
        phase_scores=phase_scores,
        summary=summary,
        identity_confidence=identity_confidence,
        metadata={"clip": clip.model_dump()},
    )


def run_scoring_session(session_id: str, student_ids: list[str] | None = None) -> list[Path]:
    actions_dir = data_path("sessions", session_id, "actions")
    if student_ids is None:
        student_ids = [p.stem for p in actions_dir.glob("*.json")]

    report_dir = data_path("sessions", session_id, "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for sid in student_ids:
        actions = _load_actions(session_id, sid)
        if not actions or not actions.clips:
            continue
        reports = [score_clip(session_id, sid, c) for c in actions.clips]
        # Use highest-confidence clip or first
        report = reports[0]
        json_path = report_dir / f"{sid}.json"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        html_path = report_dir / f"{sid}.html"
        html_path.write_text(render_report_html(report), encoding="utf-8")
        paths.append(json_path)

    return paths
