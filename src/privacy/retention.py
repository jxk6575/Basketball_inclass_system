"""Data retention and cascade deletion on consent revoke."""

from __future__ import annotations

import fnmatch
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import DATA, load_yaml
from src.privacy.audit import log_audit
from src.privacy.consent import revoke_consent


def _parse_days_map() -> dict[str, int]:
    cfg = load_yaml("privacy.yaml")
    r = cfg.get("retention", {})
    return {
        "raw_video": r.get("raw_video_days", 30),
        "embedding": r.get("embedding_days_after_session", 7),
        "pose3d_report": r.get("pose3d_and_report_days", 180),
        "audit": r.get("audit_log_days", 365),
    }


def _expand_glob(root: Path, pattern: str) -> list[Path]:
    """Simple glob for cascade paths like perception/**/{student_id}*."""
    if "**" not in pattern:
        p = root / pattern
        return [p] if p.exists() else []

    base, rest = pattern.split("**", 1)
    base = base.rstrip("/")
    rest = rest.lstrip("/")
    hits: list[Path] = []
    base_path = root / base if base else root
    if not base_path.exists():
        return hits
    for path in base_path.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if fnmatch.fnmatch(rel, f"{base}/**/{rest}" if base else f"**/{rest}"):
            hits.append(path)
    return hits


def cascade_delete_student(session_id: str, student_id: str, actor: str | None = None) -> list[str]:
    """Delete all session data for a student; returns deleted paths."""
    cfg = load_yaml("privacy.yaml")
    deleted: list[str] = []
    templates = cfg.get("deletion", {}).get("paths_on_revoke", [])

    for tpl in templates:
        path_pattern = tpl.replace("{session_id}", session_id).replace("{student_id}", student_id)
        for p in _expand_glob(DATA, path_pattern):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                deleted.append(str(p))
            elif p.is_file():
                p.unlink(missing_ok=True)
                deleted.append(str(p))

    # enrollment directory
    enroll = DATA / "enrollment" / session_id / student_id
    if enroll.exists():
        shutil.rmtree(enroll, ignore_errors=True)
        deleted.append(str(enroll))

    log_audit(
        "data_delete",
        actor=actor,
        session_id=session_id,
        student_id=student_id,
        detail={"paths": deleted},
    )
    return deleted


def revoke_and_purge(student_id: str, session_id: str, actor: str | None = None) -> list[str]:
    revoke_consent(student_id, session_id, actor=actor)
    return cascade_delete_student(session_id, student_id, actor=actor)


def run_retention_cleanup(now: datetime | None = None) -> dict[str, int]:
    """Delete expired raw videos and old embeddings by mtime."""
    now = now or datetime.now(timezone.utc)
    days = _parse_days_map()
    counts = {"raw_video": 0, "embedding": 0}

    raw_cutoff = now - timedelta(days=days["raw_video"])
    for mp4 in (DATA / "sessions").rglob("raw/*.mp4"):
        mtime = datetime.fromtimestamp(mp4.stat().st_mtime, tz=timezone.utc)
        if mtime < raw_cutoff:
            mp4.unlink(missing_ok=True)
            counts["raw_video"] += 1

    emb_cutoff = now - timedelta(days=days["embedding"])
    for emb in (DATA / "enrollment").rglob("*.npy"):
        mtime = datetime.fromtimestamp(emb.stat().st_mtime, tz=timezone.utc)
        if mtime < emb_cutoff:
            emb.unlink(missing_ok=True)
            counts["embedding"] += 1

    log_audit("retention_cleanup", detail=counts)
    return counts
