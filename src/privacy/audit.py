"""Audit logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.privacy.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_audit(
    action: str,
    actor: str | None = None,
    session_id: str | None = None,
    student_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO audit_logs (action, actor, session_id, student_id, detail, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (action, actor, session_id, student_id,
             json.dumps(detail or {}, ensure_ascii=False), _now()),
        )


def query_audit(
    session_id: str | None = None,
    limit: int = 100,
) -> list[dict]:
    with get_conn() as conn:
        if session_id:
            rows = conn.execute(
                """
                SELECT * FROM audit_logs WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]
