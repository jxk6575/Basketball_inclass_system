"""Consent management."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.privacy.audit import log_audit
from src.privacy.db import get_conn
from src.types import ConsentScope


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def grant_consent(
    student_id: str,
    session_id: str,
    scopes: list[ConsentScope | str],
    consent_text_version: str = "2026-07-v1",
    actor: str | None = None,
) -> None:
    scope_str = json.dumps([s.value if isinstance(s, ConsentScope) else s for s in scopes])
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO consent_records (student_id, session_id, consented_at, scope,
                                         consent_text_version, revoked_at)
            VALUES (?, ?, ?, ?, ?, NULL)
            ON CONFLICT(student_id, session_id) DO UPDATE SET
                consented_at = excluded.consented_at,
                scope = excluded.scope,
                consent_text_version = excluded.consent_text_version,
                revoked_at = NULL
            """,
            (student_id, session_id, _now(), scope_str, consent_text_version),
        )
    log_audit("consent_grant", actor=actor, session_id=session_id, student_id=student_id,
              detail={"scopes": json.loads(scope_str)})


def revoke_consent(student_id: str, session_id: str, actor: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE consent_records SET revoked_at = ? WHERE student_id = ? AND session_id = ?",
            (_now(), student_id, session_id),
        )
    log_audit("consent_revoke", actor=actor, session_id=session_id, student_id=student_id)


def has_consent(student_id: str, session_id: str, scope: ConsentScope | str) -> bool:
    scope_val = scope.value if isinstance(scope, ConsentScope) else scope
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT scope, revoked_at FROM consent_records
            WHERE student_id = ? AND session_id = ?
            """,
            (student_id, session_id),
        ).fetchone()
    if not row or row["revoked_at"]:
        return False
    scopes = json.loads(row["scope"])
    return scope_val in scopes


def list_session_consents(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT student_id, consented_at, scope, revoked_at
            FROM consent_records WHERE session_id = ?
            """,
            (session_id,),
        ).fetchall()
    result = []
    for r in rows:
        result.append({
            "student_id": r["student_id"],
            "consented_at": r["consented_at"],
            "scopes": json.loads(r["scope"]),
            "revoked": r["revoked_at"] is not None,
            "active": r["revoked_at"] is None,
        })
    return result


def session_all_consented(session_id: str, student_ids: list[str]) -> bool:
    for sid in student_ids:
        if not has_consent(sid, session_id, ConsentScope.VIDEO):
            return False
    return True
