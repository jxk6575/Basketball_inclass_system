"""SQLite schema and connection."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.config import data_path

DB_PATH = data_path("system.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    class_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    camera_layout TEXT NOT NULL DEFAULT 'v1_4cam_halfcourt',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    class_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS consent_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    consented_at TEXT NOT NULL,
    scope TEXT NOT NULL,
    consent_text_version TEXT NOT NULL,
    revoked_at TEXT,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    UNIQUE(student_id, session_id)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    actor TEXT,
    session_id TEXT,
    student_id TEXT,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_consent_session ON consent_records(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_logs(session_id);
"""


def init_db(db_path: Path | None = None) -> Path:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
    return path


@contextmanager
def get_conn(db_path: Path | None = None):
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
