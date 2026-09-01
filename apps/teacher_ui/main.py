"""Teacher-facing local web UI (127.0.0.1 only)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.config import data_path, load_yaml
from src.identity.enrollment import EnrollmentGallery
from src.orchestrator.session_pipeline import (
    create_session,
    get_session,
    register_student,
    run_pipeline,
    update_session_status,
)
from src.privacy.audit import query_audit
from src.privacy.consent import grant_consent, list_session_consents, revoke_consent
from src.privacy.db import init_db
from src.privacy.retention import revoke_and_purge, run_retention_cleanup
from src.types import ConsentScope, SessionStatus

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(title="篮球课堂辅助系统", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

init_db()


class CreateSessionBody(BaseModel):
    class_id: str


class ConsentBody(BaseModel):
    student_id: str
    scopes: list[str] = ["video", "face", "report"]


class StudentBody(BaseModel):
    student_id: str
    display_name: str


class PipelineBody(BaseModel):
    from_stage: str = "perception"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/sessions")
async def api_create_session(body: CreateSessionBody):
    sid = create_session(body.class_id)
    return {"session_id": sid}


@app.get("/api/sessions/{session_id}")
async def api_get_session(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    return s


@app.post("/api/sessions/{session_id}/students")
async def api_register_student(session_id: str, body: StudentBody):
    register_student(body.student_id, body.display_name)
    gallery = EnrollmentGallery(session_id)
    gallery.save_meta(body.student_id, body.display_name)
    return {"ok": True}


@app.post("/api/sessions/{session_id}/consent")
async def api_grant_consent(session_id: str, body: ConsentBody):
    scopes = [ConsentScope(s) for s in body.scopes]
    grant_consent(body.student_id, session_id, scopes, actor="teacher")
    update_session_status(session_id, SessionStatus.CONSENT_OK)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/consent")
async def api_list_consent(session_id: str):
    return list_session_consents(session_id)


@app.delete("/api/sessions/{session_id}/consent/{student_id}")
async def api_revoke_consent(session_id: str, student_id: str):
    deleted = revoke_and_purge(student_id, session_id, actor="teacher")
    return {"revoked": True, "deleted_paths": deleted}


@app.post("/api/sessions/{session_id}/enrolled")
async def api_mark_enrolled(session_id: str):
    update_session_status(session_id, SessionStatus.ENROLLED)
    return {"status": SessionStatus.ENROLLED.value}


@app.post("/api/sessions/{session_id}/recorded")
async def api_mark_recorded(session_id: str):
    update_session_status(session_id, SessionStatus.RECORDED)
    return {"status": SessionStatus.RECORDED.value}


@app.post("/api/sessions/{session_id}/pipeline")
async def api_run_pipeline(session_id: str, body: PipelineBody):
    result = run_pipeline(session_id, from_stage=body.from_stage)
    return result


@app.get("/api/sessions/{session_id}/reports")
async def api_list_reports(session_id: str):
    report_dir = data_path("sessions", session_id, "reports")
    if not report_dir.exists():
        return []
    return [
        {"student_id": p.stem, "json": str(p), "html": str(report_dir / f"{p.stem}.html")}
        for p in report_dir.glob("*.json")
    ]


@app.get("/api/sessions/{session_id}/reports/{student_id}")
async def api_get_report(session_id: str, student_id: str):
    p = data_path("sessions", session_id, "reports", f"{student_id}.json")
    if not p.exists():
        raise HTTPException(404, "report not found")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/sessions/{session_id}/reports/{student_id}/html", response_class=HTMLResponse)
async def api_report_html(session_id: str, student_id: str):
    p = data_path("sessions", session_id, "reports", f"{student_id}.html")
    if not p.exists():
        raise HTTPException(404, "report not found")
    return FileResponse(p)


@app.get("/api/audit")
async def api_audit(session_id: str | None = None):
    return query_audit(session_id=session_id)


@app.post("/api/admin/retention")
async def api_retention():
    return run_retention_cleanup()


@app.get("/api/config/cameras")
async def api_cameras():
    return load_yaml("cameras.yaml")


def main():
    import uvicorn
    host = load_yaml("privacy.yaml").get("storage", {}).get("bind_host", "127.0.0.1")
    uvicorn.run("apps.teacher_ui.main:app", host=host, port=8000, reload=False)


if __name__ == "__main__":
    main()
