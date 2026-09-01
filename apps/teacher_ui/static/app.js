let sessionId = null;

async function createSession() {
  const class_id = document.getElementById('classId').value;
  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ class_id }),
  });
  const data = await res.json();
  sessionId = data.session_id;
  document.getElementById('sessionId').textContent = sessionId;
}

async function registerStudent() {
  if (!sessionId) return alert('请先创建 Session');
  const student_id = document.getElementById('studentId').value;
  const display_name = document.getElementById('studentName').value;
  await fetch(`/api/sessions/${sessionId}/students`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id, display_name }),
  });
  alert('学生已注册');
}

async function grantConsent() {
  if (!sessionId) return alert('请先创建 Session');
  const student_id = document.getElementById('studentId').value;
  await fetch(`/api/sessions/${sessionId}/consent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id, scopes: ['video', 'face', 'report'] }),
  });
  await refreshConsent();
}

async function refreshConsent() {
  if (!sessionId) return;
  const res = await fetch(`/api/sessions/${sessionId}/consent`);
  const list = await res.json();
  const ul = document.getElementById('consentList');
  ul.innerHTML = list.map(c =>
    `<li>${c.student_id}: ${c.active ? '已同意' : '已撤回'} [${c.scopes.join(', ')}]</li>`
  ).join('');
}

async function markRecorded() {
  if (!sessionId) return;
  await fetch(`/api/sessions/${sessionId}/recorded`, { method: 'POST' });
  alert('已标记录制完成');
}

async function runPipeline() {
  if (!sessionId) return;
  document.getElementById('pipelineResult').textContent = '分析中...';
  const res = await fetch(`/api/sessions/${sessionId}/pipeline`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ from_stage: 'perception' }),
  });
  const data = await res.json();
  document.getElementById('pipelineResult').textContent = JSON.stringify(data, null, 2);
  await loadReports();
}

async function loadReports() {
  if (!sessionId) return;
  const res = await fetch(`/api/sessions/${sessionId}/reports`);
  const list = await res.json();
  const ul = document.getElementById('reportList');
  ul.innerHTML = list.map(r =>
    `<li><a href="/api/sessions/${sessionId}/reports/${r.student_id}/html" target="_blank">${r.student_id} 报告</a></li>`
  ).join('');
}

async function loadAudit() {
  const res = await fetch('/api/audit' + (sessionId ? `?session_id=${sessionId}` : ''));
  const data = await res.json();
  document.getElementById('auditLog').textContent = JSON.stringify(data, null, 2);
}
