"""HTML report renderer."""

from __future__ import annotations

from src.types import StudentReport


def render_report_html(report: StudentReport) -> str:
    rows = "".join(
        f"<tr><td>{ps.phase}</td><td>{ps.metric}</td>"
        f"<td>{ps.value:.1f}</td><td>{ps.score:.0f}</td>"
        f"<td>{ps.feedback}</td></tr>"
        for ps in report.phase_scores
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>动作分析报告 — {report.student_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    h1 {{ color: #1a365d; }}
    .score {{ font-size: 2rem; font-weight: bold; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
    th {{ background: #edf2f7; }}
  </style>
</head>
<body>
  <h1>篮球课堂动作分析报告</h1>
  <p>学生 ID：<strong>{report.student_id}</strong></p>
  <p>Session：<strong>{report.session_id}</strong></p>
  <p>动作类型：<strong>{report.action_type}</strong></p>
  <p class="score">总分：{report.total_score}</p>
  <p>{report.summary}</p>
  <p>身份置信度：{report.identity_confidence}</p>
  <table>
    <thead><tr><th>阶段</th><th>指标</th><th>实测值</th><th>得分</th><th>反馈</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""
