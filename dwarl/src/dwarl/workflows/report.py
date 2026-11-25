"""Readable report entry point; the HTML is derived from existing recorded evidence."""

import html
import json
import os
import webbrowser
from pathlib import Path

from dwarl.cli import ui


def render_html(directory):
    import markdown

    report = Path(directory) / "report"
    source = report / "README.md"
    if not source.exists():
        raise ValueError(f"Markdown report unavailable: {source}; inspect summary.json and launch logs.")
    # Escape raw HTML in source values while retaining Markdown tables and generated relative image links.
    body = markdown.markdown(html.escape(source.read_text(), quote=False), extensions=["tables", "fenced_code"])
    destination = report / "index.html"
    destination.write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>DWARL · Navigation report</title>
<style>
:root{color-scheme:light dark;--accent:#85358a}body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;
margin:auto;padding:2rem}h1,h2{line-height:1.2}a{color:var(--accent)}table{border-collapse:collapse;display:block;
overflow:auto}td,th{padding:.5rem .75rem;border-bottom:1px solid #8886;text-align:left}img{max-width:100%;height:auto}
pre{overflow:auto;padding:1rem;background:#8881}code{overflow-wrap:anywhere}h2{margin-top:3rem}
@media(prefers-color-scheme:dark){:root{--accent:#cf97d5}}@media(max-width:600px){body{padding:1rem}}
</style><main><p>DWARL · Local evaluation evidence</p>''' + body + "</main></html>\n")
    return destination


def summarize(report):
    ui.title("Test results", f"Status: {report['status']} · checkpoint {str(report.get('checkpoint', ''))[:12]}")
    rows = []
    def percent(value):
        return "unavailable" if value is None else f"{value:.1%}"

    for name, stats in report.get("simulators", {}).items():
        rows.append((name, f"{stats['completed']}/{stats['expected']}", percent(stats.get("success_rate")),
                     percent(stats.get("collision_rate"))))
    ui.table(["Backend", "Completed", "Success", "Collision"], rows)
    if report.get("error"):
        ui.finding("fail", report["error"])


def execute(directory=None, run=None, open_browser=False):
    if directory is None:
        from dwarl.runs.store import resolve_run
        from dwarl.workflows.context import selected_run

        reference = run or selected_run()
        if not reference:
            raise ValueError("No run selected; pass --run or a test directory.")
        candidates = sorted((resolve_run(reference) / "tests").glob("*/summary.json"))
        if not candidates:
            raise ValueError("This run has no recorded tests. Run `pixi run test --suite validation` first.")
        directory = candidates[-1].parent
    directory = Path(directory).expanduser().resolve()
    report = json.loads((directory / "summary.json").read_text())
    summarize(report)
    page = None
    if (directory / "report/README.md").is_file():
        page = render_html(directory)
        ui.info(f"Report: {page.as_uri()}")
        if open_browser and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            if not webbrowser.open(page.as_uri()):
                ui.info("Browser could not be opened; use the report path above.")
    else:
        ui.info(f"Raw results: {directory / 'summary.json'}")
    return {"directory": str(directory), "html": str(page) if page else None, **report}
