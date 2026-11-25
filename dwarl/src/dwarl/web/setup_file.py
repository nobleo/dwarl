"""Portable dashboard preferences, independent of runs and browser sessions."""

import json

from dwarl.runs.store import write_json

FILE = "web/dashboard-setup.json"
VERSION = "dwarl-dashboard/1"


def names(items):
    return (isinstance(items, list) and all(isinstance(s, str) and 0 < len(s) <= 512 for s in items)
            and len(set(items)) == len(items))


def validated(data):
    if not isinstance(data, dict) or data.get("schema") != VERSION:
        raise ValueError("unsupported dashboard setup schema")
    plots = data.get("plots")
    smoothing = data.get("smoothing", .6)
    if plots is not None and not names(plots):
        raise ValueError("plots must be an ordered list of unique metric names")
    if type(smoothing) not in (int, float) or not 0 <= smoothing <= .95:
        raise ValueError("smoothing must be between 0 and 0.95")
    from dwarl.web.settings import editable

    settings = data.get("settings", {})
    if not isinstance(settings, dict) or settings.keys() - {"train", "test", "deploy"}:
        raise ValueError("expected settings grouped by workflow")
    for workflow, rows in settings.items():
        if (not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows)
                or not names([r.get("path") for r in rows])
                or any(not editable(workflow, r["path"]) or r.keys() - {"path", "value"} for r in rows)):
            raise ValueError(f"invalid {workflow} settings")
    return {"schema": VERSION, "plots": plots, "smoothing": smoothing, "settings": settings}


def load(root):
    path = root / FILE
    return validated(json.loads(path.read_text()) if path.exists() else {"schema": VERSION})


def save(root, data):
    setup = validated(data)
    write_json(root / FILE, setup)
    return setup
