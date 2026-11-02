"""Compact progress backed by durable per-run logs and machine-readable measurements."""

import contextlib
import io
import time

from dwarl.cli import ui
from dwarl.runs.store import write_json


@contextlib.contextmanager
def display(runner, adapter, run, budget):
    start = time.monotonic()
    initial = runner.current_learning_iteration
    original = runner.logger.log
    with (run / "training.log").open("a") as log, ui.progress("Training", budget) as update:
        def capture(*args, **kwargs):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = original(*args, **kwargs)
            text = output.getvalue()
            log.write(text)
            log.flush()
            if ui.verbose:
                ui.info(text.rstrip())
            return result

        def iteration(index):
            elapsed = time.monotonic() - start
            measured = max(0, index - initial + 1)
            remaining = max(0, budget - index - 1)
            eta = remaining * elapsed / measured if measured else None
            state = {"iteration": index, "target": budget, "elapsed_s": elapsed, "eta_s": eta,
                     "iterations_per_s": measured / elapsed if elapsed else None}
            write_json(run / "progress.json", state)
            description = f"Training {index + 1}/{budget} · elapsed {elapsed:.0f}s · ETA {eta:.0f}s"
            update(index + 1, description)
            if not ui.console.is_terminal and (index == initial or (index + 1) % 25 == 0 or index + 1 == budget):
                ui.info(description)

        runner.logger.log = capture
        adapter.after_log.append(iteration)
        try:
            yield
        finally:
            runner.logger.log = original
            adapter.after_log.remove(iteration)
