"""Shared, line-oriented terminal UI. Human output always goes to stderr."""

import json
import os
import shlex
import sys
from contextlib import contextmanager

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.theme import Theme

console = Console(stderr=True, theme=Theme({"accent": "magenta", "ok": "green", "check": "yellow"}),
                  highlight=False, markup=False)
verbose = False
machine = False


def configure(plain=False, json_output=False, debug=False):
    global console, verbose, machine
    verbose, machine = debug, json_output
    console = Console(stderr=True, force_terminal=False if plain or json_output else None,
                      no_color=plain or json_output or "NO_COLOR" in os.environ,
                      highlight=False, markup=False)


def title(name, detail=None):
    console.print(f"DWARL · {name}", style="bold magenta")
    if detail:
        console.print(detail)


def info(message):
    console.print(str(message))


def finding(level, message, hint=None):
    style = {"pass": "green", "check": "yellow", "fail": "red", "skip": "dim"}.get(level, "")
    console.print(f"  {level.upper():5}  {message}", style=style)
    if hint:
        console.print(f"         {hint}", style="dim")


def command(*args):
    return shlex.join(["pixi", "run", *map(str, args)])


def next_step(*args):
    if not machine:
        console.print("\nNext:", style="bold")
        console.print(f"  {command(*args)}", style="magenta")


def table(columns, rows):
    grid = Table(box=None, padding=(0, 2), expand=False)
    for column in columns:
        grid.add_column(column, overflow="fold")
    for row in rows:
        grid.add_row(*(str(v) if v is not None else "—" for v in row))
    console.print(grid)


def result(data):
    if machine:
        print(json.dumps(data, indent=2, default=str, allow_nan=False))


class Back(Exception):
    """Return to the preceding wizard field."""


def ask(label, default="", *, choices=(), path=False, interactive=True):
    if not interactive:
        if default == "":
            raise ValueError(f"missing required input: {label}")
        return str(default)
    suffix = f" [{default}]" if str(default) else ""
    message = f"{label}{suffix}: "
    if sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM") != "dumb":
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import PathCompleter, WordCompleter

        completer = PathCompleter(expanduser=True) if path else WordCompleter(list(choices), match_middle=True)
        answer = prompt(message, completer=completer).strip()
    else:
        answer = input(message).strip()
    if answer == ":back":
        raise Back
    return answer or str(default)


def choose(label, choices, default=None):
    choices = list(choices)
    for i, choice in enumerate(choices, 1):
        info(f"  {i}. {choice}")
    while True:
        answer = ask(label, default or choices[0], choices=choices)
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        if answer in choices:
            return answer
        finding("check", "Choose a listed value or its number.")


@contextmanager
def progress(label, total=None):
    if not console.is_terminal or machine or verbose:
        info(label)
        yield lambda completed, description=None: None
        return
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
                  TimeElapsedColumn(), console=console, transient=True) as bar:
        task = bar.add_task(label, total=total)
        yield lambda completed, description=None: bar.update(task, completed=completed,
                                                             description=description or label)
