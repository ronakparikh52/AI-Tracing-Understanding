"""Shared helpers for the step scripts."""
import json
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path


class _Tee:
    """Write everything to several streams (terminal + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_current_file = None


def save_output(script_file: str, name: str | None = None) -> Path:
    """Mirror everything printed from now on into a .txt next to the script.

    The file is <script name>.txt, or <name>.txt if `name` is given (used to
    give each scenario its own file). Each run overwrites the previous file.
    """
    global _current_file
    if _current_file:
        _current_file.close()
    script = Path(script_file)
    path = script.parent / f"{name or script.stem}.txt"
    _current_file = open(path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _current_file)
    print(f"# {script.parent.name}/{path.name} - run at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    return path


def _jsonable(obj):
    """json.dumps fallback: SDK objects (Pydantic models) -> plain dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    raise TypeError(f"Can't serialize {type(obj)}")


class JsonlLog:
    """Append one JSON object per line - the full raw record of every turn."""

    def __init__(self, path: Path):
        self.path = path
        self.path.write_text("")  # start fresh each run
        self._lock = threading.Lock()  # parallel runs write to the same file

    def write(self, record: dict) -> None:
        line = json.dumps(record, default=_jsonable) + "\n"
        with self._lock:  # one writer at a time -> lines never interleave
            with self.path.open("a") as f:
                f.write(line)


# --- Step 7: span tracing ----------------------------------------------------
# A SPAN is one unit of work: a dict with ids, a name, start/end time and
# attributes. parent_id links spans into a tree. Same fields OpenTelemetry uses
# (trace_id, span_id, parent_span_id, start/end, attributes, status), so this
# can later be swapped for the real OTel SDK without changing call sites much.
class Tracer:
    def __init__(self, path: Path):
        self.path = path
        self.path.write_text("")       # start fresh each run
        # Spans currently open; the last one is the parent. A ContextVar so that
        # each thread has its OWN stack - with one shared list, parallel runs
        # would become each other's parents. (OpenTelemetry does exactly this.)
        self._stack: ContextVar[tuple] = ContextVar("span_stack", default=())
        self._lock = threading.Lock()  # guards the file + finished list
        self.finished: list[dict] = []

    @contextmanager
    def span(self, name: str, **attributes):
        """Usage:
            with tracer.span("tool", tool="issue_refund") as s:
                ...                       # work being timed
                s["attributes"]["ok"] = True   # add facts learned during the work
        A span opened inside another span's `with` block becomes its child.
        """
        stack = self._stack.get()
        parent = stack[-1] if stack else None
        s = {
            # Root span starts a new trace; children inherit its trace_id.
            "trace_id": parent["trace_id"] if parent else uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "parent_id": parent["span_id"] if parent else None,
            "name": name,
            "start": time.time(),
            "attributes": dict(attributes),
            "status": "ok",
        }
        token = self._stack.set(stack + (s,))  # push (for this thread only)
        try:
            yield s
        except Exception as e:
            s["status"] = "error"
            s["attributes"]["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._stack.reset(token)              # pop
            s["end"] = time.time()
            s["duration_ms"] = round((s["end"] - s["start"]) * 1000, 1)
            # Written when the span ENDS, so children land in the file before
            # their parent - same as real OTel exporters. Rebuild the tree from
            # parent_id, not from line order.
            line = json.dumps(s, default=str) + "\n"
            with self._lock:
                self.finished.append(s)
                with self.path.open("a") as f:
                    f.write(line)


def render_trace(spans: list[dict], trace_id: str) -> list[str]:
    """Draw one trace as an indented tree (what a tracing UI shows as a waterfall)."""
    spans = [s for s in spans if s["trace_id"] == trace_id]
    children: dict = {}
    for s in spans:
        children.setdefault(s["parent_id"], []).append(s)
    lines: list[str] = []

    def walk(parent_id, depth):
        for s in sorted(children.get(parent_id, []), key=lambda s: s["start"]):
            attrs = " ".join(f"{k}={v}" for k, v in s["attributes"].items())
            flag = "" if s["status"] == "ok" else f" [{s['status'].upper()}]"
            lines.append(f"{'  ' * depth}{s['name']}{flag}  {s['duration_ms']:.0f}ms  {attrs}")
            walk(s["span_id"], depth + 1)

    walk(None, 0)
    return lines
