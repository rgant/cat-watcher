"""Integration test for the ``cat-watcher logs --follow`` tail mode.

Spawns the follow loop in a background thread, writes records into the JSONL file from another
thread, and asserts the new lines surface in the viewer's output within one polling interval.
"""

import io
import json
import threading
import time
from typing import TYPE_CHECKING, cast

from cat_watcher import logs_viewer
from cat_watcher.logs_viewer import RunArgs

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _record(ts: str, msg: str) -> str:
    """Serialize a JSONL record line (with trailing newline) for fixture writes."""
    payload = {
        "ts": ts,
        "level": "INFO",
        "logger": "cat_watcher.test",
        "agent": "poller",
        "pid": 1234,
        "msg": msg,
    }
    return json.dumps(payload) + "\n"


def _run_follow_in_thread(
    internal_root: Path,
    sink: io.StringIO,
    stop: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
) -> threading.Thread:
    """Patch ``time.sleep`` process-wide so the follow worker exits promptly when ``stop`` is set."""
    original_sleep = time.sleep

    def _interruptible_sleep(seconds: float) -> None:
        original_sleep(min(seconds, 0.1))
        if stop.is_set():
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", _interruptible_sleep)

    def _target() -> None:
        args = RunArgs(
            agent="poller",
            follow=True,
            since=None,
            level=None,
            camera_filter=None,
            grep=None,
            json_mode=True,
        )
        _ = logs_viewer.run(args, internal_root=internal_root, out=sink)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread


def _appended_msgs_from_sink(sink: io.StringIO) -> list[str]:
    """Parse current sink contents into the list of `msg` strings (skipping non-string msgs)."""
    out: list[str] = []
    for line in sink.getvalue().splitlines():
        if not line:
            continue
        msg = cast("dict[str, object]", json.loads(line)).get("msg")
        if isinstance(msg, str):
            out.append(msg)
    return out


def _wait_for_messages(sink: io.StringIO, expected: list[str], timeout_s: float) -> list[str]:
    """Poll ``sink`` up to ``timeout_s`` for every message in ``expected``; return the seen list."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        seen = _appended_msgs_from_sink(sink)
        if all(m in seen for m in expected):
            return seen
        time.sleep(0.1)
    return _appended_msgs_from_sink(sink)


def test_follow_emits_new_lines_within_polling_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lines appended to the JSONL after the follow loop starts surface in stdout within ~1 polling interval."""
    log_path = tmp_path / "logs" / "poller.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ = log_path.write_text(_record("2026-05-05T10:00:00.000000+00:00", "seed-record"), encoding="utf-8")

    sink = io.StringIO()
    stop = threading.Event()
    thread = _run_follow_in_thread(tmp_path, sink, stop, monkeypatch)

    appended_msgs = ["new-record-1", "new-record-2"]
    time.sleep(0.2)
    with log_path.open("a", encoding="utf-8") as fh:
        for i, msg in enumerate(appended_msgs, start=1):
            ts = f"2026-05-05T10:0{i}:00.000000+00:00"
            _ = fh.write(_record(ts, msg))
            fh.flush()

    seen = _wait_for_messages(sink, appended_msgs, timeout_s=2.0)

    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "follow thread did not exit on stop signal"
    assert "new-record-1" in seen
    assert "new-record-2" in seen


def test_follow_handles_rotation_inode_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the JSONL is rotated (rename + new file at the same path), the follow loop picks up the new file."""
    log_path = tmp_path / "logs" / "poller.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ = log_path.write_text(_record("2026-05-05T10:00:00.000000+00:00", "before-rotation"), encoding="utf-8")

    sink = io.StringIO()
    stop = threading.Event()
    thread = _run_follow_in_thread(tmp_path, sink, stop, monkeypatch)
    time.sleep(0.3)

    rotated = log_path.with_suffix(log_path.suffix + ".1")
    _ = log_path.rename(rotated)
    _ = log_path.write_text(_record("2026-05-05T11:00:00.000000+00:00", "after-rotation"), encoding="utf-8")

    deadline = time.monotonic() + 2.0
    seen = False
    while time.monotonic() < deadline:
        out = sink.getvalue()
        if "after-rotation" in out:
            seen = True
            break
        time.sleep(0.1)

    stop.set()
    thread.join(timeout=2.0)
    assert seen, "follow loop did not pick up the new file after rotation"


def test_follow_loop_exits_cleanly_on_keyboard_interrupt(tmp_path: Path) -> None:
    """Direct unit-style test of the loop function — its KeyboardInterrupt path returns 0."""
    log_path = tmp_path / "logs" / "poller.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ = log_path.write_text(_record("2026-05-05T10:00:00.000000+00:00", "seed"), encoding="utf-8")

    captured: list[dict[str, object]] = []

    def _emit(record: dict[str, object]) -> None:
        captured.append(record)
        raise KeyboardInterrupt

    rc = logs_viewer._follow_loop(
        [log_path],
        filters=logs_viewer._Filters(since=None, level=None, camera=None, grep=None),
        emit=_emit,
        poll_seconds=0.05,
    )
    assert rc == 0
    assert len(captured) == 1
