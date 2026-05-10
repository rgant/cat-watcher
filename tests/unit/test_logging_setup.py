"""Tests for :mod:`cat_watcher.logging_setup`."""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from types import TracebackType
from typing import TYPE_CHECKING, cast

import pytest

from cat_watcher import logging_setup
from cat_watcher.logging_setup import JsonFormatter, setup_logging

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def reset_root_logger() -> Iterator[None]:
    """Snapshot the root logger's handlers + level so cross-test leakage can't mask bugs."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


type ExcInfoTuple = tuple[type[BaseException], BaseException, TracebackType | None] | tuple[None, None, None] | None


def _format_record(record: logging.LogRecord, *, agent_name: str = "poller") -> dict[str, object]:
    """Format ``record`` with a fresh ``JsonFormatter`` and parse the JSON line back to a dict."""
    formatter = JsonFormatter(agent_name=agent_name)
    line = formatter.format(record)
    assert "\n" not in line, "JsonFormatter output must be one physical line"
    return cast("dict[str, object]", json.loads(line))


def _build_record(  # noqa: PLR0913 — test-fixture builder; bundling args at the call-site is noisier.
    *,
    name: str = "cat_watcher.poller",
    level: int = logging.INFO,
    msg: str = "hello",
    args: tuple[object, ...] = (),
    exc_info: ExcInfoTuple = None,
    extra: dict[str, object] | None = None,
) -> logging.LogRecord:
    """Construct a ``LogRecord`` that mirrors the shape of one produced by a real logger call."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )
    if extra:
        for key, value in extra.items():
            record.__dict__[key] = value
    return record


def test_required_schema_fields_only_with_no_extras() -> None:
    """A vanilla log call emits exactly the six required schema fields — no spurious keys."""
    record = _build_record(msg="ingested clip")
    parsed = _format_record(record, agent_name="poller")
    assert set(parsed.keys()) == {"ts", "level", "logger", "agent", "pid", "msg"}
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "cat_watcher.poller"
    assert parsed["agent"] == "poller"
    assert parsed["msg"] == "ingested clip"
    assert isinstance(parsed["pid"], int)


def test_ts_is_iso8601_utc_with_microseconds() -> None:
    """``ts`` is ISO-8601 UTC with microsecond precision so log lines sort lexically by time."""
    record = _build_record()
    parsed = _format_record(record)
    ts = parsed["ts"]
    assert isinstance(ts, str)
    parsed_dt = datetime.fromisoformat(ts)
    offset = parsed_dt.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0
    assert "." in ts
    fractional = ts.split(".")[1]
    # ``isoformat(timespec='microseconds')`` always emits 6 digits before the offset marker.
    assert len(fractional.split("+")[0]) == 6


def test_per_call_extras_are_emitted_under_extras_key() -> None:
    """Logger ``extra=`` kwargs land under the ``extras`` key, isolated from the schema fields."""
    record = _build_record(msg="ingested clip", extra={"camera_name": "office", "clip_id": 42})
    parsed = _format_record(record)
    assert parsed["extras"] == {"camera_name": "office", "clip_id": 42}


def test_standard_logrecord_attributes_never_appear_in_extras() -> None:
    """``LogRecord`` builtins (``args``, ``msg``, etc.) must not leak into the ``extras`` dict."""
    record = _build_record(msg="formatted: %s", args=("kitchen",))
    parsed = _format_record(record)
    assert "extras" not in parsed
    assert parsed["msg"] == "formatted: kitchen"


def _raise_value_error() -> None:
    msg = "bad value"
    raise ValueError(msg)


def test_exception_capture_emits_qualified_exc_type_msg_and_traceback() -> None:
    """``exc_info`` produces the qualified type, the message, and a string traceback — operators need all three to triage."""
    exc_info: ExcInfoTuple = None
    try:
        _raise_value_error()
    except ValueError:
        exc_info = cast("ExcInfoTuple", sys.exc_info())
    record = _build_record(level=logging.ERROR, msg="caught", exc_info=exc_info)
    parsed = _format_record(record)
    assert parsed["exc_type"] == "builtins.ValueError"
    assert parsed["exc_msg"] == "bad value"
    assert isinstance(parsed["traceback"], str)
    assert "ValueError: bad value" in parsed["traceback"]
    assert "test_logging_setup" in parsed["traceback"]


def test_setup_logging_creates_logs_dir_and_writes_jsonl(tmp_path: Path) -> None:
    """First call materializes the ``logs/`` dir under ``internal_root`` and writes one record per JSONL line."""
    setup_logging(agent_name="poller", internal_root=tmp_path, level=logging.INFO)
    logging.getLogger("cat_watcher.test").info("first record")

    log_file = tmp_path / "logs" / "poller.jsonl"
    assert log_file.exists()
    line = log_file.read_text(encoding="utf-8").splitlines()[-1]
    parsed = cast("dict[str, object]", json.loads(line))
    assert parsed["msg"] == "first record"
    assert parsed["agent"] == "poller"


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    """Repeated calls leave the root logger with exactly one file + one stream handler."""
    setup_logging(agent_name="poller", internal_root=tmp_path, level=logging.INFO)
    setup_logging(agent_name="poller", internal_root=tmp_path, level=logging.INFO)
    handlers = logging.getLogger().handlers
    rotating_count = sum(1 for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler))
    # ``RotatingFileHandler`` subclasses ``StreamHandler``; exact-type comparison avoids double-counting.
    stream_count = sum(1 for h in handlers if type(h) is logging.StreamHandler)
    assert rotating_count == 1
    assert stream_count == 1


def test_setup_logging_does_not_call_basicconfig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Handlers must be configured explicitly; ``logging.basicConfig`` must not be called."""

    def _fail(*_args: object, **_kwargs: object) -> None:
        msg = "logging.basicConfig must not be called by setup_logging"
        raise AssertionError(msg)

    monkeypatch.setattr(logging, "basicConfig", _fail)
    setup_logging(agent_name="poller", internal_root=tmp_path, level=logging.INFO)


def test_rotation_creates_backup_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Forcing ``_MAX_BYTES`` low triggers rotation; both the active file and a ``.1`` backup must exist."""
    monkeypatch.setattr(logging_setup, "_MAX_BYTES", 200)
    setup_logging(agent_name="poller", internal_root=tmp_path, level=logging.INFO)

    log_file = tmp_path / "logs" / "poller.jsonl"
    logger = logging.getLogger("cat_watcher.rotation_test")
    for i in range(50):
        logger.info("padding %d %s", i, "x" * 30)

    backups = sorted((tmp_path / "logs").glob("poller.jsonl*"))
    assert log_file in backups
    rolled = [p for p in backups if p.suffix == ".1" or p.name.endswith(".jsonl.1")]
    assert rolled, f"expected at least one rotated backup, found: {backups}"


def test_stderr_handler_respects_level_parameter(tmp_path: Path) -> None:
    """The stderr handler must match the root logger level.

    Without this, ``--verbose`` (INFO) wouldn't surface INFO records on stderr — they'd only reach
    the JSONL file — and the default (WARNING) wouldn't keep stderr quiet.
    """
    setup_logging(agent_name="poller", internal_root=tmp_path, level=logging.INFO)
    handlers = logging.getLogger().handlers
    info_streams = sum(1 for h in handlers if type(h) is logging.StreamHandler and h.level == logging.INFO)
    assert info_streams == 1, f"expected one StreamHandler at INFO, got handlers={handlers!r}"

    setup_logging(agent_name="poller", internal_root=tmp_path, level=logging.WARNING)
    handlers = logging.getLogger().handlers
    warning_streams = sum(1 for h in handlers if type(h) is logging.StreamHandler and h.level == logging.WARNING)
    assert warning_streams == 1, f"expected one StreamHandler at WARNING, got handlers={handlers!r}"


def test_single_line_invariant_with_embedded_newlines() -> None:
    """Embedded newlines in a message round-trip through JSON without ever breaking the one-record-per-line contract."""
    record = _build_record(msg="line one\nline two\nline three")
    formatter = JsonFormatter(agent_name="poller")
    line = formatter.format(record)
    assert line.count("\n") == 0
    parsed = cast("dict[str, object]", json.loads(line))
    assert parsed["msg"] == "line one\nline two\nline three"


def test_extras_with_non_serializable_value_falls_back_to_str(tmp_path: Path) -> None:
    """Non-JSON-serializable extras (e.g. ``Path``) round-trip via ``default=str`` rather than crashing."""
    record = _build_record(msg="see path", extra={"path": tmp_path})
    parsed = _format_record(record)
    extras = parsed["extras"]
    assert isinstance(extras, dict)
    assert extras["path"] == str(tmp_path)
