"""Structured logging setup for cat-watcher agents.

Provides :class:`JsonFormatter` and :func:`setup_logging`. Each agent's ``main()`` calls
``setup_logging(agent_name=..., internal_root=..., level=...)`` once to wire two handlers onto the
root logger:

* A :class:`logging.handlers.RotatingFileHandler` writing JSONL records to
  ``<internal_root>/logs/<agent>.jsonl`` (10 MB rotation, 7 backups). Every record is one line of
  JSON conforming to the schema in
  ``docs/specs/2026-05-05-structured-logging-design.md``.
* A :class:`logging.StreamHandler` on stderr at the same ``level`` as the root logger, so genuine
  problems hit the LaunchAgent's ``<agent>.stderr.log`` fallback even if no one is reading the
  JSONL file. Under ``--verbose``/``level=INFO``, diagnostic detail (httpxyz requests, the
  empty-window note from amcrest_client, retries) also surfaces on stderr.

The formatter stamps each record with the agent slug and current PID, so existing
``logging.getLogger(__name__)`` call sites pick those fields up without any per-call change.
Per-call structured fields use the standard ``extra={...}`` parameter; anything in there that isn't
a standard :class:`logging.LogRecord` attribute lands under the optional ``extras`` key.
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast, final, override

if TYPE_CHECKING:
    from pathlib import Path

    from cat_watcher.config import Config


# Names every Python `logging` release sets on a `LogRecord`. Used by the formatter to compute the
# ``extras`` set as ``record.__dict__.keys() - _STANDARD_LOGRECORD_ATTRS``. Sourced from
# `logging.LogRecord.__init__` in CPython; ``message`` and ``taskName`` are added later by the
# logging machinery (the latter only when ``contextvars.Task`` is set).
_STANDARD_LOGRECORD_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",
    },
)

# Module-level so tests can monkey-patch them down to small values for fast rotation assertions.
_MAX_BYTES: int = 10 * 1024 * 1024
_BACKUP_COUNT: int = 7


def _exc_type_qualname(exc: BaseException) -> str:
    """Return ``"<module>.<class>"`` for ``exc``'s concrete type.

    Classes lacking ``__module__`` resolve to ``"<unknown>.<class>"`` so the schema's ``exc_type``
    field is always a well-formed dotted name.
    """
    cls = type(exc)
    module = getattr(cls, "__module__", "<unknown>") or "<unknown>"
    return f"{module}.{cls.__qualname__}"


@final
class JsonFormatter(logging.Formatter):
    """Format :class:`logging.LogRecord` instances as one-line JSON per the structured-logging schema."""

    def __init__(self, *, agent_name: str) -> None:
        """Pin ``agent_name`` into every emitted record's ``agent`` field."""
        super().__init__()
        self._agent_name = agent_name

    @override
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="microseconds")
        payload: dict[str, object] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "agent": self._agent_name,
            "pid": os.getpid(),
            "msg": record.getMessage(),
        }

        # Cast to ``dict[str, object]`` so callers must narrow before use (stdlib types it as Any).
        record_attrs = cast("dict[str, object]", record.__dict__)
        extras: dict[str, object] = {key: value for key, value in record_attrs.items() if key not in _STANDARD_LOGRECORD_ATTRS}
        if extras:
            payload["extras"] = extras

        if record.exc_info:
            exc = record.exc_info[1]
            if exc is not None:
                payload["exc_type"] = _exc_type_qualname(exc)
                payload["exc_msg"] = str(exc)
                payload["traceback"] = "".join(traceback.format_exception(record.exc_info[0], exc, record.exc_info[2]))

        # ``default=str`` is the safety net for non-JSON-serializable objects passed via
        # ``extra={}`` (e.g. a Path or datetime); ``ensure_ascii=False`` keeps unicode legible.
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(*, agent_name: str, internal_root: Path, level: int) -> None:
    """Wire structured logging for an agent's ``main()`` entry point.

    Idempotent: each call replaces existing root-logger handlers rather than augmenting them, so
    repeat invocations leave the root logger with the same two-handler shape.

    Parameters
    ----------
    agent_name:
        One of ``poller``, ``alerts``, ``web``, ``backup``, ``cli``. Stamped on every record's
        ``agent`` field by :class:`JsonFormatter`.
    internal_root:
        ``config.internal_root``. The JSONL file lives at ``<internal_root>/logs/<agent>.jsonl``.
    level:
        Root logger level (typically ``logging.WARNING`` by default, ``logging.INFO`` under
        ``--verbose``).

    """
    log_dir = internal_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = JsonFormatter(agent_name=agent_name)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"{agent_name}.jsonl",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger()
    # Clear-then-attach makes the file handler the canonical sink. ``handler.close()`` releases the
    # open file descriptor before the handler is dropped.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)
    root.setLevel(level)


def setup_agent_logging(*, agent_name: str, config: Config, verbose: bool = False) -> None:
    """Wire structured logging for an agent's ``main()`` from its ``Config``.

    The root logger level is ``config.log_level``, upgraded to at least ``INFO`` when ``verbose``
    is true so a ``--verbose`` CLI flag can raise visibility without dropping the configured
    baseline (e.g. a ``DEBUG`` config stays at ``DEBUG``).
    """
    config_level = logging.getLevelNamesMapping()[config.log_level]
    level = min(config_level, logging.INFO) if verbose else config_level
    setup_logging(
        agent_name=agent_name,
        internal_root=config.internal_root,
        level=level,
    )
