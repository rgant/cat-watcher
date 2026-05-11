"""Implementation of the ``cat-watcher logs`` sub-command.

Reads JSONL records written by :mod:`cat_watcher.logging_setup`, applies operator filters
(``--since``, ``--level``, ``--camera``, ``--grep``), and emits either the raw lines(``--json``) or
a pretty, color-aware columnar format. ``--follow`` tails the selected files and re-opens them on
rotation so the operator can ``pixi run logs`` and see new records as they're written.
"""

import argparse
import heapq
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import IO, TYPE_CHECKING, Final, cast, final

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

# Default subset surfaced when the operator omits the positional <agent> argument. ``cli`` is
# excluded so a manual ``cat-watcher import-local`` run doesn't drown out daemon noise; pass
# ``cli`` explicitly to opt back in.
_LAUNCHAGENT_AGENTS: Final[tuple[str, ...]] = ("poller", "alerts", "web", "backup")
_ALL_AGENTS: Final[tuple[str, ...]] = (*_LAUNCHAGENT_AGENTS, "cli")

_DURATION_RE: Final[re.Pattern[str]] = re.compile(r"^(\d+)([smhd])$")
_DURATION_UNITS: Final[dict[str, int]] = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_FOLLOW_POLL_SECONDS: Final[float] = 0.5

_ANSI_RESET: Final[str] = "\x1b[0m"
_ANSI_BY_LEVEL: Final[dict[str, str]] = {
    "DEBUG": "\x1b[2m",
    "INFO": "",
    "WARNING": "\x1b[33m",
    "ERROR": "\x1b[31m",
    "CRITICAL": "\x1b[1;31m",
}

# Right-padded to a fixed 5-char column so the pretty format's level column stays aligned across
# every level. ``WARNING`` → ``WARN``, ``CRITICAL`` → ``CRIT``.
_LEVEL_DISPLAY: Final[dict[str, str]] = {
    "DEBUG": "DEBUG",
    "INFO": "INFO ",
    "WARNING": "WARN ",
    "ERROR": "ERROR",
    "CRITICAL": "CRIT ",
}

# Agent column width. ``+ 1`` keeps at least one space between agent and msg even at the longest
# slug; ``max`` re-derives if a future agent name is added to ``_ALL_AGENTS``.
_AGENT_COL_WIDTH: Final[int] = max(len(name) for name in _ALL_AGENTS) + 1

# Parsed JSONL records have dynamic shape; ``object`` values force every read site to narrow
# explicitly via ``isinstance`` before use.
type LogRecordDict = dict[str, object]


def parse_since(value: str) -> datetime:
    """Parse a ``--since`` argument: duration shorthand (``30m``/``1h``/``7d``) or ISO 8601.

    Returns a tz-aware UTC datetime. Naive ISO inputs are interpreted as OS-local, matching the
    poller's ``--since`` semantics.
    """
    match = _DURATION_RE.match(value)
    if match is not None:
        amount = int(match.group(1))
        seconds = amount * _DURATION_UNITS[match.group(2)]
        return datetime.now(UTC) - timedelta(seconds=seconds)
    return datetime.fromisoformat(value).astimezone(UTC)


def _resolve_agent_files(internal_root: Path, agents: Iterable[str]) -> list[Path]:
    log_dir = internal_root / "logs"
    return [log_dir / f"{agent}.jsonl" for agent in agents]


def _iter_records_in(fh: IO[str]) -> Iterator[LogRecordDict]:
    """Parse one JSONL record per non-empty line from ``fh``; silently skip malformed lines.

    A malformed line almost always means a write-in-progress or a torn line at rotation. The viewer
    is read-only and should not abort on torn data; the next read sees a consistent file.
    """
    for raw in fh:
        line = raw.rstrip("\n")
        if not line:
            continue
        try:
            parsed = cast("object", json.loads(line))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield cast("LogRecordDict", parsed)


def _parse_jsonl_lines(path: Path) -> Iterator[LogRecordDict]:
    """Yield each JSONL record in ``path``; missing files yield nothing."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        yield from _iter_records_in(fh)


def _record_ts_key(record: LogRecordDict) -> str:
    """Sortable key for chronological merge (lexicographic ISO 8601 sorts chronologically)."""
    ts = record.get("ts")
    return ts if isinstance(ts, str) else ""


def _level_at_least(record_level: str, threshold: str) -> bool:
    name_to_level = logging.getLevelNamesMapping()
    record_no = name_to_level.get(record_level)
    threshold_no = name_to_level.get(threshold)
    if record_no is None or threshold_no is None:
        return False
    return record_no >= threshold_no


def _record_after(record: LogRecordDict, since: datetime) -> bool:
    ts = record.get("ts")
    if not isinstance(ts, str):
        return False
    try:
        record_dt = datetime.fromisoformat(ts)
    except ValueError:
        return False
    return record_dt >= since


def _record_at_level(record: LogRecordDict, level: str) -> bool:
    record_level = record.get("level")
    return isinstance(record_level, str) and _level_at_least(record_level, level)


def _record_camera_matches(record: LogRecordDict, camera: str) -> bool:
    extras = record.get("extras")
    if not isinstance(extras, dict):
        return False
    extras_typed = cast("dict[str, object]", extras)
    return extras_typed.get("camera_name") == camera


def _record_msg_contains(record: LogRecordDict, needle: str) -> bool:
    msg = record.get("msg")
    return isinstance(msg, str) and needle.lower() in msg.lower()


def _format_ts(raw_ts: object) -> str:
    if not isinstance(raw_ts, str):
        return "?"
    try:
        return datetime.fromisoformat(raw_ts).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw_ts


def _format_extras(extras_obj: object) -> str:
    if not isinstance(extras_obj, dict):
        return ""
    extras_typed = cast("dict[str, object]", extras_obj)
    return " ".join(f"{key}={extras_typed[key]}" for key in sorted(extras_typed))


def _format_pretty(record: LogRecordDict, *, use_color: bool) -> str:
    ts_str = _format_ts(record.get("ts"))
    level_raw = record.get("level")
    level_str = level_raw if isinstance(level_raw, str) else "?"
    level_display = _LEVEL_DISPLAY.get(level_str, level_str.ljust(5))
    agent_raw = record.get("agent")
    agent_str = agent_raw if isinstance(agent_raw, str) else "?"
    msg_raw = record.get("msg")
    msg_str = msg_raw if isinstance(msg_raw, str) else ""
    extras_pairs = _format_extras(record.get("extras"))

    line = f"{ts_str}  {level_display}  {agent_str:<{_AGENT_COL_WIDTH}}  {msg_str}"
    if extras_pairs:
        line = f"{line}  {extras_pairs}"
    if "traceback" in record:
        line = f"{line} [+traceback]"

    if use_color:
        color = _ANSI_BY_LEVEL.get(level_str, "")
        if color:
            line = f"{color}{line}{_ANSI_RESET}"
    return line


def _emit_one(record: LogRecordDict, *, out: IO[str], json_mode: bool, use_color: bool) -> None:
    rendered = json.dumps(record, ensure_ascii=False) if json_mode else _format_pretty(record, use_color=use_color)
    _ = out.write(f"{rendered}\n")


def _emit_records(
    records: Iterable[LogRecordDict],
    *,
    out: IO[str],
    json_mode: bool,
    use_color: bool,
) -> None:
    for record in records:
        _emit_one(record, out=out, json_mode=json_mode, use_color=use_color)


@final
@dataclass(frozen=True, slots=True)
class _Filters:
    """Shared filter state used by both the one-shot and ``--follow`` paths."""

    since: datetime | None
    level: str | None
    camera: str | None
    grep: str | None

    def passes(self, record: LogRecordDict) -> bool:
        """Return ``True`` iff ``record`` satisfies every active filter (``None`` filters skip)."""
        if self.since is not None and not _record_after(record, self.since):
            return False
        if self.level is not None and not _record_at_level(record, self.level):
            return False
        if self.camera is not None and not _record_camera_matches(record, self.camera):
            return False
        if self.grep is not None and not _record_msg_contains(record, self.grep):  # noqa: SIM103 — explicit returns make the early-exit chain readable.
            return False
        return True


def _collect_filtered(files: Iterable[Path], *, filters: _Filters) -> list[LogRecordDict]:
    """Read every selected file, filter, then merge chronologically (stable on ts)."""
    streams = [_parse_jsonl_lines(p) for p in files]
    merged = heapq.merge(*streams, key=_record_ts_key)
    return [r for r in merged if filters.passes(r)]


@final
class _FollowState:
    """Per-file inode + position so ``--follow`` re-opens cleanly on rotation."""

    def __init__(self) -> None:
        self._table: dict[Path, tuple[int, int]] = {}

    def get(self, path: Path, current_inode: int) -> int:
        """Return the last-known read position for ``path``; 0 if the inode has changed."""
        last_inode, last_pos = self._table.get(path, (current_inode, 0))
        if last_inode != current_inode:
            return 0
        return last_pos

    def set(self, path: Path, inode: int, position: int) -> None:
        """Record that ``path`` (currently at ``inode``) has been read up to ``position``."""
        self._table[path] = (inode, position)

    def drop(self, path: Path) -> None:
        """Forget any previous state for ``path`` (e.g. after the file disappears)."""
        _ = self._table.pop(path, None)


def _scan_new_records(path: Path, start_pos: int) -> tuple[list[LogRecordDict], int]:
    """Read ``path`` from ``start_pos`` to EOF; return parsed records + the new file position."""
    with path.open(encoding="utf-8") as fh:
        _ = fh.seek(start_pos)
        records = list(_iter_records_in(fh))
        new_pos = fh.tell()
    return records, new_pos


def _follow_loop(
    files: list[Path],
    *,
    filters: _Filters,
    emit: Callable[[LogRecordDict], None],
    poll_seconds: float = _FOLLOW_POLL_SECONDS,
) -> int:
    """Tail-and-watch the selected files; re-open on rotation (inode change or shrinking size)."""
    state = _FollowState()
    try:
        while True:
            for path in files:
                if not path.exists():
                    state.drop(path)
                    continue
                stat = path.stat()
                last_pos = state.get(path, stat.st_ino)
                # File shrank → truncated; restart from BOF.
                start = 0 if stat.st_size < last_pos else last_pos
                if stat.st_size <= start:
                    state.set(path, stat.st_ino, start)
                    continue
                records, new_pos = _scan_new_records(path, start)
                for record in records:
                    if filters.passes(record):
                        emit(record)
                state.set(path, stat.st_ino, new_pos)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        return 0


def configure_logs_parser(logs: argparse.ArgumentParser) -> None:
    """Attach the ``cat-watcher logs`` flags onto an already-created sub-parser."""
    _ = logs.add_argument(
        "agent",
        nargs="?",
        choices=[*_ALL_AGENTS],
        default=None,
        help="Agent to filter by; omit to read all four LaunchAgent files merged (cli excluded).",
    )
    _ = logs.add_argument("-f", "--follow", action="store_true", help="Tail the file(s); re-open on rotation; SIGINT exits.")
    _ = logs.add_argument(
        "--since",
        type=parse_since,
        default=None,
        help="Filter to records at or after this time. Accepts shorthand (30m/1h/7d) or ISO 8601 (naive=local).",
    )
    _ = logs.add_argument(
        "--level",
        choices=[*_ANSI_BY_LEVEL.keys()],
        default=None,
        help="Drop records below this level.",
    )
    _ = logs.add_argument(
        "--camera",
        dest="camera_filter",
        default=None,
        help="Keep only records with extras.camera_name == NAME (records without that key drop).",
    )
    _ = logs.add_argument("--grep", default=None, help="Case-insensitive substring match on the msg field.")
    _ = logs.add_argument("--json", action="store_true", help="Emit raw JSONL unchanged (for jq); skips the pretty formatter.")


class LogsNamespace(argparse.Namespace):
    """Typed namespace for ``cat-watcher logs`` (matches the dests in :func:`configure_logs_parser`).

    Subclasses (notably ``_ParsedArgs`` in :mod:`cat_watcher.__main__`) inherit these fields so the
    umbrella's parser keeps a single namespace covering every sub-command's flags while preserving
    typed access for :meth:`RunArgs.from_namespace`.
    """

    agent: str | None = None
    follow: bool = False
    since: datetime | None = None
    level: str | None = None
    camera_filter: str | None = None
    grep: str | None = None
    json: bool = False


@final
@dataclass(frozen=True, slots=True)
class RunArgs:
    """Typed view over the post-argparse fields the ``logs`` sub-command needs.

    Lets :func:`run` take a small typed parameter surface so basedpyright can flow concrete types
    through the call.
    """

    agent: str | None
    follow: bool
    since: datetime | None
    level: str | None
    camera_filter: str | None
    grep: str | None
    json_mode: bool

    @classmethod
    def from_namespace(cls, args: LogsNamespace) -> RunArgs:
        """Pack the typed namespace into ``RunArgs`` (renaming ``args.json`` → ``json_mode``)."""
        return cls(
            agent=args.agent,
            follow=args.follow,
            since=args.since,
            level=args.level,
            camera_filter=args.camera_filter,
            grep=args.grep,
            json_mode=args.json,
        )


def run(args: RunArgs, *, internal_root: Path, out: IO[str] | None = None) -> int:
    """Execute the ``logs`` sub-command. Returns a process exit code."""
    sink: IO[str] = out if out is not None else sys.stdout
    selected_agents: tuple[str, ...] = (args.agent,) if args.agent else _LAUNCHAGENT_AGENTS
    files = _resolve_agent_files(internal_root, selected_agents)
    use_color = (not args.json_mode) and bool(getattr(sink, "isatty", lambda: False)())

    filters = _Filters(since=args.since, level=args.level, camera=args.camera_filter, grep=args.grep)

    def _emit(record: LogRecordDict) -> None:
        _emit_one(record, out=sink, json_mode=args.json_mode, use_color=use_color)

    if args.follow:
        return _follow_loop(files, filters=filters, emit=_emit)

    records = _collect_filtered(files, filters=filters)
    _emit_records(records, out=sink, json_mode=args.json_mode, use_color=use_color)
    return 0


__all__ = ["LogsNamespace", "RunArgs", "configure_logs_parser", "parse_since", "run"]
