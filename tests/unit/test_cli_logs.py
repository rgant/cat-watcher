"""Tests for :mod:`cat_watcher.logs_viewer` (the ``cat-watcher logs`` sub-command)."""

import argparse
import io
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

import pytest
from tz_helpers import pinned_tz

from cat_watcher import logs_viewer
from cat_watcher.logs_viewer import RunArgs

if TYPE_CHECKING:
    from pathlib import Path


def _seed(path: Path, records: list[dict[str, object]]) -> None:
    """Write ``records`` as JSONL into ``path`` (creating parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            _ = fh.write(json.dumps(record))
            _ = fh.write("\n")


def _record(  # noqa: PLR0913 — test-fixture builder; bundling args at the call-site is noisier.
    *,
    ts: str,
    level: str = "INFO",
    msg: str = "x",
    agent: str = "poller",
    extras: dict[str, object] | None = None,
    traceback: str | None = None,
) -> dict[str, object]:
    """Build a JSONL-shaped record dict with sensible defaults for the test surface."""
    rec: dict[str, object] = {
        "ts": ts,
        "level": level,
        "logger": "cat_watcher.test",
        "agent": agent,
        "pid": 1234,
        "msg": msg,
    }
    if extras is not None:
        rec["extras"] = extras
    if traceback is not None:
        rec["traceback"] = traceback
    return rec


def _build_args(  # noqa: PLR0913 — test-fixture builder; bundling args at the call-site is noisier.
    *,
    agent: str | None = None,
    follow: bool = False,
    since: datetime | None = None,
    level: str | None = None,
    camera_filter: str | None = None,
    grep: str | None = None,
    json_mode: bool = False,
) -> RunArgs:
    """Construct a ``RunArgs`` with defaults matching the ``logs`` sub-parser."""
    return RunArgs(
        agent=agent,
        follow=follow,
        since=since,
        level=level,
        camera_filter=camera_filter,
        grep=grep,
        json_mode=json_mode,
    )


def _run(internal_root: Path, args: RunArgs) -> str:
    """Run the viewer with ``args``, capturing stdout to a StringIO; assert exit 0."""
    sink = io.StringIO()
    rc = logs_viewer.run(args, internal_root=internal_root, out=sink)
    assert rc == 0
    return sink.getvalue()


# --- parse_since ---------------------------------------------------------------------------------


def test_parse_since_duration_shorthand() -> None:
    """``parse_since('1h')`` returns roughly one hour ago in UTC."""
    before = datetime.now(UTC)
    parsed = logs_viewer.parse_since("1h")
    after = datetime.now(UTC)
    expected_lo = before - timedelta(hours=1, seconds=1)
    expected_hi = after - timedelta(hours=1) + timedelta(seconds=1)
    assert expected_lo <= parsed <= expected_hi


def test_parse_since_iso_with_offset() -> None:
    """ISO 8601 with explicit offset is honored verbatim and converted to UTC."""
    parsed = logs_viewer.parse_since("2026-05-05T12:00:00+00:00")
    assert parsed == datetime(2026, 5, 5, 12, 0, 0, tzinfo=UTC)


def test_parse_since_naive_iso_is_os_local() -> None:
    """Naive ISO input is read in OS-local time (matching the poller's `_parse_iso_datetime`)."""
    with pinned_tz("America/New_York"):
        parsed = logs_viewer.parse_since("2026-05-04T00:00:00")
        assert parsed == datetime(2026, 5, 4, 4, 0, 0, tzinfo=UTC)


# --- positional agent filter --------------------------------------------------------------------


def test_no_arg_default_excludes_cli(tmp_path: Path) -> None:
    """Without a positional agent arg, only the four LaunchAgent files are read; ``cli`` is excluded."""
    _seed(tmp_path / "logs" / "poller.jsonl", [_record(ts="2026-05-05T10:00:00.000000+00:00", msg="poller-msg", agent="poller")])
    _seed(tmp_path / "logs" / "alerts.jsonl", [_record(ts="2026-05-05T10:01:00.000000+00:00", msg="alerts-msg", agent="alerts")])
    _seed(tmp_path / "logs" / "cli.jsonl", [_record(ts="2026-05-05T10:02:00.000000+00:00", msg="cli-msg", agent="cli")])

    out = _run(tmp_path, _build_args(json_mode=True))
    assert "poller-msg" in out
    assert "alerts-msg" in out
    assert "cli-msg" not in out


def test_positional_agent_filters_to_one_file(tmp_path: Path) -> None:
    """Passing a positional agent name (e.g. ``poller``) restricts output to that single file."""
    _seed(tmp_path / "logs" / "poller.jsonl", [_record(ts="2026-05-05T10:00:00.000000+00:00", msg="poller-msg")])
    _seed(tmp_path / "logs" / "alerts.jsonl", [_record(ts="2026-05-05T10:01:00.000000+00:00", msg="alerts-msg", agent="alerts")])

    out = _run(tmp_path, _build_args(agent="poller", json_mode=True))
    assert "poller-msg" in out
    assert "alerts-msg" not in out


def test_positional_cli_opts_in(tmp_path: Path) -> None:
    """Passing ``cli`` explicitly opts the umbrella CLI's log file into the view."""
    _seed(tmp_path / "logs" / "cli.jsonl", [_record(ts="2026-05-05T10:00:00.000000+00:00", msg="cli-only", agent="cli")])
    _seed(tmp_path / "logs" / "poller.jsonl", [_record(ts="2026-05-05T10:01:00.000000+00:00", msg="poller-only")])

    out = _run(tmp_path, _build_args(agent="cli", json_mode=True))
    assert "cli-only" in out
    assert "poller-only" not in out


# --- --since -------------------------------------------------------------------------------------


def test_since_duration_filters_old_records(tmp_path: Path) -> None:
    """``--since 1h`` drops records older than that duration."""
    now = datetime.now(UTC)
    old = (now - timedelta(hours=2)).isoformat(timespec="microseconds")
    new = (now - timedelta(minutes=30)).isoformat(timespec="microseconds")
    _seed(tmp_path / "logs" / "poller.jsonl", [_record(ts=old, msg="old-msg"), _record(ts=new, msg="new-msg")])

    out = _run(tmp_path, _build_args(since=logs_viewer.parse_since("1h"), json_mode=True))
    assert "new-msg" in out
    assert "old-msg" not in out


def test_since_iso_filters_records_before(tmp_path: Path) -> None:
    """``--since`` with an explicit timestamp keeps records at-or-after that boundary."""
    _seed(
        tmp_path / "logs" / "poller.jsonl",
        [
            _record(ts="2026-05-05T11:59:00.000000+00:00", msg="before"),
            _record(ts="2026-05-05T12:00:00.000000+00:00", msg="boundary"),
            _record(ts="2026-05-05T12:01:00.000000+00:00", msg="after"),
        ],
    )
    cutoff = datetime(2026, 5, 5, 12, 0, 0, tzinfo=UTC)
    out = _run(tmp_path, _build_args(since=cutoff, json_mode=True))
    assert "before" not in out
    assert "boundary" in out
    assert "after" in out


# --- --level -------------------------------------------------------------------------------------


def test_level_drops_records_below_threshold(tmp_path: Path) -> None:
    """``--level WARNING`` keeps WARNING/ERROR/CRITICAL and drops DEBUG/INFO."""
    _seed(
        tmp_path / "logs" / "poller.jsonl",
        [
            _record(ts="2026-05-05T10:00:00.000000+00:00", level="DEBUG", msg="debug-msg"),
            _record(ts="2026-05-05T10:01:00.000000+00:00", level="INFO", msg="info-msg"),
            _record(ts="2026-05-05T10:02:00.000000+00:00", level="WARNING", msg="warn-msg"),
            _record(ts="2026-05-05T10:03:00.000000+00:00", level="ERROR", msg="error-msg"),
        ],
    )
    out = _run(tmp_path, _build_args(level="WARNING", json_mode=True))
    assert "warn-msg" in out
    assert "error-msg" in out
    assert "info-msg" not in out
    assert "debug-msg" not in out


# --- --camera ------------------------------------------------------------------------------------


def test_camera_filter_keeps_matches_drops_missing(tmp_path: Path) -> None:
    """``--camera office`` keeps only records with ``extras.camera_name == 'office'``; missing-key records drop."""
    _seed(
        tmp_path / "logs" / "poller.jsonl",
        [
            _record(ts="2026-05-05T10:00:00.000000+00:00", msg="match", extras={"camera_name": "office"}),
            _record(ts="2026-05-05T10:01:00.000000+00:00", msg="other-cam", extras={"camera_name": "pantry"}),
            _record(ts="2026-05-05T10:02:00.000000+00:00", msg="no-extras"),
        ],
    )
    out = _run(tmp_path, _build_args(camera_filter="office", json_mode=True))
    assert "match" in out
    assert "other-cam" not in out
    assert "no-extras" not in out


# --- --grep --------------------------------------------------------------------------------------


def test_grep_is_case_insensitive(tmp_path: Path) -> None:
    """``--grep ingested`` matches ``Ingested clip`` and ``successfully ingested`` (case-insensitive substring)."""
    _seed(
        tmp_path / "logs" / "poller.jsonl",
        [
            _record(ts="2026-05-05T10:00:00.000000+00:00", msg="Ingested clip"),
            _record(ts="2026-05-05T10:01:00.000000+00:00", msg="successfully ingested"),
            _record(ts="2026-05-05T10:02:00.000000+00:00", msg="something else"),
        ],
    )
    out = _run(tmp_path, _build_args(grep="ingested", json_mode=True))
    assert "Ingested clip" in out
    assert "successfully ingested" in out
    assert "something else" not in out


# --- output formats ------------------------------------------------------------------------------


def test_pretty_format_no_tty_has_no_ansi_escapes(tmp_path: Path) -> None:
    """A non-TTY sink (StringIO) gets uncolored pretty output — no ESC[ sequences."""
    _seed(tmp_path / "logs" / "poller.jsonl", [_record(ts="2026-05-05T10:00:00.000000+00:00", level="ERROR", msg="boom")])
    out = _run(tmp_path, _build_args())
    assert "\x1b[" not in out
    assert "ERROR" in out
    assert "boom" in out


def test_json_mode_emits_byte_identical_records_in_chronological_order(tmp_path: Path) -> None:
    """``--json`` emits one record per line, ordered by ts across the merged files."""
    rec_a = _record(ts="2026-05-05T10:00:00.000000+00:00", msg="a")
    rec_b = _record(ts="2026-05-05T10:01:00.000000+00:00", msg="b")
    rec_c = _record(ts="2026-05-05T10:02:00.000000+00:00", msg="c", agent="alerts")
    _seed(tmp_path / "logs" / "poller.jsonl", [rec_a, rec_b])
    _seed(tmp_path / "logs" / "alerts.jsonl", [rec_c])

    out = _run(tmp_path, _build_args(json_mode=True))
    lines = [line for line in out.splitlines() if line]
    assert len(lines) == 3
    parsed = [cast("dict[str, object]", json.loads(line)) for line in lines]
    assert [r["msg"] for r in parsed] == ["a", "b", "c"]


def test_chronological_merge_interleaves_two_files(tmp_path: Path) -> None:
    """Records from two files appear interleaved by ts in the merged output."""
    _seed(
        tmp_path / "logs" / "poller.jsonl",
        [
            _record(ts="2026-05-05T10:00:00.000000+00:00", msg="poll-1"),
            _record(ts="2026-05-05T10:02:00.000000+00:00", msg="poll-2"),
        ],
    )
    _seed(
        tmp_path / "logs" / "alerts.jsonl",
        [
            _record(ts="2026-05-05T10:01:00.000000+00:00", msg="alert-1", agent="alerts"),
            _record(ts="2026-05-05T10:03:00.000000+00:00", msg="alert-2", agent="alerts"),
        ],
    )
    out = _run(tmp_path, _build_args(json_mode=True))
    msgs = [cast("dict[str, object]", json.loads(line))["msg"] for line in out.splitlines() if line]
    assert msgs == ["poll-1", "alert-1", "poll-2", "alert-2"]


def test_exception_records_render_with_traceback_marker(tmp_path: Path) -> None:
    """Records with a ``traceback`` field render the ``[+traceback]`` marker in pretty output."""
    _seed(
        tmp_path / "logs" / "poller.jsonl",
        [
            _record(
                ts="2026-05-05T10:00:00.000000+00:00",
                level="ERROR",
                msg="boom",
                traceback="Traceback (most recent call last):\n  File ...\nValueError: bad",
            ),
        ],
    )
    out = _run(tmp_path, _build_args())
    assert "[+traceback]" in out


def test_help_text_present_for_every_documented_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """``logs --help`` includes a help line for every documented flag."""
    parser = argparse.ArgumentParser()
    logs_viewer.configure_logs_parser(parser)
    with pytest.raises(SystemExit):
        _ = parser.parse_args(["--help"])
    captured = capsys.readouterr()
    for flag in ("--follow", "--since", "--level", "--camera", "--grep", "--json"):
        assert flag in captured.out


def test_malformed_jsonl_lines_are_skipped(tmp_path: Path) -> None:
    """A torn / partially-written line is skipped silently; surrounding good records still surface."""
    log_path = tmp_path / "logs" / "poller.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ = log_path.write_text(
        "\n".join(
            [
                json.dumps(_record(ts="2026-05-05T10:00:00.000000+00:00", msg="good")),
                "{not valid json",
                json.dumps(_record(ts="2026-05-05T10:01:00.000000+00:00", msg="also good")),
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    out = _run(tmp_path, _build_args(json_mode=True))
    msgs = [cast("dict[str, object]", json.loads(line))["msg"] for line in out.splitlines() if line]
    assert msgs == ["good", "also good"]
