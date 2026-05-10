"""Tests for cat_watcher.import_local helper functions.

End-to-end import behavior (DB writes, file copies, detector dispatch, lock cooperation) is covered
in tests/integration/test_import_local_end_to_end.py.
"""

from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003  # runtime: pytest evaluates fixture annotations during collection
from zoneinfo import ZoneInfo

import pytest

from cat_watcher.import_local import (
    _DAV_FILENAME_RE,
    _find_date_dir,
    _find_jpg_dir,
    _locate_sd_thumb,
    _scan_source,
)

# --- _DAV_FILENAME_RE ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "matches"),
    [
        ("05.50.17-05.51.20[M][0@0][0].mp4", True),  # motion trigger
        ("22.49.40-22.49.47[F][0@0][0].mp4", True),  # forced/manual trigger
        ("12.00.00-12.00.05[A][0@0][0].mp4", True),  # alarm trigger
        ("23.59.59-00.00.30[M][0@0][0].mp4", True),  # midnight wrap
        ("06.47.04-06.48.58[M][0@0][0].mp4", True),
        # negatives:
        ("readme.txt", False),
        ("05.50.17-05.51.20.mp4", False),  # missing bracket suffix
        ("05.50.17-05.51.20[M].mp4", False),  # incomplete bracket suffix
        ("05.50.17-05.51.20[m][0@0][0].mp4", False),  # lowercase trigger letter
        ("05.50.17-05.51.20[MM][0@0][0].mp4", False),  # multi-letter trigger
        ("5.50.17-5.51.20[M][0@0][0].mp4", False),  # not zero-padded
        ("05.50.17-05.51.20[M][0@0][0].idx", False),  # idx not mp4
    ],
)
def test_dav_filename_re_matches_amcrest_pattern_only(name: str, matches: bool) -> None:  # noqa: FBT001  # pytest.parametrize boolean is the canonical pattern
    """Strict shape catches operator-renamed files — only zero-padded mp4 names with the ``[M][0@0][0]`` suffix match."""
    assert (_DAV_FILENAME_RE.match(name) is not None) is matches


# --- _find_date_dir ------------------------------------------------------------------------------


def test_find_date_dir_walks_to_nearest_yyyy_mm_dd_ancestor(tmp_path: Path) -> None:
    """Walking up from a clip stops at the first YYYY-MM-DD ancestor; deeper dirs (camera, NNN, dav, HH) are ignored."""
    source = tmp_path
    deep = source / "2026-04-27-camera" / "2026-04-27" / "001" / "dav" / "05" / "clip.mp4"
    deep.parent.mkdir(parents=True)
    _ = deep.write_bytes(b"")
    assert _find_date_dir(deep, source) == source / "2026-04-27-camera" / "2026-04-27"


def test_find_date_dir_returns_none_for_orphan_at_source_root(tmp_path: Path) -> None:
    """A clip at the root of source_dir has no date ancestor; covers the operator stray-orphan case."""
    source = tmp_path
    orphan = source / "06.47.04-06.48.58[M][0@0][0].mp4"
    _ = orphan.write_bytes(b"")
    assert _find_date_dir(orphan, source) is None


def test_find_date_dir_does_not_walk_above_source_dir(tmp_path: Path) -> None:
    """A YYYY-MM-DD dir above source_dir would be an operator surprise; never consulted."""
    above = tmp_path / "2026-04-27"
    source = above / "snapshot"
    clip = source / "001" / "dav" / "05" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    _ = clip.write_bytes(b"")
    assert _find_date_dir(clip, source) is None


# --- _find_jpg_dir -------------------------------------------------------------------------------


def test_find_jpg_dir_derives_parallel_jpg_path_from_dav_ancestor(tmp_path: Path) -> None:
    """For ``.../<NNN>/dav/<HH>/<file>.mp4`` the jpg dir is ``.../<NNN>/jpg/<HH>/<MM>/``."""
    mp4 = tmp_path / "2026-04-27" / "001" / "dav" / "05" / "05.50.17-05.51.20[M][0@0][0].mp4"
    mp4.parent.mkdir(parents=True)
    _ = mp4.write_bytes(b"")
    expected = tmp_path / "2026-04-27" / "001" / "jpg" / "05" / "50"
    assert _find_jpg_dir(mp4, start_hh=5, start_mm=50) == expected


def test_find_jpg_dir_returns_none_when_no_dav_ancestor(tmp_path: Path) -> None:
    """Without a ``dav`` ancestor the parallel ``jpg`` path can't be derived; caller falls back."""
    mp4 = tmp_path / "loose.mp4"
    _ = mp4.write_bytes(b"")
    assert _find_jpg_dir(mp4, start_hh=0, start_mm=0) is None


# --- _locate_sd_thumb ----------------------------------------------------------------------------


def test_locate_sd_thumb_picks_earliest_jpg_within_clip_window(tmp_path: Path) -> None:
    """The earliest jpg in the clip window is the best still — closest to the trigger frame."""
    for sec in (24, 25, 26):
        _ = (tmp_path / f"{sec:02d}[M][0@0][0].jpg").write_bytes(b"")
    chosen = _locate_sd_thumb(tmp_path, start_sec=17, duration_sec=63)
    assert chosen == tmp_path / "24[M][0@0][0].jpg"


def test_locate_sd_thumb_returns_none_when_no_candidates_in_window(tmp_path: Path) -> None:
    """No jpg within ``[start_sec, start_sec + duration_sec]`` returns None — never the closest miss."""
    _ = (tmp_path / "10[M][0@0][0].jpg").write_bytes(b"")  # before start_sec
    assert _locate_sd_thumb(tmp_path, start_sec=20, duration_sec=5) is None


def test_locate_sd_thumb_returns_none_when_directory_absent(tmp_path: Path) -> None:
    """Missing jpg directory is benign; caller falls back to ffmpeg extraction."""
    assert _locate_sd_thumb(tmp_path / "missing", start_sec=0, duration_sec=10) is None


def test_locate_sd_thumb_skips_non_amcrest_jpgs(tmp_path: Path) -> None:
    """Stray jpgs (operator screenshots, etc.) don't match the Amcrest naming pattern."""
    _ = (tmp_path / "stray.jpg").write_bytes(b"")  # no [M][0@0][0] suffix
    _ = (tmp_path / "20[M][0@0][0].jpg").write_bytes(b"")
    chosen = _locate_sd_thumb(tmp_path, start_sec=15, duration_sec=10)
    assert chosen == tmp_path / "20[M][0@0][0].jpg"


# --- _scan_source --------------------------------------------------------------------------------


_TZ = ZoneInfo("UTC")


def _make_sd_clip(root: Path, date_str: str, hour: int, minute: int, second: int) -> Path:
    """Create an empty mp4 at the canonical SD-card path; returns the file path."""
    fname = f"{hour:02d}.{minute:02d}.{second:02d}-{hour:02d}.{minute:02d}.{second + 1:02d}[M][0@0][0].mp4"
    path = root / date_str / "001" / "dav" / f"{hour:02d}" / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(b"")
    return path


def test_scan_source_returns_matched_clips_and_skipped_count(tmp_path: Path) -> None:
    """Walking the SD tree returns matched DAV clips plus a skipped count — operators see both numbers."""
    _ = _make_sd_clip(tmp_path, "2026-04-27", 5, 50, 17)
    _ = _make_sd_clip(tmp_path, "2026-04-27", 19, 1, 35)
    # non-matching file (operator stray):
    stray = tmp_path / "2026-04-27" / "001" / "dav" / "05" / "readme.txt.mp4"
    _ = stray.write_text("not a clip")
    # orphan with no date ancestor:
    orphan = tmp_path / "06.47.04-06.48.58[M][0@0][0].mp4"
    _ = orphan.write_bytes(b"")

    matched, skipped = _scan_source(tmp_path, camera_tz=_TZ)

    assert len(matched) == 2
    assert skipped == 2
    starts = sorted(clip.start_ts for clip in matched)
    assert starts[0] == datetime(2026, 4, 27, 5, 50, 17, tzinfo=UTC)
    assert starts[1] == datetime(2026, 4, 27, 19, 1, 35, tzinfo=UTC)


def test_scan_source_handles_midnight_wrap(tmp_path: Path) -> None:
    """A clip ending past midnight spans into the next local day; end_ts must roll forward."""
    fname = "23.59.59-00.00.30[M][0@0][0].mp4"
    path = tmp_path / "2026-04-27" / "001" / "dav" / "23" / fname
    path.parent.mkdir(parents=True)
    _ = path.write_bytes(b"")

    matched, _ = _scan_source(tmp_path, camera_tz=_TZ)
    assert len(matched) == 1
    clip = matched[0]
    assert clip.start_ts == datetime(2026, 4, 27, 23, 59, 59, tzinfo=UTC)
    assert clip.end_ts == datetime(2026, 4, 28, 0, 0, 30, tzinfo=UTC)


def test_scan_source_camera_tz_offsets_into_utc(tmp_path: Path) -> None:
    """Filename timestamps are in camera-local time; ``camera_tz`` shifts them to UTC for storage."""
    _ = _make_sd_clip(tmp_path, "2026-04-27", 5, 0, 0)
    matched, _ = _scan_source(tmp_path, camera_tz=ZoneInfo("America/New_York"))
    # Apr 27 2026 is EDT (UTC-4), so 05:00 local -> 09:00 UTC.
    assert matched[0].start_ts == datetime(2026, 4, 27, 9, 0, 0, tzinfo=UTC)
