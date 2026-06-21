"""Unit tests for the pure ``build_tag_summary`` composition in :mod:`cat_watcher.labels`.

``build_tag_summary`` is a pure function (no DB), so these exercise its branches directly — faster
and more precise than the HTTP-level coverage in ``test_web_clip_detail``. The per-cat frame-count
ordering it consumes is produced by ``query_cat_frame_counts`` (DB-backed, covered elsewhere); here
the counts are passed in pre-ordered, as the real caller supplies them.
"""

from datetime import UTC, datetime

from cat_watcher.db import Subject
from cat_watcher.labels import build_tag_summary

# Matches the row shape ``query_cat_frame_counts`` returns: (slug, display_name, archived_at, count).
# Annotating literals with this keeps the invariant ``list`` element type from narrowing to
# ``None`` / ``datetime`` where a given test omits the other.
type _CatCount = tuple[str, str, datetime | None, int]

_ARCHIVED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _event(slug: str, order: int = 1) -> Subject:
    """Build an unsaved event ``Subject`` (``build_tag_summary`` only reads ``.slug``)."""
    return Subject(slug=slug, display_name=slug.capitalize(), kind="event", display_order=order)


def test_empty_returns_dash() -> None:
    """No cats configured and no events tagged → the em-dash placeholder."""
    assert build_tag_summary([], set(), []) == "—"


def test_active_cats_listed_with_explicit_zero_counts() -> None:
    """Every active cat is listed with its count, including zeros (so rejected cats are visible)."""
    counts: list[_CatCount] = [("marcel", "Marcel", None, 3), ("rufus", "Rufus", None, 0)]
    assert build_tag_summary(counts, set(), []) == "marcel: 3, rufus: 0"


def test_events_only_are_bare_slugs() -> None:
    """With no cats, tagged events render as bare comma-joined slugs."""
    events = [_event("cleaning", 1), _event("person", 2)]
    assert build_tag_summary([], {"cleaning", "person"}, events) == "cleaning, person"


def test_untagged_events_are_omitted() -> None:
    """Only events with at least one tagged frame on this clip appear."""
    events = [_event("cleaning", 1), _event("person", 2)]
    assert build_tag_summary([], {"cleaning"}, events) == "cleaning"


def test_cats_and_events_separated_by_semicolon() -> None:
    """The cats group and events group are joined by '; '."""
    counts: list[_CatCount] = [("marcel", "Marcel", None, 3), ("rufus", "Rufus", None, 0)]
    events = [_event("cleaning", 1), _event("person", 2)]
    assert build_tag_summary(counts, {"cleaning", "person"}, events) == "marcel: 3, rufus: 0; cleaning, person"


def test_archived_cat_with_frames_shows_display_name_marker_and_count() -> None:
    """An archived cat with tagged frames renders as 'Display (archived): <count>' after active cats."""
    counts: list[_CatCount] = [("marcel", "Marcel", None, 2), ("fluffy", "Fluffy", _ARCHIVED_AT, 1)]
    assert build_tag_summary(counts, set(), []) == "marcel: 2, Fluffy (archived): 1"


def test_archived_cat_with_zero_frames_is_suppressed() -> None:
    """An archived cat with no frames on this clip is omitted entirely (no noise for the operator)."""
    counts: list[_CatCount] = [("marcel", "Marcel", None, 2), ("fluffy", "Fluffy", _ARCHIVED_AT, 0)]
    assert build_tag_summary(counts, set(), []) == "marcel: 2"


def test_only_archived_zero_frame_cat_and_no_events_is_dash() -> None:
    """If the only cat is archived-with-zero-frames (suppressed) and no events tag, result is '—'."""
    counts: list[_CatCount] = [("fluffy", "Fluffy", _ARCHIVED_AT, 0)]
    assert build_tag_summary(counts, set(), []) == "—"
