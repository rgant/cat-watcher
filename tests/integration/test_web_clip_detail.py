"""Integration tests for the /clips/{id} detail route view-model rebuild.

Covers: 200 for known clip, subjects_by_kind in rendered HTML, single-query invariant for
clip_frame_subjects, absence of dropped <dl> rows, 404 for unknown IDs, and the per-frame tag table
with HTMX wiring.
"""

import re
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING, NamedTuple, cast

import db_helpers
import pytest
from db_helpers import AUTH_HEADER, DEFAULT_START_TS, make_clip_frame, seed_camera_and_clip_with_files, stamp_reviewed_at
from sqlalchemy import event as sa_event

from cat_watcher.db import Camera, Clip, ClipFrame, ClipFrameSubject, Subject, create_engine, get_session
from cat_watcher.labels import query_cat_frame_counts
from cat_watcher.web import clips_routes

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


def _seed_camera_and_clip(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    storage_root: Path,
    start_ts: datetime = DEFAULT_START_TS,
    has_cat: bool = True,
) -> tuple[int, int]:
    return seed_camera_and_clip_with_files(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=start_ts,
        has_cat=has_cat,
    )


def _detail_engine_for(internal_root: Path) -> Engine:
    return create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")


def _seed_two_subjects(engine: Engine) -> tuple[int, int]:
    """Insert one cat + one event subject; return ``(cat_id, event_id)``."""
    with get_session(engine) as session:
        cat_subj = Subject(slug="nora", display_name="Nora", kind="cat", display_order=1)
        evt_subj = Subject(slug="lunchtime", display_name="Lunchtime", kind="event", display_order=1)
        session.add(cat_subj)
        session.add(evt_subj)
        session.flush()
        return cat_subj.id, evt_subj.id


def _tag_n_frames(engine: Engine, clip_id: int, *, cat_id: int, event_id: int, count: int) -> None:
    """Add ``count`` frames to ``clip_id``, each tagged with both ``cat_id`` and ``event_id``."""
    with get_session(engine) as session:
        for i in range(count):
            frame = ClipFrame(clip_id=clip_id, ordinal=i, t_offset_seconds=float(i), score=0.9, thumb_path=f"thumbs/f{i}.jpg")
            session.add(frame)
            session.flush()
            session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=cat_id))
            session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=event_id))


def _count_cfs_queries_for_render(client: TestClient, clip_id: int) -> int:
    """GET ``/clips/{clip_id}`` and return how many executed statements touched ``clip_frame_subjects``.

    Registers a SQLAlchemy ``after_cursor_execute`` hook on the app's engine for the duration of the
    request. The app engine is read from ``app.state`` (a free-form attribute bag typed ``Any``; the
    cast narrows it back to ``Engine``).
    """
    app_engine = cast("Engine", cast("FastAPI", client.app).state.engine)
    count: list[int] = [0]

    def _listener(
        _conn: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        if "clip_frame_subjects" in statement:
            count[0] += 1

    sa_event.listen(app_engine, "after_cursor_execute", _listener)
    try:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)
        assert response.status_code == 200
    finally:
        sa_event.remove(app_engine, "after_cursor_execute", _listener)
    return count[0]


def _seed_small_and_large_clips(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
    *,
    internal_root: Path,
    storage_root: Path,
) -> tuple[int, int]:
    """Seed a 2-frame clip and a 6-frame clip sharing one cat + one event subject.

    Returns ``(small_clip_id, large_clip_id)``. Both clips are tagged on every frame so the
    membership and per-cat-count queries run against differing row counts.
    """
    cam_id, small_clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        cat_id, event_id = _seed_two_subjects(engine)
        _tag_n_frames(engine, small_clip_id, cat_id=cat_id, event_id=event_id, count=2)
        large_clip_id = seed_clip(engine, camera_id=cam_id, start_ts=datetime(2026, 6, 1, 9, 0, 0, tzinfo=UTC), has_cat=True)
        _tag_n_frames(engine, large_clip_id, cat_id=cat_id, event_id=event_id, count=6)
    finally:
        engine.dispose()
    return small_clip_id, large_clip_id


def test_clip_detail_returns_200_for_known_clip(
    seeded_detail_clip: tuple[Config, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """``GET /clips/{id}`` returns 200 for a clip that exists in the database."""
    config, clip_id = seeded_detail_clip

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200


def test_clip_detail_returns_404_for_unknown_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """``GET /clips/9999`` returns 404 when no clip row exists."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with alembic_web_test_client(config) as client:
        response = client.get("/clips/9999", headers=AUTH_HEADER)

    assert response.status_code == 404


def test_clip_detail_renders_active_cat_subject_display_name(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Active cat subjects appear in the rendered HTML; archived subjects do not."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            session.add(Subject(slug="marcel", display_name="Marcel The Cat", kind="cat", display_order=1))
            session.add(
                Subject(slug="archived-cat", display_name="Retired Cat", kind="cat", display_order=99, archived_at=datetime.now(UTC)),
            )
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "Marcel The Cat" in response.text
    assert "Retired Cat" not in response.text


def test_clip_detail_renders_active_event_subject_display_name(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Active event subjects appear in the rendered HTML."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            session.add(Subject(slug="feeding", display_name="Feeding Time", kind="event", display_order=1))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "Feeding Time" in response.text


def test_clip_detail_query_count_does_not_grow_with_frame_count(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """Queries against ``clip_frame_subjects`` are constant in frame count — the binding no-N+1 guarantee.

    Renders two clips sharing the same subjects — one with 2 tagged frames, one with 6 — and asserts
    the per-render query count is *identical*. Asserting equality (not a magic number) tests the
    actual invariant: a regression that N+1-loads per frame would make the 6-frame count exceed the
    2-frame count. Uses a SQLAlchemy ``after_cursor_execute`` hook on the app's engine; the listener
    is registered inside the client context so the lifespan connection does not inflate the count.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    small_clip_id, large_clip_id = _seed_small_and_large_clips(
        db_session_factory,
        seed_clip,
        internal_root=internal_root,
        storage_root=storage_root,
    )

    with alembic_web_test_client(config) as client:
        small_count = _count_cfs_queries_for_render(client, small_clip_id)
        large_count = _count_cfs_queries_for_render(client, large_clip_id)

    assert small_count == large_count, f"query count scales with frame count: {small_count} vs {large_count}"
    # And the constant itself stays small — a clip-scoped handful, not per-frame.
    assert small_count <= 3


def test_clip_detail_dropped_manual_rows_absent(
    seeded_detail_clip: tuple[Config, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The three stale <dl> rows (manual_has_cat, manual_label_at, manual_label_notes) must not appear."""
    config, clip_id = seeded_detail_clip

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "manual_has_cat" not in response.text
    assert "manual_label_at" not in response.text
    assert "manual_label_notes" not in response.text


# ---------------------------------------------------------------------------
# Task 12: per-frame tag table rendering + HTMX wiring
# ---------------------------------------------------------------------------


def _seed_five_frames(engine: Engine, clip_id: int) -> list[int]:
    """Insert 5 ClipFrame rows for ``clip_id`` and return their ids in ordinal order."""
    with get_session(engine) as session:
        frame_ids: list[int] = []
        for i in range(5):
            frame = ClipFrame(
                clip_id=clip_id,
                ordinal=i,
                t_offset_seconds=float(i * 5),
                score=0.9,
                thumb_path=f"thumbs/frame_{clip_id}_{i}.jpg",
            )
            session.add(frame)
            session.flush()
            frame_ids.append(frame.id)
    return frame_ids


def test_tag_table_renders_five_frame_rows(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The per-frame tag table renders one row per frame in the clip."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        _ = _seed_five_frames(engine, clip_id)
        _ = db_helpers.seed_cat_subject(engine)
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    # The first row carries 'tag-row active-frame'; remaining rows carry 'tag-row"'.
    # Count both variants: exact close-quote and active-frame suffix.
    tag_row_count = response.text.count('class="tag-row"') + response.text.count('class="tag-row active-frame"')
    assert tag_row_count == 5


def test_tag_table_renders_cat_and_event_button_groups(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Each frame row renders a cat button group and an event button group."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        _ = _seed_five_frames(engine, clip_id)
        with get_session(engine) as session:
            session.add(Subject(slug="nora", display_name="Nora", kind="cat", display_order=1))
            session.add(Subject(slug="feeding", display_name="Feeding Time", kind="event", display_order=1))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    # One cat group per row x 5 rows
    assert response.text.count('class="tag-btn-group tag-btn-group-cat"') == 5
    # One event group per row x 5 rows
    assert response.text.count('class="tag-btn-group tag-btn-group-event"') == 5
    # Each row has one cat button (nora) + one event button (feeding)
    assert response.text.count('data-subject-slug="nora"') >= 5
    assert response.text.count('data-subject-slug="feeding"') >= 5


def test_tag_table_button_glyph_and_title(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Button text is the full display_name; title is the full display_name with any description."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        _ = _seed_five_frames(engine, clip_id)
        with get_session(engine) as session:
            session.add(Subject(slug="marcel", display_name="Marcel", kind="cat", display_order=1))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert 'title="Marcel"' in response.text
    # Full display_name is rendered as the button's text content (not a single-letter glyph).
    assert ">Marcel</button>" in response.text


def test_tag_table_button_title_includes_description(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Button title includes description in parentheses when the Subject has a description."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        _ = _seed_five_frames(engine, clip_id)
        with get_session(engine) as session:
            session.add(
                Subject(slug="rex", display_name="Rex", kind="cat", display_order=1, description="big orange tabby"),
            )
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert 'title="Rex (big orange tabby)"' in response.text


def test_tag_table_pressed_state_matches_frame_memberships(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A tagged frame's button has aria-pressed=true and the tag-btn-on class; untagged is false/off."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        frame_ids = _seed_five_frames(engine, clip_id)
        subject_id = db_helpers.seed_cat_subject(engine, slug="nora", display_name="Nora")
        # Tag only the first frame
        with get_session(engine) as session:
            session.add(ClipFrameSubject(clip_frame_id=frame_ids[0], subject_id=subject_id))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    # The tag table should have at least one on-state button and at least one off-state button
    assert "tag-btn-on" in response.text
    assert "tag-btn-off" in response.text
    # On-state uses hx-delete; off-state uses hx-put
    assert "hx-delete=" in response.text
    assert "hx-put=" in response.text


def test_tag_table_hx_urls_target_membership_endpoint(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """HTMX attributes target /clips/{id}/frames/{fid}/subjects/{sid}."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        _ = _seed_five_frames(engine, clip_id)
        _ = db_helpers.seed_cat_subject(engine)
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert f"/clips/{clip_id}/frames/" in response.text
    assert "/subjects/" in response.text


def test_tag_table_empty_event_kind_omits_event_group(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """When there are no event subjects, no event button group renders in any row."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        _ = _seed_five_frames(engine, clip_id)
        _ = db_helpers.seed_cat_subject(engine)
        # No event subjects seeded
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert 'class="tag-btn-group tag-btn-group-event"' not in response.text


# ---------------------------------------------------------------------------
# Task 13: Mark Reviewed control + dl rows + tag_summary
# ---------------------------------------------------------------------------


def _seed_reviewed_clip(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    storage_root: Path,
    reviewed_at: datetime | None = None,
) -> tuple[int, int]:
    """Seed a clip with optional ``reviewed_at`` and return ``(cam_id, clip_id)``."""
    cam_id, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
    )
    if reviewed_at is not None:
        engine = _detail_engine_for(internal_root)
        try:
            with get_session(engine) as session:
                clip = session.get(Clip, clip_id)
                assert clip is not None
                clip.reviewed_at = reviewed_at
        finally:
            engine.dispose()
    return cam_id, clip_id


def test_mark_reviewed_button_present_when_unreviewed(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """When ``reviewed_at IS NULL``, the page renders a 'Mark reviewed' button with hx-post."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "Mark reviewed" in response.text
    assert f'hx-post="/clips/{clip_id}/reviewed"' in response.text


def test_reopen_button_present_when_already_reviewed(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """When ``reviewed_at IS NOT NULL``, the page renders a 'Re-open for review' button with hx-delete."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # 02:00 UTC on 2026-07-02 is 2026-07-01 locally; the old 14:30 UTC seed read the same in both.
    reviewed_ts = datetime(2026, 7, 2, 2, 0, 0, tzinfo=UTC)
    _, clip_id = _seed_reviewed_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        reviewed_at=reviewed_ts,
    )

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "Re-open for review" in response.text
    assert f'hx-delete="/clips/{clip_id}/reviewed"' in response.text
    assert '<span class="review-badge">Reviewed 2026-07-01</span>' in response.text
    assert "Reviewed 2026-07-02" not in response.text


def test_dl_rows_reviewed_at_null(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The three new <dl> rows (reviewed_at, has_manual_cat, tag_summary) are present when unreviewed."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "<dt>reviewed_at</dt>" in response.text
    assert "<dt>has_manual_cat</dt>" in response.text
    assert "<dt>tag_summary</dt>" in response.text


def test_dl_rows_reviewed_at_non_null(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The reviewed_at <dl> row shows the local stamp as text, keeping UTC ISO in the attribute."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    reviewed_ts = datetime(2026, 7, 2, 2, 0, 0, tzinfo=UTC)
    _, clip_id = _seed_reviewed_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        reviewed_at=reviewed_ts,
    )

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert 'datetime="2026-07-02T02:00:00+00:00"' in response.text
    assert ">2026-07-01 22:00:00 EDT</time>" in response.text
    # The old assertion was satisfied by the attribute alone regardless of the visible text.
    assert ">2026-07-02T02:00:00+00:00</time>" not in response.text


def test_tag_summary_cats_with_frame_counts_no_events(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """tag_summary lists cats with counts (including zeros) and omits events group when none tagged."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        marcel_id = db_helpers.seed_cat_subject(engine, slug="marcel", display_name="Marcel", display_order=1)
        _ = db_helpers.seed_cat_subject(engine, slug="rufus", display_name="Rufus", display_order=2)
        with get_session(engine) as session:
            for i in range(3):
                frame = ClipFrame(
                    clip_id=clip_id,
                    ordinal=i,
                    t_offset_seconds=float(i * 5),
                    score=0.9,
                    thumb_path=f"thumbs/frame_{clip_id}_{i}.jpg",
                )
                session.add(frame)
                session.flush()
                session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=marcel_id))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "marcel: 3, rufus: 0" in response.text


def test_query_cat_frame_counts_isolates_by_clip(
    storage_dirs: tuple[Path, Path],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Counts are scoped to the requested clip — memberships on other clips must not leak in.

    Regression: a count over ``clip_frame_subjects.clip_frame_id`` (always non-null) ignored the
    ``clip_frames.clip_id`` join filter, so every clip reported the global per-cat total.
    """
    internal_root, _ = storage_dirs
    cam_id, clip_a = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_dirs[1])
    engine = _detail_engine_for(internal_root)
    try:
        marcel_id = db_helpers.seed_cat_subject(engine, slug="marcel", display_name="Marcel", display_order=1)
        _ = db_helpers.seed_cat_subject(engine, slug="rufus", display_name="Rufus", display_order=2)
        with get_session(engine) as session:
            clip_b = db_helpers.build_test_clip(cam_id, start_ts=datetime(2026, 5, 2, 8, 0, 0, tzinfo=UTC))
            session.add(clip_b)
            session.flush()
            clip_b_id = clip_b.id
            for i in range(4):
                frame = ClipFrame(clip_id=clip_a, ordinal=i, t_offset_seconds=float(i), score=0.9, thumb_path=f"thumbs/a{i}.jpg")
                session.add(frame)
                session.flush()
                session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=marcel_id))

        counts_a = {slug: count for slug, _name, _archived, count in query_cat_frame_counts(engine, clip_a)}
        counts_b = {slug: count for slug, _name, _archived, count in query_cat_frame_counts(engine, clip_b_id)}
    finally:
        engine.dispose()

    assert counts_a == {"marcel": 4, "rufus": 0}
    assert counts_b == {"marcel": 0, "rufus": 0}


def test_label_summary_endpoint_returns_recomputed_fields(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """GET /clips/{id}/label-summary returns the same tag_summary / has_manual_cat the page renders."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        marcel_id = db_helpers.seed_cat_subject(engine, slug="marcel", display_name="Marcel", display_order=1)
        _ = db_helpers.seed_cat_subject(engine, slug="rufus", display_name="Rufus", display_order=2)
        with get_session(engine) as session:
            for i in range(2):
                frame = ClipFrame(clip_id=clip_id, ordinal=i, t_offset_seconds=float(i), score=0.9, thumb_path=f"thumbs/f{i}.jpg")
                session.add(frame)
                session.flush()
                session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=marcel_id))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}/label-summary", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert response.json() == {
        "tag_summary": "marcel: 2, rufus: 0",
        "has_manual_cat": True,
        "identity_html": (
            '\n  <span class="identity-chip identity-chip-operator"\n        \n        title="Marcel (operator)">Marcel</span>\n\n'
        ),
    }


def _seed_cats_and_events_for_summary(engine: Engine, clip_id: int) -> None:
    """Seed two cats (marcel x3 frames, rufus x0) and two event subjects (cleaning, person) each tagged once."""
    marcel_id = db_helpers.seed_cat_subject(engine, slug="marcel", display_name="Marcel", display_order=1)
    _ = db_helpers.seed_cat_subject(engine, slug="rufus", display_name="Rufus", display_order=2)
    with get_session(engine) as session:
        cleaning = Subject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1)
        person = Subject(slug="person", display_name="Person", kind="event", display_order=2)
        session.add(cleaning)
        session.add(person)
        session.flush()
        for i in range(3):
            frame = ClipFrame(
                clip_id=clip_id,
                ordinal=i,
                t_offset_seconds=float(i * 5),
                score=0.9,
                thumb_path=f"thumbs/frame_{clip_id}_{i}.jpg",
            )
            session.add(frame)
            session.flush()
            session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=marcel_id))
        event_frame = ClipFrame(
            clip_id=clip_id,
            ordinal=3,
            t_offset_seconds=15.0,
            score=0.5,
            thumb_path=f"thumbs/frame_{clip_id}_3.jpg",
        )
        session.add(event_frame)
        session.flush()
        session.add(ClipFrameSubject(clip_frame_id=event_frame.id, subject_id=cleaning.id))
        session.add(ClipFrameSubject(clip_frame_id=event_frame.id, subject_id=person.id))


def test_tag_summary_cats_and_events(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """tag_summary includes cats with counts and events (bare slugs) separated by '; '."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        _seed_cats_and_events_for_summary(engine, clip_id)
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "marcel: 3, rufus: 0; cleaning, person" in response.text


def test_tag_summary_empty_state(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """tag_summary shows '—' when no cat subjects in config and no events tagged."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "<dt>tag_summary</dt>" in response.text
    assert "—" in response.text


def test_tag_summary_archived_subject_with_memberships(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """An archived cat subject with tagged frames shows 'display_name (archived): <count>' in tag_summary."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            archived_subj = Subject(
                slug="fluffy",
                display_name="Fluffy",
                kind="cat",
                display_order=1,
                archived_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            session.add(archived_subj)
            session.flush()
            frame = make_clip_frame(clip_id, 0)
            session.add(frame)
            session.flush()
            session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=archived_subj.id))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    # The full rendering interpolates the per-clip frame count after the archived marker.
    assert "Fluffy (archived): 1" in response.text


# ---------------------------------------------------------------------------
# Task 14: keyboard shortcuts + help overlay
# ---------------------------------------------------------------------------


def test_kbd_help_overlay_element_present(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The keyboard-help overlay element is present in the rendered HTML."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert 'id="kbd-help"' in response.text


def test_kbd_help_overlay_subject_rows_in_overlay_section(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The overlay section contains the display_name for each active subject."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            session.add(Subject(slug="marcel", display_name="Marcel", kind="cat", display_order=1))
            session.add(Subject(slug="rufus", display_name="Rufus", kind="cat", display_order=2))
            session.add(Subject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    overlay_start = response.text.find('id="kbd-help"')
    assert overlay_start != -1
    overlay_section = response.text[overlay_start : overlay_start + 2000]
    assert "Marcel" in overlay_section
    assert "Rufus" in overlay_section
    assert "Cleaning" in overlay_section


def test_kbd_help_overlay_digit_mapping_is_cats_then_events_in_display_order(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The overlay maps digit 1..N to subjects in cats-first, then-events, display_order sequence.

    This is the live-mapping acceptance criterion: digits are assigned by render order, not a
    hardcoded table, so the digit shown next to each subject must follow cats-before-events. With
    two cats (display_order 1, 2) and one event, the mapping is 1→Marcel, 2→Rufus, 3→Cleaning.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            session.add(Subject(slug="marcel", display_name="Marcel", kind="cat", display_order=1))
            session.add(Subject(slug="rufus", display_name="Rufus", kind="cat", display_order=2))
            session.add(Subject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert 'id="kbd-help"' in response.text
    overlay = response.text.split('id="kbd-help"', 1)[1][:2000]
    # Each subject row renders ``<kbd>{digit}</kbd>`` then the display_name. Assert the digit→subject
    # pairing and ordering by checking these markers appear in strictly increasing position.
    markers = ["<kbd>1</kbd>", "Marcel", "<kbd>2</kbd>", "Rufus", "<kbd>3</kbd>", "Cleaning"]
    positions = [overlay.find(m) for m in markers]
    assert all(p != -1 for p in positions), f"missing marker(s): {dict(zip(markers, positions, strict=True))}"
    assert positions == sorted(positions), f"digit/subject order wrong: {dict(zip(markers, positions, strict=True))}"


def test_kbd_help_overlay_grows_with_third_cat(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Adding a 3rd cat subject adds its name to the overlay."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            session.add(Subject(slug="marcel", display_name="Marcel", kind="cat", display_order=1))
            session.add(Subject(slug="rufus", display_name="Rufus", kind="cat", display_order=2))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response_two = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    engine = _detail_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            session.add(Subject(slug="whiskers", display_name="Whiskers", kind="cat", display_order=3))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response_three = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response_two.status_code == 200
    assert response_three.status_code == 200
    assert "Whiskers" not in response_two.text
    assert "Whiskers" in response_three.text
    overlay_start = response_three.text.find('id="kbd-help"')
    assert overlay_start != -1
    overlay_section = response_three.text[overlay_start : overlay_start + 3000]
    assert "Whiskers" in overlay_section


def test_kbd_help_overlay_navigation_shortcuts_present(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The overlay contains rows for ↑/↓, →/←, r, and ? shortcuts."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    overlay_start = response.text.find('id="kbd-help"')
    assert overlay_start != -1
    overlay_section = response.text[overlay_start : overlay_start + 3000]
    assert "↑" in overlay_section
    assert "↓" in overlay_section
    assert "→" in overlay_section
    assert "←" in overlay_section
    assert "?" in overlay_section


def test_tag_table_first_row_has_active_frame_class(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The first <tr> in the tag table has the active-frame class by default."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    try:
        _ = _seed_five_frames(engine, clip_id)
        _ = db_helpers.seed_cat_subject(engine)
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert 'class="tag-row active-frame"' in response.text


# ---------------------------------------------------------------------------
# Task 15: filter-scoped prev/next navigation
# ---------------------------------------------------------------------------


def _seed_three_unreviewed_clips(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
    *,
    internal_root: Path,
) -> tuple[int, int, int, int]:
    """Seed one camera and three unreviewed clips with sequential start_ts; return (cam_id, id1, id2, id3).

    clip1 is oldest, clip3 is newest. All belong to the pantry camera.
    """
    ts1 = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)
    ts2 = datetime(2026, 6, 1, 11, 0, 0, tzinfo=UTC)
    ts3 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)

    with db_session_factory(internal_root) as session:
        cam = Camera(name="pantry", display_name="Pantry Litter Box", host="cam.example.com")
        session.add(cam)
        session.flush()
        cam_id = cam.id

    engine = _detail_engine_for(internal_root)
    try:
        clip1_id = seed_clip(engine, camera_id=cam_id, start_ts=ts1, has_cat=False)
        clip2_id = seed_clip(engine, camera_id=cam_id, start_ts=ts2, has_cat=False)
        clip3_id = seed_clip(engine, camera_id=cam_id, start_ts=ts3, has_cat=False)
    finally:
        engine.dispose()

    return cam_id, clip1_id, clip2_id, clip3_id


def test_filter_scoped_next_points_to_next_unreviewed_clip_same_camera(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """Next link on /clips/{id}?reviewed=no&camera=pantry points to the next older unreviewed clip in that camera."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip1_id, clip2_id, _ = _seed_three_unreviewed_clips(
        db_session_factory,
        seed_clip,
        internal_root=internal_root,
    )

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip2_id}?reviewed=no&camera=pantry", headers=AUTH_HEADER)

    assert response.status_code == 200
    expected_next_href = f'href="/clips/{clip1_id}?reviewed=no&amp;camera=pantry" rel="next"'
    assert expected_next_href in response.text, f"Expected {expected_next_href!r} in response"


def test_filter_scoped_prev_points_to_prev_unreviewed_clip_same_camera(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """Previous link on /clips/{id}?reviewed=no&camera=pantry points to the next newer unreviewed clip in that camera."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, _, clip2_id, clip3_id = _seed_three_unreviewed_clips(
        db_session_factory,
        seed_clip,
        internal_root=internal_root,
    )

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip2_id}?reviewed=no&camera=pantry", headers=AUTH_HEADER)

    assert response.status_code == 200
    expected_prev_href = f'href="/clips/{clip3_id}?reviewed=no&amp;camera=pantry" rel="prev"'
    assert expected_prev_href in response.text, f"Expected {expected_prev_href!r} in response"


def test_filter_scoped_next_at_end_of_queue_links_back_to_clips(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """Next link on the last clip in the filtered queue points back to /clips?{filter_qs}."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip1_id, _, _ = _seed_three_unreviewed_clips(db_session_factory, seed_clip, internal_root=internal_root)

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip1_id}?reviewed=no&camera=pantry", headers=AUTH_HEADER)

    assert response.status_code == 200
    expected_next_href = 'href="/clips?reviewed=no&amp;camera=pantry" rel="next"'
    assert expected_next_href in response.text, f"Expected {expected_next_href!r} in response"


def test_no_filter_qs_falls_back_to_legacy_all_clips_nav(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """Direct URL /clips/{id} (no querystring) falls back to legacy prev/next across all clips."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip1_id, clip2_id, clip3_id = _seed_three_unreviewed_clips(
        db_session_factory,
        seed_clip,
        internal_root=internal_root,
    )

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip2_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    # Legacy behavior: prev = newer timestamp (clip3), next = older timestamp (clip1).
    # url_for generates absolute URLs in the test client (http://testserver/...).
    assert f'href="http://testserver/clips/{clip3_id}"' in response.text, "Expected legacy prev link (newer)"
    assert f'href="http://testserver/clips/{clip1_id}"' in response.text, "Expected legacy next link (older)"


def _mark_clip_reviewed_at(internal_root: Path, clip_id: int, reviewed_at: datetime) -> None:
    """Stamp ``reviewed_at`` on a clip row using a short-lived engine."""
    engine = _detail_engine_for(internal_root)
    try:
        stamp_reviewed_at(engine, clip_id, reviewed_at)
    finally:
        engine.dispose()


def test_after_marking_reviewed_next_skips_just_marked_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """After marking clip2 reviewed, the next link of clip3 (in reviewed=no queue) skips clip2 and points to clip1.

    This tests the documented limitation: the detail page is rendered from db state at request time.
    Once clip2 is marked reviewed, next skips clip2 in the unreviewed set.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip1_id, clip2_id, clip3_id = _seed_three_unreviewed_clips(
        db_session_factory,
        seed_clip,
        internal_root=internal_root,
    )

    _mark_clip_reviewed_at(internal_root, clip2_id, datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC))

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip3_id}?reviewed=no&camera=pantry", headers=AUTH_HEADER)

    assert response.status_code == 200
    expected_next_href = f'href="/clips/{clip1_id}?reviewed=no&amp;camera=pantry" rel="next"'
    assert expected_next_href in response.text, f"Expected next to skip clip2; expected {expected_next_href!r}"
    unexpected_href = f'href="/clips/{clip2_id}?reviewed=no&amp;camera=pantry"'
    assert unexpected_href not in response.text, "Neither link points to just-marked clip2"


def test_filter_scoped_reviewed_yes_nav_orders_by_start_ts(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """Under ``reviewed=yes`` prev/next follow ``start_ts`` like every other filter, not ``reviewed_at``.

    ``reviewed_at`` is seeded inverted relative to ``start_ts`` (clip1 reviewed last, clip3 reviewed
    first), so a ``reviewed_at`` order points the opposite way.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip1_id, clip2_id, clip3_id = _seed_three_unreviewed_clips(
        db_session_factory,
        seed_clip,
        internal_root=internal_root,
    )
    _mark_clip_reviewed_at(internal_root, clip1_id, datetime(2026, 6, 2, 13, 0, 0, tzinfo=UTC))
    _mark_clip_reviewed_at(internal_root, clip2_id, datetime(2026, 6, 2, 12, 0, 0, tzinfo=UTC))
    _mark_clip_reviewed_at(internal_root, clip3_id, datetime(2026, 6, 2, 11, 0, 0, tzinfo=UTC))

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip2_id}?reviewed=yes&camera=pantry", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert f'href="/clips/{clip3_id}?reviewed=yes&amp;camera=pantry" rel="prev"' in response.text
    assert f'href="/clips/{clip1_id}?reviewed=yes&amp;camera=pantry" rel="next"' in response.text


def _sorted_nav_page(
    config: Config,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    clip_id: int,
    query: str,
) -> str:
    """Return the body of the detail page of ``clip_id`` after the status and tripwire checks."""
    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}?{query}", headers=AUTH_HEADER)
    assert response.status_code == 200
    _assert_no_tripwire(response.text)
    return response.text


def test_filter_scoped_nav_follows_the_default_newest_sort(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """With no ``sort`` key, prev is the newer clip and next the older one.

    The links drop the ``page`` of the request. The back link goes to the page that holds the clip.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, older_id, middle_id, newer_id = _seed_three_unreviewed_clips(db_session_factory, seed_clip, internal_root=internal_root)

    body = _sorted_nav_page(config, alembic_web_test_client, middle_id, "reviewed=no&page=3")

    assert f'<a href="/clips/{newer_id}?reviewed=no" rel="prev">← Newer</a>' in body
    assert f'<a href="/clips/{older_id}?reviewed=no" rel="next">Older →</a>' in body
    assert '<a href="/clips?reviewed=no">All clips</a>' in body


def test_filter_scoped_nav_follows_the_oldest_sort(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """With ``sort=oldest``, prev is the older clip and next the newer one, and the labels say so.

    The links drop the ``page`` of the request.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, older_id, middle_id, newer_id = _seed_three_unreviewed_clips(db_session_factory, seed_clip, internal_root=internal_root)

    body = _sorted_nav_page(config, alembic_web_test_client, middle_id, "reviewed=no&sort=oldest&page=3")

    carried = "reviewed=no&amp;sort=oldest"
    assert f'<a href="/clips/{older_id}?{carried}" rel="prev">← Older</a>' in body
    assert f'<a href="/clips/{newer_id}?{carried}" rel="next">Newer →</a>' in body
    assert f'<a href="/clips?{carried}">All clips</a>' in body


def test_filter_scoped_nav_under_oldest_sort_ends_on_the_newest_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """The newest clip is the end of an oldest-first list, so its next link returns to the list."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, _, middle_id, newer_id = _seed_three_unreviewed_clips(db_session_factory, seed_clip, internal_root=internal_root)

    body = _sorted_nav_page(config, alembic_web_test_client, newer_id, "reviewed=no&sort=oldest")

    assert f'<a href="/clips/{middle_id}?reviewed=no&amp;sort=oldest" rel="prev">← Older</a>' in body
    assert '<a href="/clips?reviewed=no&amp;sort=oldest" rel="next">Newer →</a>' in body


class _ThreeClips(NamedTuple):
    config: Config
    internal_root: Path
    older_id: int
    middle_id: int
    newer_id: int


@pytest.fixture
def three_clips(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> _ThreeClips:
    """Seed an older, a middle, and a newer unreviewed clip, one hour apart. Build the config of their database."""
    internal_root, storage_root = storage_dirs
    _, older_id, middle_id, newer_id = _seed_three_unreviewed_clips(db_session_factory, seed_clip, internal_root=internal_root)
    return _ThreeClips(make_config(internal_root, storage_root), internal_root, older_id, middle_id, newer_id)


@pytest.mark.parametrize(
    "case",
    [
        # Newest first, two rows a page: newer and middle on page 1, older on page 2.
        ("reviewed=no", "older_id", "/clips?reviewed=no&amp;page=2"),
        ("reviewed=no&page=2", "newer_id", "/clips?reviewed=no"),
        # Oldest first: older and middle on page 1, newer on page 2.
        ("reviewed=no&sort=oldest", "newer_id", "/clips?reviewed=no&amp;sort=oldest&amp;page=2"),
        ("reviewed=no&sort=oldest&page=2", "older_id", "/clips?reviewed=no&amp;sort=oldest"),
    ],
)
def test_back_link_goes_to_the_page_that_holds_the_clip(
    three_clips: _ThreeClips,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, str, str],
) -> None:
    """The "All clips" link and the end-of-list link go to the page of the clip, not to the page of the request."""
    query, clip_field, expected_back = case
    monkeypatch.setattr(clips_routes, "_CLIPS_PAGE_SIZE", 2)
    clip_id = cast("int", getattr(three_clips, clip_field))

    body = _sorted_nav_page(three_clips.config, alembic_web_test_client, clip_id, query)

    assert f'<a href="{expected_back}">All clips</a>' in body
    assert f'<a href="{expected_back}" rel="next">' in body or f'<a href="{expected_back}" rel="prev">' in body


def test_back_link_of_a_clip_outside_the_filter_goes_to_the_page_of_its_list_position(
    three_clips: _ThreeClips,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reviewed clip under ``reviewed=no`` goes back to the page of its place in the list order. That page holds its next neighbor."""
    monkeypatch.setattr(clips_routes, "_CLIPS_PAGE_SIZE", 1)
    _mark_clip_reviewed_at(three_clips.internal_root, three_clips.middle_id, datetime(2026, 6, 2, 12, 0, 0, tzinfo=UTC))

    body = _sorted_nav_page(three_clips.config, alembic_web_test_client, three_clips.middle_id, "reviewed=no")

    assert '<a href="/clips?reviewed=no&amp;page=2">All clips</a>' in body
    assert f'<a href="/clips/{three_clips.older_id}?reviewed=no" rel="next">' in body


def _seed_start_ts_tie(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
    *,
    internal_root: Path,
) -> tuple[int, int]:
    """Seed two clips on two cameras with one ``start_ts``. Return ``(lower_id, higher_id)``."""
    with db_session_factory(internal_root) as session:
        cams = [Camera(name=name, display_name=name.title(), host="cam.example.com") for name in ("pantry", "garage")]
        session.add_all(cams)
        session.flush()
        cam_ids = [cam.id for cam in cams]
    start_ts = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)
    engine = _detail_engine_for(internal_root)
    try:
        first = seed_clip(engine, camera_id=cam_ids[0], start_ts=start_ts, has_cat=False)
        second = seed_clip(engine, camera_id=cam_ids[1], start_ts=start_ts, has_cat=False)
    finally:
        engine.dispose()
    return min(first, second), max(first, second)


def test_filter_scoped_nav_steps_to_a_start_ts_tie_under_the_newest_sort(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """Newest first lists the higher ``id`` first on a tie, so its next link is the lower ``id``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    lower_id, higher_id = _seed_start_ts_tie(db_session_factory, seed_clip, internal_root=internal_root)

    higher = _sorted_nav_page(config, alembic_web_test_client, higher_id, "reviewed=no")
    lower = _sorted_nav_page(config, alembic_web_test_client, lower_id, "reviewed=no")

    assert f'<a href="/clips/{lower_id}?reviewed=no" rel="next">Older →</a>' in higher
    assert f'<a href="/clips/{higher_id}?reviewed=no" rel="prev">← Newer</a>' in lower


def test_filter_scoped_nav_steps_to_a_start_ts_tie_under_the_oldest_sort(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """Oldest first lists the lower ``id`` first on a tie, so its next link is the higher ``id``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    lower_id, higher_id = _seed_start_ts_tie(db_session_factory, seed_clip, internal_root=internal_root)

    lower = _sorted_nav_page(config, alembic_web_test_client, lower_id, "reviewed=no&sort=oldest")
    higher = _sorted_nav_page(config, alembic_web_test_client, higher_id, "reviewed=no&sort=oldest")

    assert f'<a href="/clips/{higher_id}?reviewed=no&amp;sort=oldest" rel="next">Newer →</a>' in lower
    assert f'<a href="/clips/{lower_id}?reviewed=no&amp;sort=oldest" rel="prev">← Older</a>' in higher


def test_legacy_nav_steps_to_a_start_ts_tie(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    seed_clip: Callable[..., int],
) -> None:
    """With no filter keys, the global nav also steps from the higher ``id`` to the lower one on a tie."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    lower_id, higher_id = _seed_start_ts_tie(db_session_factory, seed_clip, internal_root=internal_root)

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{higher_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert f'href="http://testserver/clips/{lower_id}" rel="next"' in response.text


def test_review_endpoint_stamp_matches_the_rendered_page(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The string the JSON hands the browser is the string a full page load would render.

    This case lives here rather than in ``test_web_review_state.py`` because ``GET /clips/{id}``
    reads ``clip_label_summary``, and that module's fixtures create no view.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = seed_camera_and_clip_with_files(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with alembic_web_test_client(config) as client:
        posted = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)
        rendered = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert posted.status_code == 200
    body = cast("dict[str, str]", posted.json())
    assert f">{body['reviewed_at_stamp']}</time>" in rendered.text
    assert f'<span class="review-badge">Reviewed {body["reviewed_at_date"]}</span>' in rendered.text


_TRIPWIRES = ("{_fmt", "{self.", "{cam.", "{cfg.", "NoneType")


def _seed_model_frames(engine: Engine, clip_id: int, *, marcel_id: int) -> list[int]:
    """Add four frames: a confident Marcel, an unsure one, no box, and not classified."""
    specs: list[tuple[int | None, float | None, str | None]] = [
        (marcel_id, 0.95, "model@abc"),
        (marcel_id, 0.40, "model@abc"),
        (None, None, "model@abc"),
        (None, None, None),
    ]
    with get_session(engine) as session:
        frame_ids: list[int] = []
        for ordinal, (subject_id, confidence, version) in enumerate(specs):
            frame = ClipFrame(
                clip_id=clip_id,
                ordinal=ordinal,
                t_offset_seconds=float(ordinal),
                score=0.9,
                thumb_path=f"thumbs/model_{clip_id}_{ordinal}.jpg",
                predicted_subject_id=subject_id,
                prediction_confidence=confidence,
                classifier_version=version,
            )
            session.add(frame)
            session.flush()
            frame_ids.append(frame.id)
    return frame_ids


def _tag_rows(html: str) -> list[str]:
    return re.findall(r'<tr class="tag-row[^"]*">.*?</tr>', html, flags=re.DOTALL)


def _predictions(row: str) -> list[tuple[str, str]]:
    """Return ``(class, text)`` of each prediction element in a tag row."""
    return re.findall(r'<(?:span|button)\b[^>]*class="(prediction[^"]*)"[^>]*>(.*?)</', row, flags=re.DOTALL)


def _attribute_pairs(tag: str) -> list[tuple[str, str]]:
    return re.findall(r'([\w:-]+)="([^"]*)"', tag)


def _tag_buttons(row: str) -> list[dict[str, str]]:
    """Return the attributes of each ``button.tag-btn`` in a tag row, in document order."""
    tags: list[str] = re.findall(r'<button\b[^>]*class="tag-btn[^"]*"[^>]*>', row, flags=re.DOTALL)
    return [dict(_attribute_pairs(tag)) for tag in tags]


def _assert_no_tripwire(text: str) -> None:
    for tripwire in _TRIPWIRES:
        assert tripwire not in text


class _ModelClip(NamedTuple):
    config: Config
    clip_id: int
    marcel_id: int
    rufus_id: int
    frame_ids: list[int]
    engine: Engine


@pytest.fixture
def model_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> Generator[_ModelClip]:
    """Seed a cat-positive clip with four predicted frames and the cats Marcel and Rufus."""
    internal_root, storage_root = storage_dirs
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    engine = _detail_engine_for(internal_root)
    marcel_id = db_helpers.seed_cat_subject(engine, slug="marcel", display_name="Marcel", display_order=1)
    rufus_id = db_helpers.seed_cat_subject(engine, slug="rufus", display_name="Rufus", display_order=2)
    frame_ids = _seed_model_frames(engine, clip_id, marcel_id=marcel_id)
    yield _ModelClip(make_config(internal_root, storage_root), clip_id, marcel_id, rufus_id, frame_ids, engine)
    engine.dispose()


def _detail_html(
    clip: _ModelClip,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> str:
    with alembic_web_test_client(clip.config) as client:
        response = client.get(f"/clips/{clip.clip_id}", headers=AUTH_HEADER)
    assert response.status_code == 200
    _assert_no_tripwire(response.text)
    return response.text


def test_clip_detail_shows_model_label_beside_each_frame(
    model_clip: _ModelClip,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Each frame row shows its stored model result: a confident cat, unsure, no box, not classified."""
    html = _detail_html(model_clip, alembic_web_test_client)

    labels = [text.strip() for row in _tag_rows(html) for _, text in _predictions(row)]
    assert labels == ["Marcel 0.95", "unsure (0.40)", "no cat box found", "not classified"]


def test_clip_detail_shows_model_label_on_tagged_frame_and_leaves_untagged_unpressed(
    model_clip: _ModelClip,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A tagged frame shows its tag pressed and its model label. An untagged frame has no pressed button."""
    with get_session(model_clip.engine) as session:
        session.add(ClipFrameSubject(clip_frame_id=model_clip.frame_ids[1], subject_id=model_clip.marcel_id))

    rows = _tag_rows(_detail_html(model_clip, alembic_web_test_client))

    assert len(rows) == 4
    assert [row.count("tag-btn-on") for row in rows] == [0, 1, 0, 0]
    assert [row.count('aria-pressed="true"') for row in rows] == [0, 1, 0, 0]
    assert [text.strip() for _, text in _predictions(rows[1])] == ["unsure (0.40)"]


def test_model_label_is_not_a_tag_button(
    model_clip: _ModelClip,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The tag buttons keep their order and data attributes, and no prediction uses a tag class."""
    rows = _tag_rows(_detail_html(model_clip, alembic_web_test_client))

    assert len(rows) == len(model_clip.frame_ids)
    for row, frame_id in zip(rows, model_clip.frame_ids, strict=True):
        predictions = _predictions(row)
        assert predictions
        assert all("tag-btn" not in css_class for css_class, _ in predictions)
        assert [(b["data-subject-slug"], b["data-frame-id"], b["data-subject-id"]) for b in _tag_buttons(row)] == [
            ("marcel", str(frame_id), str(model_clip.marcel_id)),
            ("rufus", str(frame_id), str(model_clip.rufus_id)),
        ]


def test_clip_detail_shows_clip_identity_in_metadata(
    model_clip: _ModelClip,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The metadata list holds the clip result in dd#clip-identity."""
    match = re.search(r'<dd id="clip-identity">(.*?)</dd>', _detail_html(model_clip, alembic_web_test_client), flags=re.DOTALL)

    assert match is not None
    assert "identity-chip identity-chip-model" in match.group(1)
    assert "Marcel" in match.group(1)


def test_clip_detail_shows_no_cat_in_identity_for_a_clip_that_is_not_cat_positive(
    model_clip: _ModelClip,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A reviewed clip with no cat tag renders ``no cat`` in dd#clip-identity, never an empty cell."""
    with alembic_web_test_client(model_clip.config) as client:
        assert client.post(f"/clips/{model_clip.clip_id}/reviewed", headers=AUTH_HEADER).status_code < 300
        response = client.get(f"/clips/{model_clip.clip_id}", headers=AUTH_HEADER)
    match = re.search(r'<dd id="clip-identity">(.*?)</dd>', response.text, flags=re.DOTALL)

    assert match is not None
    assert match.group(1).strip() != ""
    assert "no cat" in match.group(1)
    assert "identity-chip" not in match.group(1)


def test_label_summary_identity_html_follows_tag_and_review_toggles(
    model_clip: _ModelClip,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """identity_html changes after a tag toggle and again after a review toggle."""
    clip_id = model_clip.clip_id
    tag_url = f"/clips/{clip_id}/frames/{model_clip.frame_ids[0]}/subjects/{model_clip.rufus_id}"

    def identity_html(client: TestClient) -> str:
        response = client.get(f"/clips/{clip_id}/label-summary", headers=AUTH_HEADER)
        assert response.status_code == 200
        _assert_no_tripwire(response.text)
        return str(cast("dict[str, object]", response.json())["identity_html"])

    with alembic_web_test_client(model_clip.config) as client:
        initial = identity_html(client)
        assert "identity-chip-model" in initial
        assert "Marcel" in initial

        assert client.put(tag_url, headers=AUTH_HEADER).status_code < 300
        tagged = identity_html(client)
        assert tagged != initial
        assert "identity-chip-operator" in tagged
        assert "Rufus" in tagged

        assert client.delete(tag_url, headers=AUTH_HEADER).status_code < 300
        assert identity_html(client) == initial

        assert client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER).status_code < 300
        assert "no cat" in identity_html(client)
        assert "identity-chip" not in identity_html(client)


def test_clip_detail_names_operator_cat_on_unreviewed_clip_the_detector_calls_no_cat(
    model_clip: _ModelClip,
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A cat tag names its cat before the cat-positive check, and the model cat stays hidden.

    A tag does not mark the clip reviewed, so the view keeps the detector verdict ``no cat``.
    """
    with get_session(model_clip.engine) as session:
        clip = session.get(Clip, model_clip.clip_id)
        assert clip is not None
        clip.has_cat = False
        session.add(ClipFrameSubject(clip_frame_id=model_clip.frame_ids[1], subject_id=model_clip.rufus_id))

    match = re.search(r'<dd id="clip-identity">(.*?)</dd>', _detail_html(model_clip, alembic_web_test_client), flags=re.DOTALL)

    assert match is not None
    assert "identity-chip identity-chip-operator" in match.group(1)
    assert "Rufus" in match.group(1)
    assert "identity-chip-model" not in match.group(1)
    assert "no cat" not in match.group(1)
