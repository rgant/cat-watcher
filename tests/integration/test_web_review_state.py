"""Integration tests for POST/DELETE /clips/{id}/reviewed.

Covers:
* POST sets ``reviewed_at``, returns the rendered timestamp.
* Re-POST is idempotent — timestamp unchanged, response repeats the original.
* DELETE clears ``reviewed_at``, returns null timestamp fields.
* Re-DELETE is a no-op, still null.
* POST then DELETE preserves ``clip_frame_subjects`` rows.
* 404 on unknown clip ID for both verbs.
* Successful POST/DELETE emit the expected JSONLines log event.
"""

import logging
from pathlib import Path  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

import pytest  # noqa: TC002  # pytest evaluates fixture annotations (LogCaptureFixture) at collection time
from sqlalchemy import select

from cat_watcher.db import Clip, ClipFrame, ClipFrameSubject, get_session
from cat_watcher.timefmt import local_date, local_stamp

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from datetime import datetime

    from fastapi.testclient import TestClient
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config

from db_helpers import (
    AUTH_HEADER,
    ROUTES_LOGGER,
    seed_camera_and_clip,
    seed_cat_subject,
    tag_clip_frame,
)  # pytest pythonpath makes this importable

_DISPLAY_TZ = ZoneInfo("America/New_York")


def _get_reviewed_at(engine: Engine, clip_id: int) -> datetime | None:
    """Read ``clips.reviewed_at`` for ``clip_id`` via the given engine."""
    with get_session(engine) as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        return clip.reviewed_at


def test_post_reviewed_sets_reviewed_at_and_returns_rendered_stamp(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """POST /clips/{id}/reviewed sets ``reviewed_at`` and returns it rendered in display_timezone."""
    config, engine, clip_id = seeded_clip_env
    with web_test_client(config) as client:
        response = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)

    assert response.status_code == 200
    stored = _get_reviewed_at(engine, clip_id)
    assert stored is not None
    body = cast("dict[str, str | None]", response.json())
    assert body["reviewed_at_iso"] == stored.isoformat()
    assert body["reviewed_at_stamp"] == local_stamp(stored, tz=_DISPLAY_TZ)
    assert body["reviewed_at_date"] == local_date(stored, tz=_DISPLAY_TZ)


def test_post_reviewed_idempotent_does_not_overwrite_timestamp(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Re-POST on an already-reviewed clip leaves ``reviewed_at`` unchanged and repeats it."""
    config, engine, clip_id = seeded_clip_env
    with web_test_client(config) as client:
        first = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)
        first_ts = _get_reviewed_at(engine, clip_id)
        second = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)
        second_ts = _get_reviewed_at(engine, clip_id)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first_ts is not None
    # The idempotent no-op must not overwrite the original timestamp.
    assert first_ts == second_ts
    assert first.json() == second.json()


def test_delete_reviewed_clears_reviewed_at_and_returns_nulls(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """DELETE /clips/{id}/reviewed clears ``reviewed_at`` and reports null timestamp fields."""
    config, engine, clip_id = seeded_clip_env
    with web_test_client(config) as client:
        _ = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)
        response = client.delete(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert response.json() == {"reviewed_at_iso": None, "reviewed_at_stamp": None, "reviewed_at_date": None}
    assert _get_reviewed_at(engine, clip_id) is None


def test_delete_reviewed_idempotent_when_already_cleared(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Re-DELETE on an already-unreviewed clip is a no-op."""
    config, engine, clip_id = seeded_clip_env
    with web_test_client(config) as client:
        first = client.delete(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)
        second = client.delete(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)

    assert first.status_code == 200
    assert second.status_code == 200
    assert _get_reviewed_at(engine, clip_id) is None


def test_post_then_delete_preserves_clip_frame_subjects(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """POST then DELETE preserves all ``clip_frame_subjects`` rows for the clip's frames.

    The re-open workflow must not evict frame tags — memberships are only modified by the per-frame
    endpoints, not by the reviewed-state toggle.
    """
    config, engine, clip_id = seeded_clip_env
    subj_id = seed_cat_subject(engine)
    tag_clip_frame(engine, clip_id=clip_id, subject_id=subj_id)

    with web_test_client(config) as client:
        _ = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)
        _ = client.delete(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)

    with get_session(engine) as session:
        frame_ids = list(session.scalars(select(ClipFrame.id).where(ClipFrame.clip_id == clip_id)))
        assert len(frame_ids) == 1, "ClipFrame row must survive the re-open"
        membership = session.scalar(
            select(ClipFrameSubject).where(ClipFrameSubject.clip_frame_id == frame_ids[0]),
        )
        assert membership is not None, "ClipFrameSubject row must survive the re-open"


def test_post_reviewed_returns_404_for_unknown_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """POST /clips/9999/reviewed on a nonexistent clip returns 404."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.post("/clips/9999/reviewed", headers=AUTH_HEADER)

    assert response.status_code == 404


def test_delete_reviewed_returns_404_for_unknown_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """DELETE /clips/9999/reviewed on a nonexistent clip returns 404."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.delete("/clips/9999/reviewed", headers=AUTH_HEADER)

    assert response.status_code == 404


def test_post_reviewed_emits_clip_reviewed_log_event(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A state-changing POST emits a ``clip_reviewed`` log record at INFO level with ``clip_id``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    clip_id = seed_camera_and_clip(db_session_factory, internal_root)
    # alembic's env.py invokes logging.config.fileConfig during the integration suite, which
    # disables every existing logger. Re-enable before capturing so the record isn't swallowed.
    logging.getLogger(ROUTES_LOGGER).disabled = False

    with caplog.at_level(logging.INFO, logger=ROUTES_LOGGER), web_test_client(config) as client:
        _ = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)

    matching = [r for r in caplog.records if r.getMessage() == "clip_reviewed"]
    assert len(matching) == 1, f"expected exactly one clip_reviewed record; got {[r.getMessage() for r in caplog.records]}"
    assert getattr(matching[0], "clip_id", None) == clip_id


def test_post_reviewed_no_log_event_on_idempotent_noop(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A no-op re-POST (already reviewed) must NOT emit a ``clip_reviewed`` event."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    clip_id = seed_camera_and_clip(db_session_factory, internal_root)
    logging.getLogger(ROUTES_LOGGER).disabled = False

    with web_test_client(config) as client:
        _ = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=ROUTES_LOGGER):
            _ = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)

    assert not any(r.getMessage() == "clip_reviewed" for r in caplog.records)


def test_delete_reviewed_emits_clip_review_reopened_log_event(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A state-changing DELETE emits a ``clip_review_reopened`` log record at INFO with ``clip_id``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    clip_id = seed_camera_and_clip(db_session_factory, internal_root)
    logging.getLogger(ROUTES_LOGGER).disabled = False

    with web_test_client(config) as client:
        _ = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)
        with caplog.at_level(logging.INFO, logger=ROUTES_LOGGER):
            _ = client.delete(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)

    matching = [r for r in caplog.records if r.getMessage() == "clip_review_reopened"]
    assert len(matching) == 1, "expected exactly one clip_review_reopened record"
    assert getattr(matching[0], "clip_id", None) == clip_id


def test_clip_detail_js_never_invents_a_timestamp() -> None:
    """The server is the only formatter; the browser inserts what it was sent.

    There is no JS test harness, so this source assertion is what pins the contract. A client-side
    ``new Date()`` renders the browser's zone, and near local midnight puts the review badge on
    tomorrow's date.
    """
    js = Path("src/cat_watcher/web/static/clip_detail.js").read_text(encoding="utf-8")
    assert "new Date(" not in js
