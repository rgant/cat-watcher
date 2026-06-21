"""Integration tests for PUT/DELETE /clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}.

Covers:
* PUT inserts the membership row, returns 204; re-PUT is idempotent (no duplicate, same 204).
* DELETE removes the row, returns 204; re-DELETE is a no-op.
* 404 on unknown clip ID.
* 404 on unknown frame ID.
* 404 on unknown subject ID.
* 404 when frame doesn't belong to the named clip (cross-clip mismatch).
* PUT against an archived subject → 409 Conflict; no row inserted.
* DELETE against an archived subject's existing membership → 204 (retraction works).
* Successful mutations emit the expected JSONLines event with the subject's slug.
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING

import pytest  # noqa: TC002  # pytest evaluates fixture annotations (LogCaptureFixture) at collection time
from sqlalchemy import select

from cat_watcher.db import Camera, Clip, ClipFrameSubject, Subject, get_session

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from fastapi.testclient import TestClient
    from sqlalchemy.engine import Engine

    from cat_watcher.config import Config

from db_helpers import (
    AUTH_HEADER,
    DEFAULT_START_TS,
    ROUTES_LOGGER,
    build_test_clip,
    make_clip_frame,
    seed_cat_subject,
)  # pytest pythonpath makes this importable


def _seed_frame(engine: Engine, *, clip_id: int, ordinal: int = 0) -> int:
    """Insert a ClipFrame for ``clip_id``; return its id."""
    with get_session(engine) as session:
        frame = make_clip_frame(clip_id, ordinal)
        session.add(frame)
        session.flush()
        return frame.id


def _seed_clip_on_first_camera(engine: Engine, *, source_filename: str, start_ts: datetime) -> int:
    """Add a second Clip on the same camera that already exists in the DB; return clip id."""
    with get_session(engine) as session:
        cam_id = session.scalar(select(Camera.id))
        assert cam_id is not None
        clip = build_test_clip(cam_id, start_ts=start_ts, source_filename=source_filename, has_cat=False)
        session.add(clip)
        session.flush()
        return clip.id


def _membership_exists(engine: Engine, *, frame_id: int, subject_id: int) -> bool:
    """Return True if the ClipFrameSubject row exists."""
    with get_session(engine) as session:
        row = session.scalar(
            select(ClipFrameSubject).where(
                ClipFrameSubject.clip_frame_id == frame_id,
                ClipFrameSubject.subject_id == subject_id,
            ),
        )
        return row is not None


def test_put_inserts_membership_and_returns_204(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """PUT inserts a ClipFrameSubject row and returns 204 No Content."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine)

    with web_test_client(config) as client:
        response = client.put(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 204
    assert _membership_exists(engine, frame_id=frame_id, subject_id=subject_id)


def test_put_membership_does_not_mark_clip_reviewed(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Tagging a frame leaves ``reviewed_at`` NULL — review state is orthogonal to tagging.

    Auto-marking a clip reviewed on first tag would silently drop it out of the ``reviewed=no``
    queue mid-labeling; review state must change only via the explicit reviewed endpoint.
    """
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine)

    with web_test_client(config) as client:
        response = client.put(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 204
    with get_session(engine) as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        assert clip.reviewed_at is None


def test_put_idempotent_no_duplicate_row(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Re-PUT is a no-op — no duplicate row inserted, still returns 204."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine)

    with web_test_client(config) as client:
        assert client.put(f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}", headers=AUTH_HEADER).status_code == 204
        assert client.put(f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}", headers=AUTH_HEADER).status_code == 204

    with get_session(engine) as session:
        rows = list(
            session.scalars(
                select(ClipFrameSubject).where(
                    ClipFrameSubject.clip_frame_id == frame_id,
                    ClipFrameSubject.subject_id == subject_id,
                ),
            ),
        )
    assert len(rows) == 1


def test_delete_removes_membership_and_returns_204(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """DELETE removes the ClipFrameSubject row and returns 204 No Content."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine)
    with get_session(engine) as session:
        session.add(ClipFrameSubject(clip_frame_id=frame_id, subject_id=subject_id))

    with web_test_client(config) as client:
        response = client.delete(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 204
    assert not _membership_exists(engine, frame_id=frame_id, subject_id=subject_id)


def test_delete_idempotent_no_error_when_absent(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Re-DELETE when membership is absent is a no-op and still returns 204."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine)

    with web_test_client(config) as client:
        first = client.delete(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )
        second = client.delete(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )

    assert first.status_code == 204
    assert second.status_code == 204


def test_put_returns_404_for_unknown_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """PUT returns 404 when the clip ID does not exist."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.put("/clips/9999/frames/1/subjects/1", headers=AUTH_HEADER)

    assert response.status_code == 404


def test_delete_returns_404_for_unknown_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """DELETE returns 404 when the clip ID does not exist."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.delete("/clips/9999/frames/1/subjects/1", headers=AUTH_HEADER)

    assert response.status_code == 404


def test_put_returns_404_for_unknown_frame(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """PUT returns 404 when the frame ID does not exist."""
    config, engine, clip_id = seeded_clip_env
    subject_id = seed_cat_subject(engine)

    with web_test_client(config) as client:
        response = client.put(
            f"/clips/{clip_id}/frames/9999/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 404


def test_delete_returns_404_for_unknown_frame(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """DELETE returns 404 when the frame ID does not exist."""
    config, engine, clip_id = seeded_clip_env
    subject_id = seed_cat_subject(engine)

    with web_test_client(config) as client:
        response = client.delete(
            f"/clips/{clip_id}/frames/9999/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 404


def test_put_returns_404_for_unknown_subject(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """PUT returns 404 when the subject ID does not exist."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)

    with web_test_client(config) as client:
        response = client.put(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/9999",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 404


def test_delete_returns_404_for_unknown_subject(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """DELETE returns 404 when the subject ID does not exist."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)

    with web_test_client(config) as client:
        response = client.delete(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/9999",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 404


def test_put_returns_404_when_frame_belongs_to_different_clip(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """PUT returns 404 when the frame ID exists but belongs to a different clip."""
    config, engine, clip_a_id = seeded_clip_env
    clip_b_id = _seed_clip_on_first_camera(engine, source_filename="070000.mp4", start_ts=DEFAULT_START_TS + timedelta(hours=1))
    frame_b_id = _seed_frame(engine, clip_id=clip_b_id)
    subject_id = seed_cat_subject(engine)

    with web_test_client(config) as client:
        # clip_a_id in path, but frame belongs to clip_b — mismatch must 404.
        response = client.put(f"/clips/{clip_a_id}/frames/{frame_b_id}/subjects/{subject_id}", headers=AUTH_HEADER)

    assert response.status_code == 404
    assert not _membership_exists(engine, frame_id=frame_b_id, subject_id=subject_id)


def test_delete_returns_404_when_frame_belongs_to_different_clip(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """DELETE returns 404 when the frame ID exists but belongs to a different clip."""
    config, engine, clip_a_id = seeded_clip_env
    clip_b_id = _seed_clip_on_first_camera(engine, source_filename="070000.mp4", start_ts=DEFAULT_START_TS + timedelta(hours=1))
    frame_b_id = _seed_frame(engine, clip_id=clip_b_id)
    subject_id = seed_cat_subject(engine)

    with web_test_client(config) as client:
        response = client.delete(f"/clips/{clip_a_id}/frames/{frame_b_id}/subjects/{subject_id}", headers=AUTH_HEADER)

    assert response.status_code == 404


def test_put_returns_409_for_archived_subject(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """PUT against an archived subject returns 409 Conflict; no row is inserted."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine)
    with get_session(engine) as session:
        subj = session.get(Subject, subject_id)
        assert subj is not None
        subj.archived_at = datetime.now(UTC)

    with web_test_client(config) as client:
        response = client.put(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 409
    assert not _membership_exists(engine, frame_id=frame_id, subject_id=subject_id)


def test_delete_allows_retraction_of_archived_subject_membership(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """DELETE against an archived subject's existing membership succeeds (204); row is removed."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine)
    with get_session(engine) as session:
        session.add(ClipFrameSubject(clip_frame_id=frame_id, subject_id=subject_id))
        subj = session.get(Subject, subject_id)
        assert subj is not None
        subj.archived_at = datetime.now(UTC)

    with web_test_client(config) as client:
        response = client.delete(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 204
    assert not _membership_exists(engine, frame_id=frame_id, subject_id=subject_id)


def test_put_emits_clip_frame_subject_added_log_event(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A state-changing PUT emits a ``clip_frame_subject_added`` record with clip_id, frame_id, and slug."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine, slug="felix")
    logging.getLogger(ROUTES_LOGGER).disabled = False

    with caplog.at_level(logging.INFO, logger=ROUTES_LOGGER), web_test_client(config) as client:
        _ = client.put(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )

    matching = [r for r in caplog.records if r.getMessage() == "clip_frame_subject_added"]
    assert len(matching) == 1, f"expected exactly one clip_frame_subject_added record; got {[r.getMessage() for r in caplog.records]}"
    assert getattr(matching[0], "clip_id", None) == clip_id
    assert getattr(matching[0], "frame_id", None) == frame_id
    assert getattr(matching[0], "subject_slug", None) == "felix"


def test_put_no_log_event_on_idempotent_noop(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A no-op re-PUT (row already exists) must NOT emit a ``clip_frame_subject_added`` event."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine)
    logging.getLogger(ROUTES_LOGGER).disabled = False

    with web_test_client(config) as client:
        _ = client.put(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=ROUTES_LOGGER):
            _ = client.put(
                f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
                headers=AUTH_HEADER,
            )

    assert not any(r.getMessage() == "clip_frame_subject_added" for r in caplog.records)


def test_delete_emits_clip_frame_subject_removed_log_event(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A state-changing DELETE emits a ``clip_frame_subject_removed`` record with clip_id, frame_id, and slug."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine, slug="felix")
    with get_session(engine) as session:
        session.add(ClipFrameSubject(clip_frame_id=frame_id, subject_id=subject_id))
    logging.getLogger(ROUTES_LOGGER).disabled = False

    with caplog.at_level(logging.INFO, logger=ROUTES_LOGGER), web_test_client(config) as client:
        _ = client.delete(
            f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
            headers=AUTH_HEADER,
        )

    matching = [r for r in caplog.records if r.getMessage() == "clip_frame_subject_removed"]
    assert len(matching) == 1, f"expected exactly one clip_frame_subject_removed record; got {[r.getMessage() for r in caplog.records]}"
    assert getattr(matching[0], "clip_id", None) == clip_id
    assert getattr(matching[0], "frame_id", None) == frame_id
    assert getattr(matching[0], "subject_slug", None) == "felix"


def test_delete_no_log_event_on_idempotent_noop(
    seeded_clip_env: tuple[Config, Engine, int],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A no-op re-DELETE (row already absent) must NOT emit a ``clip_frame_subject_removed`` event."""
    config, engine, clip_id = seeded_clip_env
    frame_id = _seed_frame(engine, clip_id=clip_id)
    subject_id = seed_cat_subject(engine)
    logging.getLogger(ROUTES_LOGGER).disabled = False

    with web_test_client(config) as client:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=ROUTES_LOGGER):
            _ = client.delete(
                f"/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}",
                headers=AUTH_HEADER,
            )

    assert not any(r.getMessage() == "clip_frame_subject_removed" for r in caplog.records)
