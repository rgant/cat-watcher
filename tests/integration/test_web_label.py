"""Integration tests for the cat-watcher manual-labeling routes (Task 22).

Covers ``POST /clips/{id}/label`` (sets ``manual_has_cat`` / ``manual_label_notes`` /
``manual_label_at`` and redirects 303 to the detail page) and ``DELETE /clips/{id}/label``
(clears all three columns back to ``NULL``). Both endpoints share the auth chain — the auth
behavior itself is exhaustively covered by ``test_web_health.py``, so this module just attaches a
constant ``Authorization`` header to every request.

These tests do not exercise on-disk media at all — labeling only mutates the ``clips`` row — so
seeding bypasses the ``storage_root`` filesystem layout used by ``test_web_clips.py``.
"""

import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING

from sqlalchemy import select

from cat_watcher.db import Base, Camera, Clip, create_engine, get_session

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from fastapi.testclient import TestClient

    from cat_watcher.config import Config


_AUTH_HEADER = {"Authorization": f"Basic {base64.b64encode(b'admin:pw').decode()}"}
_DEFAULT_START_TS = datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)


def _seed_clip(internal_root: Path) -> int:
    """Seed one camera + one clip in the test DB and return the clip's id.

    Materializes the schema first so the test can run without depending on Alembic migrations
    (the ``web_test_client`` fixture also calls ``create_all``, but seeding through a fresh engine
    must not race with the app's lifespan).
    """
    engine = create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")
    try:
        Base.metadata.create_all(engine)
        with get_session(engine) as session:
            cam = Camera(name="pantry", display_name="Pantry", host="cam.example.com")
            session.add(cam)
            session.flush()
            clip = Clip(
                camera_id=cam.id,
                source_filename="064704.mp4",
                start_ts=_DEFAULT_START_TS,
                end_ts=_DEFAULT_START_TS + timedelta(seconds=30),
                duration_seconds=30.0,
                file_path="clips/pantry/2026-05-01/064704.mp4",
                thumb_path="thumbs/pantry/2026-05-01/064704.jpg",
                file_size_bytes=1024,
                has_cat=False,
                detector_version="yolov11n@deadbeef",
                ingested_at=datetime.now(UTC),
            )
            session.add(clip)
            session.flush()
            return clip.id
    finally:
        engine.dispose()


def _read_clip(internal_root: Path, clip_id: int) -> Clip:
    """Read ``clip_id`` back from the DB outside any session and return the detached instance."""
    engine = create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            clip = session.scalar(select(Clip).where(Clip.id == clip_id))
            assert clip is not None
            session.expunge(clip)
            return clip
    finally:
        engine.dispose()


def test_label_post_sets_manual_has_cat_true_and_notes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """``POST /clips/{id}/label`` with ``has_cat=true`` writes the three manual columns."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    clip_id = _seed_clip(internal_root)

    with web_test_client(config) as client:
        response = client.post(
            f"/clips/{clip_id}/label",
            data={"has_cat": "true", "notes": "Rufus paws"},
            headers=_AUTH_HEADER,
            follow_redirects=False,
        )

    assert response.status_code == 303
    clip = _read_clip(internal_root, clip_id)
    assert clip.manual_has_cat is True
    assert clip.manual_label_notes == "Rufus paws"
    assert clip.manual_label_at is not None


def test_label_post_sets_manual_has_cat_false(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """``has_cat=false`` persists as ``False`` (not ``NULL``) so the override is recoverable."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    clip_id = _seed_clip(internal_root)

    with web_test_client(config) as client:
        response = client.post(
            f"/clips/{clip_id}/label",
            data={"has_cat": "false", "notes": ""},
            headers=_AUTH_HEADER,
            follow_redirects=False,
        )

    assert response.status_code == 303
    clip = _read_clip(internal_root, clip_id)
    assert clip.manual_has_cat is False
    # Empty form input collapses to NULL — distinct from a non-empty notes string.
    assert clip.manual_label_notes is None
    assert clip.manual_label_at is not None


def test_label_post_redirects_back_to_clip_detail(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The ``303 See Other`` ``Location`` header points back at ``/clips/{id}``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    clip_id = _seed_clip(internal_root)

    with web_test_client(config) as client:
        response = client.post(
            f"/clips/{clip_id}/label",
            data={"has_cat": "true", "notes": "x"},
            headers=_AUTH_HEADER,
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"].endswith(f"/clips/{clip_id}")


def test_label_post_returns_404_for_unknown_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Posting to an id that doesn't exist yields 404, not 500."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.post(
            "/clips/9999/label",
            data={"has_cat": "true", "notes": ""},
            headers=_AUTH_HEADER,
            follow_redirects=False,
        )

    assert response.status_code == 404


def test_label_delete_clears_manual_fields(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A DELETE after a POST returns ``manual_has_cat`` / ``_notes`` / ``_at`` to ``NULL``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    clip_id = _seed_clip(internal_root)

    with web_test_client(config) as client:
        post_resp = client.post(
            f"/clips/{clip_id}/label",
            data={"has_cat": "true", "notes": "stash"},
            headers=_AUTH_HEADER,
            follow_redirects=False,
        )
        assert post_resp.status_code == 303
        delete_resp = client.delete(f"/clips/{clip_id}/label", headers=_AUTH_HEADER)

    assert delete_resp.status_code == 204
    clip = _read_clip(internal_root, clip_id)
    assert clip.manual_has_cat is None
    assert clip.manual_label_notes is None
    assert clip.manual_label_at is None


def test_label_post_overwrites_existing_label(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A second POST replaces the first label's columns and re-stamps ``manual_label_at``.

    Documents the real user workflow (label wrong → fix it) and proves the route does an UPDATE,
    not an INSERT-or-error.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    clip_id = _seed_clip(internal_root)

    with web_test_client(config) as client:
        first = client.post(
            f"/clips/{clip_id}/label",
            data={"has_cat": "true", "notes": "first"},
            headers=_AUTH_HEADER,
            follow_redirects=False,
        )
        assert first.status_code == 303
        first_at = _read_clip(internal_root, clip_id).manual_label_at
        assert first_at is not None

        second = client.post(
            f"/clips/{clip_id}/label",
            data={"has_cat": "false", "notes": "second"},
            headers=_AUTH_HEADER,
            follow_redirects=False,
        )
        assert second.status_code == 303

    clip = _read_clip(internal_root, clip_id)
    assert clip.manual_has_cat is False
    assert clip.manual_label_notes == "second"
    assert clip.manual_label_at is not None
    # Use ``>=`` rather than ``>`` so a coarse system clock (datetime.now resolution can collapse
    # two same-microsecond reads on some platforms) doesn't make the test flaky; the meaningful
    # signal is "the row was rewritten", which the column-value asserts above already establish.
    assert clip.manual_label_at >= first_at


def test_label_post_missing_has_cat_returns_422(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A POST that omits the required ``has_cat`` field yields 422 (FastAPI form validation).

    Documents the contract: the Annotated[bool, Form()] declaration on ``set_label`` makes
    ``has_cat`` mandatory; the route never sees the request and the clip row is untouched.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    clip_id = _seed_clip(internal_root)

    with web_test_client(config) as client:
        response = client.post(
            f"/clips/{clip_id}/label",
            data={"notes": "no cat field"},
            headers=_AUTH_HEADER,
            follow_redirects=False,
        )

    assert response.status_code == 422
    clip = _read_clip(internal_root, clip_id)
    assert clip.manual_has_cat is None
    assert clip.manual_label_notes is None
    assert clip.manual_label_at is None
