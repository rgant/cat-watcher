"""Integration tests for the cat-watcher clip-management routes (Task 21).

Covers ``GET /clips``, ``GET /clips/{id}``, ``GET /media/clip/{id}.mp4`` (HTTP byte-Range), and
``GET /media/thumb/{id}.jpg`` — plus the storage-offline degradation path (spec §4.13). The
manual-label form is asserted on (form HTML lives on the detail page) but its POST/DELETE endpoints
land in Task 22.

Tests share the project-standard fixtures (``storage_dirs``, ``make_config``, ``web_test_client``)
from ``tests/conftest.py``. The auth path itself is exhaustively covered by ``test_web_health.py``;
this module just attaches a constant ``Authorization`` header to every request.
"""

import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import desc, select

from cat_watcher.db import Camera, Clip, ClipFrame

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from fastapi.testclient import TestClient
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


_AUTH_HEADER = {"Authorization": f"Basic {base64.b64encode(b'admin:pw').decode()}"}
_DEFAULT_START_TS = datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)


def _seed_camera_and_clip(  # noqa: PLR0913  # test-fixture builder; bundling args at the call-site is noisier
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    storage_root: Path,
    clip_bytes: bytes = b"\x00" * 1024,
    thumb_bytes: bytes = b"\xff\xd8\xff\xe0",
    start_ts: datetime = _DEFAULT_START_TS,
    has_cat: bool = True,
    write_files: bool = True,
    camera_name: str = "pantry",
    camera_display_name: str = "Pantry Litter Box",
) -> tuple[int, int]:
    """Seed one camera + one clip; optionally write the clip + thumb bytes to disk.

    Returns ``(camera_id, clip_id)``. Setting ``write_files=False`` simulates the storage-offline
    degradation case: DB row exists, files don't.
    """
    rel_clip, rel_thumb = _relative_paths_for(camera_name, start_ts)
    if write_files:
        _materialize_clip_files(storage_root, rel_clip, clip_bytes, rel_thumb, thumb_bytes)

    with db_session_factory(internal_root) as session:
        cam = Camera(name=camera_name, display_name=camera_display_name, host="cam.example.com")
        session.add(cam)
        session.flush()
        clip = _build_clip(cam.id, rel_clip, rel_thumb, start_ts, len(clip_bytes), has_cat=has_cat)
        session.add(clip)
        session.flush()
        return cam.id, clip.id


def _relative_paths_for(camera_name: str, start_ts: datetime) -> tuple[str, str]:
    """Return ``(rel_clip_path, rel_thumb_path)`` derived from the camera + UTC timestamp."""
    fname = start_ts.strftime("%H%M%S")
    date_dir = start_ts.strftime("%Y-%m-%d")
    return (
        f"clips/{camera_name}/{date_dir}/{fname}.mp4",
        f"thumbs/{camera_name}/{date_dir}/{fname}.jpg",
    )


def _materialize_clip_files(storage_root: Path, rel_clip: str, clip_bytes: bytes, rel_thumb: str, thumb_bytes: bytes) -> None:
    """Write the clip + thumb bytes to disk under ``storage_root``, creating parent dirs."""
    for rel, payload in ((rel_clip, clip_bytes), (rel_thumb, thumb_bytes)):
        full = storage_root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        _ = full.write_bytes(payload)


def _build_clip(  # noqa: PLR0913  # constructor wrapper; flat kwargs map 1:1 to ORM columns
    camera_id: int,
    rel_clip: str,
    rel_thumb: str,
    start_ts: datetime,
    file_size: int,
    *,
    has_cat: bool,
) -> Clip:
    """Build a populated :class:`Clip` ORM instance with the test-suite default detection scores."""
    return Clip(
        camera_id=camera_id,
        source_filename=rel_clip.rsplit("/", 1)[-1],
        start_ts=start_ts,
        end_ts=start_ts + timedelta(seconds=114),
        duration_seconds=114.0,
        file_path=rel_clip,
        thumb_path=rel_thumb,
        file_size_bytes=file_size,
        has_cat=has_cat,
        max_score=0.92,
        frames_sampled=5,
        frames_with_cat=4,
        detector_version="yolov11n@deadbeef",
        ingested_at=datetime.now(UTC),
    )


def _seed_clip_frame(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    storage_root: Path,
    clip_id: int,
    frame_bytes: bytes | None = b"\xff\xd8\xff\xe0frame-bytes",
) -> int:
    """Seed one ``ClipFrame`` row tied to ``clip_id`` and optionally write its JPEG to disk.

    Returns the new ``ClipFrame.id``. ``frame_bytes=None`` simulates the row-without-file drift the
    410 path covers. The relpath matches the production layout
    (``thumbs/<cam>/<YYYY-MM-DD>/<HHMMSS>/<NN>.jpg``) so filesystem-coupled regressions surface here
    instead of getting masked by a synthetic test path. The ordinal is pinned to 0 — sufficient for
    the route's behavioral coverage; tests that care about multi-frame layout build their own.
    """
    rel_thumb = _frame_relpath_from_clip(db_session_factory, internal_root=internal_root, clip_id=clip_id)
    if frame_bytes is not None:
        full = storage_root / rel_thumb
        full.parent.mkdir(parents=True, exist_ok=True)
        _ = full.write_bytes(frame_bytes)

    with db_session_factory(internal_root) as session:
        frame = ClipFrame(clip_id=clip_id, ordinal=0, t_offset_seconds=0.0, score=0.91, thumb_path=rel_thumb)
        session.add(frame)
        session.flush()
        return frame.id


def _frame_relpath_from_clip(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    clip_id: int,
) -> str:
    """Compute the per-frame relpath ``thumbs/<cam>/<date>/<HHMMSS>/00.jpg`` from the seeded clip."""
    with db_session_factory(internal_root) as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        camera = session.get(Camera, clip.camera_id)
        assert camera is not None
        date_dir = clip.start_ts.strftime("%Y-%m-%d")
        hhmmss = clip.start_ts.strftime("%H%M%S")
        return f"thumbs/{camera.name}/{date_dir}/{hhmmss}/00.jpg"


def _seed_clip_frame_at(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    clip_id: int,
    spec: tuple[int, float],
    score: float = 0.5,
) -> int:
    """Seed one ``ClipFrame`` row at ``(ordinal, t_offset_seconds)`` with ``score``; no JPEG written.

    The contact-sheet tests don't render the JPEG bytes (the ``<img>`` URL is what matters), so we
    skip the on-disk write that :func:`_seed_clip_frame` performs and keep ``thumb_path`` distinct
    per ordinal so a regression in the per-frame relpath would still surface.
    """
    ordinal, t_offset_seconds = spec
    rel_thumb = f"thumbs/clip-{clip_id}/{ordinal:02d}.jpg"
    with db_session_factory(internal_root) as session:
        frame = ClipFrame(clip_id=clip_id, ordinal=ordinal, t_offset_seconds=t_offset_seconds, score=score, thumb_path=rel_thumb)
        session.add(frame)
        session.flush()
        return frame.id


def _seed_extra_clip(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    internal_root: Path,
    *,
    source_filename: str,
    start_ts: datetime,
    has_cat: bool,
) -> None:
    """Add a second clip on the same (already-seeded) camera so filter tests can compare results."""
    with db_session_factory(internal_root) as session:
        cam = session.scalar(select(Camera))
        assert cam is not None
        date_dir = start_ts.strftime("%Y-%m-%d")
        session.add(
            Clip(
                camera_id=cam.id,
                source_filename=source_filename,
                start_ts=start_ts,
                end_ts=start_ts + timedelta(seconds=30),
                duration_seconds=30.0,
                file_path=f"clips/{cam.name}/{date_dir}/{source_filename}",
                thumb_path=f"thumbs/{cam.name}/{date_dir}/{source_filename}.jpg",
                file_size_bytes=512,
                has_cat=has_cat,
                detector_version="yolov11n@deadbeef",
                ingested_at=datetime.now(UTC),
            ),
        )


def test_clips_list_returns_200_and_renders_camera_display_name(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``GET /clips`` renders the camera's display name for each row."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with web_test_client(config) as client:
        response = client.get("/clips", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "Pantry Litter Box" in response.text


def test_clips_list_renders_start_ts_in_display_timezone(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Each row's ``Start`` cell renders the clip start in ``web.display_timezone``, matching the
    OSD time burned into the video. The ``<time datetime="…">`` attribute keeps UTC ISO for HTML5
    semantics; only the visible text is localized.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # 18:47:04 UTC on 2026-05-01 → 14:47:04 EDT (default display_timezone is America/New_York).
    start_ts = datetime(2026, 5, 1, 18, 47, 4, tzinfo=UTC)
    _ = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=start_ts,
    )

    with web_test_client(config) as client:
        response = client.get("/clips", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "2026-05-01 14:47:04 EDT" in response.text
    assert 'datetime="2026-05-01T18:47:04+00:00"' in response.text


def test_clips_list_filter_by_camera_name(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``?camera=<name>`` restricts the rendered list to clips for that camera."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, pantry_clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="pantry",
        camera_display_name="Pantry",
    )
    _, garage_clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="garage",
        camera_display_name="Garage Watch",
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
    )

    with web_test_client(config) as client:
        response = client.get("/clips?camera=pantry", headers=_AUTH_HEADER)

    assert response.status_code == 200
    # Both display names live in the filter <select> regardless of the active filter, so the
    # actual signal is which clip-detail links land in the table body.
    assert f"/clips/{pantry_clip_id}" in response.text
    assert f"/clips/{garage_clip_id}" not in response.text


def test_clips_list_filter_by_has_cat_true(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``?has_cat=true`` returns only cat-positive clips."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, has_cat=True)
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="070000.mp4",
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        has_cat=False,
    )

    with web_test_client(config) as client:
        response = client.get("/clips?has_cat=true", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "064704" in response.text  # the cat-positive clip's filename
    assert "070000" not in response.text  # the no-cat clip should be filtered out


def test_clips_list_filter_by_date_str(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``?date_str=YYYY-MM-DD`` restricts to clips whose ``start_ts`` falls on that UTC day."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC),
    )
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="100000.mp4",
        start_ts=datetime(2026, 5, 2, 10, 0, 0, tzinfo=UTC),
        has_cat=False,
    )

    with web_test_client(config) as client:
        response = client.get("/clips?date_str=2026-05-02", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "100000" in response.text
    assert "064704" not in response.text


def test_clips_list_returns_200_with_no_clips(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """An empty database renders the page (no rows) without raising."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.get("/clips", headers=_AUTH_HEADER)

    assert response.status_code == 200


def test_clip_detail_renders_video_player_targeting_media_route(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``GET /clips/{id}`` renders a ``<video>`` element pointing at the media route."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert f"/media/clip/{clip_id}.mp4" in response.text
    assert "0.92" in response.text  # max_score
    assert "yolov11n@deadbeef" in response.text  # detector_version


def test_clip_detail_heading_renders_in_display_timezone(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Heading time-of-day is in ``web.display_timezone``, matching the camera-OSD time burned into
    the video. The ``<time datetime="…">`` attribute keeps UTC ISO for HTML5 semantics, but the
    visible text uses the configured display zone — ``Clip.start_ts`` is stored UTC, so a raw
    ``isoformat()`` would disagree with the on-screen video timestamp by the tz offset.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # 18:47:04 UTC on 2026-05-01 → 14:47:04 EDT (default display_timezone is America/New_York,
    # which is UTC-4 in May).
    start_ts = datetime(2026, 5, 1, 18, 47, 4, tzinfo=UTC)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=start_ts,
    )

    with web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "2026-05-01 14:47:04 EDT" in response.text
    assert 'datetime="2026-05-01T18:47:04+00:00"' in response.text


def test_clip_detail_renders_manual_label_form(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``GET /clips/{id}`` renders a label form posting to the ``set_label`` endpoint."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=_AUTH_HEADER)

    assert response.status_code == 200
    # The action is rendered as an absolute URL by ``url_for(...)`` (Task 22 cleanup); pin the tail
    # of the path so the test stays robust to host/port changes in the test client.
    assert f'/clips/{clip_id}/label"' in response.text
    # The form must offer both has_cat values plus a notes field per Task 22's contract.
    assert 'name="has_cat"' in response.text
    assert 'name="notes"' in response.text


def test_clip_detail_returns_404_for_unknown_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A nonexistent clip id yields ``404`` (not a 500)."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.get("/clips/9999", headers=_AUTH_HEADER)

    assert response.status_code == 404


def test_clip_detail_renders_prev_next_navigation(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Detail page links to the chronologically-newer (``← Newer``) and older (``Older →``) clips.

    Three clips at distinct timestamps; visit the middle one and assert both neighbors are linked
    by id. Pin the rel attributes so a regression that swaps prev/next surfaces here.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, oldest_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="070000.mp4",
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        has_cat=True,
    )
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="080000.mp4",
        start_ts=datetime(2026, 5, 1, 8, 0, 0, tzinfo=UTC),
        has_cat=True,
    )
    with db_session_factory(internal_root) as session:
        rows = list(session.scalars(select(Clip).order_by(Clip.start_ts.asc())).all())
    middle_id = rows[1].id
    newest_id = rows[2].id

    with web_test_client(config) as client:
        response = client.get(f"/clips/{middle_id}", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert f'href="http://testserver/clips/{newest_id}" rel="prev"' in response.text
    assert f'href="http://testserver/clips/{oldest_id}" rel="next"' in response.text


def test_clip_detail_disables_navigation_at_endpoints(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The newest clip has ``← Newer`` rendered as a disabled span; the oldest clip's ``Older →``
    is the disabled one. Asserts via the ``clip-nav-disabled`` CSS class so a refactor that drops
    the visual cue surfaces here.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, oldest_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="070000.mp4",
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        has_cat=True,
    )
    with db_session_factory(internal_root) as session:
        newest_id = session.scalar(select(Clip.id).order_by(desc(Clip.start_ts)).limit(1))
    assert newest_id is not None

    with web_test_client(config) as client:
        newest_response = client.get(f"/clips/{newest_id}", headers=_AUTH_HEADER)
        oldest_response = client.get(f"/clips/{oldest_id}", headers=_AUTH_HEADER)

    assert '<span class="clip-nav-disabled" aria-disabled="true">← Newer</span>' in newest_response.text
    assert 'rel="next"' in newest_response.text
    assert '<span class="clip-nav-disabled" aria-disabled="true">Older →</span>' in oldest_response.text
    assert 'rel="prev"' in oldest_response.text


def test_media_clip_returns_full_file_when_no_range_header(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A request without ``Range`` returns the entire MP4 with ``200 OK``."""
    payload = b"\x00\x01\x02\x03" * 256  # 1024 bytes, distinct
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, clip_bytes=payload)

    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"].startswith("video/mp4")


def test_media_clip_honors_range_header_returns_206(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``Range: bytes=0-15`` returns ``206`` with the requested 16-byte segment + ``Content-Range``."""
    payload = bytes(range(256)) * 4  # 1024 distinct bytes
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, clip_bytes=payload)

    headers = dict(_AUTH_HEADER)
    headers["Range"] = "bytes=0-15"
    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=headers)

    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 0-15/{len(payload)}"
    assert response.headers["content-length"] == "16"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.content == payload[:16]


def test_media_clip_honors_open_ended_range_header(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``Range: bytes=N-`` (no end) returns from N to EOF."""
    payload = bytes(range(256)) * 4  # 1024 distinct bytes
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, clip_bytes=payload)

    headers = dict(_AUTH_HEADER)
    headers["Range"] = "bytes=512-"
    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=headers)

    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 512-{len(payload) - 1}/{len(payload)}"
    assert response.content == payload[512:]


def test_media_clip_returns_503_when_storage_root_unmounted(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    tmp_path: Path,
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Spec §4.13: when ``storage_root`` is not accessible, ``/media/clip`` returns ``503``.

    Simulated by pointing the config at a non-existent ``storage_root`` directory — the row exists
    but the route's storage probe fails. The 503 response is what the timeline-template's
    ``onerror`` handler uses to decide between rendering the clip thumbnail and falling back to the
    bundled placeholder SVG; the same handler also drives the storage-offline banner.
    """
    internal_root, _ = storage_dirs
    missing_storage = tmp_path / "drive-not-mounted"
    config = make_config(internal_root, missing_storage)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=missing_storage,
        write_files=False,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=_AUTH_HEADER)

    assert response.status_code == 503


def test_media_clip_returns_410_when_file_missing_but_storage_mounted(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """DB row exists, ``storage_root`` is mounted, but the specific file is gone — ``410 Gone``.

    This is data-integrity drift (e.g. retention sweep removed the file but the row hasn't been
    pruned yet). It's distinct from the 503 case because the drive itself is fine — only this one
    resource is unavailable. Returning 410 keeps the operator-visible signal in logs distinct.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        write_files=False,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=_AUTH_HEADER)

    assert response.status_code == 410


def test_media_clip_returns_404_for_unknown_id(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """No row at all → ``404`` (distinct from 410, which means "row exists, file gone")."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.get("/media/clip/9999.mp4", headers=_AUTH_HEADER)

    assert response.status_code == 404


def test_media_thumb_returns_jpeg_bytes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``/media/thumb/{id}.jpg`` returns the on-disk thumbnail with a JPEG content type."""
    payload = b"\xff\xd8\xff\xe0" + b"thumb-bytes"
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        thumb_bytes=payload,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/thumb/{clip_id}.jpg", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"].startswith("image/jpeg")


def test_media_thumb_returns_503_when_storage_unmounted(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    tmp_path: Path,
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Same 503 contract as ``/media/clip`` — drive offline → 503."""
    internal_root, _ = storage_dirs
    missing_storage = tmp_path / "drive-not-mounted"
    config = make_config(internal_root, missing_storage)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=missing_storage,
        write_files=False,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/thumb/{clip_id}.jpg", headers=_AUTH_HEADER)

    assert response.status_code == 503


def test_media_thumb_returns_410_when_thumb_missing(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Same 410 contract as ``/media/clip`` — drive mounted but thumb file gone."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        write_files=False,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/thumb/{clip_id}.jpg", headers=_AUTH_HEADER)

    assert response.status_code == 410


def test_media_frame_returns_jpeg_bytes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``/media/frame/{id}.jpg`` returns the per-frame JPEG with a JPEG content type."""
    payload = b"\xff\xd8\xff\xe0" + b"per-frame-bytes"
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
    )
    frame_id = _seed_clip_frame(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        clip_id=clip_id,
        frame_bytes=payload,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/frame/{frame_id}.jpg", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"].startswith("image/jpeg")


def test_media_frame_returns_404_for_unknown_id(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """No ``ClipFrame`` row → ``404`` (distinct from 410, which means "row exists, file gone")."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.get("/media/frame/9999.jpg", headers=_AUTH_HEADER)

    assert response.status_code == 404


def test_media_frame_returns_503_when_storage_offline(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    tmp_path: Path,
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Same 503 contract as ``/media/thumb`` — drive offline → 503."""
    internal_root, _ = storage_dirs
    missing_storage = tmp_path / "drive-not-mounted"
    config = make_config(internal_root, missing_storage)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=missing_storage,
        write_files=False,
    )
    frame_id = _seed_clip_frame(
        db_session_factory,
        internal_root=internal_root,
        storage_root=missing_storage,
        clip_id=clip_id,
        frame_bytes=None,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/frame/{frame_id}.jpg", headers=_AUTH_HEADER)

    assert response.status_code == 503


def test_media_frame_returns_410_when_file_missing(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Same 410 contract as ``/media/thumb`` — drive mounted but per-frame file gone."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        write_files=False,
    )
    frame_id = _seed_clip_frame(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        clip_id=clip_id,
        frame_bytes=None,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/frame/{frame_id}.jpg", headers=_AUTH_HEADER)

    assert response.status_code == 410


@pytest.mark.parametrize(
    ("range_header", "expected_status"),
    [
        ("bytes=invalid", 400),  # malformed — RFC 7233 § 3.1
        ("0-15", 400),  # missing ``bytes=`` prefix
        ("bytes=100-50", 400),  # start > end (semantically invalid)
        ("bytes=99999-", 416),  # start past EOF — RFC 7233 § 4.4 unsatisfiable
    ],
    ids=["malformed", "missing-prefix", "inverted", "start-past-eof"],
)
def test_media_clip_returns_rfc_correct_status_for_bad_range_headers(  # noqa: PLR0913  # pylint: disable=too-many-positional-arguments  # parametrized: 4 fixtures + 2 parametrize values
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    range_header: str,
    expected_status: int,
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Bad Range headers get RFC 7233 status codes (400 / 416), not 200 or 500.

    Pins the actual behavior of the underlying ``starlette.responses.FileResponse``: malformed
    syntax gets ``400 Bad Request``; ranges that fall entirely past EOF get ``416 Range Not
    Satisfiable``. A regression that swallowed the bad header silently (returning 200 + full
    content) or that crashed (500) would break ``<video>`` clients that handle 416 by retrying
    without ``Range``.
    """
    payload = bytes(range(256)) * 4  # 1024 distinct bytes
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, clip_bytes=payload)

    headers = dict(_AUTH_HEADER)
    headers["Range"] = range_header
    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=headers)

    assert response.status_code == expected_status


def test_clips_list_renders_in_start_ts_descending_order(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``GET /clips`` orders rows by ``start_ts`` descending — newest clips render first.

    Seed three clips out-of-chronological-insert-order; the rendered HTML must show their
    detail-page links in newest-first order regardless of insert sequence. Pins the
    ``ORDER BY start_ts DESC`` clause; a regression dropping it (or flipping to ASC) would
    silently render the list reversed and operators wouldn't notice immediately on a low-volume
    day.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # Insert oldest, newest, middle — different order than the expected render order so the
    # test can't accidentally pass on insertion order alone.
    _, oldest_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _, newest_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 3, 6, 0, 0, tzinfo=UTC),
        camera_name="garage",
        camera_display_name="Garage",
    )
    _, middle_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 2, 6, 0, 0, tzinfo=UTC),
        camera_name="bedroom",
        camera_display_name="Bedroom",
    )

    with web_test_client(config) as client:
        response = client.get("/clips", headers=_AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    positions = [body.find(f"/clips/{cid}") for cid in (newest_id, middle_id, oldest_id)]
    assert all(p > 0 for p in positions), "every clip's detail link must appear in body"
    assert positions == sorted(positions), "rows must render newest-first (start_ts DESC)"


def test_clips_list_filters_compose_with_and_semantics(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """All three filters combined narrow to the intersection — pins AND semantics.

    Seed four clips covering the cross-product of (camera in {pantry, garage}) x (has_cat in
    {true, false}) on May 2; only one matches all three filters
    (``camera=pantry & has_cat=true & date_str=2026-05-02``). A regression that swaps the
    chained ``.where()`` calls for an OR-equivalent (or drops one accidentally) would
    silently broaden the result set.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    target_ts = datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC)
    # The clip that should pass all three filters.
    _, target_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="pantry",
        camera_display_name="Pantry",
        has_cat=True,
        start_ts=target_ts,
    )
    # Right camera + day, wrong has_cat.
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="120100.mp4",
        start_ts=target_ts + timedelta(minutes=1),
        has_cat=False,
    )
    # Wrong camera, right has_cat + day.
    _, garage_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="garage",
        camera_display_name="Garage",
        has_cat=True,
        start_ts=target_ts + timedelta(minutes=2),
    )
    # Right camera + has_cat, wrong day.
    _, wrong_day_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="bedroom",  # extra camera so the seed stays unique
        camera_display_name="Bedroom",
        has_cat=True,
        start_ts=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
    )

    with web_test_client(config) as client:
        response = client.get(
            "/clips?camera=pantry&has_cat=true&date_str=2026-05-02",
            headers=_AUTH_HEADER,
        )

    assert response.status_code == 200
    body = response.text
    assert f"/clips/{target_id}" in body, "the only clip matching all three filters must render"
    assert f"/clips/{garage_id}" not in body, "wrong camera must be excluded"
    assert f"/clips/{wrong_day_id}" not in body, "wrong day must be excluded"
    # The right-camera-wrong-has_cat clip is added via ``_seed_extra_clip`` to the same
    # camera as the target; its ID we don't track but its filename ``120100.mp4`` is unique.
    assert "120100" not in body, "wrong has_cat must be excluded"


def _frame_button_slice(body: str, frame_id: int) -> str:
    """Return ``body`` between the frame's media URL and the next ``</button>``.

    Slicing keeps a neighbour frame's markers from leaking into a substring check on this
    frame's wrapper.
    """
    start = body.find(f"/media/frame/{frame_id}.jpg")
    return body[start : body.find("</button>", start)]


def test_clip_detail_renders_contact_sheet_in_ordinal_order(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Contact sheet renders one keyboard-accessible click-to-seek button per frame, ordinal-asc.

    Frames are inserted with shuffled ordinals so the ordering signal comes from the route's
    relationship-ordered read, not insert order. Scores are mixed above/below the default 0.35
    threshold so the same response covers ordinal ordering, ``data-seek-seconds`` carriage,
    and the threshold-styling cue (``contact-sheet-score-below``) on a single page render.
    """
    internal_root, storage_root = storage_dirs
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    # Insert order 2/0/3/1 is distinct from the expected ordinal-asc render order, so a regression
    # that dropped the relationship's ``order_by`` would surface as out-of-order positions below.
    # Scores: ordinal 0 below threshold (0.10), 2 below (0.20), 1 above (0.50), 3 above (0.90).
    rows: list[tuple[tuple[int, float], float]] = [((2, 10.0), 0.20), ((0, 0.0), 0.10), ((3, 15.0), 0.90), ((1, 5.0), 0.50)]
    ids_by_ordinal: dict[int, int] = {
        spec[0]: _seed_clip_frame_at(db_session_factory, internal_root=internal_root, clip_id=clip_id, spec=spec, score=score)
        for spec, score in rows
    }

    with web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get(f"/clips/{clip_id}", headers=_AUTH_HEADER).text

    assert 'class="contact-sheet"' in body
    assert 'class="contact-sheet-button"' in body
    positions = [body.find(f"/media/frame/{ids_by_ordinal[o]}.jpg") for o in (0, 1, 2, 3)]
    assert all(p > 0 for p in positions), "every frame's media-frame URL must appear in body"
    assert positions == sorted(positions), "contact-sheet must render in ordinal-ascending order"
    for spec, _ in rows:
        assert f'data-seek-seconds="{spec[1]}"' in body
    # Sub-threshold frame (ordinal 0, score 0.10) carries the muted class; above-threshold (ordinal
    # 1, score 0.50) does not.
    assert "contact-sheet-score-below" in _frame_button_slice(body, ids_by_ordinal[0])
    assert "contact-sheet-score-below" not in _frame_button_slice(body, ids_by_ordinal[1])


def test_clip_detail_hides_contact_sheet_for_legacy_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A clip with no ``ClipFrame`` rows must not render the contact-sheet section at all."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert 'class="contact-sheet"' not in response.text
