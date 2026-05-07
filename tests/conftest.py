"""Shared pytest fixtures for the cat-watcher test suite."""

import httpxyz_compat  # noqa: F401, I001  # pyright: ignore[reportUnusedImport]  # side-effect import; must run first; see module docstring.

from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from make_clip import make_clip
from pydantic import SecretStr

from cat_watcher.config import (
    AlertConfig,
    BackupConfig,
    CameraConfig,
    CameraSecrets,
    Config,
    DetectorConfig,
    EmailRulesConfig,
    EmailSecrets,
    MacOsRulesConfig,
    PollerConfig,
    RetentionConfig,
    StorageConfig,
    WebAuth,
    WebConfig,
)
from cat_watcher.db import Base, Camera, Clip, PollStatus, create_engine, get_session
from cat_watcher.web.app import build_app

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from contextlib import AbstractContextManager
    from datetime import datetime
    from pathlib import Path

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session


@pytest.fixture(scope="session")
def synthetic_clip_path() -> Path:
    """A 2-second H.264 MP4 synthesized via ffmpeg's testsrc filter; built once per session."""
    return make_clip()


@pytest.fixture
def db_engine(tmp_path: Path) -> Iterator[Engine]:
    """Per-test file-backed SQLite engine with the schema materialized.

    File-backed (not ``:memory:``) because some tests rely on the WAL-mode PRAGMA, which SQLite
    cannot enable on in-memory databases. The engine is disposed in teardown so SQLAlchemy releases
    its sqlite3 handles before pytest's ``filterwarnings = error`` escalates a ``ResourceWarning``
    from a GC-finalized connection.
    """
    db_path = tmp_path / "test.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


_DEFAULT_CAMERAS: tuple[CameraConfig, ...] = (
    CameraConfig(name="pantry", display_name="Pantry", host="cam.example.com", port=80, timezone="UTC"),
)


@pytest.fixture
def storage_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Materialize ``(internal_root, storage_root)`` under ``tmp_path`` and return both paths.

    Mirrors the directory layout every agent expects in production: separate roots for the local
    SSD-backed DB / logs (``internal_root``) and the bulk-storage drive (``storage_root``). Tests
    that exercise ``run_alerts_tick`` / ``run_tick`` / ``build_app`` / etc. need both pre-created
    so :class:`cat_watcher.config.Config` validation and ``ensure_storage_layout`` succeed.
    """
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    internal_root.mkdir()
    storage_root.mkdir()
    return internal_root, storage_root


@pytest.fixture
def make_config() -> Callable[..., Config]:
    """Factory for a minimal valid :class:`Config` populated with one ``pantry`` camera.

    The default host/port/timezone match what the e2e tests expect (respx mocks
    ``cam.example.com:80``; the UTC timezone makes camera-local clip-path computation
    deterministic). Tests that need a different camera topology (e.g. multi-camera
    error-isolation) pass ``cameras=[...]`` explicitly.
    """

    def _build(internal_root: Path, storage_root: Path, *, cameras: list[CameraConfig] | None = None) -> Config:
        return Config(
            internal_root=internal_root,
            storage_root=storage_root,
            cameras=cameras if cameras is not None else list(_DEFAULT_CAMERAS),
            detector=DetectorConfig(),
            alerts=AlertConfig(email=EmailRulesConfig(), macos=MacOsRulesConfig()),
            web=WebConfig(public_url="http://localhost:8000"),
            storage=StorageConfig(),
            retention=RetentionConfig(),
            backup=BackupConfig(),
            poller=PollerConfig(),
            camera_secrets=CameraSecrets(username="u", password=SecretStr("p")),
            email=EmailSecrets(
                gmail_user="alerts@example.com",
                gmail_app_password=SecretStr("pw"),
                alert_to_addresses=("me@example.com",),
            ),
            web_auth=WebAuth(username="admin", password=SecretStr("pw")),
        )

    return _build


@pytest.fixture
def seed_camera() -> Callable[..., int]:
    """Factory: insert a ``Camera`` row with sensible defaults; kwargs override individual fields.

    Returned callable signature: ``(engine: Engine, **overrides: object) -> int``. Defaults match
    the ``pantry`` camera in :data:`_DEFAULT_CAMERAS` (``host="cam.example.com"``,
    ``poll_status=OK``) so the row is consistent with the rest of the test infrastructure. Tests
    that need different state (a stale ``last_polled_at``, a non-OK ``poll_status``, etc.) pass
    those as kwargs.
    """

    def _seed(engine: Engine, **overrides: object) -> int:
        defaults: dict[str, object] = {
            "name": "pantry",
            "display_name": "Pantry",
            "host": "cam.example.com",
            "poll_status": PollStatus.OK,
        }
        defaults.update(overrides)
        cam = Camera(**defaults)
        with get_session(engine) as session:
            session.add(cam)
            session.flush()
            return cam.id

    return _seed


@pytest.fixture
def seed_clip() -> Callable[..., None]:
    """Factory: insert a ``Clip`` row with deterministic paths derived from ``start_ts``.

    Callable signature: ``(engine, *, camera_id, start_ts, has_cat, manual_has_cat=None) -> None``.
    File paths use the ``HHMMSS`` of ``start_ts`` so callers seeding multiple clips just vary
    ``start_ts`` to keep ``(camera_id, source_filename)`` unique.
    """

    def _seed(
        engine: Engine,
        *,
        camera_id: int,
        start_ts: datetime,
        has_cat: bool,
        manual_has_cat: bool | None = None,
    ) -> None:
        fname = f"{start_ts.strftime('%H%M%S')}.mp4"
        date_dir = start_ts.strftime("%Y-%m-%d")
        clip = Clip(
            camera_id=camera_id,
            source_filename=fname,
            start_ts=start_ts,
            end_ts=start_ts + timedelta(seconds=2),
            duration_seconds=2.0,
            file_path=f"clips/pantry/{date_dir}/{fname}",
            thumb_path=f"thumbs/pantry/{date_dir}/{fname}.jpg",
            file_size_bytes=10,
            has_cat=has_cat,
            manual_has_cat=manual_has_cat,
            detector_version="test@deadbeef",
            ingested_at=start_ts,
        )
        with get_session(engine) as session:
            session.add(clip)

    return _seed


@pytest.fixture
def web_test_client() -> Callable[[Config], AbstractContextManager[TestClient]]:
    """Factory: materialize the SQLite schema under ``config.internal_root`` and yield a ``TestClient``.

    Calling ``web_test_client(config)`` runs ``Base.metadata.create_all`` **eagerly** so the test
    can seed rows / files between the call and the ``with``-statement entry. The returned context
    manager only runs the FastAPI lifespan (which spawns the heartbeat task) — that's what needs
    the start-and-stop bracket. SQLite WAL mode lets the test process keep read-writing the same
    file concurrently with the app's session.

    Usage::

        config = make_config(internal_root, storage_root)
        # ... seed the DB / disk via fresh engines pointed at config.internal_root ...
        with web_test_client(config) as client:
            response = client.get(...)
    """

    def _factory(config: Config) -> AbstractContextManager[TestClient]:
        engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
        Base.metadata.create_all(engine)
        engine.dispose()
        app = build_app(config)

        @contextmanager
        def _enter_lifespan() -> Generator[TestClient]:
            with TestClient(app) as client:
                yield client

        return _enter_lifespan()

    return _factory


@pytest.fixture
def db_session_factory() -> Callable[[Path], AbstractContextManager[Session]]:
    """Factory: open a Session over the test DB at ``internal_root``, materialize schema, dispose on exit.

    Use when an integration test needs to seed rows BEFORE entering ``web_test_client``'s lifespan
    — the lifespan opens its own engine on the same SQLite file, so seeding through a separate,
    short-lived engine and disposing it before the app boots avoids cross-engine connection
    interference. SQLite WAL mode (set on the engine by :mod:`cat_watcher.db`) keeps subsequent
    reader/writer engines compatible.

    Usage::

        with db_session_factory(internal_root) as session:
            cam = Camera(name="pantry", display_name="Pantry", host="cam.example.com")
            session.add(cam)
            session.flush()
            cam_id = cam.id
    """

    @contextmanager
    def _open(internal_root: Path) -> Generator[Session]:
        engine = create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")
        try:
            Base.metadata.create_all(engine)
            with get_session(engine) as session:
                yield session
        finally:
            engine.dispose()

    return _open
