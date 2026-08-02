"""Shared pytest fixtures for the cat-watcher test suite."""

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from db_helpers import (
    apply_clip_label_summary_view,
    build_test_clip,
    seed_camera_and_clip,
    seed_camera_and_clip_with_files,
)
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
from cat_watcher.db import Base, Camera, PollStatus, create_engine, get_session
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
    """Session-scoped synthetic mp4 path; one encode is reused across every test that needs one."""
    return make_clip()


@pytest.fixture
def restore_root_logger() -> Iterator[logging.Logger]:
    """Snapshot and restore the root logger's level + handlers around a test.

    Yields the root logger so tests that exercise ``setup_agent_logging`` (or any other
    handler-attaching code) can assert on it without leaking handler state to other tests.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield root
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


@pytest.fixture
def db_engine(tmp_path: Path) -> Iterator[Engine]:
    """File-backed SQLite engine — WAL-mode PRAGMA cannot be enabled on ``:memory:`` databases.

    Disposed in teardown so SQLAlchemy releases its sqlite3 handles before pytest's
    ``filterwarnings = error`` escalates a ``ResourceWarning`` from a GC-finalized connection.
    """
    db_path = tmp_path / "test.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def alembic_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """File-backed SQLite engine with full schema + views applied via Alembic.

    Use instead of ``db_engine`` when tests touch code that joins through views created by
    migrations (e.g. ``clip_label_summary``).  ``Base.metadata.create_all`` only emits ORM table
    DDL — view DDL lives in migration scripts and is absent without Alembic.
    """
    db_path = tmp_path / "test.sqlite"
    monkeypatch.setenv("CAT_WATCHER_DB_URL", f"sqlite:///{db_path}")
    alembic_cfg = AlembicConfig("alembic.ini")
    alembic_cfg.set_main_option("script_location", "migrations")
    command.upgrade(alembic_cfg, "head")
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        yield engine
    finally:
        engine.dispose()


_DEFAULT_CAMERAS: tuple[CameraConfig, ...] = (
    CameraConfig(name="pantry", display_name="Pantry", host="cam.example.com", port=80, timezone="UTC"),
)


@pytest.fixture
def storage_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Pre-create ``(internal_root, storage_root)`` so ``Config`` validation succeeds.

    Separate roots model production: local SSD-backed DB / logs vs the bulk-storage drive.
    ``ensure_storage_layout`` also needs both to exist before it runs.
    """
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    internal_root.mkdir()
    storage_root.mkdir()
    return internal_root, storage_root


@pytest.fixture
def make_config() -> Callable[..., Config]:
    """Return a Config factory whose default camera matches the respx mocks at ``cam.example.com:80``.

    The UTC timezone keeps camera-local clip-path computation deterministic. Override
    ``cameras=[...]`` for multi-camera topologies.
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
def disable_alert_channels() -> Callable[[Config], Config]:
    """Return a transformer that disables ``alerts.email`` and ``alerts.macos`` on a ``Config``.

    The :class:`AlertConfig` wires both channels enabled by default; tests that exercise the alert
    pipeline rely on :mod:`cat_watcher.notifier`'s ``enabled=False`` short-circuit (returns
    ``ok=True``) to avoid real SMTP / osascript I/O.
    """

    def _apply(base_config: Config) -> Config:
        return base_config.model_copy(
            update={
                "alerts": base_config.alerts.model_copy(
                    update={
                        "email": EmailRulesConfig(enabled=False),
                        "macos": MacOsRulesConfig(enabled=False),
                    },
                ),
            },
        )

    return _apply


@pytest.fixture
def seed_camera() -> Callable[..., int]:
    """Return a Camera-row seeder whose defaults match the ``pantry`` camera in :data:`_DEFAULT_CAMERAS`.

    Matching defaults keep the row consistent with the rest of the test infrastructure.
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
def seed_clip() -> Callable[..., int]:
    """Return a Clip-row seeder that derives file paths from the ``HHMMSS`` of ``start_ts``.

    Callers seeding multiple clips just vary ``start_ts`` to keep ``(camera_id, source_filename)``
    unique.  Returns the new row's ``id`` so callers that need it (e.g. for tagging) can capture it.
    """

    def _seed(
        engine: Engine,
        *,
        camera_id: int,
        start_ts: datetime,
        has_cat: bool,
    ) -> int:
        clip = build_test_clip(camera_id, start_ts=start_ts, has_cat=has_cat)
        with get_session(engine) as session:
            session.add(clip)
        return clip.id

    return _seed


@pytest.fixture
def web_test_client() -> Callable[[Config], AbstractContextManager[TestClient]]:
    """Run ``Base.metadata.create_all`` **eagerly** so tests can seed rows/files before lifespan entry.

    Calling ``web_test_client(config)`` runs the schema creation; the returned context manager only
    runs the FastAPI lifespan (which spawns the heartbeat task). SQLite WAL mode lets the test
    process keep read-writing the same file concurrently with the app's session.

    Use ``alembic_web_test_client`` instead when the test exercises routes that join views defined
    by migrations (e.g. ``clip_label_summary`` on ``/timeline`` and ``/stats``).
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
def alembic_web_test_client() -> Callable[[Config], AbstractContextManager[TestClient]]:
    """Apply ORM tables + view DDL before launching the test app.

    Use instead of ``web_test_client`` when the route under test joins a migration-defined view
    such as ``clip_label_summary``.  ``Base.metadata.create_all`` only emits ORM table DDL — view
    DDL lives in migration scripts, so this fixture adds the view via raw SQL after the table DDL.

    Safe to call after ``db_session_factory`` has already created the ORM tables (create_all is
    idempotent via its default checkfirst=True); the view creation uses ``CREATE VIEW IF NOT EXISTS``
    for the same reason.
    """

    def _factory(config: Config) -> AbstractContextManager[TestClient]:
        engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
        Base.metadata.create_all(engine)
        apply_clip_label_summary_view(engine)
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
    """Return a short-lived Session factory for seeding rows BEFORE entering ``web_test_client``'s lifespan.

    The lifespan opens its own engine on the same SQLite file, so seeding through a separate
    short-lived engine and disposing it before the app boots avoids cross-engine connection
    interference. SQLite WAL mode keeps subsequent reader/writer engines compatible.
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


@pytest.fixture
def seeded_clip_env(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> Iterator[tuple[Config, Engine, int]]:
    """Yield ``(config, engine, clip_id)`` with one seeded Camera/Clip; dispose the engine on teardown.

    Covers web-clip endpoint tests that arrange a single clip and read DB state back through the engine.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    clip_id = seed_camera_and_clip(db_session_factory, internal_root)
    engine = create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")
    try:
        yield config, engine, clip_id
    finally:
        engine.dispose()


@pytest.fixture
def seeded_detail_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> tuple[Config, int]:
    """Provide a config + a file-backed seeded Clip for clip-detail page tests; return (config, clip_id)."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _cam_id, clip_id = seed_camera_and_clip_with_files(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    return config, clip_id
