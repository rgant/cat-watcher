"""Shared pytest fixtures for the cat-watcher test suite."""

from typing import TYPE_CHECKING

import pytest
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
from cat_watcher.db import Base, create_engine

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from sqlalchemy.engine import Engine


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
