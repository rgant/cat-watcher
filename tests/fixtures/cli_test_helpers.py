"""Shared helpers for ``test_cli*.py``.

Lives under ``tests/fixtures/`` because pytest puts that directory on ``pythonpath``; both
``tests/unit/test_cli.py`` and ``tests/unit/test_cli_reanalyze.py`` import directly from it.
"""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select

from cat_watcher.__main__ import _ParsedArgs
from cat_watcher.db import Base, Camera, Clip, PollStatus, create_engine, get_session

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


_DB_FILENAME = "cat_watcher.sqlite"


def init_schema(internal_root: Path) -> None:
    """Schema must exist before the CLI handler opens its own engine on the same SQLite file;
    create + dispose a one-shot engine so no pooled connections remain.
    """
    engine = create_engine(f"sqlite:///{internal_root / _DB_FILENAME}")
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()


def config_with_dirs(tmp_path: Path, make_config: Callable[..., Config]) -> Config:
    """Build a Config and create the matching internal/storage roots so it satisfies validation."""
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    internal_root.mkdir()
    storage_root.mkdir()
    return make_config(internal_root, storage_root)


def make_handler_args(config_path: Path | None = None, **overrides: object) -> _ParsedArgs:
    """Build a ``_ParsedArgs`` for handler tests; ``overrides`` mirror argparse field names."""
    args = _ParsedArgs()
    args.config = config_path
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@contextmanager
def open_seed_session(config: Config) -> Generator[Session]:
    """Disposing the engine here keeps the seed connection from holding a write lock when the
    handler opens its own session against the same SQLite file.
    """
    engine = create_engine(f"sqlite:///{config.internal_root / _DB_FILENAME}")
    try:
        with get_session(engine) as session:
            yield session
    finally:
        engine.dispose()


def seed_camera(config: Config, **overrides: object) -> int:
    """Insert a Camera row in its own session so the seed connection's write lock releases before the handler opens one."""
    cam = Camera(
        **{
            "name": "pantry",
            "display_name": "Pantry",
            "host": "cam.example.com",
            "poll_status": PollStatus.OK,
            **overrides,
        },
    )
    with open_seed_session(config) as session:
        session.add(cam)
        session.flush()
        return int(cam.id)


def seed_clip(  # noqa: PLR0913  # pylint: disable=too-many-arguments  # single coherent test-data builder; bundling kwargs into a dataclass would just shadow the ORM constructor
    config: Config,
    *,
    camera_id: int,
    start_ts: datetime | None = None,
    has_cat: bool = False,
    manual_has_cat: bool | None = None,
    analysis_error: str | None = None,
    detector_version: str = "yolov11n@old",
    source_filename: str | None = None,
    file_path: str = "clips/pantry/test.mp4",
    thumb_path: str = "thumbs/pantry/test.jpg",
) -> int:
    """``source_filename`` defaults to ``YYYYMMDD-HHMMSSffffff.mp4`` derived from ``start_ts`` so a
    test seeding multiple clips per camera only needs to vary ``start_ts`` to keep the
    ``(camera_id, source_filename)`` uniqueness constraint satisfied.
    """
    start = start_ts or datetime.now(UTC)
    fname = source_filename or f"{start.strftime('%Y%m%d-%H%M%S%f')}.mp4"
    clip = Clip(
        camera_id=camera_id,
        source_filename=fname,
        start_ts=start,
        end_ts=start + timedelta(seconds=10),
        duration_seconds=10.0,
        file_path=file_path,
        thumb_path=thumb_path,
        file_size_bytes=1024,
        has_cat=has_cat,
        manual_has_cat=manual_has_cat,
        detector_version=detector_version,
        ingested_at=start,
        analysis_error=analysis_error,
    )
    with open_seed_session(config) as session:
        session.add(clip)
        session.flush()
        return int(clip.id)


def read_clip(config: Config, clip_id: int) -> Clip:
    """Read a clip back outside the handler's session; the row is detached on return."""
    engine = create_engine(f"sqlite:///{config.internal_root / _DB_FILENAME}")
    try:
        with get_session(engine) as session:
            clip = session.scalar(select(Clip).where(Clip.id == clip_id))
            assert clip is not None
            session.expunge(clip)
            return clip
    finally:
        engine.dispose()
