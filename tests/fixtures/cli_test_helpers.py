"""Shared helpers for ``test_cli*.py``.

Lives under ``tests/fixtures/`` because pytest puts that directory on ``pythonpath`` (see
``[tool.pytest.ini_options].pythonpath`` in ``pyproject.toml``); both ``tests/unit/test_cli.py``
and ``tests/unit/test_cli_reanalyze.py`` import directly from it. The split keeps each test
module under pylint's ``too-many-lines`` cap without duplicating seed-data builders.
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
    """Materialize the SQLAlchemy schema at ``<internal_root>/cat_watcher.sqlite``.

    The CLI handlers open their own engine via ``_open_engine(config)`` against this exact path; we
    create + dispose a one-shot engine here so the schema exists before the handler runs without
    leaving any pooled connections alive.
    """
    engine = create_engine(f"sqlite:///{internal_root / _DB_FILENAME}")
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()


def config_with_dirs(tmp_path: Path, make_config: Callable[..., Config]) -> Config:
    """Build a Config wired to fresh internal/storage roots under ``tmp_path``."""
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    internal_root.mkdir()
    storage_root.mkdir()
    return make_config(internal_root, storage_root)


def make_handler_args(config_path: Path | None = None, **overrides: object) -> _ParsedArgs:
    """Construct a typed ``_ParsedArgs`` with ``config`` pre-set; per-handler fields via kwargs."""
    args = _ParsedArgs()
    args.config = config_path
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@contextmanager
def open_seed_session(config: Config) -> Generator[Session]:
    """Yield a writer Session against the live test DB; engine is disposed on exit.

    Tests use this to seed rows BEFORE entering a CLI handler. Handlers create their own engine
    against the same SQLite file; SQLite WAL mode lets the two engines coexist briefly. Disposing
    here keeps the seed connection from holding a write lock when the handler opens its session.
    """
    engine = create_engine(f"sqlite:///{config.internal_root / _DB_FILENAME}")
    try:
        with get_session(engine) as session:
            yield session
    finally:
        engine.dispose()


def seed_camera(config: Config, **overrides: object) -> int:
    """Insert one ``Camera`` row and return its id. Defaults match a single ``pantry`` camera."""
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
    """Insert one ``Clip`` row with deterministic defaults; return its id.

    ``source_filename`` defaults to ``YYYYMMDD-HHMMSSffffff.mp4`` derived from ``start_ts`` so a
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
    """Read a clip back from the DB outside the handler's session, detached on return."""
    engine = create_engine(f"sqlite:///{config.internal_root / _DB_FILENAME}")
    try:
        with get_session(engine) as session:
            clip = session.scalar(select(Clip).where(Clip.id == clip_id))
            assert clip is not None
            session.expunge(clip)
            return clip
    finally:
        engine.dispose()
