"""Shared pytest fixtures for the cat-watcher test suite."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from make_clip import make_clip

from cat_watcher.db import Base, create_engine

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.engine import Engine


_CAT_IMAGE = Path(__file__).parent / "fixtures" / "cat_image.jpg"


@pytest.fixture(scope="session")
def synthetic_clip_path() -> Path:
    """A 2-second H.264 MP4 of the calico-kitten fixture image; built once per pytest session."""
    return make_clip(_CAT_IMAGE)


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
