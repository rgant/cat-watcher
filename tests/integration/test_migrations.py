"""Integration tests for the Alembic migration set.

Each test points Alembic at a fresh ``tmp_path`` SQLite file via the ``CAT_WATCHER_DB_URL``
test-only override (see :mod:`alembic.env`). Together they verify that the upgrade / downgrade
chain runs cleanly *and* that ``upgrade head`` produces the full set of tables the schema defines,
catching the class of bug where a manually-edited migration drops a table by accident.
"""

from typing import TYPE_CHECKING, cast

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from cat_watcher.db import Base

if TYPE_CHECKING:
    from pathlib import Path


# Tables the schema defines, plus Alembic's bookkeeping table.
EXPECTED_TABLES: frozenset[str] = frozenset(
    {
        "agent_starts",
        "alembic_version",
        "alerts_sent",
        "cameras",
        "clip_frames",
        "clips",
        "heartbeats",
    },
)


@pytest.fixture
def alembic_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Build an Alembic ``Config`` pointed at a tmp_path SQLite via ``CAT_WATCHER_DB_URL``.

    Setting the env var (rather than mutating the Config object) exercises the same precedence
    branch in ``alembic/env.py`` that integration tests will use everywhere else.
    """
    db_path = tmp_path / "round.sqlite"
    monkeypatch.setenv("CAT_WATCHER_DB_URL", f"sqlite:///{db_path}")
    cfg = Config("alembic.ini")
    cfg.set_main_option("script_location", "alembic")
    return cfg


def test_upgrade_head_then_downgrade_base(alembic_cfg: Config) -> None:
    """``upgrade head`` followed by ``downgrade base`` runs without error.

    Guards against the common bug where ``downgrade()`` is missing a table-drop or has the wrong
    drop order relative to its FKs — the chain would fail and this test would catch it.
    """
    command.upgrade(alembic_cfg, "head")
    command.downgrade(alembic_cfg, "base")


def test_upgrade_head_creates_all_tables(alembic_cfg: Config, tmp_path: Path) -> None:
    """After ``upgrade head``, the inspector reports every table the schema defines."""
    command.upgrade(alembic_cfg, "head")

    engine = create_engine(f"sqlite:///{tmp_path / 'round.sqlite'}")
    try:
        inspector = inspect(engine)
        tables = frozenset(inspector.get_table_names())
    finally:
        engine.dispose()

    assert tables == EXPECTED_TABLES


def test_no_db_url_raises_rather_than_creating_stub_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no CAT_WATCHER_DB_URL and no config.toml, upgrade raises ValueError loudly.

    A typo'd config or missing env var fails with a clear diagnostic instead of leaving a confusing
    leftover SQLite stub on disk at the alembic.ini fallback path.
    """
    monkeypatch.delenv("CAT_WATCHER_DB_URL", raising=False)
    # Point CAT_WATCHER_CONFIG at a path that doesn't exist so load_config never fires.
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(tmp_path / "no-such-config.toml"))
    cfg = Config("alembic.ini")  # script_location resolves via %(here)s in the ini

    with pytest.raises(ValueError, match=r"CAT_WATCHER_DB_URL"):
        command.upgrade(cfg, "head")

    # And critically — no stub DB file was written anywhere under tmp_path.
    assert not list(tmp_path.glob("**/*.sqlite"))


def test_upgrade_head_matches_current_models(alembic_cfg: Config, tmp_path: Path) -> None:
    """``compare_metadata`` after ``upgrade head`` reports zero diffs.

    Catches the model/migration drift bug: a developer adds or modifies a column in
    :mod:`cat_watcher.db` and forgets to ``pixi run db-revision``. ``EXPECTED_TABLES`` only
    checks table names; this test compares the full schema (columns, types, indices, FKs,
    constraints) against ``Base.metadata`` using the same comparison logic Alembic's
    ``--autogenerate`` uses internally.
    """
    command.upgrade(alembic_cfg, "head")

    engine = create_engine(f"sqlite:///{tmp_path / 'round.sqlite'}")
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(
                conn,
                opts={
                    "compare_type": True,
                    "compare_server_default": True,
                    "render_as_batch": True,
                },
            )
            # ``compare_metadata`` returns a list of operation tuples; Alembic's stubs type it as
            # ``Any``. Cast at the boundary to keep the rest of the test typed cleanly.
            diff = cast("list[object]", compare_metadata(context, Base.metadata))
    finally:
        engine.dispose()

    assert diff == [], f"model/migration drift: {diff}"
