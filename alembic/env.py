"""Alembic environment for cat-watcher.

URL resolution priority (see ``alembic.ini`` for the matching prose):

1. ``CAT_WATCHER_DB_URL`` env var. Test-only override so integration tests can point Alembic at a
   ``tmp_path`` SQLite file without writing a config. Never set in production.
2. ``<internal_root>/cat_watcher.sqlite`` computed from the loaded application config. This is the
   production path: ``alembic upgrade head`` after a ``git pull`` reads the same TOML the running
   agents do.

If neither is available, ``_resolve_url`` raises ``ValueError`` rather than silently creating a stub
SQLite file at some default path. A typo'd ``CAT_WATCHER_CONFIG`` should fail loudly, not spawn an
empty database.

``render_as_batch=True`` is set in both modes because SQLite has no ``ALTER TABLE`` support for
column changes; Alembic emulates ALTER via CREATE + COPY when batch mode is on.
"""

import os
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context
from cat_watcher.config import load_config
from cat_watcher.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _resolve_url() -> str:
    """Resolve the database URL via the documented 2-tier precedence; raise if neither resolves."""
    env_url = os.environ.get("CAT_WATCHER_DB_URL", "").strip()
    if env_url:
        return env_url

    config_path = os.environ.get("CAT_WATCHER_CONFIG", "./config.toml")
    if Path(config_path).is_file():
        cfg = load_config()  # ConfigError propagates: a typo'd config should fail loudly.
        return f"sqlite:///{cfg.internal_root}/cat_watcher.sqlite"

    msg = (
        "no DB URL available: set CAT_WATCHER_DB_URL (test override) or place a valid config.toml "
        f"at {config_path!r} so the application's internal_root can be resolved"
    )
    raise ValueError(msg)


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without opening a DB connection)."""
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _resolve_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
