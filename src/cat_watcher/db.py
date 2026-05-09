"""SQLAlchemy 2.0 ORM models, engine factory, and transactional session for the cat-watcher DB.

Single source of truth for the SQLite schema consumed by every long-running agent:

* The **poller** writes ``Camera`` updates, ``Clip`` rows, ``AgentStart``, and ``Heartbeat``.
* The **alerts** agent writes ``AlertSent`` (with cool-down lookups by
  ``(camera_id, alert_type, sent_at)``) plus ``Heartbeat`` and ``AgentStart``.
* The **web** agent reads everything and writes ``Heartbeat`` / ``AgentStart``.

The **backup** agent does NOT use this module — it opens its own raw ``sqlite3.Connection`` to drive
the SQLite online-backup API.

All datetime columns are timezone-aware UTC. The connect-time PRAGMA listener enables WAL mode
(concurrent readers + single writer), enforces foreign keys (off by default in SQLite), and sets
``synchronous=NORMAL`` (the standard, fsync-friendly companion to WAL).
"""

import enum
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, override

from sqlalchemy import (
    Boolean,
    DateTime,
    Dialect,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy import (
    create_engine as _sa_create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)
from sqlalchemy.types import JSON, TypeDecorator, TypeEngine

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy.engine import Engine
    from sqlalchemy.engine.interfaces import DBAPIConnection
    from sqlalchemy.pool import ConnectionPoolEntry
    from sqlalchemy.sql.schema import SchemaItem


__all__ = (
    "AgentStart",
    "AlertSent",
    "AlertType",
    "Base",
    "Camera",
    "Clip",
    "ClipFrame",
    "Heartbeat",
    "PollStatus",
    "UtcDateTime",
    "create_engine",
    "get_session",
)


class UtcDateTime(TypeDecorator[datetime]):  # pylint: disable=too-many-ancestors  # SQLAlchemy TypeDecorator MRO
    """``DateTime`` that always stores + returns timezone-aware UTC.

    SQLite has no native timezone-aware datetime storage; ``DateTime(timezone=True)`` on SQLite
    silently strips ``tzinfo`` on the way out, leaving callers with naive datetimes. The spec for
    cat-watcher requires every persisted datetime to be tz-aware UTC, so this decorator:

    * On bind: rejects naive datetimes (loud failure beats silent timezone drift) and converts any
      tz-aware datetime to UTC before handing it to the dialect.
    * On result: stamps a UTC tzinfo on every returned datetime.

    Other dialects (Postgres, MySQL with proper config) round-trip tz-aware datetimes natively; this
    decorator is a no-op-ish layer over those — it still normalizes to UTC on the way in and
    asserts UTC on the way out. The ORM's ``Mapped[datetime]`` annotations therefore reliably
    mean "UTC-aware datetime" everywhere.
    """

    impl: TypeEngine[datetime] | type[TypeEngine[datetime]] = DateTime(timezone=True)
    # Pure, instance-stateless transforms → safe to cache compiled SQL.
    # Adding constructor params later requires reconsidering this flag.
    cache_ok: bool | None = True

    @override
    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        del dialect  # required by base signature; unused here.
        if value is None:
            return None
        if value.tzinfo is None:
            msg = "naive datetime rejected; cat-watcher requires tz-aware UTC datetimes"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @override
    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        del dialect  # required by base signature; unused here.
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @override
    def process_literal_param(self, value: datetime | None, dialect: Dialect) -> str:
        # Used only when SQLAlchemy needs to render a literal in an inlined SQL string;
        # round-trip through ``process_bind_param`` to enforce the same UTC normalization.
        normalized = self.process_bind_param(value, dialect)
        return repr(normalized)

    @property
    @override
    def python_type(self) -> type[datetime]:
        return datetime


class PollStatus(enum.Enum):
    """Poll-loop health for a single camera; surfaced in the web UI status badge."""

    OK = "ok"
    UNREACHABLE = "unreachable"
    ERROR = "error"


class AlertType(enum.Enum):
    """Discriminator for ``AlertSent`` rows; drives cool-down lookups + alert routing."""

    INACTIVITY = "INACTIVITY"
    FREQUENCY = "FREQUENCY"
    POLLER_STUCK = "POLLER_STUCK"
    WEB_DOWN = "WEB_DOWN"
    WEB_FLAPPING = "WEB_FLAPPING"
    ALERTS_STUCK = "ALERTS_STUCK"
    DISK_LOW = "DISK_LOW"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    BACKUP_STALE = "BACKUP_STALE"


class Base(DeclarativeBase):
    """Declarative base for all ORM models. No extra surface — keep it boring."""


class Camera(Base):
    """One row per camera in ``config.toml``; updated in-place by the poller."""

    __tablename__: str = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    last_polled_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    last_clip_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    last_cat_seen_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    poll_status: Mapped[PollStatus] = mapped_column(Enum(PollStatus, name="poll_status"), nullable=False, default=PollStatus.OK)
    poll_status_since: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    poll_error: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # ``cascade="all, delete-orphan"`` mirrors the FK ``ondelete="CASCADE"`` on Clip.camera_id so
    # removing a Camera (rare — only when the operator deletes it from config) drops its clips.
    # Alerts are intentionally NOT cascaded — see ``AlertSent.camera_id``.
    clips: Mapped[list[Clip]] = relationship(back_populates="camera", cascade="all, delete-orphan", passive_deletes=True)
    alerts: Mapped[list[AlertSent]] = relationship(back_populates="camera")


class Clip(Base):
    """One row per ingested clip; ``(camera_id, source_filename)`` is the idempotency key."""

    __tablename__: str = "clips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[int] = mapped_column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    source_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    start_ts: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    end_ts: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    thumb_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    has_cat: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    max_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    frames_sampled: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frames_with_cat: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Phase-2 ROI overlap will populate this from the best-scoring frame; nullable today.
    best_box_xyxy: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    detector_version: Mapped[str] = mapped_column(String(128), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    analysis_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Three-state user correction: True (yes), False (no), NULL (no manual label).
    # The web UI projects ``COALESCE(manual_has_cat, has_cat)``; all three states matter.
    manual_has_cat: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    manual_label_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    manual_label_notes: Mapped[str | None] = mapped_column(String(500), nullable=True)

    camera: Mapped[Camera] = relationship(back_populates="clips")
    # ``passive_deletes=True`` defers row removal to the DB-level ``ondelete=CASCADE`` on
    # ``ClipFrame.clip_id`` (cheaper than letting the ORM emit per-child DELETEs); requires the
    # connect-time ``PRAGMA foreign_keys=ON`` set by ``create_engine``.
    frames: Mapped[list[ClipFrame]] = relationship(
        back_populates="clip",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ClipFrame.ordinal",
    )

    __table_args__: tuple[SchemaItem, ...] = (
        UniqueConstraint("camera_id", "source_filename", name="uq_clips_camera_source"),
        Index("ix_clips_camera_start", "camera_id", "start_ts"),
        Index("ix_clips_camera_hascat_start", "camera_id", "has_cat", "start_ts"),
    )


class ClipFrame(Base):
    """One row per detector-sampled frame inside a ``Clip``; ``Clip.thumb_path`` points at the best.

    The detector samples N frames per clip; each scored frame yields a JPEG thumbnail and a
    ``ClipFrame`` row. ``ordinal`` is a 0-based stable index over the sampled frames (not raw video
    frame numbers), so ``(clip_id, ordinal)`` is the natural identity. ``score`` is the YOLO
    max-cat-score for the frame (0.0 when no qualifying detection).
    """

    __tablename__: str = "clip_frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    clip_id: Mapped[int] = mapped_column(Integer, ForeignKey("clips.id", ondelete="CASCADE"), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    t_offset_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    thumb_path: Mapped[str] = mapped_column(String(512), nullable=False)

    clip: Mapped[Clip] = relationship(back_populates="frames")

    __table_args__: tuple[SchemaItem, ...] = (
        UniqueConstraint("clip_id", "ordinal", name="uq_clip_frames_clip_ordinal"),
        Index("ix_clip_frames_clip", "clip_id"),
    )


class AlertSent(Base):
    """One row per alert dispatched; queried by the alerts agent for cool-down windows."""

    __tablename__: str = "alerts_sent"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_type: Mapped[AlertType] = mapped_column(Enum(AlertType, name="alert_type"), nullable=False)
    # Nullable + no ``ondelete=CASCADE``: keep the alert history even if a camera is later
    # deleted from config. The relationship side also has no cascade.
    camera_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("cameras.id"), nullable=True)
    sent_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    # Rendered body can be many lines; ``Text`` (no length cap) avoids a surprise truncation.
    body: Mapped[str] = mapped_column(Text, nullable=False)
    email_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    macos_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    delivery_error: Mapped[str | None] = mapped_column(String(500), nullable=True)

    camera: Mapped[Camera | None] = relationship(back_populates="alerts")

    __table_args__: tuple[SchemaItem, ...] = (Index("ix_alerts_camera_type_sent", "camera_id", "alert_type", "sent_at"),)


class Heartbeat(Base):
    """One row per long-running agent (``poller`` / ``alerts`` / ``web``).

    Application contract (not enforced by the DB): ``agent_name`` is one of ``poller``, ``alerts``,
    or ``web``. The backup agent intentionally does NOT write a heartbeat — its once-daily cadence
    would always look stale to the alerts watchdog, so backup health is monitored via mtime on
    backup files instead.
    """

    __tablename__: str = "heartbeats"

    agent_name: Mapped[str] = mapped_column(String(32), primary_key=True)
    last_seen_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)


class AgentStart(Base):
    """One row per agent process start; surfaces flapping in the web UI + alerts."""

    __tablename__: str = "agent_starts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    __table_args__: tuple[SchemaItem, ...] = (Index("ix_agent_starts_name_started", "agent_name", "started_at"),)


def create_engine(url: str) -> Engine:
    """Build an ``Engine`` for ``url`` with WAL + foreign-key PRAGMAs applied per connection.

    Wraps :func:`sqlalchemy.create_engine` and registers a ``connect`` event listener that runs
    three PRAGMAs on every new DB-API connection:

    * ``journal_mode=WAL`` — concurrent readers alongside a single writer; persistent on the file.
    * ``foreign_keys=ON`` — SQLite ships with FK enforcement off; per-connection setting.
    * ``synchronous=NORMAL`` — the standard, fsync-friendly companion to WAL (still safe).

    Accepts any SQLAlchemy URL; production passes ``sqlite:///<path>`` and tests pass a
    ``tmp_path``-derived URL (an in-memory ``sqlite:///:memory:`` URL also works but cannot enable
    WAL — file-based SQLite is the only way to verify the journal_mode PRAGMA).

    A non-SQLite URL fails fast here at engine-build time rather than later when the SQLite-only
    PRAGMA listener fires on the first connection against the wrong dialect.
    """
    engine = _sa_create_engine(url, future=True)
    if engine.dialect.name != "sqlite":
        msg = f"cat_watcher.db.create_engine requires sqlite; got dialect {engine.dialect.name!r}"
        raise ValueError(msg)

    def set_sqlite_pragmas(dbapi_conn: DBAPIConnection, _record: ConnectionPoolEntry) -> None:
        cursor = dbapi_conn.cursor()
        try:
            for pragma in ("PRAGMA journal_mode=WAL", "PRAGMA foreign_keys=ON", "PRAGMA synchronous=NORMAL"):
                # PEP 249 ``cursor.execute`` is typed to return ``Any``; we discard it.
                _ = cursor.execute(pragma)  # pyright: ignore[reportAny]
        finally:
            cursor.close()

    event.listen(engine, "connect", set_sqlite_pragmas)

    return engine


@contextmanager
def get_session(engine: Engine) -> Generator[Session]:
    """Yield a transactional :class:`Session` bound to ``engine``.

    On clean exit the session is committed; any exception inside the ``with`` block triggers a
    rollback before the exception propagates. The session is always closed.

    Used as::

        with get_session(engine) as session:
            session.add(row)
            # commit happens on successful exit
    """
    session = Session(bind=engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except BaseException:  # rollback must run on KeyboardInterrupt / SystemExit too; bare ``raise`` re-propagates.
        session.rollback()
        raise
    finally:
        session.close()
