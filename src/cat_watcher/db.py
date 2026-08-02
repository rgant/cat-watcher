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
    CheckConstraint,
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
    text,
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
    from pathlib import Path

    from sqlalchemy.engine import Engine
    from sqlalchemy.engine.interfaces import DBAPIConnection
    from sqlalchemy.pool import ConnectionPoolEntry
    from sqlalchemy.sql.schema import SchemaItem


__all__ = (
    "DB_FILENAME",
    "AgentStart",
    "AlertSent",
    "AlertType",
    "Base",
    "Camera",
    "Clip",
    "ClipFrame",
    "ClipFrameSubject",
    "ClipLabelSummary",
    "Heartbeat",
    "PollStatus",
    "Subject",
    "UtcDateTime",
    "create_engine",
    "engine_for",
    "get_session",
)


DB_FILENAME = "cat_watcher.sqlite"
"""Filename of the live SQLite database under ``internal_root`` (see :func:`engine_for`)."""


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
    # Pure, instance-stateless transforms; safe to cache compiled SQL. Adding constructor params
    # later requires reconsidering this flag.
    cache_ok: bool | None = True

    @override
    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None:
            msg = "naive datetime rejected; cat-watcher requires tz-aware UTC datetimes"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @override
    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @override
    def process_literal_param(self, value: datetime | None, dialect: Dialect) -> str:
        # Round-trip through ``process_bind_param`` so inlined SQL literals get the same UTC
        # normalization as bound parameters.
        normalized = self.process_bind_param(value, dialect)
        return repr(normalized)

    @property
    @override
    def python_type(self) -> type[datetime]:
        return datetime


def _utc_now() -> datetime:
    return datetime.now(UTC)


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
    POLLER_EMPTY_AFTER_QUIET = "POLLER_EMPTY_AFTER_QUIET"
    WEB_DOWN = "WEB_DOWN"
    WEB_FLAPPING = "WEB_FLAPPING"
    ALERTS_STUCK = "ALERTS_STUCK"
    DISK_LOW = "DISK_LOW"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    BACKUP_STALE = "BACKUP_STALE"
    CAMERA_CLOCK = "CAMERA_CLOCK"


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


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
    # Clock-sync state, written by the poller each tick. The camera's own NTP client is kept off,
    # so the host is the only thing correcting these clocks. ``clock_correction_streak`` counts
    # consecutive ticks that needed a correction, which is what distinguishes an expected one-off
    # (a camera unplugged for cleaning) from a clock that will not hold.
    clock_drift_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    clock_checked_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    # ``server_default`` matches the migration, which needs it to backfill pre-existing rows.
    clock_correction_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    clock_ntp_enabled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # ``cascade="all, delete-orphan"`` mirrors the FK ``ondelete="CASCADE"`` on Clip.camera_id so
    # removing a Camera (rare — only when the operator deletes it from config) drops its clips.
    # Alerts are intentionally NOT cascaded — see ``AlertSent.camera_id``.
    clips: Mapped[list[Clip]] = relationship(back_populates="camera", cascade="all, delete-orphan", passive_deletes=True)
    alerts: Mapped[list[AlertSent]] = relationship(back_populates="camera")


class Clip(Base):
    """One row per ingested clip; ``(camera_id, source_filename)`` is the idempotency key.

    Detector verdict lives on ``has_cat``; operator confirmation is stamped on ``reviewed_at``.
    The ``clip_label_summary`` view derives ``effective_has_cat`` from frame-subject tagging when
    the clip is reviewed, falling back to the detector verdict otherwise.
    """

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
    reviewed_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)

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
        Index("ix_clips_reviewed_at_start", "reviewed_at", "start_ts"),
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
    activity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    bbox_xyxy: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)

    clip: Mapped[Clip] = relationship(back_populates="frames")
    subjects: Mapped[list[ClipFrameSubject]] = relationship(
        back_populates="frame",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__: tuple[SchemaItem, ...] = (
        UniqueConstraint("clip_id", "ordinal", name="uq_clip_frames_clip_ordinal"),
        Index("ix_clip_frames_clip", "clip_id"),
    )


class Subject(Base):
    """One row per taggable subject (cat or event) shown in the review UI.

    ``kind`` is constrained to ``'cat'`` or ``'event'`` via a CHECK constraint in the migration.
    Active (non-archived) subjects are unique on ``(kind, display_order)`` via the partial index
    ``ux_subjects_kind_order_active`` — archived subjects may reuse old display positions.
    """

    __tablename__: str = "subjects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    color: Mapped[str | None] = mapped_column(String(16), nullable=True)
    archived_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=_utc_now)

    frame_subjects: Mapped[list[ClipFrameSubject]] = relationship(
        back_populates="subject",
    )

    __table_args__: tuple[SchemaItem, ...] = (
        CheckConstraint("kind IN ('cat', 'event')", name="ck_subjects_kind"),
        Index(
            "ux_subjects_kind_order_active",
            "kind",
            "display_order",
            unique=True,
            sqlite_where=text("archived_at IS NULL"),
        ),
    )


class ClipFrameSubject(Base):
    """Junction row linking a ``ClipFrame`` to a ``Subject``; composite PK ``(clip_frame_id, subject_id)``.

    FK to ``clip_frames.id`` uses ``ON DELETE CASCADE`` so deleting a frame removes its taggings.
    FK to ``subjects.id`` uses ``ON DELETE RESTRICT`` to prevent accidental subject deletion while
    tags exist. The reverse index on ``subject_id`` supports looking up all frames tagged with a
    given subject.
    """

    __tablename__: str = "clip_frame_subjects"

    clip_frame_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("clip_frames.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    subject_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("subjects.id", ondelete="RESTRICT"),
        primary_key=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, default=_utc_now)

    frame: Mapped[ClipFrame] = relationship(back_populates="subjects")
    subject: Mapped[Subject] = relationship(back_populates="frame_subjects")

    __table_args__: tuple[SchemaItem, ...] = (Index("ix_clip_frame_subjects_subject", "subject_id", "clip_frame_id"),)


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


class _ViewBase(DeclarativeBase):
    """Declarative base for read-only DB views.

    Its ``metadata`` stays separate from ``Base.metadata`` so ``Base.metadata.create_all`` never
    emits table DDL for a view — the migration owns the ``CREATE VIEW``.
    """


class ClipLabelSummary(_ViewBase):
    """Typed ORM binding for the read-only ``clip_label_summary`` view (one row per clip)."""

    __tablename__: str = "clip_label_summary"

    clip_id: Mapped[int] = mapped_column(primary_key=True)
    has_manual_cat: Mapped[bool] = mapped_column()
    effective_has_cat: Mapped[bool] = mapped_column()
    tagged_subject_slugs: Mapped[str] = mapped_column()


# Canonical, single-source definition of the ``clip_label_summary`` view. It lives in app code (not
# the migration) because both the HEAD migration and the test fixtures need the exact same text and
# the ``migrations`` package is not importable under pytest. This is the *current* definition: any
# change to the view must ship as a new migration that drops and recreates it, updating this constant
# in lockstep. Historical migrations keep their own frozen copies so an old ``upgrade`` reproduces
# the schema as it was authored. ``slug`` uses the aggregate ``ORDER BY`` form (SQLite >= 3.44) so
# the comma-joined ordering is well-defined rather than relying on undefined aggregate input order.
CLIP_LABEL_SUMMARY_VIEW_SQL = """
CREATE VIEW clip_label_summary AS
SELECT
    c.id AS clip_id,
    CAST(EXISTS (
        SELECT 1
        FROM clip_frames cf
        JOIN clip_frame_subjects cfs ON cfs.clip_frame_id = cf.id
        JOIN subjects s ON s.id = cfs.subject_id
        WHERE cf.clip_id = c.id AND s.kind = 'cat'
    ) AS INTEGER) AS has_manual_cat,
    CAST(
        CASE
            WHEN c.reviewed_at IS NULL THEN c.has_cat
            ELSE EXISTS (
                SELECT 1
                FROM clip_frames cf
                JOIN clip_frame_subjects cfs ON cfs.clip_frame_id = cf.id
                JOIN subjects s ON s.id = cfs.subject_id
                WHERE cf.clip_id = c.id AND s.kind = 'cat'
            )
        END
    AS INTEGER) AS effective_has_cat,
    COALESCE((
        SELECT GROUP_CONCAT(slug_distinct.slug ORDER BY slug_distinct.kind, slug_distinct.display_order)
        FROM (
            SELECT DISTINCT s.slug AS slug, s.kind AS kind, s.display_order AS display_order
            FROM clip_frames cf
            JOIN clip_frame_subjects cfs ON cfs.clip_frame_id = cf.id
            JOIN subjects s ON s.id = cfs.subject_id
            WHERE cf.clip_id = c.id
        ) AS slug_distinct
    ), '') AS tagged_subject_slugs
FROM clips c
"""

DROP_CLIP_LABEL_SUMMARY_VIEW_SQL = "DROP VIEW IF EXISTS clip_label_summary"


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
                _ = cursor.execute(pragma)  # pyright: ignore[reportAny]  # DBAPI cursor.execute() is untyped (returns Any)
        finally:
            cursor.close()

    event.listen(engine, "connect", set_sqlite_pragmas)

    return engine


def engine_for(internal_root: Path) -> Engine:
    """Build an :class:`Engine` for the live DB (:data:`DB_FILENAME`) under ``internal_root``."""
    return create_engine(f"sqlite:///{internal_root / DB_FILENAME}")


@contextmanager
def get_session(engine: Engine) -> Generator[Session]:
    """Yield a transactional :class:`Session` bound to ``engine``.

    Commits on clean exit; rolls back on any exception (including ``KeyboardInterrupt`` and
    ``SystemExit``); always closes.
    """
    session = Session(bind=engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
