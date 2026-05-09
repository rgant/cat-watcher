"""Initial schema for the cat-watcher DB.

Creates the five tables defined in :mod:`cat_watcher.db`: ``cameras``, ``clips``, ``alerts_sent``,
``heartbeats``, and ``agent_starts``, plus all indices, the ``uq_clips_camera_source`` unique
constraint, and the ``ON DELETE CASCADE`` FK from ``clips`` to ``cameras``.

Revision ID: fda18cd0b832
Revises:
Create Date: 2026-05-03 15:23:44.344876
"""

from typing import TYPE_CHECKING

import sqlalchemy as sa

import cat_watcher.db
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


# revision identifiers, used by Alembic.
revision: str = "fda18cd0b832"  # pragma: allowlist secret
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # ``op.create_table`` returns the ``Table`` it created; we never use it (the side effect IS the
    # migration), so each call assigns to ``_`` to satisfy basedpyright's reportUnusedCallResult.
    _ = op.create_table(
        "agent_starts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_name", sa.String(length=32), nullable=False),
        sa.Column("started_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("agent_starts", schema=None) as batch_op:
        batch_op.create_index("ix_agent_starts_name_started", ["agent_name", "started_at"], unique=False)

    _ = op.create_table(
        "cameras",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("host", sa.String(length=255), nullable=False),
        sa.Column("last_polled_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=True),
        sa.Column("last_clip_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=True),
        sa.Column("last_cat_seen_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=True),
        sa.Column("poll_status", sa.Enum("OK", "UNREACHABLE", "ERROR", name="poll_status"), nullable=False),
        sa.Column("poll_status_since", cat_watcher.db.UtcDateTime(timezone=True), nullable=True),
        sa.Column("poll_error", sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    _ = op.create_table(
        "heartbeats",
        sa.Column("agent_name", sa.String(length=32), nullable=False),
        sa.Column("last_seen_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("agent_name"),
    )

    _ = op.create_table(
        "alerts_sent",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "alert_type",
            sa.Enum(
                "INACTIVITY",
                "FREQUENCY",
                "POLLER_STUCK",
                "WEB_DOWN",
                "WEB_FLAPPING",
                "ALERTS_STUCK",
                "DISK_LOW",
                "STORAGE_UNAVAILABLE",
                "BACKUP_STALE",
                name="alert_type",
            ),
            nullable=False,
        ),
        sa.Column("camera_id", sa.Integer(), nullable=True),
        sa.Column("sent_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("email_ok", sa.Boolean(), nullable=False),
        sa.Column("macos_ok", sa.Boolean(), nullable=False),
        sa.Column("delivery_error", sa.String(length=500), nullable=True),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("alerts_sent", schema=None) as batch_op:
        batch_op.create_index("ix_alerts_camera_type_sent", ["camera_id", "alert_type", "sent_at"], unique=False)

    _ = op.create_table(
        "clips",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=False),
        sa.Column("source_filename", sa.String(length=255), nullable=False),
        sa.Column("start_ts", cat_watcher.db.UtcDateTime(timezone=True), nullable=False),
        sa.Column("end_ts", cat_watcher.db.UtcDateTime(timezone=True), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("thumb_path", sa.String(length=512), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("has_cat", sa.Boolean(), nullable=False),
        sa.Column("max_score", sa.Float(), nullable=False),
        sa.Column("frames_sampled", sa.Integer(), nullable=False),
        sa.Column("frames_with_cat", sa.Integer(), nullable=False),
        sa.Column("best_box_xyxy", sa.JSON(), nullable=True),
        sa.Column("detector_version", sa.String(length=128), nullable=False),
        sa.Column("ingested_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=False),
        sa.Column("analysis_error", sa.String(length=500), nullable=True),
        sa.Column("manual_has_cat", sa.Boolean(), nullable=True),
        sa.Column("manual_label_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=True),
        sa.Column("manual_label_notes", sa.String(length=500), nullable=True),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("camera_id", "source_filename", name="uq_clips_camera_source"),
    )
    with op.batch_alter_table("clips", schema=None) as batch_op:
        batch_op.create_index("ix_clips_camera_hascat_start", ["camera_id", "has_cat", "start_ts"], unique=False)
        batch_op.create_index("ix_clips_camera_start", ["camera_id", "start_ts"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("clips", schema=None) as batch_op:
        batch_op.drop_index("ix_clips_camera_start")
        batch_op.drop_index("ix_clips_camera_hascat_start")
    op.drop_table("clips")

    with op.batch_alter_table("alerts_sent", schema=None) as batch_op:
        batch_op.drop_index("ix_alerts_camera_type_sent")
    op.drop_table("alerts_sent")

    op.drop_table("heartbeats")
    op.drop_table("cameras")

    with op.batch_alter_table("agent_starts", schema=None) as batch_op:
        batch_op.drop_index("ix_agent_starts_name_started")
    op.drop_table("agent_starts")
