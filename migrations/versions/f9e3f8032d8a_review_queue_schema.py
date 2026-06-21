"""review queue schema.

Revision ID: f9e3f8032d8a
Revises: 3c9ff40695f4
Create Date: 2026-06-17 11:35:44.245883
"""
# pylint: disable=duplicate-code  # Alembic migration boilerplate (imports + upgrade/downgrade skeleton) necessarily repeats across revisions

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "f9e3f8032d8a"  # pragma: allowlist secret
down_revision: str | Sequence[str] | None = "3c9ff40695f4"  # pragma: allowlist secret
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The view definition is the canonical contract every downstream caller reads through clip_label_summary.
_CREATE_LABEL_SUMMARY_VIEW_SQL = """
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

_DROP_LABEL_SUMMARY_VIEW_SQL = "DROP VIEW IF EXISTS clip_label_summary"


def upgrade() -> None:
    """Upgrade schema."""
    _ = op.create_table(
        "subjects",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("display_order", sa.Integer(), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("color", sa.String(length=16), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind IN ('cat', 'event')", name="ck_subjects_kind"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_subjects_slug"),
    )
    with op.batch_alter_table("subjects", schema=None) as batch_op:
        batch_op.create_index(
            "ux_subjects_kind_order_active",
            ["kind", "display_order"],
            unique=True,
            sqlite_where=sa.text("archived_at IS NULL"),
        )

    _ = op.create_table(
        "clip_frame_subjects",
        sa.Column("clip_frame_id", sa.Integer(), nullable=False),
        sa.Column("subject_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["clip_frame_id"], ["clip_frames.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subject_id"], ["subjects.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("clip_frame_id", "subject_id"),
    )
    with op.batch_alter_table("clip_frame_subjects", schema=None) as batch_op:
        batch_op.create_index(
            "ix_clip_frame_subjects_subject",
            ["subject_id", "clip_frame_id"],
            unique=False,
        )

    with op.batch_alter_table("clip_frames", schema=None) as batch_op:
        batch_op.add_column(sa.Column("activity", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("bbox_xyxy", sa.JSON(), nullable=True))

    with op.batch_alter_table("clips", schema=None) as batch_op:
        batch_op.add_column(sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_index("ix_clips_reviewed_at_start", ["reviewed_at", "start_ts"], unique=False)

    op.execute(sa.text(_CREATE_LABEL_SUMMARY_VIEW_SQL))


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(sa.text(_DROP_LABEL_SUMMARY_VIEW_SQL))

    with op.batch_alter_table("clips", schema=None) as batch_op:
        batch_op.drop_index("ix_clips_reviewed_at_start")
        batch_op.drop_column("reviewed_at")

    with op.batch_alter_table("clip_frames", schema=None) as batch_op:
        batch_op.drop_column("bbox_xyxy")
        batch_op.drop_column("activity")

    with op.batch_alter_table("clip_frame_subjects", schema=None) as batch_op:
        batch_op.drop_index("ix_clip_frame_subjects_subject")
    op.drop_table("clip_frame_subjects")

    with op.batch_alter_table("subjects", schema=None) as batch_op:
        batch_op.drop_index("ux_subjects_kind_order_active")
    op.drop_table("subjects")
