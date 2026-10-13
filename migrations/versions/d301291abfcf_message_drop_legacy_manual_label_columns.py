"""drop legacy manual label columns.

Revision ID: d301291abfcf
Revises: f9e3f8032d8a
Create Date: 2026-06-20 11:43:28.447255
"""
# pylint: disable=duplicate-code  # Alembic migration boilerplate (imports + upgrade/downgrade skeleton) necessarily repeats across revisions

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

import cat_watcher.db

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "d301291abfcf"  # pragma: allowlist secret
down_revision: str | Sequence[str] | None = "f9e3f8032d8a"  # pragma: allowlist secret
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# This migration keeps its own frozen copy of the view SQL, so its ``upgrade`` reproduces the schema
# as it was authored. The view references ``clips``. The batch rename of ``clips`` breaks the view,
# so the migration drops the view before the batch and creates it again after.
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
    op.execute(sa.text(_DROP_LABEL_SUMMARY_VIEW_SQL))
    # Labels live in clip_frame_subjects, so downgrade restoring these columns empty is acceptable.
    with op.batch_alter_table("clips", schema=None) as batch_op:
        batch_op.drop_column("manual_has_cat")
        batch_op.drop_column("manual_label_at")
        batch_op.drop_column("manual_label_notes")
    op.execute(sa.text(_CREATE_LABEL_SUMMARY_VIEW_SQL))


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(sa.text(_DROP_LABEL_SUMMARY_VIEW_SQL))
    with op.batch_alter_table("clips", schema=None) as batch_op:
        batch_op.add_column(sa.Column("manual_label_notes", sa.String(length=500), nullable=True))
        batch_op.add_column(sa.Column("manual_label_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("manual_has_cat", sa.Boolean(), nullable=True))
    op.execute(sa.text(_CREATE_LABEL_SUMMARY_VIEW_SQL))
