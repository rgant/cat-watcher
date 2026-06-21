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

# This is the HEAD migration, so its view recreation reads the live single-source definition from
# ``cat_watcher.db`` (the prior revision keeps its own frozen copy for historical reproducibility).
# The view references ``clips``; batch_alter_table renames clips → _alembic_tmp_clips, which breaks
# the view, so it is dropped before the batch and restored after.


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(sa.text(cat_watcher.db.DROP_CLIP_LABEL_SUMMARY_VIEW_SQL))
    # Labels live in clip_frame_subjects, so downgrade restoring these columns empty is acceptable.
    with op.batch_alter_table("clips", schema=None) as batch_op:
        batch_op.drop_column("manual_has_cat")
        batch_op.drop_column("manual_label_at")
        batch_op.drop_column("manual_label_notes")
    op.execute(sa.text(cat_watcher.db.CLIP_LABEL_SUMMARY_VIEW_SQL))


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(sa.text(cat_watcher.db.DROP_CLIP_LABEL_SUMMARY_VIEW_SQL))
    with op.batch_alter_table("clips", schema=None) as batch_op:
        batch_op.add_column(sa.Column("manual_label_notes", sa.String(length=500), nullable=True))
        batch_op.add_column(sa.Column("manual_label_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("manual_has_cat", sa.Boolean(), nullable=True))
    op.execute(sa.text(cat_watcher.db.CLIP_LABEL_SUMMARY_VIEW_SQL))
