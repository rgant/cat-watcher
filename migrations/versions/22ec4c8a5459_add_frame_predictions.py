"""Add frame predictions.

Revision ID: 22ec4c8a5459
Revises: 8e3979c693dc
Create Date: 2026-10-12 12:00:00.000000
"""

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

import cat_watcher.db

if TYPE_CHECKING:
    from collections.abc import Sequence

# Alembic reads these revision identifiers. The pragmas stop detect-secrets on the IDs.
revision: str = "22ec4c8a5459"  # pragma: allowlist secret
down_revision: str | Sequence[str] | None = "8e3979c693dc"  # pragma: allowlist secret
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FK_NAME = "fk_clip_frames_predicted_subject_id_subjects"

# The batch rebuild renames ``clip_frames``, which breaks the view. The migration drops the view
# before the batch and creates it again after. As the HEAD migration, it reads the live view SQL
# from ``cat_watcher.db``.


def upgrade() -> None:
    """Upgrade schema."""
    # "No box" has one form: SQL NULL. The downgrade does not restore JSON 'null', because the
    # ORM reads both forms as None.
    op.execute(sa.text("UPDATE clip_frames SET bbox_xyxy = NULL WHERE bbox_xyxy = 'null'"))
    op.execute(sa.text("UPDATE clips SET best_box_xyxy = NULL WHERE best_box_xyxy = 'null'"))
    op.execute(sa.text(cat_watcher.db.DROP_CLIP_LABEL_SUMMARY_VIEW_SQL))
    with op.batch_alter_table("clip_frames", schema=None) as batch_op:
        batch_op.add_column(sa.Column("predicted_subject_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("prediction_confidence", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("classifier_version", sa.String(length=255), nullable=True))
        batch_op.create_foreign_key(_FK_NAME, "subjects", ["predicted_subject_id"], ["id"], ondelete="SET NULL")
    op.execute(sa.text(cat_watcher.db.CLIP_LABEL_SUMMARY_VIEW_SQL))


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(sa.text(cat_watcher.db.DROP_CLIP_LABEL_SUMMARY_VIEW_SQL))
    with op.batch_alter_table("clip_frames", schema=None) as batch_op:
        batch_op.drop_constraint(_FK_NAME, type_="foreignkey")
        batch_op.drop_column("classifier_version")
        batch_op.drop_column("prediction_confidence")
        batch_op.drop_column("predicted_subject_id")
    op.execute(sa.text(cat_watcher.db.CLIP_LABEL_SUMMARY_VIEW_SQL))
