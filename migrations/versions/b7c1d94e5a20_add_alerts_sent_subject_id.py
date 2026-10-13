"""Add alerts_sent subject_id.

Revision ID: b7c1d94e5a20
Revises: 22ec4c8a5459
Create Date: 2026-10-12 14:00:00.000000
"""

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# Alembic reads these revision identifiers. The pragmas stop detect-secrets on the IDs.
revision: str = "b7c1d94e5a20"  # pragma: allowlist secret
down_revision: str | Sequence[str] | None = "22ec4c8a5459"  # pragma: allowlist secret
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FK_NAME = "fk_alerts_sent_subject_id_subjects"


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("alerts_sent", schema=None) as batch_op:
        batch_op.add_column(sa.Column("subject_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(_FK_NAME, "subjects", ["subject_id"], ["id"])
        batch_op.drop_index("ix_alerts_camera_type_sent")
        batch_op.create_index(
            "ix_alerts_camera_subject_type_sent",
            ["camera_id", "subject_id", "alert_type", "sent_at"],
            unique=False,
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("alerts_sent", schema=None) as batch_op:
        batch_op.drop_index("ix_alerts_camera_subject_type_sent")
        batch_op.create_index("ix_alerts_camera_type_sent", ["camera_id", "alert_type", "sent_at"], unique=False)
        batch_op.drop_constraint(_FK_NAME, type_="foreignkey")
        batch_op.drop_column("subject_id")
