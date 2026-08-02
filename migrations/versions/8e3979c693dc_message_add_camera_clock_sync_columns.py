"""Add camera clock sync columns.

Revision ID: 8e3979c693dc
Revises: d301291abfcf
Create Date: 2026-08-02 16:02:24.776674
"""

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

import cat_watcher.db

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic; pragmas silence detect-secrets on the IDs.
revision: str = "8e3979c693dc"  # pragma: allowlist secret
down_revision: str | Sequence[str] | None = "d301291abfcf"  # pragma: allowlist secret
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("cameras", schema=None) as batch_op:
        batch_op.add_column(sa.Column("clock_drift_seconds", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("clock_checked_at", cat_watcher.db.UtcDateTime(timezone=True), nullable=True))
        # server_default backfills existing rows, which a bare NOT NULL add cannot do.
        batch_op.add_column(sa.Column("clock_correction_streak", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("clock_ntp_enabled", sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("cameras", schema=None) as batch_op:
        batch_op.drop_column("clock_ntp_enabled")
        batch_op.drop_column("clock_correction_streak")
        batch_op.drop_column("clock_checked_at")
        batch_op.drop_column("clock_drift_seconds")
