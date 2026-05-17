"""Add ``POLLER_EMPTY_AFTER_QUIET`` to the ``AlertType`` enum.

Revision ID: 3c9ff40695f4
Revises: 9a64f905fb03
Create Date: 2026-05-10 21:54:25.996824
"""

# pylint: disable=duplicate-code
# Each enum-altering migration must restate the full AlertType value list; importing from
# cat_watcher.db would couple frozen history to evolving code. R0801 fires structurally.

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


# revision identifiers, used by Alembic.
revision: str = "3c9ff40695f4"  # pragma: allowlist secret
down_revision: str | Sequence[str] | None = "9a64f905fb03"  # pragma: allowlist secret
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("alerts_sent", schema=None) as batch_op:
        batch_op.alter_column(
            "alert_type",
            existing_type=sa.Enum(
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
            type_=sa.Enum(
                "INACTIVITY",
                "FREQUENCY",
                "POLLER_STUCK",
                "POLLER_EMPTY_AFTER_QUIET",
                "WEB_DOWN",
                "WEB_FLAPPING",
                "ALERTS_STUCK",
                "DISK_LOW",
                "STORAGE_UNAVAILABLE",
                "BACKUP_STALE",
                name="alert_type",
            ),
            existing_nullable=False,
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("alerts_sent", schema=None) as batch_op:
        batch_op.alter_column(
            "alert_type",
            existing_type=sa.Enum(
                "INACTIVITY",
                "FREQUENCY",
                "POLLER_STUCK",
                "POLLER_EMPTY_AFTER_QUIET",
                "WEB_DOWN",
                "WEB_FLAPPING",
                "ALERTS_STUCK",
                "DISK_LOW",
                "STORAGE_UNAVAILABLE",
                "BACKUP_STALE",
                name="alert_type",
            ),
            type_=sa.Enum(
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
            existing_nullable=False,
        )
