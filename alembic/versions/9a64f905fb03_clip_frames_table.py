"""Add the ``clip_frames`` table for per-frame thumbnails.

Revision ID: 9a64f905fb03
Revises: fda18cd0b832
Create Date: 2026-05-08 14:19:45.536948
"""

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


# revision identifiers, used by Alembic.
revision: str = "9a64f905fb03"
down_revision: str | Sequence[str] | None = "fda18cd0b832"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    _ = op.create_table(
        "clip_frames",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("clip_id", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("t_offset_seconds", sa.Float(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("thumb_path", sa.String(length=512), nullable=False),
        sa.ForeignKeyConstraint(["clip_id"], ["clips.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("clip_id", "ordinal", name="uq_clip_frames_clip_ordinal"),
    )
    with op.batch_alter_table("clip_frames", schema=None) as batch_op:
        batch_op.create_index("ix_clip_frames_clip", ["clip_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("clip_frames", schema=None) as batch_op:
        batch_op.drop_index("ix_clip_frames_clip")
    op.drop_table("clip_frames")
