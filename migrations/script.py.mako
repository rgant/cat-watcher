"""${message}.

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op
${imports if imports else ""}
if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic; pragmas silence detect-secrets on the IDs.
revision: str = ${repr(up_revision)}  # pragma: allowlist secret
down_revision: str | Sequence[str] | None = ${repr(down_revision)}  # pragma: allowlist secret
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


def upgrade() -> None:
    """Upgrade schema."""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Downgrade schema."""
    ${downgrades if downgrades else "pass"}
