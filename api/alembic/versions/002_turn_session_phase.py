"""Add session_phase to turns

Revision ID: 002
Revises: 001
Create Date: 2026-04-28
"""

from typing import Sequence, Union
from alembic import op

revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('ALTER TABLE turns ADD COLUMN IF NOT EXISTS session_phase TEXT')


def downgrade() -> None:
    op.execute('ALTER TABLE turns DROP COLUMN IF EXISTS session_phase')
