"""
add is_indexed to utterances

Revision ID: 5a2c3b7a9f10
Revises: 
Create Date: 2025-10-09 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5a2c3b7a9f10'
down_revision = '1a2b3c4d5e6f'  # Depends on initial schema
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'utterances',
        sa.Column('is_indexed', sa.Boolean(), nullable=False, server_default=sa.text('false'))
    )


def downgrade() -> None:
    op.drop_column('utterances', 'is_indexed')

