"""
add chart query indexes

Revision ID: 6b4d8e2f1a21
Revises: 5a2c3b7a9f10
Create Date: 2025-10-27 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6b4d8e2f1a21'
down_revision = '5a2c3b7a9f10'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index('idx_utterance_date', 'utterances', ['date'])
    op.create_index('idx_utterance_speaker', 'utterances', ['speaker'])
    op.create_index('idx_utterance_date_speaker', 'utterances', ['date', 'speaker'])


def downgrade() -> None:
    op.drop_index('idx_utterance_date_speaker', table_name='utterances')
    op.drop_index('idx_utterance_speaker', table_name='utterances')
    op.drop_index('idx_utterance_date', table_name='utterances')
