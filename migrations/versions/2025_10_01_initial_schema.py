"""
initial schema - create base tables

Revision ID: 1a2b3c4d5e6f
Revises: 
Create Date: 2025-10-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1a2b3c4d5e6f'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create analyses table
    op.create_table(
        'analyses',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_filename', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_analyses_id'), 'analyses', ['id'], unique=False)
    op.create_index(op.f('ix_analyses_source_filename'), 'analyses', ['source_filename'], unique=False)

    # Create utterances table
    op.create_table(
        'utterances',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('analysis_id', sa.Integer(), nullable=True),
        sa.Column('date', sa.String(), nullable=True),
        sa.Column('timestamp', sa.String(), nullable=True),
        sa.Column('speaker', sa.String(), nullable=True),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('predictions', sa.JSON(), nullable=True),
        sa.Column('aggregated_scores', sa.JSON(), nullable=True),
        sa.Column('sa_labels', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_utterances_id'), 'utterances', ['id'], unique=False)
    op.create_index(op.f('ix_utterances_speaker'), 'utterances', ['speaker'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_utterances_speaker'), table_name='utterances')
    op.drop_index(op.f('ix_utterances_id'), table_name='utterances')
    op.drop_table('utterances')
    op.drop_index(op.f('ix_analyses_source_filename'), table_name='analyses')
    op.drop_index(op.f('ix_analyses_id'), table_name='analyses')
    op.drop_table('analyses')
