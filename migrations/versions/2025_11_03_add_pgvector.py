"""
add pgvector extension and embedding column

Revision ID: 7c5e9f3b2d32
Revises: 6b4d8e2f1a21
Create Date: 2025-11-03 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = '7c5e9f3b2d32'
down_revision = '6b4d8e2f1a21'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension (Postgres only, ignored on SQLite)
    conn = op.get_bind()
    if conn.dialect.name == 'postgresql':
        conn.execute(sa.text('CREATE EXTENSION IF NOT EXISTS vector'))
        
        # Add embedding column for semantic search
        op.add_column('utterances', 
            sa.Column('embedding', Vector(1536), nullable=True)
        )
        
        # Create vector index for fast similarity search
        # Using ivfflat with 50 lists (good for ~2K-10K rows)
        conn.execute(sa.text(
            'CREATE INDEX IF NOT EXISTS idx_utterances_embedding '
            'ON utterances USING ivfflat (embedding vector_cosine_ops) '
            'WITH (lists = 50)'
        ))
        
        # Optimize for vector queries
        conn.execute(sa.text('ANALYZE utterances'))


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == 'postgresql':
        op.drop_index('idx_utterances_embedding', table_name='utterances')
        op.drop_column('utterances', 'embedding')
        conn.execute(sa.text('DROP EXTENSION IF EXISTS vector'))
