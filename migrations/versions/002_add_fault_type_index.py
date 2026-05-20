"""add fault type index

Revision ID: 002
Revises: 001
Create Date: 2024-05-19 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    op.create_index(op.f('ix_detections_ground_truth'), 'detections', ['ground_truth'], unique=False)
    op.create_index(op.f('ix_detections_created_at'), 'detections', ['created_at'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_detections_created_at'), table_name='detections')
    op.drop_index(op.f('ix_detections_ground_truth'), table_name='detections')
