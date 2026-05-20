"""initial schema

Revision ID: 001
Revises: 
Create Date: 2024-05-19 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # 1. 创建 detections 表
    op.create_table(
        'detections',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('event_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('zone_id', sa.String(length=32), nullable=False),
        sa.Column('equipment_id', sa.String(length=32), nullable=False),
        sa.Column('detector_source', sa.String(length=32), nullable=False),
        sa.Column('violated_policy', sa.String(length=64), nullable=False),
        sa.Column('trigger_signal', sa.String(length=64), nullable=False),
        sa.Column('anomaly_index', sa.Float(), nullable=False),
        sa.Column('alert_level', sa.String(length=16), nullable=False),
        sa.Column('ground_truth', sa.String(length=32), nullable=True),
        sa.Column('predicted_fault', sa.String(length=32), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_detections_event_time'), 'detections', ['event_time'], unique=False)
    op.create_index(op.f('ix_detections_zone_id'), 'detections', ['zone_id'], unique=False)
    op.create_index(op.f('ix_detections_alert_level'), 'detections', ['alert_level'], unique=False)
    op.create_index(op.f('ix_detections_violated_policy'), 'detections', ['violated_policy'], unique=False)

    # 2. 创建 pipeline_jobs 表
    op.create_table(
        'pipeline_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('scenario', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('error_msg', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('records_processed', sa.Integer(), nullable=True),
        sa.Column('anomalies_found', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_pipeline_jobs_status'), 'pipeline_jobs', ['status'], unique=False)

def downgrade():
    op.drop_table('pipeline_jobs')
    op.drop_table('detections')
