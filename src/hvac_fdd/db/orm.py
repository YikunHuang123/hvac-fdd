from sqlalchemy import Column, Integer, String, DateTime, Float, Text, func
from src.hvac_fdd.db.base import Base

class DetectionORM(Base):
    __tablename__ = "detections"
    id              = Column(Integer, primary_key=True)
    event_time      = Column(DateTime(timezone=True), nullable=False, index=True)
    zone_id         = Column(String(32), nullable=False, index=True)
    equipment_tag   = Column(String(32), nullable=False)
    violated_policy = Column(String(64), nullable=False, index=True)
    trigger_signal  = Column(String(64), nullable=False)
    anomaly_index   = Column(Float, nullable=False)
    alert_level     = Column(String(16), nullable=False, index=True)
    ground_truth    = Column(String(32), nullable=False, index=True)  # Filename label
    predicted_fault = Column(String(32), nullable=True)               # Classifier output
    confidence      = Column(Float, nullable=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now(), index=True)

class PipelineJobORM(Base):
    __tablename__ = "pipeline_jobs"
    id          = Column(Integer, primary_key=True)
    scenario    = Column(String(64))           # Which CSV or "all"
    status      = Column(String(16), index=True)  # pending | running | done | failed
    error_msg   = Column(Text, nullable=True)
    started_at  = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True), nullable=True)
    records_processed = Column(Integer, nullable=True)
    anomalies_found   = Column(Integer, nullable=True)
