from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "postgresql://hvac:hvac@localhost:5432/hvac_fdd"

    # Connection pool
    db_pool_size:    int = 5
    db_max_overflow: int = 10

    # Data paths
    lbnl_data_dir:      Path = Path("data/LBNL_FDD_Data_Sets_SDAHU_all_3/LBNL_FDD_Dataset_SDAHU")
    processed_data_dir: Path = Path("data/processed")
    models_dir:         Path = Path("models")

    # Model selection
    unsupervised_model: str = Field(default="gmm")  # options: 'if', 'gmm', 'kan'

    # GMM parameters
    gmm_n_components:    int = Field(default=5, ge=1)
    gmm_covariance_type: str = Field(default="full")

    # Isolation Forest parameters
    if_contamination:    float = Field(default=0.05, ge=0.0, le=0.5)
    if_n_estimators:     int   = Field(default=100, ge=1)

    # GNN parameters
    gnn_hidden_dim:      int = Field(default=32, ge=8)
    gnn_epochs:          int = Field(default=5, ge=1)
    gnn_batch_size:      int = Field(default=8192, ge=64)
    gnn_learning_rate:   float = Field(default=0.001, gt=0)

    # TCN parameters
    tcn_seq_len:         int = Field(default=30, ge=1)
    tcn_hidden_dim:      int = Field(default=32, ge=8)
    tcn_kernel_size:     int = Field(default=3, ge=2)
    tcn_levels:          int = Field(default=3, ge=1)
    tcn_epochs:          int = Field(default=5, ge=1)
    tcn_batch_size:      int = Field(default=2048, ge=64)
    tcn_learning_rate:   float = Field(default=0.001, gt=0)

    random_state:        int = 42

    # Classifier parameters
    supervised_model: str = Field(default="xgboost")  # options: 'random_forest', 'xgboost', 'tcn', 'hierarchical_xgb'
    clf_n_estimators: int = Field(default=200, ge=10)

    # Equipment
    default_equipment_id: str = "AHU-1"

    # LBNL detection thresholds
    sa_temp_error_threshold_c:      float = 1.5   # SA_TEMP - SA_TEMPSPT alert threshold (°C)
    valve_tracking_threshold_pct:   float = 10.0  # Chilled-water valve tracking error threshold (%)
    damper_tracking_threshold_pct:  float = 10.0  # Outside-air damper tracking error threshold (%)
    sustained_fault_minutes:        int   = 15    # Minutes a condition must persist before alerting

    # Classifier thresholds
    classifier_conf_critical: float = 0.80
    classifier_conf_warning:  float = 0.50

    # API
    api_host:       str       = "0.0.0.0"
    api_port:       int       = 8000
    cors_origins:   list[str] = ["*"]
    log_level:      str       = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
