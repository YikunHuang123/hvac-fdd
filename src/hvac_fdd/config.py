from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "postgresql://hvac:hvac@localhost:5432/hvac_fdd"

    # Data paths
    lbnl_data_dir: Path = Path("data/LBNL_FDD_Data_Sets_SDAHU_all_3/LBNL_FDD_Dataset_SDAHU")
    processed_data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")

    # Isolation Forest parameters
    if_contamination: float = 0.05
    if_n_estimators: int = 100
    random_state: int = 42

    # LBNL detection thresholds
    sa_temp_error_threshold_c: float = 1.5      # SA_TEMP - SA_TEMPSPT threshold
    valve_tracking_threshold_pct: float = 10.0  # Valve tracking error threshold (%)
    damper_tracking_threshold_pct: float = 10.0 # Damper tracking error threshold (%)
    fan_speed_tracking_threshold_pct: float = 8.0
    sustained_fault_minutes: int = 15           # Minutes to sustain before alerting

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8501
    cors_origins: list[str] = ["*"]

@lru_cache
def get_settings() -> Settings:
    return Settings()
