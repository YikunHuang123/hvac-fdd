# HVAC-FDD

HVAC Fault Detection and Diagnostics (FDD) for the LBNL SDAHU (Single-Duct Air Handling Unit) dataset. The project combines ingestion, feature engineering, physics-based rules, unsupervised anomaly detection, supervised fault classification, evaluation, a REST API, and a Streamlit presentation layer.

> This project is currently intended for offline experiments and end-to-end demonstrations, not as a production building-control system. Model metrics should come from command-line evaluation scripts and explicit split protocols. UI metrics are computed only from detection events stored in the database.

## Features

- Load LBNL SDAHU CSV files and infer fault types from filenames.
- Convert zone-wide data to long format, normalize units and column names, and create engineered features.
- Run interpretable physics-based binary anomaly rules.
- Run GMM or Isolation Forest unsupervised detectors; KAN, GNN, and TCN are experimental implementations, not current defaults.
- Use XGBoost (with optional Random Forest, hierarchical XGBoost, or TCN) as an auxiliary fault-type classifier.
- Persist detection events and pipeline-job status, then query them through FastAPI.
- Use Streamlit to inspect events, statistics, job status, and stored evaluation results.

## Architecture

    LBNL CSV
      -> ingestion: load / transform / feature engineering
      -> detection: rules + unsupervised detector + optional classifier
      -> evaluation: temporal split / leave-one-severity-out
      -> PostgreSQL (events and jobs) + command-line reports
      -> FastAPI + Streamlit UI

The recommended configuration is rules as the primary detector, GMM as an optional detector with a controlled normal-operation false-positive rate, and XGBoost as an auxiliary fault classifier. TCN, GNN, KAN, and Isolation Forest are comparison paths, not the current best model.

## Fault types

| Identifier | Meaning |
|---|---|
| normal | Normal operation |
| coi_bias | Coil-inlet temperature sensor bias |
| coi_leakage | Chilled-water coil leakage |
| coi_stuck | Coil valve stuck |
| damper_stuck | Outdoor-air damper stuck |
| oa_bias | Outdoor-air temperature sensor bias |

The rules detector answers whether an anomaly exists and which rule was triggered; it is not itself a fault-type classifier. Detection and classification metrics must therefore be reported separately.

## Data and evaluation splits

The raw LBNL data is not included. Download it separately and set LBNL_DATA_DIR. The default path is:

    data/LBNL_FDD_Data_Sets_SDAHU_all_3/LBNL_FDD_Dataset_SDAHU/

Unsupervised models must be trained on normal rows only. Supported protocols are:

1. Temporal split: configure training and evaluation windows with evaluation-window.
2. Leave-one-severity-out: scripts/run_leave_one_severity.py holds one severity file out for testing while the remaining severities are used for training.

Every metric should state the model, split protocol, threshold policy, and whether a normal reference set was included.

## Installation

The project is primarily run in the WSL Ubuntu Conda environment named hvac:

    conda create -n hvac python=3.11 -y
    conda activate hvac
    pip install -e ".[dev]"
    cp .env.example .env

Default PostgreSQL configuration:

    DATABASE_URL=postgresql://hvac:hvac@localhost:5432/hvac_fdd
    LBNL_DATA_DIR=data/LBNL_FDD_Data_Sets_SDAHU_all_3/LBNL_FDD_Dataset_SDAHU
    PROCESSED_DATA_DIR=data/processed
    MODELS_DIR=models
    API_HOST=0.0.0.0
    API_PORT=8000
    DASHBOARD_PORT=8501

After creating the database, run alembic upgrade head. Unit tests use test settings and SQLite fixtures; that does not configure production PostgreSQL.

## Usage

### Preprocess data

    python scripts/preprocess_data.py

### Train and run detectors

    python scripts/run_pipeline.py --train-unsup --train-clf
    python scripts/run_pipeline.py --use-rules --use-unsup --use-clf --persist
    python scripts/run_pipeline.py --use-rules --use-unsup --use-clf --evaluate
    python scripts/run_pipeline.py --help

Example leave-one-severity-out experiment:

    python scripts/run_leave_one_severity.py --model rules_gmm --output-dir artifacts/holdout_severity/rules_gmm

### Start the API

    uvicorn hvac_fdd.api:create_app --factory --host 0.0.0.0 --port 8000

Main endpoints:

- GET /health/live and GET /health/ready
- GET /api/v1/detections/ (filtering and pagination)
- GET /stats/ and GET /stats/by-fault-type
- POST /pipeline/run
- GET /pipeline/jobs/{job_id}
- /docs

### Start the Streamlit UI

    streamlit run src/hvac_fdd/ui/Dashboard.py --server.port 8501

Start the API before opening the UI:

| Page | Purpose |
|---|---|
| Dashboard | Detection counts, alert levels, fault distribution, and recent events |
| Detections | Filter events by time, zone, alert level, and fault type |
| Analytics | Trend and distribution analysis of stored events |
| Pipeline | Trigger an asynchronous job and inspect its status |
| Evaluation | Auxiliary plots for database events with ground_truth |

The UI is a presentation and operation layer, not a replacement for strict evaluation scripts. Without complete experiment output in the database, UI metrics must not be treated as final model conclusions.

### Tests

    pytest -v

## Project layout

    src/hvac_fdd/ingestion/       loading, transforms, and feature engineering
    src/hvac_fdd/detection/       rules, GMM, Isolation Forest, classifiers, and experiments
    src/hvac_fdd/evaluation/      metrics and reports
    src/hvac_fdd/db/              SQLAlchemy ORM and repositories
    src/hvac_fdd/api/              FastAPI routes and schemas
    src/hvac_fdd/ui/               Streamlit pages and API client
    scripts/                       preprocessing, main pipeline, leave-one-severity-out
    tests/                         unit and API tests
    migrations/                    Alembic migrations
    data/                          raw data and Parquet (normally not committed)
    models/                        trained artifacts (commit selectively)
    artifacts/                     experiment outputs (normally not committed)

## Current project conclusions

- Rules are currently the most stable and interpretable primary detector.
- GMM is a complementary detector, but its threshold and normal-reference FPR must be reported.
- XGBoost is suitable as an auxiliary fault-type classifier; its output does not prove that the binary detector identified the specific fault.
- TCN has insufficient classification generalization in leave-one-severity-out experiments and is retained only for research comparison.
- GNN, KAN, and Isolation Forest are optional or experimental paths, not defaults.

## License

This project is released under the MIT License. Use of the LBNL dataset is subject to the publisher's license and citation requirements.

