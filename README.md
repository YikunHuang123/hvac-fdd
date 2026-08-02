# HVAC-FDD

HVAC fault detection and diagnostics for the LBNL SDAHU single-duct air-handling-unit dataset.

> **Status:** Offline research and end-to-end demonstration. Report metrics with the model, split protocol, threshold policy, and normal-reference false-positive rate.

## Features

- Loads LBNL SDAHU CSV files, infers fault labels from filenames, and engineers normalized features.
- Detects anomalies with Rules plus optional GMM, Isolation Forest, or KAN detectors.
- Adds optional fault-type predictions with XGBoost, Random Forest, Hierarchical XGBoost, or TCN.
- Evaluates temporal splits and leave-one-severity-out generalization.
- Persists detection events and pipeline-job status in PostgreSQL.
- Exposes stored data through FastAPI and PostgreSQL analytical views for Power BI.

## Architecture

```text
LBNL CSV
   |
   +--> Ingestion: load -> transform -> feature engineering
   |
   +--> Detection: Rules + (GMM / Isolation Forest / KAN)
   |                 + optional classifier
   |                   (XGBoost / Random Forest / Hierarchical XGBoost / TCN)
   |
   +--> Evaluation: temporal split / leave-one-severity-out
   |
   +--> PostgreSQL: detections and pipeline jobs
          +--> FastAPI: programmatic data access
          +--> analytics views: Power BI reporting
```

The recommended configuration is **Rules + GMM + XGBoost**. GNN is retained only as a separate experimental implementation and is not part of the default CLI pipeline. TCN is available for comparison but showed weak leave-one-severity-out classification generalization.

| Layer | Available models | Role |
|---|---|---|
| Binary detector | Rules | Interpretable primary detector |
| Unsupervised complement | GMM, Isolation Forest, KAN | Anomaly-score comparison |
| Fault classifier | XGBoost, Random Forest, Hierarchical XGBoost, TCN | Optional fault-type prediction |

## Data and evaluation

The raw LBNL dataset is not included. Download it separately and set `LBNL_DATA_DIR` to `data/LBNL_FDD_Data_Sets_SDAHU_all_3/LBNL_FDD_Dataset_SDAHU`.

Supported protocols:

1. Annual temporal split: January-September training, October-November validation, December hold-out test.
2. Common temporal split: `--split-protocol common` when every scenario must appear in each period.
3. Leave-one-severity-out: hold one severity CSV file out for testing while the remaining files are used for training.

Unsupervised models are trained on normal rows only. Binary detection metrics and fault-type classification metrics are reported separately.

## Run locally

Use a Python 3.11 environment and a PostgreSQL instance.

```bash
conda create -n hvac python=3.11 -y
conda activate hvac
pip install -e ".[dev]"
cp .env.example .env
```

Set `DATABASE_URL=postgresql://hvac:hvac@localhost:5432/hvac_fdd` and point `LBNL_DATA_DIR` at the downloaded data. Initialize the schema:

```bash
alembic upgrade head
```

Select and train models:

```bash
export UNSUPERVISED_MODEL=gmm       # gmm, if, or kan
export SUPERVISED_MODEL=xgboost     # xgboost, random_forest, hierarchical_xgb, or tcn
python scripts/preprocess_data.py
python scripts/run_pipeline.py --train-unsup --train-clf
```

Run detection and persist events:

```bash
python scripts/run_pipeline.py --use-rules --use-unsup --use-clf --persist
```

Then start the API:

```bash
uvicorn hvac_fdd.api:create_app --factory --host 0.0.0.0 --port 8000
```

API documentation is available at <http://localhost:8000/docs>. The API reads and writes PostgreSQL data; it does not replace the offline training/evaluation CLI.

## Run with Docker Compose

Docker Compose starts PostgreSQL and FastAPI. It does not run training or detection automatically.

```bash
docker compose up --build -d
docker compose ps
curl http://localhost:8000/health/live
```

The API is at <http://localhost:8000>; PostgreSQL is available at `localhost:5432` (`hvac` / `hvac_fdd`). Keep the database container running, then execute the training and detection CLI from a Python environment configured with the same database URL. Apply Power BI views with:

```bash
psql postgresql://hvac:hvac@localhost:5432/hvac_fdd -f powerbi/sql/analytics_views.sql
```

Stop services with `docker compose down`. Use `docker compose down -v` only when the PostgreSQL volume should also be deleted.

## API reference

- `GET /health/live` and `GET /health/ready`
- `GET /api/v1/detections/`
- `GET /stats/` and `GET /stats/by-fault-type`
- `POST /pipeline/run`
- `GET /pipeline/jobs/{job_id}`
- `/docs`

## Power BI analytical layer

Power BI is the primary reporting and analysis interface. Connect Power BI Desktop to PostgreSQL and use the `analytics` schema; reports read database views rather than raw CSV files or application UI state.

The SQL file creates:

- `analytics.vw_detection_events` - event-level detection details;
- `analytics.vw_detection_daily` - daily detection aggregates;
- `analytics.vw_pipeline_runs` - pipeline status, duration, and processing counts.

## Recorded offline comparison

This benchmark uses January-November training and a December hold-out test. It is not a live database metric.

| Configuration | Recall | Precision | F1 | Role |
|---|---:|---:|---:|---|
| Rules baseline | 66.92% | 97.97% | 0.795 | Interpretable baseline |
| Rules + Isolation Forest | 74.13% | 96.67% | 0.839 | Comparison |
| Rules + GMM | 77.94% | 96.81% | 0.864 | Recommended ensemble |
| Rules + GMM, calibrated threshold | 79.32% | 96.73% | 0.872 | Threshold trade-off |

`coi_stuck` and `damper_stuck` were strongest; `oa_bias` and `coi_bias` remained harder because control-loop compensation can mask part of the bias. Detection success does not prove exact fault-type identification; those are separate claims.

## Project layout

```text
src/hvac_fdd/ingestion/   loading, transforms, and feature engineering
src/hvac_fdd/detection/   rules, GMM, Isolation Forest, and classifiers
src/hvac_fdd/evaluation/  metrics and reports
src/hvac_fdd/db/          SQLAlchemy ORM and repositories
src/hvac_fdd/api/         FastAPI routes and schemas
powerbi/                  PostgreSQL analytical views and connection notes
scripts/                  preprocessing, pipeline, and evaluation runners
tests/                    unit and API tests
migrations/               Alembic migrations
data/                     raw data and processed files (normally not committed)
models/                   trained artifacts (commit selectively)
artifacts/                experiment outputs (normally not committed)
```

## Tests

```bash
pytest -v
```

## License

MIT. Use of the LBNL dataset is subject to the publisher's license and citation requirements.
