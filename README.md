# HVAC-FDD: End-to-End Fault Detection and Diagnostics System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HVAC-FDD** is a production-oriented, end-to-end fault detection and diagnostics prototype for Heating, Ventilation, and Air Conditioning (HVAC) environments. Built on the LBNL SDAHU dataset, it transforms AHU time-series data into traceable detection events and PostgreSQL-backed Power BI analytics.

The project integrates **expert physical rules** with **machine learning models**, establishing a complete pipeline from ETL data ingestion, model inference, and PostgreSQL persistence to a FastAPI service and BI analytics.

---

## 🚀 Core Features & Engineering Contributions

- **Hybrid-Driven Detection Engine**
  - **Expert Rules Layer (APAR)**: Utilizes domain physics knowledge to rapidly filter out definitive control anomalies, ensuring high interpretability.
  - **Unsupervised Anomaly Detection**: Supports Gaussian Mixture Models (GMM), Isolation Forest, and **Kolmogorov-Arnold Networks (KAN-AD)** to capture implicit anomalies under unknown operational conditions.
  - **Supervised Fault Classification**: Supports XGBoost, Random Forest, Hierarchical XGBoost, and **Temporal Convolutional Networks (TCN)** to assist in labeling fault root causes.
- **Model Evaluation System**
  - Supports **Annual / Common Temporal Split** to strictly prevent future data leakage.
  - Utilizes **Leave-one-severity-out** evaluation to measure generalization across unseen fault severities.
- **Deployment & Performance Engineering**
  - **Lazy temporal windows**: TCN experiments use scenario-aware, on-demand tensor slicing instead of materializing the complete sliding-window dataset in memory to reduce memory pressure.
  - **Parquet + DuckDB ingestion path**: Converts feature-engineered CSV outputs into per-file Parquet caches and uses DuckDB column projection to materialize only the columns required by detection. On 54.09 million zone-time rows, subsequent ingestion/read time decreased from 337.34 s to 31.76 s (10.62×), peak RSS decreased by 80.52%, and materialized columns decreased from 83 to 40.
  - **Batched persistence**: The pipeline converts detected domain events to ORM rows and flushes them in one transaction, reducing per-event persistence overhead.
  - **API & BI integration**: Separates binary anomaly detection from fault-type labeling, exposes REST endpoints through FastAPI, and provides PostgreSQL analytical views for Power BI.

## 📊 Business Intelligence & Data Visualization (Power BI)

The project integrates Power BI with PostgreSQL analytical views for historical and operational monitoring. The current reference setup uses Power BI Desktop Import mode; automatic refresh and Power BI Service deployment are outside the current repository scope.

### System Overview
*Detection events on the LBNL holdout dataset*
![Power BI System Overview](assets/powerbi_overview.png)

### Fault Diagnostics Drill-down
*Combined fault localization using physical rule triggers (Violated Policy) and algorithmic anomaly scoring (Anomaly Index)*
![Power BI Fault Drill-down](assets/powerbi_drilldown.png)

## ⚡ Data Engineering Optimization: Parquet + DuckDB

The reference CSV path performs parsing, wide-to-long zone expansion, unit conversion, rolling/lag feature engineering, and detection input construction on every run. To avoid repeating this work, the project provides a per-file Parquet cache and a DuckDB projection reader:

```text
CSV → one-time feature engineering → per-file Parquet cache
   → DuckDB column projection → Pandas detection/evaluation
```

On the 21 LBNL SDAHU scenario files (54,094,505 zone-time rows), the benchmark recorded the following results:

| Metric | CSV + Pandas streaming path | Parquet + DuckDB projection path |
| :--- | ---: | ---: |
| Subsequent ingestion/read time | 337.34 s | 31.76 s (10.62× faster) |
| Peak RSS memory | 7,119.8 MB | 1,386.7 MB (80.52% lower) |
| Processed rows | 54,094,505 | 54,094,505 |
| Materialized columns | 83 | 40 |

The same annual/final rules evaluation produced F1 scores of 0.8140 on the CSV path and 0.8139 on the projected Parquet path. The small event-count difference of 170 rows is attributed to backend floating-point and boundary behavior. These figures describe cached ingestion and feature-read performance, not a 10.62× model-inference speedup. Initial cache generation took approximately 559 seconds and produced about 8.97 GB of Parquet files, so the optimization is intended for repeated training, evaluation, benchmarking, and demonstrations.

## 🛠️ Tech Stack

| Domain | Core Technologies |
| :--- | :--- |
| **Core Services** | Python 3.11, FastAPI, Uvicorn, Pydantic |
| **Algorithms & DL** | Scikit-learn, XGBoost, PyTorch (TCN, KAN) |
| **Data Engineering** | Pandas, NumPy, PyArrow, Parquet, DuckDB |
| **Storage & DB** | PostgreSQL, Alembic (ORM migrations), SQLAlchemy |
| **Deploy & BI** | Docker, Docker Compose, Power BI |

## 🏗️ System Architecture

```mermaid
graph LR
    A[(LBNL CSV\nTime-series)] --> B[Ingestion\nETL & Features]
    B --> C{Hybrid Detection\nEngine}
    
    subgraph Prediction Flow
        C -->|1. Prior Physics| D[Rules APAR]
        C -->|2. Unknown States| E[GMM / IF / KAN]
        C -->|3. Root Cause| F[XGBoost / TCN]
    end
    
    C --> G[Evaluation\nRigorous metrics]
    
    G --> H[(PostgreSQL\nPersistence)]
    H --> I[FastAPI\nMicroservice]
    H --> J[Analytics Views\nSQL Schema]
    J --> K[Power BI\nDashboards]
```

## 📊 Model Evaluation Performance

The primary benchmark uses a common temporal protocol: Apr–Aug for training, Sep for validation, and Oct as the final holdout. Metrics below are zone-time, row-level metrics on the LBNL scenario files. A separate leave-one-severity-out protocol evaluates generalization to a completely held-out fault scenario.

### Stage 1: Binary Anomaly Detection
The first layer is responsible for core alerting. Under the common temporal protocol on the Oct final window, the performance of each model is as follows. Normal-scenario FPR is reported separately because the LBNL final window has an artificial fault-heavy class balance.

| Detection Architecture | Normal Scenario FPR | Fault Recall | Precision | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Physical Rules (Rules)** | 20.11% | 67.62% | 98.54% | 80.20% |
| **Rules + GMM** (FPR Constrained) | 30.16% | 75.77% | 98.05% | 85.48% |
| **Temporal Deep Model (TCN)** | 2.76% | 80.03% | 99.83% | 88.84% |

> **Stage 1 Architecture Decision**: Rules is the primary interpretable detector and safety fallback. GMM is retained as a validation-controlled recall enhancement or shadow module, not as an unconditionally enabled production detector. TCN achieved 2.76% normal-scenario FPR, 80.03% recall, and 88.84% F1 on the common Oct final window after validation calibration, but its first held-out severity experiments reached only 61.12% and 66.55% recall. It is therefore retained as a research comparison rather than the final production model.

### Stage 2: Fault Type Classification
The second layer is responsible for specific fault root cause labeling (6-class) on the identified anomalies. We evolved multiple cutting-edge architectures during R&D:

| Diagnostic Architecture | Evaluation Protocol | Accuracy | Macro-F1 | Architectural Features & Insights |
| :--- | :--- | :--- | :--- | :--- |
| **GNN (Graph Neural Network)** | Annual Temporal Split | 77.26% | 0.6974 | Lack of complex real device topology; forced graph building caused computation overhead and poor performance. |
| **TCN (Temporal ConvNet)** | Annual Temporal Split | 86.28% | 0.8079 | Incorporates temporal context through scenario-aware sliding-window inference; retained as a research comparison. |
| **Hierarchical Cascade XGBoost** | Annual Temporal Split | - | 0.8571 | Step-by-step diagnosis by physical subsystems; requires attention to error amplification during probability cascading. |
| **Flat XGBoost** | Same-scenario future month | **95.59%** | **0.9472** | Serves as a baseline: demonstrates stable classification performance on tabular scalar data. |
| **Flat XGBoost** | Leave-one unseen severity | 99.30%* | **0.1661** | **Generalization warning**: *A highly imbalanced single unseen scenario caused inflated Accuracy, while Macro-F1 exposed poor cross-severity fault-type generalization.* |

The common-protocol XGBoost result (Accuracy 95.59%, Macro-F1 0.9472) is a same-scenario future-month benchmark, not an unseen-severity result. On a held-out severity, Macro-F1 fell to 0.1661. The classifier is therefore used for auxiliary fault-type labeling rather than as the binary alert boundary.

### 🎯 Default Production Architecture
Based on the validation and holdout results, the current reference architecture is:
1. **Primary binary alert detector**: **Expert Physical Rules (APAR)** provide the interpretable safety baseline.
2. **Candidate recall enhancement**: **Gaussian Mixture Model (GMM)** can supplement recall only after validation-set FPR calibration; the common Oct final benchmark reached 75.77% recall, 98.05% precision, 85.48% F1, and 30.16% normal-scenario FPR at the 10% validation target.
3. **Auxiliary fault labeling**: **XGBoost (6-class)** labels the fault type after an anomaly is triggered.
4. **Research-only comparisons**: TCN and GNN remain available for experiments; neither is part of the final production path.

## 📊 Data Preparation

The raw LBNL dataset is not included in the repository to save space. After downloading, set the environment variable `LBNL_DATA_DIR` to point to:

```text
data/LBNL_FDD_Data_Sets_SDAHU_all_3/LBNL_FDD_Dataset_SDAHU
```

## 💻 Quick Start (Local Run)

**Step 1: Environment Setup**
We recommend using Conda. Ensure PostgreSQL is installed locally.
```bash
conda create -n hvac python=3.11 -y
conda activate hvac
pip install -e .
cp .env.example .env
```
For the runtime pipeline, `pip install -e .` is sufficient. Install the optional development dependencies with `pip install -e ".[dev]"` when running tests or the Parquet benchmark. Edit `.env` to configure `DATABASE_URL` (e.g., `postgresql://hvac:hvac@localhost:5432/hvac_fdd`) and `LBNL_DATA_DIR`.

The base installation does not include the optional PyTorch and `pykan` packages required by the experimental TCN, GNN, and KAN implementations. Install compatible versions separately only when reproducing those experiments.

**Step 2: Database Initialization**
```bash
alembic upgrade head
```

**Step 3: Model Training (Select algorithms)**
```bash
export UNSUPERVISED_MODEL=kan       # Options: gmm, if, kan
export SUPERVISED_MODEL=tcn         # Options: xgboost, random_forest, hierarchical_xgb, tcn

# Preprocess and train
python scripts/preprocess_data.py
python scripts/run_pipeline.py --train-unsup --train-clf
```

To generate the current per-file Parquet cache once:

```bash
python scripts/preprocess_data.py --force
```

To reuse that cache and let DuckDB project only the columns required by the detection pipeline:

```bash
python scripts/run_pipeline.py --input-format parquet --use-rules --use-unsup --use-clf --evaluate
```

The Parquet path is a cached ingestion and feature-reading optimization, not a claim that model inference itself is 10x faster. Existing caches with an older schema must be regenerated with `--force`.

**Step 4: Execute Detection Pipeline**
```bash
python scripts/run_pipeline.py --use-rules --use-unsup --use-clf --persist
```

**Step 5: Launch API Service**
```bash
uvicorn hvac_fdd.api:create_app --factory --host 0.0.0.0 --port 8000
```
Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## 🐳 Docker Compose Start

Use Docker Compose to quickly spin up PostgreSQL and FastAPI. We recommend running training and inference pipelines natively on the host CLI to fully leverage GPU/CPU resources.

```bash
# Start DB and API services
docker compose up --build -d

# Check service health
curl http://localhost:8000/health/live
```
After events have been written by the pipeline, initialize the Power BI views:
```bash
psql postgresql://hvac:hvac@localhost:5432/hvac_fdd -f powerbi/sql/analytics_views.sql
```

## 📁 Project Layout

```text
src/hvac_fdd/
 ├── ingestion/   # Loading, transforms, and feature engineering
 ├── detection/   # Core algorithms (Rules, KAN, TCN, XGB, etc.)
 ├── evaluation/  # Metrics calculation and data splits
 ├── db/          # SQLAlchemy ORM, repositories, transactions
 └── api/         # FastAPI routes, schemas, middlewares
powerbi/          # PostgreSQL analytical views definitions
scripts/          # Entrypoints for training and pipelines
migrations/       # Alembic migrations scripts
```

## 📄 License

MIT License. Use of the LBNL dataset is subject to the publisher's license and citation requirements.
