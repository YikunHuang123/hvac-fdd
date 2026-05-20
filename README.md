# HVAC-FDD

**A modular Fault Detection and Diagnostics (FDD) system for Single-Duct Air Handling Units (SDAHU).** Combines physics-based rules, unsupervised anomaly detection, and supervised classification to identify six HVAC fault types, then serves results through a REST API and an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [How It Works](#️-how-it-works)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Follow-up Development Plans](#-follow-up-development-plans)
- [Contributing](#-contributing)
- [License & Contact](#-license--contact)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔧 Rule-Based Detection | Five physics-based fault rules derived from ASHRAE Guideline 36 |
| 🌲 Isolation Forest | Unsupervised anomaly detection trained on fault-free baseline data |
| 🎯 Fault Classifier | Supervised RandomForest distinguishing six fault classes |
| 🔗 Modular Detectors | Each strategy is independent; combine freely at runtime via CLI flags |
| ⚙️ Feature Engineering | 16 engineered features — tracking errors, temperature deltas, rolling statistics, lags |
| 📦 Chunked Pipeline | Iterator-based processing prevents memory pressure on large datasets |
| 📊 Evaluation Suite | Binary detection P/R/F1, multi-class confusion matrices, AUC, time-to-detect |
| 🗄️ Persistence Layer | Detection events stored in PostgreSQL via Repository pattern |
| 🌐 REST API | FastAPI backend with detection queries, pipeline triggers, and statistics |
| 🖥️ Dashboard | Multi-page Streamlit UI for live monitoring, analytics, and evaluation |

---

## 🎬 Architecture

### System Architecture

```
┌──────────────────────────┐
│      Streamlit UI        │  streamlit run src/hvac_fdd/ui/Dashboard.py
│      (port 8501)         │
└────────────┬─────────────┘
             │  HTTP (requests)
             ▼
┌──────────────────────────────────────────────────────────┐
│                 FastAPI  (port 8000)                      │
│  GET /api/v1/detections  │  POST /pipeline/run           │
│  GET /stats/             │  GET  /health/live|ready      │
└──────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│                 Detection Engine                          │
│  ┌───────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │  Rules        │  │ IsolationForest  │  │ RandomForest│
│  │  (ASHRAE G36) │  │ (unsupervised)   │  │ (6-class) │ │
│  └───────────────┘  └──────────────────┘  └───────────┘ │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐  ┌────────┐  ┌──────────┐
   │ PostgreSQL │  │Parquet │  │  Models  │
   │ (events)   │  │ (data) │  │ (joblib) │
   └────────────┘  └────────┘  └──────────┘
```

### Detection Pipeline

```
Raw LBNL CSVs
     │
     ▼  Load (LBNLDataLoader)
     │   · 21 CSV files, fault type inferred from filename
     │
     ▼  Transform (transforms.py)
     │   · wide_zones_to_long  → 5 zone columns → 1 column + zone_id
     │   · fahrenheit_to_celsius
     │   · normalize_column_names
     │
     ▼  Feature Engineering (features.py)
     │   · Tracking errors (valve, damper, fan)
     │   · Temperature deltas (mixed-outside, return-supply)
     │   · Rolling means (15 min and 60 min windows)
     │   · Lag-1 features
     │
     ▼  Detect
     │   · Rules  →  policy violations (threshold-based)
     │   · IF     →  anomaly score (trained on NORMAL rows)
     │   · Clf    →  predicted fault + confidence
     │
     ▼  Persist
         · DetectionORM → PostgreSQL
         · PipelineJobORM → job status tracking
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **Web Framework** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit + Plotly |
| **ML / Detection** | scikit-learn (IsolationForest, RandomForestClassifier) |
| **Data Processing** | Pandas, NumPy, PyArrow (Parquet) |
| **Database** | PostgreSQL 16+ via SQLAlchemy 2.0 |
| **Migrations** | Alembic |
| **Config** | Pydantic v2 + pydantic-settings |
| **Model Serialization** | joblib |
| **Testing** | pytest, httpx |
| **Package Manager** | uv |

---

## ⚙️ How It Works

### Fault Types

The system targets six fault conditions found in the LBNL SDAHU dataset:

| Fault Type | Description |
|---|---|
| `NORMAL` | No fault — baseline operation |
| `COI_BIAS` | Coil inlet temperature sensor bias |
| `COI_LEAKAGE` | Chilled-water coil leakage |
| `COI_STUCK` | Coil valve stuck |
| `DAMPER_STUCK` | Outdoor air damper stuck |
| `OA_BIAS` | Outdoor air temperature sensor bias |

### Detection Strategies

**Rules Detector (`LBNLRulesDetector`)**  
Five physics-based rules encoded from ASHRAE Guideline 36. Each rule checks a specific sensor condition (e.g., supply air temperature falls below setpoint while the coil valve is commanded closed, indicating coil leakage) and emits an `INFO`, `WARNING`, or `CRITICAL` event depending on the rule. No training required.

**Isolation Forest (`IsolationForestDetector`)**  
Trained exclusively on `NORMAL` rows. During inference, anomaly scores are mapped to alert levels via configurable thresholds. Effective for detecting novel or combined faults outside the training distribution.

**Fault Classifier (`FaultClassifier`)**  
A supervised RandomForest trained on all six classes. Returns the predicted fault type and a confidence score. Only non-NORMAL predictions generate a detection event.

### Evaluation

The evaluation module computes:
- **Binary metrics**: Precision, Recall, F1 at event level
- **Multi-class metrics**: Per-class P/R/F1, confusion matrix
- **Time-to-detect**: Minutes elapsed from fault onset to first detection

Temporal splitting is used: January–September (months ≤ 9) for training, October–December for evaluation.

---

## 📂 Dataset

The project uses the **LBNL FDD Data Sets — SDAHU (Single-Duct Air Handling Unit)**, published by Lawrence Berkeley National Laboratory (LBNL). This dataset is **not included** in the repository and must be downloaded separately.

### Dataset Overview

| Property | Detail |
|---|---|
| **Source** | Lawrence Berkeley National Laboratory (LBNL) |
| **System type** | Single-Duct Air Handling Unit (SDAHU) |
| **Files** | 21 CSV files covering 6 fault conditions |
| **Columns** | 31 sensor and actuator channels per record |
| **Time resolution** | 1-minute interval measurements |
| **Usage in project** | Training (January–September, months ≤ 9) and evaluation (October–December) splits |

### Download Steps

1. Visit the LBNL Buildings Technology and Urban Systems division website and search for **"LBNL FDD Data Sets SDAHU"**, or locate the dataset via the publication associated with this benchmark.

2. Download the dataset archive (typically named `LBNL_FDD_Data_Sets_SDAHU_all_3.zip` or similar).

3. Extract the archive so the CSV files are placed at:

   ```
   hvac-fdd/
   └── data/
       └── LBNL_FDD_Data_Sets_SDAHU_all_3/
           └── LBNL_FDD_Dataset_SDAHU/
               ├── SDAHU_COI_BIAS_1.csv
               ├── SDAHU_COI_BIAS_2.csv
               ├── SDAHU_COI_LEAKAGE_1.csv
               ├── ...                         # 21 CSV files total
               └── SDAHU_NORMAL_5.csv
   ```

4. Confirm the path matches `LBNL_DATA_DIR` in your `.env`:

   ```env
   LBNL_DATA_DIR=data/LBNL_FDD_Data_Sets_SDAHU_all_3/LBNL_FDD_Dataset_SDAHU
   ```

> The fault type for each file is automatically inferred from its filename by `LBNLDataLoader`. No manual labelling is required.

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (Anaconda or Miniconda)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/YikunHuang123/hvac-fdd.git
cd hvac-fdd
```

**2. Create and activate a conda environment**

```bash
conda create -n hvac python=3.11 -y
conda activate hvac
```

**3. Install dependencies**

```bash
pip install -e ".[dev]"
```

**4. Configure environment**

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/hvac_fdd

# Data paths
LBNL_DATA_DIR=data/LBNL_FDD_Data_Sets_SDAHU_all_3/LBNL_FDD_Dataset_SDAHU
PROCESSED_DATA_DIR=data/processed
MODELS_DIR=models

# API
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard
DASHBOARD_PORT=8501
```

**5. Apply database migrations**

```bash
alembic upgrade head
```

**6. Download and place the dataset**

Follow the [Dataset](#-dataset) section above.

---

## 💡 Usage

### Run the Ingestion Pipeline

Process the raw LBNL CSVs into Parquet and store detection events:

```bash
# Run all detectors
python scripts/run_pipeline.py --use-rules --use-if --use-clf

# Train Isolation Forest, then run it
python scripts/run_pipeline.py --train-if --use-if

# Train classifier, then run it
python scripts/run_pipeline.py --train-clf --use-clf

# Train both, run all detectors, persist to DB, and evaluate
python scripts/run_pipeline.py --train-if --train-clf --use-rules --use-if --use-clf --persist --evaluate
```

### Start the API Server

```bash
uvicorn hvac_fdd.api:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs are available at `http://localhost:8000/docs`.

**Key endpoints:**

```bash
# Query detection events (with filters and pagination)
GET /api/v1/detections?fault_type=COI_BIAS&alert_level=CRITICAL&limit=50

# Trigger the pipeline asynchronously (returns 202 Accepted)
POST /pipeline/run

# Check pipeline job status
GET /pipeline/jobs/{job_id}

# Summary statistics
GET /stats/

# Stats grouped by fault type
GET /stats/by-fault-type

# Health checks
GET /health/live
GET /health/ready
```

### Launch the Dashboard

```bash
streamlit run src/hvac_fdd/ui/Dashboard.py
```

Navigate to `http://localhost:8501`. The dashboard has four pages:

| Page | Content |
|---|---|
| **Detections** | Live detection event log with filters |
| **Analytics** | Fault frequency and trend charts |
| **Pipeline** | Run history and job status |
| **Evaluation** | Detector performance metrics and confusion matrices |

### Run Tests

```bash
pytest -v
```

Tests use an in-memory SQLite engine with auto-rollback transactions — no external database required.

---

## 🗂 Project Structure

```
hvac-fdd/
├── src/hvac_fdd/
│   ├── config.py               # Pydantic Settings — all env vars with defaults
│   ├── domain.py               # FaultType, AlertLevel, DetectionEvent
│   ├── exceptions.py           # Custom exception hierarchy
│   ├── ingestion/
│   │   ├── base.py             # DataLoaderBase abstract class
│   │   ├── lbnl_loader.py      # Reads 21 LBNL CSV files; infers fault type from filename
│   │   ├── pipeline.py         # Load → transform → features → save orchestration
│   │   ├── transforms.py       # wide_zones_to_long, F→C, column normalisation
│   │   └── features.py         # 16 engineered features (tracking errors, deltas, rolling stats)
│   ├── detection/
│   │   ├── base.py             # DetectorBase with fit/predict interface; 24 feature columns
│   │   ├── rules.py            # LBNLRulesDetector — 5 ASHRAE Guideline 36 rules
│   │   ├── isolation_forest.py # IsolationForestDetector — trains on NORMAL rows only
│   │   ├── classifier.py       # FaultClassifier — RandomForest, 6-class
│   │   └── ensemble.py         # EnsembleDetector — combines rules + IF + classifier
│   ├── evaluation/
│   │   ├── metrics.py          # detection_report, classification_report_extended, time_to_detect
│   │   └── report.py           # generate_report
│   ├── db/
│   │   ├── base.py             # Engine/session factory, declarative Base
│   │   ├── orm.py              # DetectionORM, PipelineJobORM
│   │   ├── detections.py       # DetectionRepository (filter, paginate)
│   │   └── jobs.py             # JobRepository
│   ├── api/
│   │   ├── __init__.py         # create_app() factory with lifespan management
│   │   ├── health.py           # /health/live  &  /health/ready
│   │   ├── detections.py       # GET /api/v1/detections
│   │   ├── pipeline.py         # POST /pipeline/run  &  GET /pipeline/jobs/{job_id}
│   │   ├── stats.py            # GET /stats/  &  GET /stats/by-fault-type
│   │   ├── schemas.py          # Request / response DTOs
│   │   ├── deps.py             # FastAPI dependency injection
│   │   └── middleware.py       # Request tracing
│   └── ui/
│       ├── Dashboard.py        # Streamlit entry point
│       ├── _shared.py          # Shared components and styling
│       ├── api_client.py       # HTTP client for the backend API
│       └── pages/
│           ├── 1_Detections.py
│           ├── 2_Analytics.py
│           ├── 3_Pipeline.py
│           └── 4_Evaluation.py
├── tests/
│   ├── conftest.py             # Fixtures: test_settings, SQLite engine, db_session
│   ├── test_ingestion.py       # Pipeline, transforms, feature engineering
│   ├── test_detection.py       # Rules, isolation forest, classifier
│   ├── test_db.py              # Repository CRUD, filtering, pagination
│   ├── test_api.py             # Endpoint response schemas and error handling
│   └── test_evaluation.py      # Metrics correctness
├── scripts/
│   └── run_pipeline.py         # CLI orchestrator (--train-if / --use-rules / --evaluate …)
├── migrations/
│   └── versions/
│       ├── 001_initial_schema.py
│       └── 002_add_fault_type_index.py
├── data/
│   ├── LBNL_FDD_Data_Sets_SDAHU_all_3/   # Raw CSV files (not in repo — download separately)
│   └── processed/                         # Parquet output
├── models/                                # Trained models: isolation_forest.joblib, classifier.joblib
├── pyproject.toml
├── alembic.ini
└── .env.example
```

---

## 🔮 Follow-up Development Plans


---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome.

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** — follow the existing code style (Ruff formatting).

3. **Add or update tests** for any new behaviour:
   ```bash
   pytest -v
   ```

4. **Commit** with a descriptive message:
   ```bash
   git commit -m "feat: add XYZ detection rule"
   ```

5. **Open a Pull Request** against `main`. Include:
   - A clear description of the change and its motivation
   - Steps to reproduce (for bug fixes)
   - Any relevant config or environment changes

**Reporting bugs:** Open a GitHub Issue with the label `bug`, your Python version, and a minimal reproduction snippet.

---

## 📄 License & Contact

This project is licensed under the **MIT License**.

**Author:** Yikun Huang  
**Email:** q1945948369@gmail.com  
**GitHub:** [@YikunHuang123](https://github.com/YikunHuang123)

> Built as an end-to-end HVAC fault detection engineering showcase — covering multi-strategy ML detection, async REST API design, interactive dashboard development, and production-grade data pipeline architecture.
