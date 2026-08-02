# HVAC-FDD: End-to-End Fault Detection and Diagnostics System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HVAC-FDD** is an enterprise-grade, end-to-end fault detection and diagnostics system for Heating, Ventilation, and Air Conditioning (HVAC) environments. Built on the LBNL SDAHU dataset, it transforms vast amounts of complex air-handling-unit (AHU) time-series data into traceable detection events and Power BI-ready reports.

The project integrates **expert physical rules** with **machine learning models**, establishing a complete pipeline from ETL data ingestion, model inference, and PostgreSQL persistence to FastAPI microservices and BI analytics.

---

## 🚀 Core Features

- **Hybrid-Driven Detection Engine**
  - **Expert Rules Layer (APAR)**: Utilizes domain physics knowledge to rapidly filter out definitive control anomalies, ensuring high interpretability.
  - **Unsupervised Anomaly Detection**: Supports Gaussian Mixture Models (GMM), Isolation Forest, and **Kolmogorov-Arnold Networks (KAN-AD)** to capture implicit anomalies under unknown operational conditions.
  - **Supervised Fault Classification**: Supports XGBoost, Random Forest, Hierarchical XGBoost, and **Temporal Convolutional Networks (TCN)** to assist in labeling fault root causes.
- **Model Evaluation System**
  - Supports **Annual / Common Temporal Split** to strictly prevent future data leakage.
  - Utilizes **Leave-one-severity-out** cross-validation to guarantee robust generalization across unseen, extreme fault conditions.
- **System Deployment**
  - Decouples binary anomaly detection from fault classification, supporting low-memory streaming inference.
  - Persists detection events and pipeline job states to PostgreSQL, exposes them via standard RESTful APIs with FastAPI, and natively integrates with Power BI analytics views.

## 🛠️ Tech Stack

| Domain | Core Technologies |
| :--- | :--- |
| **Core Services** | Python 3.11, FastAPI, Uvicorn, Pydantic |
| **Algorithms & DL** | Scikit-learn, XGBoost, PyTorch (TCN, KAN) |
| **Data Engineering** | Pandas, Numpy |
| **Storage & DB** | PostgreSQL, Alembic, SQLAlchemy |
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

System evaluation is based on rigorous real-world temporal splits (Apr–Aug training, Oct final holdout test) and an unseen severity protocol (Leave-one-severity-out).

### Stage 1: Binary Anomaly Detection
The first layer is responsible for core alerting. Under the same-scenario temporal split, the performance of each model is as follows:

| Detection Architecture | Normal Scenario FPR | Fault Recall | Precision | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Physical Rules (Rules)** | 20.11% | 67.62% | 98.54% | 80.20% |
| **Rules + GMM** (FPR Constrained) | 30.16% | 75.77% | 98.05% | 85.48% |
| **Temporal Deep Model (TCN)** | 2.76% | 80.03% | 99.83% | 88.84% |

> **Stage 1 Architecture Decision**: Although TCN performed exceptionally well in the same-scenario temporal split, rigorous cross-scenario testing with unseen severities revealed that its cross-domain Recall is less stable than Rules+GMM. Based on industrial safety fallback considerations, the system ultimately adopts **"Rules-led detection + GMM-constrained enhancement"** as the primary production choice.

### Stage 2: Fault Type Classification
The second layer is responsible for specific fault root cause labeling (6-class) on the identified anomalies. We evolved multiple cutting-edge architectures during R&D:

| Diagnostic Architecture | Evaluation Protocol | Accuracy | Macro-F1 | Architectural Features & Insights |
| :--- | :--- | :--- | :--- | :--- |
| **GNN (Graph Neural Network)** | Annual Temporal Split | 77.26% | 0.6974 | Lack of complex real device topology; forced graph building caused computation overhead and poor performance. |
| **TCN (Temporal ConvNet)** | Annual Temporal Split | 86.28% | 0.8079 | Incorporates long-term thermal inertia and supports large-scale sliding windows via memory optimization (Zero-copy Slicing). |
| **Hierarchical Cascade XGBoost** | Annual Temporal Split | - | 0.8571 | Step-by-step diagnosis by physical subsystems; requires attention to error amplification during probability cascading. |
| **Flat XGBoost** | Same-scenario future month | **95.59%** | **0.9472** | Serves as a baseline: demonstrates stable classification performance on tabular scalar data. |
| **Flat XGBoost** | Leave-one unseen severity | 99.30%* | **0.1661** | **Generalization Trap Warning**: *A highly imbalanced single unseen scenario caused inflated Accuracy, but Macro-F1 revealed the true cross-domain failure of the classifier, validating the necessity of the decoupled two-stage fallback.* |

### 🎯 Default Production Architecture
Based on the validation results above, to ensure generalization and interpretability in industrial environments, the project defaults to the following **hybrid detection architecture**:
1. **Binary Alert Backbone**: **Expert Physical Rules (APAR)** serve as the baseline, providing highly confident and fully interpretable anomaly detection.
2. **Alert Recall Enhancement**: **Gaussian Mixture Model (GMM)** supplements the detection of subtle anomalies under a constrained False Positive Rate (FPR).
3. **Fault Root-cause Labeling**: **XGBoost (6-class)** infers specific fault root-cause labels only after an anomaly is triggered.

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
pip install -e ".[dev]"
cp .env.example .env
```
Edit `.env` to configure `DATABASE_URL` (e.g., `postgresql://hvac:hvac@localhost:5432/hvac_fdd`) and `LBNL_DATA_DIR`.

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
