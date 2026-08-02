# HVAC-FDD

面向 LBNL SDAHU 单风道空调机组数据集的 HVAC 故障检测与诊断项目。

> **项目状态：** 离线研究与端到端演示，不是生产楼宇控制系统。报告指标时应同时注明模型、数据切分、阈值策略和正常参考数据误报率。

## 项目功能

- 加载 LBNL SDAHU CSV，从文件名推断故障标签并完成特征工程。
- 使用 Rules 以及可选的 GMM、Isolation Forest、KAN 检测异常。
- 使用 XGBoost、Random Forest、Hierarchical XGBoost 或 TCN 做可选故障类型预测。
- 支持时间切分和 leave-one-severity-out 泛化测试。
- 将检测事件和 Pipeline 任务状态持久化到 PostgreSQL。
- 通过 FastAPI 提供数据接口，并通过 PostgreSQL 分析视图连接 Power BI。

## 系统架构

```text
LBNL CSV -> 摄取 -> 检测 -> 评估 -> PostgreSQL
                                      |-> FastAPI 数据接口
                                      |-> analytics 视图 -> Power BI
```

推荐配置为 **Rules + GMM + XGBoost**。GNN 仅保留为实验实现；TCN 可用于对比，但在 leave-one-severity-out 分类泛化测试中表现较弱。

| 层级 | 可用模型 | 作用 |
|---|---|---|
| 二元检测 | Rules | 可解释的主检测器 |
| 无监督补充 | GMM、Isolation Forest、KAN | 异常分数对比 |
| 故障分类 | XGBoost、Random Forest、Hierarchical XGBoost、TCN | 可选故障类型预测 |

## 数据与评估

原始 LBNL 数据集不包含在仓库中。下载后将 `LBNL_DATA_DIR` 指向 `data/LBNL_FDD_Data_Sets_SDAHU_all_3/LBNL_FDD_Dataset_SDAHU`。支持年度时间切分、common 时间切分和 leave-one-severity-out。无监督模型只使用正常行训练；检测指标和故障分类指标分别报告。

## 本地运行

```bash
conda create -n hvac python=3.11 -y
conda activate hvac
pip install -e ".[dev]"
cp .env.example .env
alembic upgrade head
```

设置 `DATABASE_URL=postgresql://hvac:hvac@localhost:5432/hvac_fdd`，配置数据目录，选择并训练模型：

```bash
export UNSUPERVISED_MODEL=gmm       # gmm、if 或 kan
export SUPERVISED_MODEL=xgboost     # xgboost、random_forest、hierarchical_xgb 或 tcn
python scripts/preprocess_data.py
python scripts/run_pipeline.py --train-unsup --train-clf
python scripts/run_pipeline.py --use-rules --use-unsup --use-clf --persist
uvicorn hvac_fdd.api:create_app --factory --host 0.0.0.0 --port 8000
```

API 文档：<http://localhost:8000/docs>。

## Docker Compose 运行

Compose 只启动 PostgreSQL 和 FastAPI，不会自动训练或检测：

```bash
docker compose up --build -d
docker compose ps
curl http://localhost:8000/health/live
```

API 地址为 <http://localhost:8000>，数据库为 `localhost:5432`（用户 `hvac`，密码 `hvac_fdd`）。创建 Power BI 视图：

```bash
psql postgresql://hvac:hvac@localhost:5432/hvac_fdd -f powerbi/sql/analytics_views.sql
```

停止服务：`docker compose down`。只有需要删除 PostgreSQL 数据卷时才使用 `docker compose down -v`。

## API 接口

`GET /health/live`、`GET /health/ready`、`GET /api/v1/detections/`、`GET /stats/`、`GET /stats/by-fault-type`、`POST /pipeline/run`、`GET /pipeline/jobs/{job_id}` 和 `/docs`。

## Power BI 分析层

Power BI 是项目的主要报表和分析界面。Power BI Desktop 连接 PostgreSQL 的 `analytics` schema，读取 `vw_detection_events`、`vw_detection_daily` 和 `vw_pipeline_runs`，而不是原始 CSV 或应用页面状态。

## 已记录的离线对比

该基准使用 1-11 月训练、12 月留出测试，不是实时数据库指标：

| 配置 | Recall | Precision | F1 |
|---|---:|---:|---:|
| Rules baseline | 66.92% | 97.97% | 0.795 |
| Rules + Isolation Forest | 74.13% | 96.67% | 0.839 |
| Rules + GMM | 77.94% | 96.81% | 0.864 |
| Rules + GMM，阈值校准 | 79.32% | 96.73% | 0.872 |

`coi_stuck` 和 `damper_stuck` 效果最好；`oa_bias` 与 `coi_bias` 较难检测。检测成功不等于准确识别故障类型。

## 目录结构

```text
src/hvac_fdd/ingestion/   数据加载、转换和特征工程
src/hvac_fdd/detection/   检测器和分类器
src/hvac_fdd/evaluation/  指标与报告
src/hvac_fdd/db/          数据库模型与仓储
src/hvac_fdd/api/         FastAPI 路由与 schema
powerbi/                  PostgreSQL 分析视图
scripts/                  预处理、训练、检测和评估脚本
tests/                    单元测试与 API 测试
migrations/               Alembic 迁移
```

## 测试

```bash
pytest -v
```
