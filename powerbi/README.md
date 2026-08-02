# Power BI 数据层

本目录保存 Power BI 使用的 PostgreSQL 分析视图定义。Power BI 连接分析视图，不直接读取 Streamlit 页面或原始 CSV。

## 应用视图

在 PostgreSQL 中执行：

```bash
psql "$DATABASE_URL" -f powerbi/sql/analytics_views.sql
```

当前视图位于 `analytics` schema：

- `analytics.vw_detection_events`：检测事件明细；
- `analytics.vw_detection_daily`：按日期、区域、设备、故障和告警等级聚合；
- `analytics.vw_pipeline_runs`：Pipeline 作业状态、耗时和处理量。

## Power BI 连接建议

使用 Power BI Desktop 的 PostgreSQL connector 连接数据库，并优先选择 Import 模式。报表页面建议分为管理总览、故障分析、Pipeline 与模型监控。生产环境再配置 Power BI Service 的网关或托管 PostgreSQL，避免把本地 Docker 数据库直接暴露到公网。

模型评估结果暂未写入 PostgreSQL，因此第一阶段不创建对应视图；待评估结果具有稳定表结构后再接入，避免把日志文件直接当作分析数据源。
