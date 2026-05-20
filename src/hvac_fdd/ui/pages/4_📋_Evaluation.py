"""Evaluation page — per-fault precision/recall/F1 and confusion matrix
computed from ground_truth vs predicted_fault in the detections table."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="HVAC FDD — Evaluation",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hvac_fdd.ui._shared import (
    PLOTLY_LAYOUT,
    connection_banner,
    inject_css,
    kpi_card,
    page_header,
    sec_title,
)
from hvac_fdd.ui.api_client import FAULT_LABELS, FAULT_PALETTE, HVACAPIClient

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Connection")
    api_url: str = st.text_input("API Base URL", value="http://localhost:8000", key="api_url_eval")
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.78rem;color:#3D5070;line-height:1.6'>
            <b style='color:#64748B'>About this page</b><br>
            Metrics are computed client-side from stored detection events that
            have both <code>ground_truth</code> and <code>predicted_fault</code>
            fields populated (i.e., labelled events from the LBNL dataset).
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("↺  Refresh"):
        st.cache_data.clear()
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────


@st.cache_resource
def _client(url: str) -> HVACAPIClient:
    return HVACAPIClient(url)


@st.cache_data(ttl=60)
def _fetch_all(url: str) -> list[dict]:
    try:
        return _client(url).get_detections(page_size=1000).get("items", [])
    except Exception:
        return []


def _per_class_metrics(
    y_true: pd.Series, y_pred: pd.Series, classes: list[str]
) -> pd.DataFrame:
    """Binary one-vs-rest P/R/F1 for each class."""
    rows = []
    for cls in classes:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append({"fault": cls, "label": FAULT_LABELS.get(cls, cls),
                     "P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn})
    return pd.DataFrame(rows)


# ── Page header ───────────────────────────────────────────────────────────────

page_header("📋", "Model Evaluation", "Precision, recall, F1, and confusion matrix from stored detections")

healthy = _client(api_url).is_live()
connection_banner(healthy, api_url)
if not healthy:
    st.warning("API is not reachable.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────

raw = _fetch_all(api_url)
if not raw:
    st.info("No detections found. Run the ingestion pipeline first.")
    st.stop()

df = pd.DataFrame(raw)

# Keep only labelled rows (have ground_truth)
labelled = df.dropna(subset=["ground_truth", "predicted_fault"]).copy()
labelled = labelled[labelled["ground_truth"] != "normal"]

if labelled.empty:
    st.warning(
        "No labelled events (with `ground_truth` populated) found in the database. "
        "Evaluation requires events generated from the LBNL labelled dataset."
    )
    st.stop()

y_true = labelled["ground_truth"].str.strip()
y_pred = labelled["predicted_fault"].str.strip()

classes = sorted(y_true.unique().tolist())
metrics_df = _per_class_metrics(y_true, y_pred, classes)

overall_f1 = metrics_df["F1"].mean()
overall_p = metrics_df["P"].mean()
overall_r = metrics_df["R"].mean()
n_labelled = len(labelled)

# ── KPI cards ─────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    accent = "#10B981" if overall_f1 >= 0.7 else "#F59E0B" if overall_f1 >= 0.5 else "#EF4444"
    st.markdown(
        kpi_card("Macro F1", f"{overall_f1:.3f}", "avg across fault types", "🎯", accent),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        kpi_card("Macro Precision", f"{overall_p:.3f}", "avg one-vs-rest", "🔬", "#7C3AED"),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        kpi_card("Macro Recall", f"{overall_r:.3f}", "avg one-vs-rest", "📡", "#3B82F6"),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        kpi_card("Evaluated Events", f"{n_labelled:,}", "with ground truth", "🗂️", "#00D4B4"),
        unsafe_allow_html=True,
    )

# ── Per-fault-type performance ────────────────────────────────────────────────

sec_title("Per-Fault-Type Performance")

colors = [FAULT_PALETTE.get(f, "#6B7280") for f in metrics_df["fault"]]

fig = go.Figure()
for metric, opacity in [("F1", 0.90), ("P", 0.65), ("R", 0.45)]:
    fig.add_trace(
        go.Bar(
            name={"F1": "F1 Score", "P": "Precision", "R": "Recall"}[metric],
            x=metrics_df["label"],
            y=metrics_df[metric],
            marker=dict(
                color=[c + f"{int(255 * opacity):02X}" for c in colors],
                line=dict(color=[c + "44" for c in colors], width=1),
            ),
            text=[f"{v:.2f}" for v in metrics_df[metric]],
            textposition="auto",
            textfont=dict(size=10, color="#E2E8F0"),
            hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.3f}<extra></extra>",
        )
    )

fig.update_layout(
    **PLOTLY_LAYOUT,
    barmode="group",
    height=360,
    yaxis_range=[0, 1.08],
    yaxis_title="Score",
    xaxis_title="",
    legend=dict(**PLOTLY_LAYOUT["legend"], orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Confusion matrix ──────────────────────────────────────────────────────────

sec_title("Confusion Matrix")

cm_col, detail_col = st.columns([3, 2])

with cm_col:
    all_classes = sorted(set(y_true.unique()) | set(y_pred.unique()))
    class_labels = [FAULT_LABELS.get(c, c) for c in all_classes]

    cm_data = []
    for true_cls in all_classes:
        row = []
        for pred_cls in all_classes:
            row.append(int(((y_true == true_cls) & (y_pred == pred_cls)).sum()))
        cm_data.append(row)

    cm_array = np.array(cm_data, dtype=float)
    row_sums = cm_array.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_array / row_sums, 0.0)

    hover_text = [
        [f"True: <b>{class_labels[i]}</b><br>Pred: <b>{class_labels[j]}</b><br>Count: {int(cm_data[i][j])}<br>Rate: {cm_norm[i][j]:.1%}"
         for j in range(len(all_classes))]
        for i in range(len(all_classes))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=cm_norm,
            x=class_labels,
            y=class_labels,
            text=[[str(int(v)) for v in row] for row in cm_data],
            texttemplate="%{text}",
            textfont=dict(size=12, color="#F1F5F9"),
            colorscale=[[0, "#080C17"], [0.3, "#003050"], [0.6, "#007090"], [1.0, "#00D4B4"]],
            showscale=True,
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            colorbar=dict(
                title=dict(text="Rate", font=dict(color="#64748B", size=11)),
                tickformat=".0%",
                tickfont=dict(color="#64748B", size=10),
                outlinecolor="#1E2D40",
                outlinewidth=1,
            ),
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(300, len(all_classes) * 55 + 80),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], side="bottom"),
        margin=dict(l=100, r=60, t=30, b=80),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with detail_col:
    sec_title("Metrics Table")
    display = metrics_df[["label", "F1", "P", "R", "TP", "FP", "FN"]].rename(
        columns={"label": "Fault Type", "P": "Precision", "R": "Recall"}
    )
    display["F1"] = display["F1"].map("{:.3f}".format)
    display["Precision"] = display["Precision"].map("{:.3f}".format)
    display["Recall"] = display["Recall"].map("{:.3f}".format)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <div style='background:#0C1220;border:1px solid #1A2540;
                    border-radius:10px;padding:14px 16px;margin-top:16px;
                    font-size:0.78rem;color:#4A6080;line-height:1.7'>
            <b style='color:#64748B'>Methodology</b><br>
            Metrics are binary one-vs-rest per fault type.<br>
            Precision_X = TP / (TP + FP)<br>
            Recall_X = TP / (TP + FN)<br>
            F1 = harmonic mean of P and R<br><br>
            <b style='color:#64748B'>Dataset</b><br>
            {n_labelled:,} labelled events · {len(classes)} fault classes
        </div>
        """,
        unsafe_allow_html=True,
    )
