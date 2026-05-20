"""Analytics page — time-series trends, heatmaps, score distributions."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="HVAC FDD — Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hvac_fdd.ui._shared import (
    PLOTLY_LAYOUT,
    connection_banner,
    inject_css,
    page_header,
    sec_title,
)
from hvac_fdd.ui.api_client import ALERT_PALETTE, FAULT_LABELS, FAULT_PALETTE, HVACAPIClient

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Connection")
    api_url: str = st.text_input("API Base URL", value="http://localhost:8000", key="api_url_ana")
    st.divider()

    st.markdown("### 📊 Options")
    time_grain = st.selectbox("Time Granularity", ["Hourly", "Daily", "Weekly"], index=1)
    top_n_equip = st.slider("Top N Equipment", 3, 15, 8)

    if st.button("↺  Refresh"):
        st.cache_data.clear()
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────


@st.cache_resource
def _client(url: str) -> HVACAPIClient:
    return HVACAPIClient(url)


@st.cache_data(ttl=60)
def _all_detections(url: str) -> list[dict]:
    """Fetch up to 1000 records for client-side analytics."""
    try:
        return _client(url).get_detections(page_size=1000).get("items", [])
    except Exception:
        return []


@st.cache_data(ttl=60)
def _stats(url: str) -> dict:
    try:
        return _client(url).get_stats()
    except Exception:
        return {}


# ── Page header ───────────────────────────────────────────────────────────────

page_header("📈", "Analytics & Trends", "Patterns, distributions, and equipment insights")

healthy = _client(api_url).is_live()
connection_banner(healthy, api_url)
if not healthy:
    st.warning("API is not reachable.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────

raw = _all_detections(api_url)
stats = _stats(api_url)

if not raw:
    st.info("No detection events found. Run the ingestion pipeline to populate data.")
    st.stop()

df = pd.DataFrame(raw)
df["event_time"] = pd.to_datetime(df["event_time"])
df["predicted_fault"] = df["predicted_fault"].fillna("unknown")
df["fault_label"] = df["predicted_fault"].map(FAULT_LABELS).fillna(df["predicted_fault"])

grain_map = {"Hourly": "h", "Daily": "D", "Weekly": "W"}
freq = grain_map[time_grain]

# ── Detections over time (stacked bar) ───────────────────────────────────────

sec_title("Detections Over Time")

df_time = (
    df.set_index("event_time")
    .groupby([pd.Grouper(freq=freq), "alert_level"])
    .size()
    .reset_index(name="count")
    .rename(columns={"event_time": "period"})
)

if not df_time.empty:
    fig = px.bar(
        df_time,
        x="period",
        y="count",
        color="alert_level",
        color_discrete_map=ALERT_PALETTE,
        barmode="stack",
        labels={"period": "", "count": "Events", "alert_level": "Alert Level"},
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=290,
        bargap=0.15,
        legend_title_text="Alert Level",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Two-column: fault type trend + anomaly score distribution ────────────────

sec_title("Fault Type Trends & Score Distribution")
col_l, col_r = st.columns([3, 2])

with col_l:
    st.markdown(
        "<p style='font-size:0.82rem;color:#4A6080;margin-bottom:6px'>Fault Type Over Time</p>",
        unsafe_allow_html=True,
    )
    df_fault_time = (
        df[df["predicted_fault"] != "normal"]
        .set_index("event_time")
        .groupby([pd.Grouper(freq=freq), "fault_label"])
        .size()
        .reset_index(name="count")
        .rename(columns={"event_time": "period"})
    )
    if not df_fault_time.empty:
        unique_faults = df_fault_time["fault_label"].unique()
        color_map = {FAULT_LABELS.get(k, k): FAULT_PALETTE.get(k, "#6B7280") for k in FAULT_PALETTE}

        fig = px.area(
            df_fault_time,
            x="period",
            y="count",
            color="fault_label",
            color_discrete_map=color_map,
            labels={"period": "", "count": "Events", "fault_label": "Fault Type"},
        )
        fig.update_traces(opacity=0.75, line=dict(width=1.5))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            legend_title_text="Fault Type",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No fault data in the selected window.")

with col_r:
    st.markdown(
        "<p style='font-size:0.82rem;color:#4A6080;margin-bottom:6px'>Anomaly Score Distribution</p>",
        unsafe_allow_html=True,
    )
    if "anomaly_index" in df.columns:
        fig = go.Figure()
        for level, color in ALERT_PALETTE.items():
            subset = df[df["alert_level"] == level]["anomaly_index"]
            if subset.empty:
                continue
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    name=level,
                    nbinsx=25,
                    marker_color=color + "BB",
                    marker_line=dict(color=color + "44", width=0.5),
                    opacity=0.8,
                    hovertemplate=f"<b>{level}</b><br>Score: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
                )
            )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="overlay",
            height=320,
            xaxis_title="Anomaly Score",
            yaxis_title="Frequency",
            legend_title_text="Alert Level",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Zone × Fault heatmap ──────────────────────────────────────────────────────

sec_title("Zone × Fault Type Heatmap")

if "zone_id" in df.columns and "predicted_fault" in df.columns:
    heatmap_df = (
        df[df["predicted_fault"] != "normal"]
        .groupby(["zone_id", "fault_label"])
        .size()
        .reset_index(name="count")
    )

    if not heatmap_df.empty:
        pivot = heatmap_df.pivot(index="zone_id", columns="fault_label", values="count").fillna(0)

        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[[0, "#0A0E1A"], [0.3, "#00405A"], [0.6, "#00A09A"], [1.0, "#00D4B4"]],
                showscale=True,
                hovertemplate="Zone: <b>%{y}</b><br>Fault: <b>%{x}</b><br>Count: %{z}<extra></extra>",
                colorbar=dict(
                    tickfont=dict(color="#64748B", size=10),
                    outlinecolor="#1E2D40",
                    outlinewidth=1,
                ),
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=max(280, len(pivot) * 40 + 80),
            xaxis_title="",
            yaxis_title="",
            margin=dict(l=90, r=60, t=30, b=60),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No non-normal fault events to display in heatmap.")

# ── Top equipment ranked by detections ───────────────────────────────────────

sec_title("Equipment Ranking")

if "equipment_id" in df.columns:
    equip_counts = (
        df.groupby(["equipment_id", "alert_level"])
        .size()
        .reset_index(name="count")
    )
    top_equip = (
        equip_counts.groupby("equipment_id")["count"]
        .sum()
        .nlargest(top_n_equip)
        .index.tolist()
    )
    equip_filtered = equip_counts[equip_counts["equipment_id"].isin(top_equip)]

    if not equip_filtered.empty:
        fig = px.bar(
            equip_filtered,
            x="count",
            y="equipment_id",
            color="alert_level",
            color_discrete_map=ALERT_PALETTE,
            orientation="h",
            barmode="stack",
            labels={"count": "Event Count", "equipment_id": "", "alert_level": "Alert Level"},
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=max(240, top_n_equip * 40 + 80),
            margin=dict(l=90, r=40, t=20, b=40),
            legend_title_text="Alert Level",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Summary stats row ─────────────────────────────────────────────────────────

sec_title("Quick Statistics")
q1, q2, q3, q4 = st.columns(4)

with q1:
    if "anomaly_index" in df.columns:
        st.metric("Mean Anomaly Score", f"{df['anomaly_index'].mean():.3f}")
with q2:
    if "anomaly_index" in df.columns:
        st.metric("Max Anomaly Score", f"{df['anomaly_index'].max():.3f}")
with q3:
    unique_zones = df["zone_id"].nunique() if "zone_id" in df.columns else 0
    st.metric("Unique Zones", str(unique_zones))
with q4:
    unique_detectors = df["detector_source"].nunique() if "detector_source" in df.columns else 0
    st.metric("Detector Sources", str(unique_detectors))
