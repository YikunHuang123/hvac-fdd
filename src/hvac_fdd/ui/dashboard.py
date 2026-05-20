"""HVAC FDD Dashboard — Overview page (app entry point).

Run with:
    streamlit run src/hvac_fdd/ui/Dashboard.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="HVAC FDD — Overview",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import plotly.graph_objects as go

from hvac_fdd.ui._shared import (
    PLOTLY_LAYOUT,
    badge_html,
    connection_banner,
    get_api_url,
    inject_css,
    kpi_card,
    page_header,
    sec_title,
)
from hvac_fdd.ui.api_client import ALERT_PALETTE, FAULT_LABELS, FAULT_PALETTE, HVACAPIClient

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    if st.button("↺  Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ── Cached data fetchers ──────────────────────────────────────────────────────

api_url = get_api_url()


@st.cache_resource
def _client(url: str) -> HVACAPIClient:
    return HVACAPIClient(url)


@st.cache_data(ttl=30)
def _stats(url: str) -> dict:
    try:
        return _client(url).get_stats()
    except Exception:
        return {}


@st.cache_data(ttl=30)
def _recent(url: str, n: int = 10) -> list[dict]:
    try:
        return _client(url).get_detections(page_size=n).get("items", [])
    except Exception:
        return []


# ── Resolve data ──────────────────────────────────────────────────────────────

healthy = _client(api_url).is_live()
stats = _stats(api_url)
items = _recent(api_url)

# ── Page header + connection status ──────────────────────────────────────────

page_header("🌡️", "HVAC Fault Detection & Diagnostics", "Real-time system monitoring overview")
connection_banner(healthy, api_url)

if not healthy:
    st.warning(
        "The backend API is not reachable. "
        "Start it with: `uvicorn hvac_fdd.api:create_app --factory --port 8000`"
    )
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────

total: int = stats.get("total", 0)
by_alert: dict = stats.get("by_alert_level", {})
by_fault: dict = stats.get("by_fault_type", {})
by_equip: dict = stats.get("by_equipment", {})

critical = by_alert.get("CRITICAL", 0)
active_equip = len(by_equip)
active_faults = sum(1 for k, v in by_fault.items() if k not in ("unknown", "normal") and v > 0)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        kpi_card("Total Detections", f"{total:,}", "events recorded", "🔔", "#00D4B4"),
        unsafe_allow_html=True,
    )
with c2:
    accent = "#EF4444" if critical > 0 else "#10B981"
    st.markdown(
        kpi_card("Critical Faults", f"{critical:,}", "need immediate action", "🚨", accent),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        kpi_card("Active Equipment", str(active_equip), "units with events", "🏢", "#7C3AED"),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        kpi_card("Fault Types Active", str(active_faults), "distinct anomaly classes", "⚡", "#F59E0B"),
        unsafe_allow_html=True,
    )

# ── Charts row ────────────────────────────────────────────────────────────────

sec_title("Fault Analysis")
col_left, col_right = st.columns(2)

# Fault type donut
with col_left:
    st.markdown(
        "<p style='font-size:0.82rem;color:#4A6080;margin-bottom:6px'>Fault Type Distribution</p>",
        unsafe_allow_html=True,
    )
    if by_fault:
        keys = list(by_fault.keys())
        vals = list(by_fault.values())
        labels = [FAULT_LABELS.get(k, k) for k in keys]
        colors = [FAULT_PALETTE.get(k, "#6B7280") for k in keys]

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=vals,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#080C17", width=2)),
                textfont=dict(size=11, color="#E2E8F0"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[
                dict(
                    text=f"<b>{total:,}</b><br>events",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16, color="#F1F5F9"),
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No fault data yet.")

# Alert level bar
with col_right:
    st.markdown(
        "<p style='font-size:0.82rem;color:#4A6080;margin-bottom:6px'>Alert Level Breakdown</p>",
        unsafe_allow_html=True,
    )
    if by_alert:
        levels = list(by_alert.keys())
        counts = list(by_alert.values())
        colors = [ALERT_PALETTE.get(lv, "#6B7280") for lv in levels]

        fig = go.Figure(
            go.Bar(
                x=levels,
                y=counts,
                marker=dict(
                    color=[c + "CC" for c in colors],
                    line=dict(color=[c + "66" for c in colors], width=1),
                ),
                text=[f"{v:,}" for v in counts],
                textposition="outside",
                textfont=dict(color="#64748B", size=12),
                hovertemplate="<b>%{x}</b><br>%{y:,} events<extra></extra>",
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            margin=dict(l=40, r=20, t=20, b=40),
            yaxis_title="Event Count",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No alert data yet.")

# Equipment breakdown
if by_equip:
    sec_title("Equipment Overview")
    top_equip = sorted(by_equip.items(), key=lambda x: x[1], reverse=True)[:12]
    equip_names = [e[0] for e in top_equip]
    equip_counts = [e[1] for e in top_equip]

    n = len(top_equip)
    bar_colors = [
        f"rgba({int(0 + 60 * i / max(n - 1, 1))},{int(180 + 30 * i / max(n - 1, 1))},{int(160 + 20 * i / max(n - 1, 1))},0.8)"
        for i in range(n)
    ]

    fig = go.Figure(
        go.Bar(
            x=equip_counts,
            y=equip_names,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)", width=0)),
            hovertemplate="<b>%{y}</b><br>%{x:,} detections<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        margin=dict(l=90, r=40, t=20, b=40),
        height=max(220, n * 36),
        xaxis_title="Detection Count",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Recent events table ───────────────────────────────────────────────────────

sec_title("Recent Detection Events")
if items:
    df = pd.DataFrame(items)

    col_map = {
        "event_time": "Timestamp",
        "zone_id": "Zone",
        "equipment_id": "Equipment",
        "alert_level": "Alert",
        "predicted_fault": "Fault Type",
        "anomaly_index": "Anomaly Score",
        "detector_source": "Detector",
        "confidence": "Confidence",
    }
    df = df[[c for c in col_map if c in df.columns]].rename(columns=col_map)

    if "Fault Type" in df.columns:
        df["Fault Type"] = df["Fault Type"].apply(lambda x: FAULT_LABELS.get(str(x), str(x)) if x else "—")
    if "Anomaly Score" in df.columns:
        df["Anomaly Score"] = df["Anomaly Score"].round(3)
    if "Confidence" in df.columns:
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No events in the database. Trigger the pipeline from the Pipeline page.")
