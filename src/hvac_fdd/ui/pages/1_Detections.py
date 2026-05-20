"""Detection Events page — filterable browser with timeline scatter."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="HVAC FDD — Detections",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import plotly.express as px

from hvac_fdd.ui._shared import (
    PLOTLY_LAYOUT,
    badge_html,
    connection_banner,
    get_api_url,
    inject_css,
    page_header,
    sec_title,
)
from hvac_fdd.ui.api_client import ALERT_PALETTE, FAULT_LABELS, FAULT_PALETTE, HVACAPIClient

inject_css()

# ── Sidebar — filters ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔎 Filters")

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("From", value=None, key="det_start")
    with date_col2:
        end_date = st.date_input("To", value=None, key="det_end")

    zone_filter: str = st.text_input("Zone ID", placeholder="e.g. ZONE-1", key="det_zone")

    alert_opts = ["", "CRITICAL", "WARNING", "INFO"]
    alert_filter: str = st.selectbox("Alert Level", alert_opts, index=0, key="det_alert")

    fault_opts = [""] + list(FAULT_LABELS.keys())
    fault_filter: str = st.selectbox(
        "Fault Type",
        fault_opts,
        format_func=lambda x: FAULT_LABELS.get(x, x) if x else "All",
        index=0,
        key="det_fault",
    )

    page_size: int = st.slider("Results per page", 50, 1000, 250, 50, key="det_pagesize")

    if st.button("↺  Apply Filters"):
        st.cache_data.clear()
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

api_url = get_api_url()


@st.cache_resource
def _client(url: str) -> HVACAPIClient:
    return HVACAPIClient(url)


@st.cache_data(ttl=20)
def _fetch(
    url: str,
    start: str | None,
    end: str | None,
    zone: str | None,
    alert: str | None,
    fault: str | None,
    size: int,
    skip: int,
) -> dict:
    try:
        return _client(url).get_detections(
            start=start, end=end,
            zone_id=zone or None, alert_level=alert or None,
            fault_type=fault or None,
            page_size=size, skip=skip,
        )
    except Exception as exc:
        return {"items": [], "total": 0, "error": str(exc)}


# ── Page header ───────────────────────────────────────────────────────────────

page_header("🔍", "Detection Events", "Browse and filter all recorded fault events")

healthy = _client(api_url).is_live()
if not healthy:
    st.warning("API is not reachable.")
    st.stop()

# ── Pagination state ──────────────────────────────────────────────────────────

if "det_skip" not in st.session_state:
    st.session_state["det_skip"] = 0

# ── Fetch data ────────────────────────────────────────────────────────────────

start_iso = start_date.isoformat() + "T00:00:00Z" if start_date else None
end_iso = end_date.isoformat() + "T23:59:59Z" if end_date else None

result = _fetch(
    api_url, start_iso, end_iso,
    zone_filter or None, alert_filter or None, fault_filter or None,
    page_size, st.session_state["det_skip"],
)

items = result.get("items", [])
total: int = result.get("total", 0)

if "error" in result:
    st.error(f"API error: {result['error']}")
    st.stop()

# ── Applied-filter chips ──────────────────────────────────────────────────────

active_filters = []
if start_date:
    active_filters.append(f"From: {start_date}")
if end_date:
    active_filters.append(f"To: {end_date}")
if zone_filter:
    active_filters.append(f"Zone: {zone_filter}")
if alert_filter:
    active_filters.append(f"Alert: {alert_filter}")
if fault_filter:
    active_filters.append(f"Fault: {FAULT_LABELS.get(fault_filter, fault_filter)}")

filter_html = " ".join(badge_html(f, "INFO") for f in active_filters) if active_filters else ""
st.markdown(
    f"<div style='margin-bottom:16px;font-size:0.82rem;color:#4A6080'>"
    f"Showing <b style='color:#F1F5F9'>{len(items)}</b> of "
    f"<b style='color:#F1F5F9'>{total:,}</b> total &nbsp; {filter_html}</div>",
    unsafe_allow_html=True,
)

if not items:
    st.info("No events match the current filters.")
    st.stop()

# ── Build dataframe ───────────────────────────────────────────────────────────

df = pd.DataFrame(items)
df["event_time"] = pd.to_datetime(df["event_time"])
df["predicted_fault_label"] = df.get("predicted_fault", pd.Series(dtype=str)).apply(
    lambda x: FAULT_LABELS.get(str(x), str(x)) if pd.notna(x) else "—"
)

# ── Timeline scatter ──────────────────────────────────────────────────────────

sec_title("Detection Timeline")

if "anomaly_index" in df.columns:
    color_map = {lv: ALERT_PALETTE[lv] for lv in ALERT_PALETTE}

    fig = px.scatter(
        df,
        x="event_time",
        y="anomaly_index",
        color="alert_level",
        color_discrete_map=color_map,
        hover_data={
            "zone_id": True,
            "predicted_fault_label": True,
            "detector_source": True,
            "event_time": "|%Y-%m-%d %H:%M",
        },
        labels={
            "event_time": "Time",
            "anomaly_index": "Anomaly Score",
            "alert_level": "Alert Level",
        },
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0)))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis_title="",
        yaxis_title="Anomaly Score",
        yaxis_range=[0, max(df["anomaly_index"].max() * 1.1, 1.05)],
        legend_title_text="Alert",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Data table ────────────────────────────────────────────────────────────────

sec_title("Event Records")

col_map = {
    "event_time": "Timestamp",
    "zone_id": "Zone",
    "equipment_id": "Equipment",
    "alert_level": "Alert",
    "predicted_fault_label": "Fault Type",
    "anomaly_index": "Anomaly Score",
    "confidence": "Confidence",
    "detector_source": "Detector",
    "violated_policy": "Policy",
    "trigger_signal": "Signal",
    "ground_truth": "Ground Truth",
}

display_df = df.rename(columns=col_map)[[c for c in col_map.values() if c in df.rename(columns=col_map).columns]]

if "Anomaly Score" in display_df.columns:
    display_df["Anomaly Score"] = display_df["Anomaly Score"].round(4)
if "Confidence" in display_df.columns:
    display_df["Confidence"] = display_df["Confidence"].apply(
        lambda x: f"{x:.0%}" if pd.notna(x) else "—"
    )
if "Timestamp" in display_df.columns:
    display_df["Timestamp"] = display_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

st.dataframe(display_df, use_container_width=True, hide_index=True, height=420)

# ── Pagination ────────────────────────────────────────────────────────────────

pc1, _, pc3 = st.columns([1, 6, 1])
skip = st.session_state["det_skip"]
with pc1:
    if skip > 0 and st.button("← Previous"):
        st.session_state["det_skip"] = max(0, skip - page_size)
        st.cache_data.clear()
        st.rerun()
with pc3:
    if skip + page_size < total and st.button("Next →"):
        st.session_state["det_skip"] = skip + page_size
        st.cache_data.clear()
        st.rerun()
