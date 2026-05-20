"""Pipeline Control page — trigger ingestion jobs and monitor status."""
from __future__ import annotations

import time

import streamlit as st

st.set_page_config(
    page_title="HVAC FDD — Pipeline",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd

from hvac_fdd.ui._shared import (
    badge_html,
    connection_banner,
    get_api_url,
    inject_css,
    page_header,
    sec_title,
)
from hvac_fdd.ui.api_client import SCENARIOS, HVACAPIClient

inject_css()

# ── Helpers ───────────────────────────────────────────────────────────────────

api_url = get_api_url()


@st.cache_resource
def _client(url: str) -> HVACAPIClient:
    return HVACAPIClient(url)


def _status_color(status: str) -> str:
    return {
        "pending": "#9CA3AF",
        "running": "#3B82F6",
        "done": "#10B981",
        "failed": "#EF4444",
    }.get(status, "#6B7280")


# ── Page header ───────────────────────────────────────────────────────────────

page_header("⚙️", "Pipeline Control", "Trigger ingestion jobs and monitor execution status")

healthy = _client(api_url).is_live()
if not healthy:
    st.warning("API is not reachable.")
    st.stop()

# ── Trigger panel ─────────────────────────────────────────────────────────────

sec_title("Run Pipeline")

trig_col, info_col = st.columns([2, 3])

with trig_col:
    st.markdown(
        """
        <div style='background:linear-gradient(145deg,#0F1624,#141E30);
                    border:1px solid #1A2740;border-radius:16px;padding:24px;'>
            <div style='font-size:0.72rem;font-weight:600;color:#3D5070;
                        text-transform:uppercase;letter-spacing:1.3px;margin-bottom:12px'>
                Ingestion Scenario
            </div>
        """,
        unsafe_allow_html=True,
    )
    scenario = st.selectbox(
        "Scenario",
        SCENARIOS,
        index=0,
        format_func=lambda x: x.replace("_", " ").title() if x != "all" else "All Scenarios",
        label_visibility="collapsed",
        key="pipe_scenario",
    )
    trigger_btn = st.button("▶  Trigger Pipeline", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    st.markdown(
        """
        <div style='background:#0C1220;border:1px solid #1A2540;border-radius:12px;
                    padding:20px;font-size:0.82rem;color:#4A6080;line-height:1.7'>
            <b style='color:#94A3B8'>How it works:</b><br>
            Triggering a pipeline run starts an asynchronous background job that:<br>
            &nbsp;&nbsp;① Loads LBNL CSV data files from the configured data directory<br>
            &nbsp;&nbsp;② Applies transforms and feature engineering<br>
            &nbsp;&nbsp;③ Runs the ensemble fault detector<br>
            &nbsp;&nbsp;④ Persists detection events to the database<br><br>
            The job returns immediately with status <code>pending</code>.
            Poll status below.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Handle trigger ────────────────────────────────────────────────────────────

if trigger_btn:
    with st.spinner("Submitting pipeline job…"):
        try:
            job = _client(api_url).trigger_pipeline(scenario)
            st.session_state["last_job_id"] = job["id"]
            st.success(f"Pipeline job #{job['id']} created — status: `{job['status']}`")
        except Exception as exc:
            st.error(f"Failed to trigger pipeline: {exc}")

# ── Live job status ───────────────────────────────────────────────────────────

if "last_job_id" in st.session_state:
    job_id = st.session_state["last_job_id"]
    sec_title(f"Job #{job_id} — Status")

    status_placeholder = st.empty()
    poll_rounds = 0
    max_polls = 30  # poll up to 30 × 2s = 60s before giving up

    while poll_rounds < max_polls:
        try:
            job = _client(api_url).get_job(job_id)
        except Exception as exc:
            status_placeholder.error(f"Could not fetch job status: {exc}")
            break

        status = job.get("status", "unknown")
        records = job.get("records_processed")
        anomalies = job.get("anomalies_found")
        error_msg = job.get("error_msg")
        finished = job.get("finished_at")

        badge = badge_html(status.upper(), status)
        detail_rows = ""
        if records is not None:
            detail_rows += f"<tr><td style='color:#4A6080'>Records processed</td><td style='color:#F1F5F9'>{records:,}</td></tr>"
        if anomalies is not None:
            detail_rows += f"<tr><td style='color:#4A6080'>Anomalies found</td><td style='color:#F1F5F9'>{anomalies:,}</td></tr>"
        if finished:
            detail_rows += f"<tr><td style='color:#4A6080'>Finished at</td><td style='color:#F1F5F9'>{finished}</td></tr>"
        if error_msg:
            detail_rows += f"<tr><td style='color:#EF4444'>Error</td><td style='color:#EF4444'>{error_msg}</td></tr>"

        status_placeholder.markdown(
            f"""
            <div style='background:#0F1624;border:1px solid #1A2740;
                        border-radius:12px;padding:20px;'>
                <table style='width:100%;border-collapse:collapse'>
                    <tr>
                        <td style='color:#4A6080;width:160px'>Job ID</td>
                        <td style='color:#F1F5F9'>#{job_id}</td>
                    </tr>
                    <tr>
                        <td style='color:#4A6080'>Status</td>
                        <td>{badge}</td>
                    </tr>
                    <tr>
                        <td style='color:#4A6080'>Scenario</td>
                        <td style='color:#F1F5F9'>{job.get('scenario','—')}</td>
                    </tr>
                    {detail_rows}
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if status in ("done", "failed"):
            break

        time.sleep(2)
        poll_rounds += 1

# ── Job history ───────────────────────────────────────────────────────────────

sec_title("Recent Jobs")

st.markdown(
    "<p style='font-size:0.8rem;color:#3D5070;margin-bottom:12px'>"
    "Past pipeline jobs are stored in the database. "
    "Fetch individual job IDs above. "
    "Full job history requires a dedicated API endpoint (future work).</p>",
    unsafe_allow_html=True,
)

if "last_job_id" in st.session_state:
    job_ids_to_show = list(range(
        max(1, st.session_state["last_job_id"] - 9),
        st.session_state["last_job_id"] + 1,
    ))
    rows = []
    for jid in reversed(job_ids_to_show):
        try:
            j = _client(api_url).get_job(jid)
            rows.append({
                "Job ID": f"#{j['id']}",
                "Scenario": j.get("scenario", "—"),
                "Status": j.get("status", "—").upper(),
                "Started": (j.get("started_at") or "—")[:19].replace("T", " "),
                "Finished": (j.get("finished_at") or "—")[:19].replace("T", " "),
                "Records": j.get("records_processed") or "—",
                "Anomalies": j.get("anomalies_found") or "—",
            })
        except Exception:
            pass  # job doesn't exist yet

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No job history to display.")
else:
    st.info("Trigger a pipeline run above to see job history here.")
