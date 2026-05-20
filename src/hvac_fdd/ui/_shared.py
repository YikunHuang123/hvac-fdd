from __future__ import annotations

import streamlit as st
from hvac_fdd.config import get_settings

def get_api_url() -> str:
    """Retrieve the API base URL from settings."""
    settings = get_settings()
    # If the host is 0.0.0.0 (bind all), use localhost for the UI connection.
    host = "localhost" if settings.api_host == "0.0.0.0" else settings.api_host
    return f"http://{host}:{settings.api_port}"

# ── Plotly chart layout (applied to every chart) ──────────────────────────────

PLOTLY_LAYOUT: dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94A3B8", size=12),
    title_font=dict(color="#E2E8F0", size=14, family="Inter, sans-serif"),
    xaxis=dict(
        gridcolor="#1E2D40",
        linecolor="#1E2D40",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#94A3B8"),
    ),
    yaxis=dict(
        gridcolor="#1E2D40",
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#94A3B8"),
    ),
    legend=dict(
        bgcolor="rgba(14,20,35,0.9)",
        bordercolor="#1E2D40",
        borderwidth=1,
        font=dict(color="#94A3B8", size=11),
    ),
    margin=dict(l=50, r=20, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#1A2340",
        bordercolor="#1E2D40",
        font=dict(color="#F1F5F9", size=12),
    ),
)

# ── CSS injected into every page ──────────────────────────────────────────────

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ──────────────────────────────────────────────────────────────── */
html, body, .stApp {
    background-color: #080C17 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 2rem !important;
    max-width: 1440px !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0C1220 !important;
    border-right: 1px solid #1A2540 !important;
}
[data-testid="stSidebar"] .stMarkdown h3 { 
    color: #00D4B4 !important; 
    font-size: 1.3rem !important;
    margin-bottom: 20px !important;
}

/* Sidebar Toggle Buttons (Collapse & Expand) */
/* Target the buttons in both the sidebar and the main header */
header[data-testid="stHeader"] button, 
[data-testid="stSidebar"] button[kind="headerNoPadding"] {
    background-color: transparent !important;
    color: #00D4B4 !important;
}

header[data-testid="stHeader"] button svg, 
[data-testid="stSidebar"] button[kind="headerNoPadding"] svg {
    fill: #00D4B4 !important;
    width: 28px !important;
    height: 28px !important;
}

/* Ensure the header doesn't block clicks but is transparent */
header[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !important;
    color: #00D4B4 !important;
}

/* Sidebar Navigation Links (Page Names) */
[data-testid="stSidebarNav"] ul li a span {
    color: #E2E8F0 !important;
    font-weight: 500 !important;
    font-size: 1.05rem !important; /* Larger text for nav */
}

/* Inject icons for all sidebar navigation items */
[data-testid="stSidebarNav"] ul li:nth-child(1) a span::before { content: "🌡️ "; margin-right: 8px; }
[data-testid="stSidebarNav"] ul li:nth-child(2) a span::before { content: "🔍 "; margin-right: 8px; }
[data-testid="stSidebarNav"] ul li:nth-child(3) a span::before { content: "📈 "; margin-right: 8px; }
[data-testid="stSidebarNav"] ul li:nth-child(4) a span::before { content: "⚙️ "; margin-right: 8px; }
[data-testid="stSidebarNav"] ul li:nth-child(5) a span::before { content: "📋 "; margin-right: 8px; }

/* Active Page Link */
[data-testid="stSidebarNav"] ul li [aria-current="page"] {
    background-color: rgba(0, 212, 180, 0.12) !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
}
[data-testid="stSidebarNav"] ul li [aria-current="page"] span {
    color: #00D4B4 !important;
    font-weight: 700 !important;
}

/* Hover Effect for Links */
[data-testid="stSidebarNav"] ul li a:hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
}

/* Sidebar Labels and functional elements sizing */
[data-testid="stSidebar"] label { 
    color: #E2E8F0 !important; 
    font-size: 0.95rem !important; /* Larger labels */
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}

/* Make Inputs and Selectboxes larger */
[data-testid="stSidebar"] div[data-baseweb="input"] {
    height: 48px !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] div[data-baseweb="input"] input {
    font-size: 1rem !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    height: 48px !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    display: flex;
    align-items: center;
}

/* Sidebar Buttons */
[data-testid="stSidebar"] .stButton button {
    width: 100% !important;
    height: 52px !important;
    font-size: 1.1rem !important;
    border-radius: 10px !important;
    margin-top: 15px !important;
    letter-spacing: 0.5px !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #94A3B8 !important;
    font-size: 0.9rem !important;
}
/* Style for small metadata or help text in sidebar */
.sidebar-note {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 10px;
}

/* ── KPI Cards ───────────────────────────────────────────────────────────── */
.kpi-grid { display: flex; gap: 16px; margin-bottom: 24px; }
.kpi-card {
    flex: 1;
    background: linear-gradient(145deg, #0F1624 0%, #141E30 100%);
    border: 1px solid #1A2740;
    border-radius: 16px;
    padding: 22px 20px 18px;
    position: relative;
    overflow: hidden;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    cursor: default;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 40px rgba(0, 212, 180, 0.1);
}
.kpi-top-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-icon {
    position: absolute;
    top: 14px; right: 16px;
    font-size: 1.7rem;
    opacity: 0.12;
    user-select: none;
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #3D5070;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1.1;
    letter-spacing: -1.5px;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #3D5070;
    margin-top: 6px;
}

/* ── Page Header ─────────────────────────────────────────────────────────── */
.pg-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
    padding-bottom: 18px;
    border-bottom: 1px solid #1A2540;
}
.pg-icon {
    width: 46px; height: 46px;
    background: linear-gradient(135deg, #00D4B4 0%, #0080A0 100%);
    border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
}
.pg-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1.2;
}
.pg-sub {
    font-size: 0.8rem;
    color: #3D5070;
    margin-top: 3px;
}

/* ── Section Titles ──────────────────────────────────────────────────────── */
.sec-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #4A6080;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    border-left: 3px solid #00D4B4;
    padding-left: 10px;
    margin: 28px 0 14px;
}

/* ── Connection Banners ──────────────────────────────────────────────────── */
.conn-ok, .conn-err {
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.82rem;
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 22px;
}
.conn-ok  { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); color: #10B981; }
.conn-err { background: rgba(239,68,68,0.08);  border: 1px solid rgba(239,68,68,0.2);  color: #EF4444; }
.conn-url { color: #2A4060; margin-left: 4px; font-family: monospace; font-size: 0.78rem; }

/* ── Alert Badges ────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 100px;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    border: 1px solid;
}
.badge-CRITICAL { background: rgba(239,68,68,0.12);   color: #EF4444; border-color: rgba(239,68,68,0.25); }
.badge-WARNING  { background: rgba(245,158,11,0.12);  color: #F59E0B; border-color: rgba(245,158,11,0.25); }
.badge-INFO     { background: rgba(59,130,246,0.12);  color: #60A5FA; border-color: rgba(59,130,246,0.25); }
.badge-ok       { background: rgba(16,185,129,0.12);  color: #10B981; border-color: rgba(16,185,129,0.25); }
.badge-pending  { background: rgba(156,163,175,0.12); color: #9CA3AF; border-color: rgba(156,163,175,0.25); }
.badge-running  { background: rgba(59,130,246,0.12);  color: #60A5FA; border-color: rgba(59,130,246,0.25); }
.badge-done     { background: rgba(16,185,129,0.12);  color: #10B981; border-color: rgba(16,185,129,0.25); }
.badge-failed   { background: rgba(239,68,68,0.12);   color: #EF4444; border-color: rgba(239,68,68,0.25); }

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: #1A2540 !important; margin: 16px 0 !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00D4B4 0%, #009090 100%) !important;
    color: #080C17 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 22px !important;
    transition: box-shadow 0.2s, transform 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(0, 212, 180, 0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #141E30 !important;
    color: #64748B !important;
    border: 1px solid #1A2740 !important;
}

/* ── Hide branding ───────────────────────────────────────────────────────── */
#MainMenu, footer { visibility: hidden !important; }
.stAppDeployButton { display: none !important; }
[data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
div[data-testid="stToolbar"] { display: none !important; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str = "") -> None:
    sub_html = f'<div class="pg-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="pg-header">
            <div class="pg-icon">{icon}</div>
            <div>
                <div class="pg-title">{title}</div>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    sub: str = "",
    icon: str = "",
    accent: str = "#00D4B4",
) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-top-bar" style="background:{accent}"></div>'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"</div>"
    )


def connection_banner(healthy: bool, url: str) -> None:
    if healthy:
        st.markdown(
            f'<div class="conn-ok">● &nbsp;API Connected'
            f'<span class="conn-url">{url}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="conn-err">✕ &nbsp;API Unreachable'
            f'<span class="conn-url">{url}</span>'
            f" &nbsp;—&nbsp; run: "
            f"<code>uvicorn hvac_fdd.api:create_app --factory</code></div>",
            unsafe_allow_html=True,
        )


def sec_title(text: str) -> None:
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)


def badge_html(text: str, kind: str = "INFO") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'
