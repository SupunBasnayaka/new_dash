import os
import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from database import initialize_database
from model import load_or_train_model, irrigation_features, IRR_TYPES
from agent import get_paddy_sql_agent, ask_agent

# ─────────────────────────────────────────────
#  LOAD ENV
# ─────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lanka Paddy Intelligence Suite",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --green-primary: #1a7a3c;
    --green-light:   #2ecc71;
    --green-dark:    #145e2d;
    --gold:          #f4a926;
    --bg:            #0d1117;
    --surface:       #161b22;
    --surface2:      #21262d;
    --border:        rgba(255,255,255,0.08);
    --text:          #e6edf3;
    --muted:         #7d8590;
    --radius:        12px;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: radial-gradient(ellipse at top left, #0f1f12 0%, #0d1117 50%, #0f1520 100%);
    min-height: 100vh;
}

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
section[data-testid="stSidebarContent"] { display: none; }
.stDeployButton { display: none; }
button[kind="header"] { display: none; }

/* ── Remove ALL top padding so tabs sit at the very top ── */
.block-container {
    padding-top: 0.6rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1600px !important;
}
/* Streamlit injects extra top space via this element */
[data-testid="stAppViewContainer"] > section > div:first-child {
    padding-top: 0 !important;
}

/* ══════════════════════════════
   COMPACT PAGE HEADER
══════════════════════════════ */
.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0 8px 0;
    border-bottom: 1px solid rgba(46,204,113,0.12);
    margin-bottom: 10px;
}
.page-header-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.page-header-icon { font-size: 1.4rem; }
.page-header-title {
    font-size: 0.95rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.2px;
}
.page-header-sub {
    font-size: 0.65rem;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    font-weight: 600;
}
.page-header-right {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    color: #4ade80;
    font-weight: 600;
    background: rgba(46,204,113,0.1);
    border: 1px solid rgba(46,204,113,0.22);
    padding: 4px 10px;
    border-radius: 20px;
}
.nav-dot {
    width: 6px; height: 6px;
    background: #2ecc71;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
.nav-meta {
    font-size: 0.68rem;
    color: #3d4756;
    font-weight: 500;
}

/* ══════════════════════════════
   PLACEHOLDER (kept for reference but no longer used)
══════════════════════════════ */
.top-navbar {
    display: none; /* removed */
    background: rgba(13, 17, 23, 0.95);
    border-bottom: 1px solid rgba(46,204,113,0.15);
    backdrop-filter: blur(20px);
    position: sticky;
    top: 0;
    z-index: 999;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 64px;
    margin: 0 -2rem 2rem -2rem;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.nav-brand-icon { font-size: 1.6rem; }
.nav-brand-text {
    font-size: 1rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.3px;
}
.nav-brand-sub {
    font-size: 0.65rem;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-top: 1px;
}
.nav-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    color: #4ade80;
    font-weight: 600;
    background: rgba(46,204,113,0.1);
    border: 1px solid rgba(46,204,113,0.25);
    padding: 5px 12px;
    border-radius: 20px;
}
.nav-dot {
    width: 7px; height: 7px;
    background: #2ecc71;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ══════════════════════════════
   STREAMLIT TABS — CUSTOM STYLE
══════════════════════════════ */
.stTabs {
    margin-top: 0;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 5px;
    gap: 3px;
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: all 0.2s ease;
    white-space: nowrap;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #c8d0de !important;
    background: rgba(255,255,255,0.05) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a7a3c 0%, #145e2d 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(26,122,60,0.35) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ══════════════════════════════
   KPI CARDS
══════════════════════════════ */
.kpi-card {
    background: linear-gradient(135deg, #161f19 0%, #161b22 100%);
    border: 1px solid rgba(46,204,113,0.15);
    border-radius: var(--radius);
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.5);
    border-color: rgba(46,204,113,0.3);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1a7a3c, #2ecc71, #f4a926);
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: -30px; right: -30px;
    width: 90px; height: 90px;
    background: radial-gradient(circle, rgba(46,204,113,0.07), transparent 70%);
    border-radius: 50%;
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 10px; display: block; }
.kpi-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    letter-spacing: -0.5px;
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    color: var(--muted);
    margin-top: 5px;
}
.kpi-delta {
    font-size: 0.75rem;
    font-weight: 500;
    color: #4ade80;
    margin-top: 8px;
}

/* ══════════════════════════════
   SECTION HEADERS
══════════════════════════════ */
.section-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: #cdd9e5;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 2px;
}
.section-sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 14px;
}

/* ══════════════════════════════
   BRAND BAR (per-page)
══════════════════════════════ */
.brand-bar {
    background: rgba(26,122,60,0.07);
    border: 1px solid rgba(26,122,60,0.18);
    border-radius: var(--radius);
    padding: 14px 20px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.brand-bar-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #4ade80;
}
.brand-bar-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 2px;
}
.brand-badge {
    background: rgba(46,204,113,0.12);
    border: 1px solid rgba(46,204,113,0.25);
    color: #2ecc71;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}

/* ── Fancy divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(46,204,113,0.25), transparent);
    margin: 20px 0;
}

/* ══════════════════════════════
   PREDICTION RESULT
══════════════════════════════ */
.pred-result {
    background: linear-gradient(135deg, #0d2b1a 0%, #0a1f2e 100%);
    border: 1px solid rgba(46,204,113,0.3);
    border-radius: var(--radius);
    padding: 32px;
    text-align: center;
    margin-top: 20px;
}
.pred-value {
    font-size: 3rem;
    font-weight: 800;
    color: #2ecc71;
    letter-spacing: -1px;
}
.pred-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
}
.info-pill {
    display: inline-block;
    background: rgba(26,122,60,0.15);
    border: 1px solid rgba(26,122,60,0.3);
    color: #80e0a0;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 3px 2px;
}

/* ══════════════════════════════
   FORM / INPUTS
══════════════════════════════ */
.stForm {
    background: rgba(22, 27, 34, 0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
}
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #1a7a3c, #145e2d) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 13px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 8px 24px rgba(26,122,60,0.4) !important;
    transform: translateY(-1px) !important;
}
.stSelectbox label, .stNumberInput label,
.stSlider label, .stTextInput label { color: #8b92a5 !important; font-size: 0.8rem !important; }

/* ── Chat popup ── */
.stChatMessage {
    background: rgba(22,27,34,0.9) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}
div[data-testid="stPopoverBody"] {
    border-radius: 16px !important;
    border: 1px solid rgba(46,204,113,0.2) !important;
    background: #161b22 !important;
    box-shadow: 0 24px 60px rgba(0,0,0,0.6) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; }

/* ── Alert ── */
.stAlert { border-radius: 10px !important; }


/* ══════════════════════════════
   FLOATING CHAT BUTTON (round, headphone icon, bottom-right)
══════════════════════════════ */
div[data-testid="stPopover"] {
    position: fixed !important;
    bottom: 24px !important;
    right: 24px !important;
    z-index: 999999 !important;
    width: 60px !important;
    height: 60px !important;
    min-width: 0 !important;
}
div[data-testid="stPopover"] > button,
div[data-testid="stPopover"] button[data-testid="stPopoverButton"] {
    width: 60px !important;
    height: 60px !important;
    min-height: 0 !important;
    padding: 0 !important;
    border-radius: 50% !important;
    border: 2px solid rgba(74,222,128,0.55) !important;
    background: linear-gradient(135deg, #22a853, #0f4d24) !important;
    box-shadow: 0 8px 28px rgba(26,122,60,0.6) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: transform .2s ease, box-shadow .2s ease !important;
    /* hide the text label and dropdown arrow, show only the icon */
    font-size: 0 !important;
    color: transparent !important;
    overflow: hidden !important;
}
div[data-testid="stPopover"] > button > *,
div[data-testid="stPopover"] button[data-testid="stPopoverButton"] > * {
    display: none !important;
}
div[data-testid="stPopover"] > button::before,
div[data-testid="stPopover"] button[data-testid="stPopoverButton"]::before {
    content: "";
    display: block;
    width: 30px;
    height: 30px;
    background: no-repeat center / contain url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 14v-2a9 9 0 0 1 18 0v2'/><path d='M21 15a2 2 0 0 1-2 2h-1a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h3z'/><path d='M3 15a2 2 0 0 0 2 2h1a1 1 0 0 0 1-1v-3a1 1 0 0 0-1-1H3z'/><path d='M21 17v1a3 3 0 0 1-3 3h-4'/></svg>");
}
div[data-testid="stPopover"] > button:hover,
div[data-testid="stPopover"] button[data-testid="stPopoverButton"]:hover {
    transform: translateY(-3px) scale(1.07) !important;
    box-shadow: 0 14px 40px rgba(26,122,60,0.8) !important;
}
/* Chat window that opens above the button */
div[data-testid="stPopoverBody"] {
    width: min(440px, calc(100vw - 32px)) !important;
    max-height: min(640px, calc(100vh - 120px)) !important;
    overflow-y: auto !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INIT
# ─────────────────────────────────────────────
initialize_database()
bundle = load_or_train_model()
model = bundle["pipeline"]
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paddy_data.db")


@st.cache_data
def load_data(_db_version):
    """Returns the three views the dashboard needs.

    df_paddy : district rows, irrigation_type='Total'  (the only rows that carry production;
               excludes national / scheme rows so nothing is double counted)
    df_irr   : district rows split by Major / Minor / Rainfed (extent + yield only)
    df_all   : every district row, for the raw-data table
    """
    conn = sqlite3.connect(DB_FILE)
    allrows = pd.read_sql("SELECT * FROM paddy_stats", conn)
    conn.close()
    d = allrows[allrows["area_type"] == "district"]
    return (d[d["irrigation_type"] == "Total"].copy(),
            d[d["irrigation_type"] != "Total"].copy(),
            d.copy())


df_paddy_all, df_irr_all, df_all_all = load_data(os.path.getmtime(DB_FILE))

# ── Dashboard scope ──
# Data: ALL districts and ALL years are loaded. Only the *views* are limited:
#   * year charts open on 2015–2025 but hold every year (drag / pan left to see earlier years)
#   * charts with a district / province axis show only the top districts / provinces
#   * the scatter plot and the yield box plot keep their original scope (top-10 districts, 2015–2025)
Y_LO, Y_HI = 2015, 2025        # default viewing window
TOP_N = 10                     # districts shown in district-axis plots
TOP_PROV_N = 5                 # provinces shown in province-axis plots
ALL_DISTRICTS = sorted(df_paddy_all["district"].unique())
N_DISTRICTS = len(ALL_DISTRICTS)


def in_window(d):
    return d[d["year"].between(Y_LO, Y_HI)]


_win = in_window(df_paddy_all)
_district_prod = _win.groupby("district")["production_mt"].sum()
TOP_DISTRICTS = list(_district_prod.nlargest(TOP_N).index)          # ranked, biggest first
TOP_SHARE = _district_prod[TOP_DISTRICTS].sum() / _district_prod.sum() * 100

# Full data (all districts, all years)
df_paddy, df_irr, df_all = df_paddy_all, df_irr_all, df_all_all


def complete_year_list(d, n_dist):
    """Years where both seasons have production for (almost) every district, so a partly
    published year is never drawn as a fake collapse."""
    have = d[d["production_mt"].notna()].groupby(["year", "season"])["district"].nunique().unstack()
    need = n_dist - 2
    return sorted(have[(have.get("Maha", 0) >= need) & (have.get("Yala", 0) >= need)].index)


complete_years = complete_year_list(df_paddy, N_DISTRICTS)
df_c = df_paddy[df_paddy["year"].isin(complete_years) & df_paddy["production_mt"].notna()]   # all districts, all years
df_cw = in_window(df_c)                                                                        # same, 2015–2025 only (KPIs, pies)
YEAR_MIN, YEAR_MAX = Y_LO, Y_HI
PROD_MIN, PROD_MAX = int(min(df_cw["year"])), int(max(df_cw["year"]))
ALL_MIN, ALL_MAX = int(df_c["year"].min()), int(df_c["year"].max())

# Original top-10 / 2015–2025 subset, kept only for the scatter plot and the box plot
df_top = in_window(df_paddy)[in_window(df_paddy)["district"].isin(TOP_DISTRICTS)].copy()
_top_years = complete_year_list(df_top, TOP_N)
df_c_top = df_top[df_top["year"].isin(_top_years) & df_top["production_mt"].notna()]

# Top provinces (by production in the window, all districts) for province-axis plots
_prov_prod = df_cw.groupby("province")["production_mt"].sum()
TOP_PROVINCES = list(_prov_prod.nlargest(TOP_PROV_N).index)


def year_axes(fig, d_years, ymax=None, y_pad=1.12):
    """Open on Y_LO–Y_HI but keep every year loaded: drag (pan) sideways to see earlier years.
    The y-axis is fixed to the full-data maximum so older years are never clipped while panning."""
    lo, hi = int(min(d_years)), int(max(d_years))
    fig.update_xaxes(range=[Y_LO - 0.5, Y_HI + 0.5], dtick=1,
                     minallowed=lo - 0.5, maxallowed=max(hi, Y_HI) + 0.5)
    if ymax is not None:
        fig.update_yaxes(range=[0, ymax * y_pad], fixedrange=True)
    fig.update_layout(dragmode="pan")
    return fig


PAN_CFG = {"scrollZoom": False, "displaylogo": False,
           "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


def wmean(g, v="average_yield_kg_ha", w="harvested_extent_ha"):
    """Harvest-weighted mean that ignores missing values."""
    m = g[v].notna() & g[w].notna() & (g[w] > 0)
    return float(np.average(g.loc[m, v], weights=g.loc[m, w])) if m.any() else np.nan


# ─────────────────────────────────────────────
#  PLOT THEME HELPERS
# ─────────────────────────────────────────────
GREEN   = "#2ecc71"
GOLD    = "#f4a926"
PALETTE = ["#2ecc71", "#f4a926", "#3498db", "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6", "#fd79a8"]

def apply_theme(fig, legend=True):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c8d0de", size=12),
        margin=dict(l=0, r=0, t=28, b=0),
        legend=dict(
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(size=11),
        ) if legend else dict(visible=False),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.05)", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.05)", tickfont=dict(size=11)),
    )
    return fig

def kpi_card(icon, value, label, delta=""):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""

def brand_bar(title, subtitle, badge):
    st.markdown(f"""
    <div class="brand-bar">
        <div>
            <div class="brand-bar-title">{title}</div>
            <div class="brand-bar-sub">{subtitle}</div>
        </div>
        <div class="brand-badge">{badge}</div>
    </div>""", unsafe_allow_html=True)

def note(text):
    st.caption(text)

# ─────────────────────────────────────────────
#  COMPACT PAGE HEADER (no tall navbar)
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div class="page-header-brand">
        <span class="page-header-icon">🌾</span>
        <div>
            <div class="page-header-title">Paddy Intelligence Suite</div>
            <div class="page-header-sub">Sri Lanka · All {N_DISTRICTS} districts · {ALL_MIN}–{ALL_MAX} (view {YEAR_MIN}–{YEAR_MAX})</div>
        </div>
    </div>
    <div class="page-header-right">
        <span class="nav-meta">Dept. of Census &amp; Statistics data · University of Colombo</span>
        <div class="nav-status"><div class="nav-dot"></div> Live Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN TABS  — right at the top
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Executive Overview",
    "🗺️  District Deep-Dive",
    "💧  Irrigation Analysis",
    "🤖  ML Predictor",
])


# ═══════════════════════════════════════════════════════
#  TAB 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════
with tab1:
    brand_bar(
        "🌾 Sri Lanka Paddy Production &amp; Food Security",
        f"Interactive EDA suite · All {N_DISTRICTS} districts · opens on {YEAR_MIN}–{YEAR_MAX}",
        "LIVE ANALYTICS"
    )

    # KPIs  (all districts, 2015–2025, complete years only for production)
    total_prod   = df_cw["production_mt"].sum() / 1e6
    avg_yield    = wmean(in_window(df_paddy))
    total_sown   = in_window(df_paddy)["sown_extent_ha"].sum() / 1e6
    by_district  = df_cw.groupby("district")["production_mt"].sum()
    top_district = by_district.idxmax()
    max_prod_k   = by_district.max() / 1e3
    top_prov     = df_paddy.loc[df_paddy["district"] == top_district, "province"].iloc[0]
    yearly_prod  = df_cw.groupby("year")["production_mt"].sum()
    latest_prod  = yearly_prod.iloc[-1]
    yoy_pct      = (yearly_prod.iloc[-1] - yearly_prod.iloc[-2]) / yearly_prod.iloc[-2] * 100
    sign         = "▲" if yoy_pct >= 0 else "▼"

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(kpi_card("📦", f"{total_prod:.1f}M MT", f"Total Production · All {N_DISTRICTS} Districts", f"{PROD_MIN}–{PROD_MAX} · both seasons"), unsafe_allow_html=True)
    with k2: st.markdown(kpi_card("🌾", f"{avg_yield:,.0f} kg/ha", "Mean Yield · All Districts", "Harvest-weighted"), unsafe_allow_html=True)
    with k3: st.markdown(kpi_card("🌍", f"{total_sown:.1f}M Ha", "Total Sown Extent · All Districts", f"{Y_LO}–{Y_HI}, both seasons"), unsafe_allow_html=True)
    with k4: st.markdown(kpi_card("🏆", top_district, "Top District", f"{max_prod_k:,.0f}k MT · {top_prov} Province"), unsafe_allow_html=True)
    with k5: st.markdown(kpi_card("📈", f"{latest_prod/1e6:.2f}M MT", f"{PROD_MAX} Production", f"{sign} {abs(yoy_pct):.1f}% vs {PROD_MAX - 1}"), unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Row 1 — Trend + Pie
    r1a, r1b = st.columns([3, 2], gap="large")
    with r1a:
        st.markdown('<div class="section-header">📈 Annual Production Trend by Season</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Maha vs. Yala contribution, all {N_DISTRICTS} districts · opens on {Y_LO}–{Y_HI} · drag left for earlier years ({ALL_MIN}+)</div>', unsafe_allow_html=True)
        trend_df = df_c.groupby(["year", "season"])["production_mt"].sum().reset_index()
        fig_trend = px.area(trend_df, x="year", y="production_mt", color="season",
                            line_shape="spline", color_discrete_map={"Maha": GREEN, "Yala": GOLD},
                            labels={"production_mt": "Production (MT)", "year": "Year"})
        fig_trend.update_traces(opacity=0.75, line_width=2.5)
        apply_theme(fig_trend)
        # stacked area: y max is the largest yearly total
        year_axes(fig_trend, trend_df["year"], ymax=df_c.groupby("year")["production_mt"].sum().max())
        st.plotly_chart(fig_trend, width="stretch", config=PAN_CFG)

    with r1b:
        st.markdown('<div class="section-header">⚖️ Season Contribution Share</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">All districts · production split by Maha / Yala, {PROD_MIN}–{PROD_MAX}</div>', unsafe_allow_html=True)
        season_df = df_cw.groupby("season")["production_mt"].sum().reset_index()
        fig_pie = px.pie(season_df, names="season", values="production_mt", hole=0.55,
                         color="season", color_discrete_map={"Maha": GREEN, "Yala": GOLD})
        fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                              pull=[0.04] * len(season_df), textfont_size=12)
        apply_theme(fig_pie, legend=False)
        fig_pie.update_layout(annotations=[dict(text="🌾", x=0.5, y=0.5, font_size=28, showarrow=False)])
        st.plotly_chart(fig_pie, width="stretch")

    # Row 2 — Province bar + Scatter
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    r2a, r2b = st.columns([2, 3], gap="large")
    with r2a:
        st.markdown('<div class="section-header">🏛️ Province-Level Production</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Top {TOP_PROV_N} provinces (all districts counted), cumulative {PROD_MIN}–{PROD_MAX}</div>', unsafe_allow_html=True)
        prov_df = (df_cw.groupby("province")["production_mt"].sum().reset_index()
                   .nlargest(TOP_PROV_N, "production_mt").sort_values("production_mt", ascending=True))
        fig_prov = px.bar(prov_df, x="production_mt", y="province", orientation="h",
                          color="production_mt", color_continuous_scale=["#145e2d", "#2ecc71"],
                          labels={"production_mt": "Total Production (MT)", "province": ""}, text_auto=".2s")
        fig_prov.update_traces(textfont_size=10, marker_line_width=0)
        fig_prov.update_coloraxes(showscale=False)
        apply_theme(fig_prov, legend=False)
        st.plotly_chart(fig_prov, width="stretch")

    with r2b:
        st.markdown('<div class="section-header">🔍 Sown Extent vs. Production Correlation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Top {TOP_N} districts, {Y_LO}–{Y_HI} · bubble size = average yield · colour = province · trendline = OLS</div>', unsafe_allow_html=True)
        sc_df = df_c_top.dropna(subset=["sown_extent_ha", "production_mt", "average_yield_kg_ha"])
        fig_scatter = px.scatter(sc_df, x="sown_extent_ha", y="production_mt", color="province",
                                 size="average_yield_kg_ha", hover_data=["district", "year", "season"],
                                 trendline="ols", color_discrete_sequence=PALETTE, opacity=0.75,
                                 labels={"sown_extent_ha": "Sown Extent (Ha)", "production_mt": "Production (MT)"})
        apply_theme(fig_scatter)
        st.plotly_chart(fig_scatter, width="stretch")

    # Row 3 — YoY bar
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Year-on-Year National Production</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Total MT of all {N_DISTRICTS} districts, both seasons · green = growth · red = decline · drag left for earlier years</div>', unsafe_allow_html=True)
    yearly = df_c.groupby("year")["production_mt"].sum().reset_index()
    yearly["color"] = [GREEN] + [GREEN if x >= 0 else "#e74c3c" for x in yearly["production_mt"].diff().iloc[1:]]
    fig_yoy = go.Figure(go.Bar(x=yearly["year"], y=yearly["production_mt"],
                               marker_color=yearly["color"],
                               text=[f"{v/1e3:.0f}k" for v in yearly["production_mt"]],
                               textposition="outside", textfont=dict(size=9)))
    apply_theme(fig_yoy, legend=False)
    fig_yoy.update_layout(height=300)
    year_axes(fig_yoy, yearly["year"], ymax=yearly["production_mt"].max(), y_pad=1.2)
    st.plotly_chart(fig_yoy, width="stretch", config=PAN_CFG)
    note(f"KPIs and the season pie cover all {N_DISTRICTS} districts for {PROD_MIN}–{PROD_MAX}. Year charts open on {Y_LO}–{Y_HI} and can be dragged back to {ALL_MIN}. "
         f"The scatter plot uses the top {TOP_N} districts ({', '.join(TOP_DISTRICTS)}). "
         "Years where a season is only partly published are left out so they don't look like a collapse.")


# ═══════════════════════════════════════════════════════
#  TAB 2 — DISTRICT DEEP-DIVE
# ═══════════════════════════════════════════════════════
with tab2:
    brand_bar("🗺️ Regional &amp; District Performance Analysis",
              "Granular district metrics · Yield distributions · Inter-provincial disparities",
              "DISTRICT VIEW")

    # all districts are selectable, ranked by production (top ones first)
    _rest = [d for d in ALL_DISTRICTS if d not in TOP_DISTRICTS]
    district_list = TOP_DISTRICTS + sorted(_rest)
    sel_district = st.selectbox(f"🔎 Select District (all {N_DISTRICTS}; top producers listed first)", district_list, index=0, key="dist_sel")
    dist_df = df_paddy[df_paddy["district"] == sel_district]
    dist_c = dist_df[dist_df["production_mt"].notna()]
    dist_yield = wmean(dist_df)

    dk1, dk2, dk3, dk4 = st.columns(4)
    with dk1: st.markdown(kpi_card("📦", f"{dist_c['production_mt'].sum()/1e3:,.0f}k MT", "Total Production", f"{sel_district}, {int(dist_c['year'].min())}–{int(dist_c['year'].max())}"), unsafe_allow_html=True)
    with dk2: st.markdown(kpi_card("🌾", "n/a" if pd.isna(dist_yield) else f"{dist_yield:,.0f} kg/ha", "Average Yield", "Harvest-weighted"), unsafe_allow_html=True)
    with dk3: st.markdown(kpi_card("🌍", f"{dist_df['sown_extent_ha'].sum()/1e3:,.0f}k Ha", "Total Sown Area", f"{int(dist_df['year'].min())}–{int(dist_df['year'].max())}, both seasons"), unsafe_allow_html=True)
    with dk4: st.markdown(kpi_card("🏛️", dist_df["province"].iloc[0], "Province", "Administrative Region"), unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2, gap="large")
    with d1:
        st.markdown(f'<div class="section-header">📅 Historical Production – {sel_district}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Maha vs. Yala per year · opens on 2015–2025, drag left for earlier years (gaps = no data published)</div>', unsafe_allow_html=True)
        fig_dist = px.bar(dist_c, x="year", y="production_mt", color="season", barmode="group",
                          color_discrete_map={"Maha": GREEN, "Yala": GOLD},
                          labels={"production_mt": "Production (MT)", "year": "Year"})
        fig_dist.update_traces(marker_line_width=0)
        apply_theme(fig_dist)
        year_axes(fig_dist, dist_c["year"], ymax=dist_c["production_mt"].max())
        st.plotly_chart(fig_dist, width="stretch", config=PAN_CFG)

    with d2:
        st.markdown('<div class="section-header">📊 Yield Distribution – Top 10 Districts</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Top 10 districts, 2015–2025 · ordered by median yield · colour = province</div>', unsafe_allow_html=True)
        fig_box = px.box(df_top.dropna(subset=["average_yield_kg_ha"]), x="district", y="average_yield_kg_ha", color="province",
                         color_discrete_sequence=PALETTE,
                         labels={"average_yield_kg_ha": "Avg Yield (kg/ha)", "district": ""})
        fig_box.update_layout(xaxis={"categoryorder": "median descending"})
        apply_theme(fig_box)
        st.plotly_chart(fig_box, width="stretch")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">📈 Yield Trend Over Time – {sel_district}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Average yield kg/ha by year and season · opens on 2015–2025, drag left for earlier years</div>', unsafe_allow_html=True)
    _yt = dist_df.dropna(subset=["average_yield_kg_ha"])
    fig_yt = px.line(_yt, x="year", y="average_yield_kg_ha", color="season",
                     markers=True, line_shape="spline",
                     color_discrete_map={"Maha": GREEN, "Yala": GOLD},
                     labels={"average_yield_kg_ha": "Avg Yield (kg/ha)", "year": "Year"})
    fig_yt.update_traces(line_width=2.5, marker_size=7)
    apply_theme(fig_yt)
    year_axes(fig_yt, _yt["year"], ymax=_yt["average_yield_kg_ha"].max())
    st.plotly_chart(fig_yt, width="stretch", config=PAN_CFG)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">🗂️ Raw Data – {sel_district}</div>', unsafe_allow_html=True)
    show_cols = ["year", "season", "province", "irrigation_type", "sown_extent_ha", "harvested_extent_ha",
                 "average_yield_kg_ha", "production_mt"]
    raw = df_all[df_all["district"] == sel_district][show_cols].sort_values(["year", "season", "irrigation_type"])
    st.dataframe(raw.reset_index(drop=True), width="stretch", height=260)
    note("Empty cells are values the source did not publish. Production is only published for the 'Total' rows.")


# ═══════════════════════════════════════════════════════
#  TAB 3 — IRRIGATION ANALYSIS
# ═══════════════════════════════════════════════════════
with tab3:
    brand_bar("💧 Irrigation &amp; Water Management Analysis",
              "Major · Minor · Rainfed — extent and yield by water source",
              "AGRI VIEW")
    note("Production is only published as district totals, so these views use sown extent and yield. "
         f"All {N_DISTRICTS} districts are included. The slider drives the pie, heatmap and season bars; "
         f"the over-time chart always holds every year and opens on {Y_LO}–{Y_HI}.")

    irr_years = sorted(df_irr.dropna(subset=["sown_extent_ha"])["year"].unique())
    _lo_def = max(Y_LO, int(min(irr_years)))
    _hi_def = min(Y_HI, int(max(irr_years)))
    yr_lo, yr_hi = st.slider("📅 Year range", min_value=int(min(irr_years)), max_value=int(max(irr_years)),
                             value=(_lo_def, _hi_def), key="irr_years")
    irr_sel = df_irr[(df_irr["year"] >= yr_lo) & (df_irr["year"] <= yr_hi)]
    IRR_COLORS = {"Major": "#2ecc71", "Minor": "#3498db", "Rainfed": "#f4a926"}

    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown('<div class="section-header">💧 Sown Extent by Irrigation Type</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Share of sown hectares in the selected years</div>', unsafe_allow_html=True)
        irr_df = irr_sel.groupby("irrigation_type")["sown_extent_ha"].sum().reset_index()
        fig_irr = px.pie(irr_df, names="irrigation_type", values="sown_extent_ha", hole=0.52,
                         color="irrigation_type", color_discrete_map=IRR_COLORS)
        fig_irr.update_traces(textinfo="percent+label", pull=[0.03] * len(irr_df), textfont_size=12)
        apply_theme(fig_irr, legend=False)
        st.plotly_chart(fig_irr, width="stretch")

    with a2:
        st.markdown('<div class="section-header">📈 Sown Extent Over Time</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Hectares per year by irrigation type (both seasons) · opens on 2015–2025, drag left for earlier years</div>', unsafe_allow_html=True)
        irr_trend = df_irr.groupby(["year", "irrigation_type"])["sown_extent_ha"].sum().reset_index()
        fig_it = px.area(irr_trend, x="year", y="sown_extent_ha", color="irrigation_type",
                         color_discrete_map=IRR_COLORS,
                         labels={"sown_extent_ha": "Sown Extent (Ha)", "year": "Year", "irrigation_type": ""})
        apply_theme(fig_it)
        year_axes(fig_it, irr_trend["year"], ymax=irr_trend.groupby("year")["sown_extent_ha"].sum().max())
        st.plotly_chart(fig_it, width="stretch", config=PAN_CFG)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🌡️ Yield Heatmap – Province × Irrigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Harvest-weighted average yield (kg/ha), top 5 provinces by production, all districts, selected years</div>', unsafe_allow_html=True)
    heat = (irr_sel.groupby(["province", "irrigation_type"])
            .apply(wmean, include_groups=False).reset_index(name="yield"))
    heatmap_data = (heat.pivot(index="province", columns="irrigation_type", values="yield")
                    .reindex([p for p in TOP_PROVINCES if p in set(heat["province"])]))
    fig_heat = px.imshow(heatmap_data, text_auto=".0f", aspect="auto",
                         color_continuous_scale=["#0d2b1a", "#1a7a3c", "#2ecc71"],
                         labels=dict(color="Yield (kg/ha)", x="", y=""))
    fig_heat.update_traces(textfont_size=13)
    apply_theme(fig_heat, legend=False)
    fig_heat.update_layout(height=340)
    st.plotly_chart(fig_heat, width="stretch")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Irrigation Performance by Season</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Harvest-weighted yield per irrigation type, split by Maha/Yala</div>', unsafe_allow_html=True)
    irr_season = (irr_sel.groupby(["irrigation_type", "season"])
                  .apply(wmean, include_groups=False).reset_index(name="yield"))
    fig_irr_s = px.bar(irr_season, x="irrigation_type", y="yield",
                       color="season", barmode="group",
                       color_discrete_map={"Maha": GREEN, "Yala": GOLD},
                       labels={"yield": "Avg Yield (kg/ha)", "irrigation_type": ""},
                       text_auto=".0f")
    fig_irr_s.update_traces(marker_line_width=0, textfont_size=10)
    apply_theme(fig_irr_s)
    st.plotly_chart(fig_irr_s, width="stretch")


# ═══════════════════════════════════════════════════════
#  TAB 4 — ML PREDICTOR
# ═══════════════════════════════════════════════════════
with tab4:
    m = bundle["metrics"]
    brand_bar("🤖 ML Production Predictor",
              f"Random Forest Regressor · Trained on {bundle['n_rows']:,} district-season records ({bundle['min_year']}–{bundle['max_year']}) · district, province, season, sown extent &amp; irrigation mix",
              "AI POWERED")

    ml_left, ml_right = st.columns([3, 2], gap="large")
    all_districts = sorted(df_paddy_all["district"].unique())
    prov_of = df_paddy_all.drop_duplicates("district").set_index("district")["province"].to_dict()

    with ml_left:
        st.markdown('<div class="section-header">🔮 Configure Prediction Parameters</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Choose the district and season, enter the sown extent under each irrigation type, then run the model (year is not an input)</div>', unsafe_allow_html=True)

        # District / season sit outside the form so the irrigation defaults can follow them.
        sc1, sc2 = st.columns(2)
        with sc1:
            input_district = st.selectbox("📍 District", all_districts,
                                          index=all_districts.index(TOP_DISTRICTS[0]), key="ml_district")
        with sc2:
            input_season = st.selectbox("🌤️ Season", ["Maha", "Yala"], key="ml_season")
        st.caption(f"🏛️ {prov_of[input_district]} Province")

        # Defaults = this district-season's sown extent per irrigation type in the latest year.
        _last = df_irr_all[(df_irr_all["district"] == input_district) & (df_irr_all["season"] == input_season)
                           & (df_irr_all["year"] == bundle["max_year"])]
        default_ha = _last.groupby("irrigation_type")["sown_extent_ha"].sum().to_dict()

        with st.form("prediction_form"):
            st.markdown("**💧 Sown extent by irrigation type (Ha)**")
            ic = st.columns(3)
            ha_in = {}
            for col, typ in zip(ic, IRR_TYPES):
                with col:
                    ha_in[typ] = st.number_input(f"{typ}", min_value=0.0, max_value=200000.0,
                                                 value=float(round(default_ha.get(typ, 0.0))), step=100.0,
                                                 key=f"ha_{typ}_{input_district}_{input_season}")
            st.caption(f"Pre-filled with {input_district} {input_season} {bundle['max_year']} figures. "
                       "Total sown extent = Major + Minor + Rainfed.")
            submitted = st.form_submit_button("🔮  Run Prediction", width="stretch")

        if submitted:
            if sum(ha_in.values()) <= 0:
                st.error("Enter a sown extent greater than zero for at least one irrigation type.")
            else:
                feats = irrigation_features(ha_in["Major"], ha_in["Minor"], ha_in["Rainfed"])
                input_data = pd.DataFrame([{
                    "district": input_district, "province": prov_of[input_district],
                    "season": input_season, **feats,
                }])[bundle["features"]]
                prediction = float(model.predict(input_data)[0])
                hist = _win[(_win["district"] == input_district) & (_win["season"] == input_season)
                            & _win["production_mt"].notna()]["production_mt"]
                hist_txt = (f"Historical median for {input_district} {input_season} ({Y_LO}–{Y_HI}): {hist.median():,.0f} MT"
                            if len(hist) else "")
                mix = " · ".join(f"{t} {feats[t.lower() + '_share']:.0%}" for t in IRR_TYPES)
                st.markdown(f"""
                <div class="pred-result">
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:1.5px; color:#4ade80; font-weight:700; margin-bottom:14px;">
                        ✅ Prediction Complete
                    </div>
                    <div class="pred-value">{prediction:,.0f} MT</div>
                    <div class="pred-label">Estimated Paddy Production</div>
                    <div style="margin-top:18px; display:flex; justify-content:center; gap:6px; flex-wrap:wrap;">
                        <span class="info-pill">📍 {input_district}</span>
                        <span class="info-pill">🏛️ {prov_of[input_district]}</span>
                        <span class="info-pill">🌤️ {input_season}</span>
                        <span class="info-pill">🌾 {feats['sown_extent_ha']:,.0f} Ha</span>
                        <span class="info-pill">💧 Mostly {feats['irrigation_type']}</span>
                    </div>
                    <div style="margin-top:10px; font-size:0.75rem; color:#80e0a0;">Irrigation mix: {mix}</div>
                    <div style="margin-top:10px; font-size:0.75rem; color:#7d8590;">{hist_txt}</div>
                </div>""", unsafe_allow_html=True)
        note(f"Accuracy on years the model never saw ({m['test_years']}): R² = {m['r2_test']:.2f}, "
             f"average error ≈ {m['mae_test_mt']:,.0f} MT per district-season. "
             "Year is not a model input: the prediction depends on district, season, sown extent and irrigation mix only.")

    with ml_right:
        st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Share of the model\'s decisions driven by each input</div>', unsafe_allow_html=True)
        label_map = {"sown_extent_ha": "Sown Extent", "district": "District", "province": "Province",
                     "season": "Season", "irrigation": "Irrigation Type / Mix"}
        fi_df = pd.DataFrame({
            "Feature": [label_map[k] for k in bundle["importance"]],
            "Importance": list(bundle["importance"].values()),
        }).sort_values("Importance")
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=["#145e2d", "#2ecc71"],
                        text=[f"{v:.0%}" for v in fi_df["Importance"]],
                        labels={"Importance": "Relative Importance", "Feature": ""})
        fig_fi.update_traces(textfont_size=10, marker_line_width=0)
        fig_fi.update_coloraxes(showscale=False)
        apply_theme(fig_fi, legend=False)
        fig_fi.update_layout(height=290)
        st.plotly_chart(fig_fi, width="stretch")

        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px;">
            <div class="kpi-card" style="padding:14px 16px;">
                <div class="kpi-icon" style="font-size:1.2rem; margin-bottom:6px;">🌲</div>
                <div class="kpi-value" style="font-size:1.2rem;">{bundle['n_estimators']}</div>
                <div class="kpi-label">Estimators</div>
            </div>
            <div class="kpi-card" style="padding:14px 16px;">
                <div class="kpi-icon" style="font-size:1.2rem; margin-bottom:6px;">📈</div>
                <div class="kpi-value" style="font-size:1.2rem;">R² {m['r2_test']:.2f}</div>
                <div class="kpi-label">Held-out Score ({m['test_years']})</div>
            </div>
            <div class="kpi-card" style="padding:14px 16px;">
                <div class="kpi-icon" style="font-size:1.2rem; margin-bottom:6px;">🗂️</div>
                <div class="kpi-value" style="font-size:1.2rem;">{bundle['n_rows']:,}</div>
                <div class="kpi-label">Training Rows</div>
            </div>
            <div class="kpi-card" style="padding:14px 16px;">
                <div class="kpi-icon" style="font-size:1.2rem; margin-bottom:6px;">🔢</div>
                <div class="kpi-value" style="font-size:1.2rem;">{len(bundle['features'])}</div>
                <div class="kpi-label">Input Features</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  FLOATING AI CHAT — fixed bottom-right via CSS
#  (placed at end of script; CSS positions it visually)
# ═══════════════════════════════════════════════════════
@st.cache_resource
def cached_agent(api_key):
    return get_paddy_sql_agent(api_key)


with st.popover("Agri-AI chat", help="Chat with the Agri-AI assistant"):
    st.markdown("### 🌾 Agri-AI Assistant")
    st.caption("Ask anything about paddy production, yields, or districts in natural language.")
    st.markdown(
        '<div style="height:3px;background:linear-gradient(90deg,#1a7a3c,#2ecc71);'
        'border-radius:2px;margin-bottom:14px;"></div>',
        unsafe_allow_html=True,
    )
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        st.warning("⚠️ Add your OpenRouter API key to the `.env` file to activate the assistant.")
        st.code("OPENROUTER_API_KEY=sk-or-xxxxxxxx", language="bash")
    else:
        agent = cached_agent(OPENROUTER_API_KEY)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("Ask about paddy data…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        resp = ask_agent(agent, prompt)
                        st.markdown(resp)
                        st.session_state.messages.append({"role": "assistant", "content": resp})
                    except Exception as e:
                        st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════
#  FLOATING MAP BUTTON + PAD POPUP — fixed bottom-right, just above the Agri-AI button
#  Opens a centred pop-up with a Sri Lanka district map of total (Maha + Yala) paddy
#  production and a year selector above the map (default 2025).
# ═══════════════════════════════════════════════════════
import json

GEOJSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lk-district-boundaries.geojson")
# The database spells it "Mullativu"; the boundary file spells it "Mullaitivu"
_DB_TO_GEO = {"Mullativu": "Mullaitivu"}

# Light-to-rich green -> gold gradient. Stops are packed toward the low end because a few
# dry-zone districts dwarf the rest; the colour bar itself stays linear.
MAP_COLORSCALE = [
    [0.00, "#16402b"], [0.06, "#1a7a3c"], [0.18, "#2ecc71"],
    [0.40, "#b6e04a"], [0.70, "#f4d03f"], [1.00, "#f4a926"],
]

st.markdown("""
<style>
/* Map button: same round style as the Agri-AI button, stacked 84px above it */
.st-key-map_fab {
    position: fixed !important;
    bottom: 96px !important;
    right: 24px !important;
    z-index: 999999 !important;
    width: 60px !important;
    height: 60px !important;
    min-width: 0 !important;
}
.st-key-map_fab button {
    width: 60px !important;
    height: 60px !important;
    min-height: 0 !important;
    padding: 0 !important;
    border-radius: 50% !important;
    border: 2px solid rgba(244,169,38,0.6) !important;
    background: linear-gradient(135deg, #22a853, #0f4d24) !important;
    box-shadow: 0 8px 28px rgba(26,122,60,0.6) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: transform .2s ease, box-shadow .2s ease !important;
    font-size: 0 !important;
    color: transparent !important;
    overflow: hidden !important;
}
.st-key-map_fab button > * { display: none !important; }
.st-key-map_fab button::before {
    content: "";
    display: block;
    width: 30px;
    height: 30px;
    background: no-repeat center / contain url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polygon points='1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6'/><line x1='8' y1='2' x2='8' y2='18'/><line x1='16' y1='6' x2='16' y2='22'/></svg>");
}
.st-key-map_fab button:hover {
    transform: translateY(-3px) scale(1.07) !important;
    box-shadow: 0 14px 40px rgba(26,122,60,0.8) !important;
}
/* Pop-up window */
div[data-testid="stDialog"] div[role="dialog"] {
    background: #161b22 !important;
    border: 1px solid rgba(46,204,113,0.2) !important;
    border-radius: 16px !important;
}
</style>
""", unsafe_allow_html=True)


def _ring_area(r):
    return sum(r[i][0] * r[i + 1][1] - r[i + 1][0] * r[i][1] for i in range(len(r) - 1)) / 2


def _rdp(pts, eps):
    """Douglas-Peucker line simplification (iterative). eps in degrees."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (x1, y1), (x2, y2) = pts[a], pts[b]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5
        dmax, idx = 0.0, -1
        for i in range(a + 1, b):
            x0, y0 = pts[i]
            d = (abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / norm) if norm else ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
            if d > dmax:
                dmax, idx = d, i
        if dmax > eps and idx != -1:
            keep[idx] = True
            stack += [(a, idx), (idx, b)]
    return [p for p, k in zip(pts, keep) if k]


def _fix_ring(ring, clockwise):
    """Round to ~100 m, drop repeated points, simplify (invisible at map size, much smaller
    payload = faster redraw), and force the winding direction."""
    pts = []
    for x, y in ring:
        p = [round(x, 3), round(y, 3)]
        if not pts or p != pts[-1]:
            pts.append(p)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    simp = _rdp(pts, 0.003)
    if len(simp) >= 4:
        pts = simp
    if (_ring_area(pts) < 0) != clockwise:
        pts.reverse()
    return pts


@st.cache_data
def load_geojson(path):
    """District boundaries, re-wound for Plotly (outer rings clockwise, holes counter-clockwise).
    Without this the districts are filled 'inside out' and cover the whole map."""
    with open(path, encoding="utf-8") as fh:
        geo = json.load(fh)
    for f in geo["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        fixed = [[_fix_ring(r, clockwise=(i == 0)) for i, r in enumerate(p)] for p in polys]
        # drop specks that collapsed to nothing after rounding (and holes that did)
        fixed = [[r for i, r in enumerate(p) if len(r) >= 4 and abs(_ring_area(r)) > 1e-8] for p in fixed]
        fixed = [p for p in fixed if p]
        g["coordinates"] = fixed[0] if g["type"] == "Polygon" else fixed
    return geo


@st.cache_data(show_spinner=False)
def build_paddy_map(year):
    """Choropleth of total (Maha + Yala) production per district for one year, in thousand Mt."""
    geo = load_geojson(GEOJSON_PATH)
    d = df_paddy[(df_paddy["year"] == year) & df_paddy["production_mt"].notna()]
    tot = d.groupby("district")["production_mt"].sum().rename(index=_DB_TO_GEO) / 1000   # thousand Mt

    names = [f["properties"]["shapeName"] for f in geo["features"]]
    prod = pd.Series({n: tot.get(n, np.nan) for n in names})          # thousand Mt

    fig = go.Figure()
    # grey base layer, only needed when some district has no published data that year
    if prod.isna().any():
        fig.add_trace(go.Choropleth(
            geojson=geo, locations=names, z=[0] * len(names), featureidkey="properties.shapeName",
            colorscale=[[0, "#2a313c"], [1, "#2a313c"]], showscale=False,
            marker_line_color="#ffffff", marker_line_width=0.8,
            hovertemplate="<b>%{location}</b><br>No data published<extra></extra>",
        ))
    have = prod.dropna()
    fig.add_trace(go.Choropleth(
        geojson=geo, locations=list(have.index), z=have.round(1).tolist(),
        featureidkey="properties.shapeName",
        colorscale=MAP_COLORSCALE, zmin=0, zmax=float(have.max()) if len(have) else 1,
        marker_line_color="#ffffff", marker_line_width=0.8,
        hovertemplate="<b>%{location}</b><br>%{z:,.1f} thousand metric tonnes<extra></extra>",
        colorbar=dict(title=dict(text="Thousand<br>metric tonnes", font=dict(size=10, color="#c8d0de")),
                      thickness=10, len=0.6, x=1.0, xanchor="right", xpad=0, tickfont=dict(size=10, color="#c8d0de"),
                      outlinewidth=0, bgcolor="rgba(0,0,0,0)"),
    ))
    # Fixed frame around Sri Lanka so the island fills the whole map area (no pan / zoom)
    fig.update_geos(visible=False, bgcolor="rgba(0,0,0,0)", projection_type="equirectangular",
                    lonaxis_range=[79.45, 82.0], lataxis_range=[5.8, 9.95],
                    domain=dict(x=[0, 1], y=[0, 1]))
    apply_theme(fig, legend=False)
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0), dragmode=False,
                      geo=dict(bgcolor="rgba(0,0,0,0)"))
    return fig, prod


@st.dialog("Sri Lanka Paddy Production Map", width="small")
def paddy_map_dialog():
    years = sorted((int(y) for y in complete_years), reverse=True)
    default_idx = years.index(2025) if 2025 in years else 0
    # key changes on every open -> dropdown and map always start together on the default year
    year = st.selectbox("Year", years, index=default_idx,
                        key=f"paddy_map_year_{st.session_state.get('paddy_map_open_n', 0)}")
    st.caption(f"Total paddy production (Yala + Maha), {year} · thousand metric tonnes")
    fig, prod = build_paddy_map(year)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "displayModeBar": False,
                                                        "scrollZoom": False, "doubleClick": False})
    if prod.notna().any():
        st.caption(f"National total {prod.sum():,.1f} thousand metric tonnes · highest: "
                   f"{prod.idxmax()} ({prod.max():,.1f})"
                   + (" · grey = no data published" if prod.isna().any() else ""))


@st.fragment
def map_button():
    # Fragment: clicking only re-runs this block, not the whole dashboard, so the pop-up opens instantly
    with st.container(key="map_fab"):
        if st.button("Paddy production map", key="map_fab_btn", help="Sri Lanka paddy production map"):
            st.session_state["paddy_map_open_n"] = st.session_state.get("paddy_map_open_n", 0) + 1   # new dropdown each open -> default year (2025)
            paddy_map_dialog()


map_button()
