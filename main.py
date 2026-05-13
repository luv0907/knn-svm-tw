from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

ROOT = Path(__file__).resolve().parent

# ─── Matplotlib global style ─────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1C1A15",
    "axes.facecolor": "#1C1A15",
    "axes.edgecolor": "#4A3F2F",
    "axes.labelcolor": "#D4B896",
    "xtick.color": "#A08060",
    "ytick.color": "#A08060",
    "text.color": "#E8D5B0",
    "grid.color": "#2E2820",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Custom colormaps
CREAM_MAP = LinearSegmentedColormap.from_list("cream", ["#1C1A15", "#4A3F2F", "#C8A96E", "#F5E6C8"])
SAGE_MAP  = LinearSegmentedColormap.from_list("sage",  ["#1C1A15", "#2A3B2E", "#5C8B6A", "#A8D4B0"])

# ─── Path helpers ─────────────────────────────────────────────────────────────
def prefer_existing_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback

KNN_REGRESSION_CSV = prefer_existing_path(
    ROOT / "KNN" / "regrssion" / "house_prices_deploy.csv",
    ROOT / "KNN" / "regrssion" / "house_prices.csv",
)
SVM_CLASSIFICATION_CSV = ROOT / "SVM" / "classification" / "smart_healthcare_dataset.csv"
SVM_REGRESSION_CSV = prefer_existing_path(
    ROOT / "SVM" / "Regression" / "genz_social_media_usage_200k.csv",
    ROOT / "SVM" / "Regression" / "genz_social_media_usage_1M.csv",
)
SVM_REGRESSION_FEATURES = [
    "age", "gender", "daily_usage_hours", "primary_platform",
    "num_platforms_used", "avg_session_minutes", "night_usage",
    "addiction_level", "screen_time_before_sleep",
]

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Creamy ML Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Mono:wght@300;400;500&family=Cormorant+Garamond:wght@300;400;600&display=swap');

    /* ── Root palette ── */
    :root {
        --bg:        #0F0E0B;
        --bg2:       #16140F;
        --panel:     rgba(28, 24, 16, 0.88);
        --border:    rgba(180, 140, 80, 0.18);
        --border2:   rgba(180, 140, 80, 0.32);
        --cream0:    #F5E6C8;
        --cream1:    #D4B896;
        --cream2:    #A08060;
        --gold:      #C8A96E;
        --gold2:     #E8C87A;
        --sage:      #6B8F71;
        --sage2:     #A8C5AD;
        --rust:      #C47A50;
        --ink:       #2A2218;
        --text:      #D4C8B4;
        --text-dim:  #7A6A52;
        --glow:      rgba(200, 169, 110, 0.12);
    }

    /* ── Base ── */
    html, body, .stApp {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Mono', monospace !important;
    }

    /* Animated grain overlay */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
        opacity: 0.45;
        pointer-events: none;
        z-index: 0;
    }

    .block-container {
        padding: 1.2rem 2rem 3rem 2rem !important;
        max-width: 1280px !important;
        position: relative;
        z-index: 1;
    }

    /* ── Hero ── */
    .hero-wrap {
        background: linear-gradient(135deg, rgba(28,24,16,0.95) 0%, rgba(38,32,20,0.90) 100%);
        border: 1px solid var(--border2);
        border-radius: 20px;
        padding: 2.4rem 2.6rem 2rem 2.6rem;
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(200,169,110,0.08), 0 24px 48px rgba(0,0,0,0.5);
        animation: heroIn 0.7s cubic-bezier(.16,1,.3,1) forwards;
    }
    @keyframes heroIn {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .hero-wrap::before {
        content: "";
        position: absolute;
        top: -80px; right: -80px;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(200,169,110,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-wrap::after {
        content: "";
        position: absolute;
        bottom: -40px; left: 60px;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(107,143,113,0.10) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.28rem 0.85rem;
        border-radius: 999px;
        border: 1px solid rgba(200,169,110,0.4);
        background: rgba(200,169,110,0.10);
        color: var(--gold);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 900;
        color: var(--cream0);
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin: 0 0 0.5rem 0;
    }
    .hero-title span {
        background: linear-gradient(90deg, var(--gold2), var(--sage2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: var(--text-dim);
        font-size: 0.88rem;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.02em;
        max-width: 540px;
    }
    .hero-dots {
        display: flex;
        gap: 0.45rem;
        margin-top: 1.4rem;
        flex-wrap: wrap;
    }
    .hero-dot {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 6px;
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border);
        color: var(--text-dim);
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        font-family: 'DM Mono', monospace;
    }

    /* ── Panels ── */
    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        backdrop-filter: blur(8px);
        transition: border-color 280ms ease, box-shadow 280ms ease, transform 280ms ease;
        animation: panelIn 0.5s cubic-bezier(.16,1,.3,1) both;
    }
    .panel:hover {
        border-color: var(--border2);
        box-shadow: 0 12px 40px rgba(200,169,110,0.07), 0 0 0 1px rgba(200,169,110,0.05);
        transform: translateY(-2px);
    }
    @keyframes panelIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .panel-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: var(--cream1);
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .panel-title::before {
        content: "";
        display: inline-block;
        width: 3px; height: 1em;
        background: linear-gradient(180deg, var(--gold), var(--sage));
        border-radius: 2px;
    }

    /* ── Metric cards ── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.85rem;
        margin-bottom: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(40,34,22,0.95), rgba(28,24,16,0.95));
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        transition: transform 220ms ease, border-color 220ms ease;
        animation: metricPop 0.6s cubic-bezier(.16,1,.3,1) both;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(200,169,110,0.35);
    }
    .metric-card::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--gold), var(--sage), transparent);
    }
    @keyframes metricPop {
        from { opacity: 0; transform: scale(0.96); }
        to   { opacity: 1; transform: scale(1); }
    }
    .metric-label {
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-dim);
        font-family: 'DM Mono', monospace;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--cream0);
        line-height: 1;
    }
    .metric-accent {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: var(--sage2);
        margin-top: 0.25rem;
    }

    /* ── Module cards (overview) ── */
    .module-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.85rem;
        margin-top: 0.5rem;
    }
    .module-card {
        background: linear-gradient(135deg, rgba(34,28,18,0.96), rgba(22,18,12,0.96));
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        cursor: default;
        transition: all 280ms cubic-bezier(.16,1,.3,1);
        position: relative;
        overflow: hidden;
    }
    .module-card:hover {
        border-color: rgba(200,169,110,0.4);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(200,169,110,0.08);
    }
    .module-card::before {
        content: attr(data-num);
        position: absolute;
        top: 0.8rem; right: 1rem;
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 900;
        color: rgba(200,169,110,0.06);
        line-height: 1;
        pointer-events: none;
    }
    .module-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    .module-name {
        font-family: 'Playfair Display', serif;
        font-size: 1rem;
        color: var(--cream1);
        margin-bottom: 0.25rem;
    }
    .module-desc {
        font-size: 0.78rem;
        color: var(--text-dim);
        line-height: 1.5;
        font-family: 'DM Mono', monospace;
    }
    .module-tag {
        display: inline-block;
        margin-top: 0.6rem;
        padding: 0.18rem 0.55rem;
        border-radius: 5px;
        font-size: 0.65rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: 'DM Mono', monospace;
    }
    .tag-knn  { background: rgba(200,169,110,0.12); color: var(--gold); border: 1px solid rgba(200,169,110,0.2); }
    .tag-svm  { background: rgba(107,143,113,0.12); color: var(--sage2); border: 1px solid rgba(107,143,113,0.2); }
    .tag-cls  { background: rgba(196,122,80,0.12);  color: #D4946A; border: 1px solid rgba(196,122,80,0.2); }
    .tag-reg  { background: rgba(120,100,160,0.12); color: #B4A0D4; border: 1px solid rgba(120,100,160,0.2); }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D0C09 0%, #131108 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] > div {
        padding: 1.2rem 1rem;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
        font-family: 'DM Mono', monospace !important;
    }
    section[data-testid="stSidebar"] h1 {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.1rem !important;
        color: var(--cream0) !important;
    }
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border2), transparent);
        margin: 0.8rem 0;
    }
    .sidebar-section {
        font-size: 0.65rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 0.5rem;
        font-family: 'DM Mono', monospace;
    }

    /* ── Streamlit widgets ── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--cream1) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    div[data-baseweb="select"] svg { color: var(--gold) !important; }
    label, .stSelectbox label, .stNumberInput label, .stSlider label {
        color: var(--text-dim) !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        font-family: 'DM Mono', monospace !important;
    }
    /* Slider track */
    div[data-testid="stSlider"] > div > div > div {
        background: var(--border) !important;
    }
    div[data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, var(--gold), var(--sage)) !important;
    }
    div[data-testid="stSlider"] > div > div > div > div > div {
        background: var(--cream0) !important;
        border: 2px solid var(--gold) !important;
        box-shadow: 0 0 10px rgba(200,169,110,0.3) !important;
    }

    /* Radio buttons */
    div[data-testid="stRadio"] label {
        text-transform: none !important;
        letter-spacing: 0 !important;
        color: var(--text) !important;
        font-size: 0.85rem !important;
    }

    /* Buttons */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, rgba(200,169,110,0.15), rgba(107,143,113,0.12)) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 10px !important;
        color: var(--cream0) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.06em !important;
        padding: 0.5rem 1.4rem !important;
        transition: all 200ms ease !important;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, rgba(200,169,110,0.25), rgba(107,143,113,0.20)) !important;
        border-color: rgba(200,169,110,0.5) !important;
        box-shadow: 0 0 20px rgba(200,169,110,0.15) !important;
        transform: translateY(-1px) !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] summary {
        color: var(--cream1) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    /* Success / info boxes */
    div[data-testid="stSuccess"] {
        background: rgba(107,143,113,0.12) !important;
        border: 1px solid rgba(107,143,113,0.3) !important;
        border-radius: 10px !important;
        color: var(--sage2) !important;
    }
    div[data-testid="stInfo"] {
        background: rgba(200,169,110,0.08) !important;
        border: 1px solid rgba(200,169,110,0.2) !important;
        border-radius: 10px !important;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* Form submit button */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, var(--gold), var(--sage)) !important;
        border: none !important;
        color: var(--ink) !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        box-shadow: 0 4px 24px rgba(200,169,110,0.3) !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        box-shadow: 0 8px 32px rgba(200,169,110,0.45) !important;
        transform: translateY(-2px) !important;
    }

    /* Caption / small text */
    div[data-testid="stCaptionContainer"] p {
        color: var(--text-dim) !important;
        font-size: 0.75rem !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg2); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--gold); }

    /* Horizontal rule */
    hr { border-color: var(--border) !important; }

    /* stMetric default override */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.025) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
    }
    div[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-family: 'DM Mono', monospace !important; }
    div[data-testid="stMetricValue"] { color: var(--cream0) !important; font-family: 'Playfair Display', serif !important; }

    /* Pulse animation for active elements */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(200,169,110,0); }
        50%       { box-shadow: 0 0 0 6px rgba(200,169,110,0.08); }
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Helper utilities ─────────────────────────────────────────────────────────
def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def parse_sqft(value) -> float:
    if pd.isna(value): return np.nan
    text = str(value).lower()
    token = "".join(c for c in text if c.isdigit() or c == ".")
    try: return float(token)
    except: return np.nan


def detect_house_price_target(df: pd.DataFrame) -> str:
    for col in ["Price", "price", "SalePrice", "sale_price", "target"]:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols: return numeric_cols[-1]
    raise ValueError("No target column found in house price dataset.")


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = X.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", one_hot_encoder())]), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric_cols, cat_cols


# ─── Custom metric row ────────────────────────────────────────────────────────
def render_metric_row(metrics: list[tuple]) -> None:
    cols_html = ""
    accent_map = ["✦ model score", "✦ error metric", "✦ dataset size"]
    for i, (label, value, _) in enumerate(metrics):
        if isinstance(value, float) and value < 10:
            display = f"{value:.4f}"
        elif isinstance(value, float):
            display = f"{value:,.0f}"
        else:
            display = str(value)
        accent = accent_map[i % len(accent_map)]
        cols_html += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{display}</div>
            <div class="metric-accent">{accent}</div>
        </div>"""

    st.markdown(f'<div class="metric-row">{cols_html}</div>', unsafe_allow_html=True)


# ─── Plot helpers ─────────────────────────────────────────────────────────────
def _fig_base(figsize=(6, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#12100C")
    ax.set_facecolor("#12100C")
    ax.grid(True, color="#2A2418", linestyle="--", alpha=0.6, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3A3020")
    return fig, ax


def plot_confusion_matrix(cm: np.ndarray, labels: list[str]) -> plt.Figure:
    fig, ax = _fig_base((5, 4))
    cmap = LinearSegmentedColormap.from_list("cm", ["#12100C", "#3A3020", "#C8A96E"])
    im = ax.imshow(cm, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5, color="#A08060")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5, color="#A08060")
    ax.set_xlabel("Predicted", fontsize=8, color="#7A6A52", labelpad=6)
    ax.set_ylabel("Actual", fontsize=8, color="#7A6A52", labelpad=6)
    ax.set_title("Confusion Matrix", fontsize=10, color="#D4B896", pad=12, fontfamily="serif")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="#F5E6C8" if cm[i, j] > thresh else "#6A5A40", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="#7A6A52", labelsize=7)
    fig.tight_layout(pad=1.5)
    return fig


def plot_regression_scatter(y_test, preds, title="Predicted vs Actual") -> plt.Figure:
    fig, ax = _fig_base((5.5, 4.2))
    # Scatter with density coloring
    from matplotlib.colors import Normalize
    y_arr = np.array(y_test); p_arr = np.array(preds)
    residuals = np.abs(y_arr - p_arr)
    norm = Normalize(vmin=residuals.min(), vmax=residuals.max())
    colors = plt.cm.RdYlGn_r(norm(residuals))
    ax.scatter(y_arr, p_arr, c=colors, alpha=0.6, s=18, edgecolors="none")
    mn = min(y_arr.min(), p_arr.min()); mx = max(y_arr.max(), p_arr.max())
    ax.plot([mn, mx], [mn, mx], "--", color="#C8A96E", alpha=0.7, linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("Actual", fontsize=8, color="#7A6A52", labelpad=6)
    ax.set_ylabel("Predicted", fontsize=8, color="#7A6A52", labelpad=6)
    ax.set_title(title, fontsize=10, color="#D4B896", pad=12, fontfamily="serif")
    ax.legend(fontsize=7.5, facecolor="#1C1A15", edgecolor="#3A3020", labelcolor="#A08060")
    fig.tight_layout(pad=1.5)
    return fig


def plot_residuals(y_test, preds, title="Residual Distribution") -> plt.Figure:
    fig, ax = _fig_base((5.5, 4.2))
    residuals = np.array(y_test) - np.array(preds)
    n, bins, patches = ax.hist(residuals, bins=28, edgecolor="none", alpha=0.9)
    # Color gradient on bars
    cmap = LinearSegmentedColormap.from_list("r", ["#6B8F71", "#C8A96E", "#C47A50"])
    col_norm = plt.Normalize(bins.min(), bins.max())
    for patch, left in zip(patches, bins):
        patch.set_facecolor(cmap(col_norm(left + (bins[1]-bins[0])/2)))
    ax.axvline(0, color="#F5E6C8", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Residual", fontsize=8, color="#7A6A52", labelpad=6)
    ax.set_ylabel("Count", fontsize=8, color="#7A6A52", labelpad=6)
    ax.set_title(title, fontsize=10, color="#D4B896", pad=12, fontfamily="serif")
    fig.tight_layout(pad=1.5)
    return fig


def plot_decision_boundary(model, X_train, y_train, pca, target_names) -> plt.Figure:
    fig, ax = _fig_base((7, 5))
    scaled = model.named_steps["scale"].transform(X_train)
    train_2d = pca.transform(scaled)
    xx, yy = np.meshgrid(
        np.linspace(train_2d[:, 0].min()-1.2, train_2d[:, 0].max()+1.2, 300),
        np.linspace(train_2d[:, 1].min()-1.2, train_2d[:, 1].max()+1.2, 300),
    )
    grid = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    knn_step = model.named_steps.get("knn") or model.named_steps.get("svm")
    Z = knn_step.predict(grid).reshape(xx.shape)

    # Muted fill colors
    cmap_fill = LinearSegmentedColormap.from_list("fill", ["#1E2E22", "#2E2018", "#1C2030"])
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_fill, levels=[-0.5, 0.5, 1.5, 2.5])
    ax.contour(xx, yy, Z, colors=["#4A3020", "#3A4830", "#302840"], linewidths=0.6, alpha=0.5)

    colors_pts = ["#C8A96E", "#6B8F71", "#C47A50"]
    for cls_idx in np.unique(y_train):
        mask = y_train == cls_idx
        ax.scatter(train_2d[mask, 0], train_2d[mask, 1],
                   color=colors_pts[cls_idx % len(colors_pts)],
                   edgecolors="#1C1A15", linewidths=0.5,
                   s=55, alpha=0.88, label=target_names[cls_idx],
                   zorder=3)

    ax.set_xlabel("PCA Component 1", fontsize=8, color="#7A6A52", labelpad=6)
    ax.set_ylabel("PCA Component 2", fontsize=8, color="#7A6A52", labelpad=6)
    ax.set_title("Decision Boundary (PCA 2D)", fontsize=10, color="#D4B896", pad=12, fontfamily="serif")
    leg = ax.legend(fontsize=8, facecolor="#1A180F", edgecolor="#3A3020")
    for text in leg.get_texts(): text.set_color("#C8A96E")
    fig.tight_layout(pad=1.5)
    return fig


# ─── Prediction form ──────────────────────────────────────────────────────────
def build_prediction_inputs(source, feature_columns, key_prefix, free_text_columns=None):
    free_text_columns = free_text_columns or set()
    with st.form(f"{key_prefix}_form"):
        st.markdown('<div class="panel-title">⬡ Prediction inputs</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        values: dict = {}
        for i, col in enumerate(feature_columns):
            tgt = left if i % 2 == 0 else right
            series = source[col] if col in source.columns else pd.Series(dtype="object")
            with tgt:
                if col in free_text_columns:
                    default = str(series.dropna().iloc[0]) if not series.dropna().empty else ""
                    values[col] = st.text_input(col, value=default, key=f"{key_prefix}_{col}")
                elif pd.api.types.is_numeric_dtype(series):
                    default = float(series.median()) if not series.empty else 0.0
                    mn = float(series.min()) if not series.empty else 0.0
                    mx = float(series.max()) if not series.empty else max(default + 1, 1.0)
                    if np.isfinite(mn) and np.isfinite(mx) and mn != mx:
                        values[col] = st.number_input(col, min_value=mn, max_value=mx, value=default, key=f"{key_prefix}_{col}")
                    else:
                        values[col] = st.number_input(col, value=default, key=f"{key_prefix}_{col}")
                else:
                    opts = series.dropna().astype(str).unique().tolist() if not series.empty else []
                    if 1 <= len(opts) <= 20:
                        values[col] = st.selectbox(col, opts, index=0, key=f"{key_prefix}_{col}")
                    else:
                        default = str(series.dropna().astype(str).mode().iloc[0]) if not series.empty and not series.dropna().empty else ""
                        values[col] = st.text_input(col, value=default, key=f"{key_prefix}_{col}")
        submitted = st.form_submit_button("✦ Run Prediction")
    if not submitted: return None
    return pd.DataFrame([values], columns=feature_columns)


# ─── App header ───────────────────────────────────────────────────────────────
def app_header() -> None:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">✦ ML STUDIO · CONCEPT LAB</div>
        <div class="hero-title">Creamy<br><span>ML Studio</span></div>
        <p class="hero-sub">Interactive KNN & SVM workflows with live metrics, animated charts, and clean prediction interfaces.</p>
        <div class="hero-dots">
            <span class="hero-dot">KNN Classification</span>
            <span class="hero-dot">KNN Regression</span>
            <span class="hero-dot">SVM Classification</span>
            <span class="hero-dot">SVM Regression</span>
            <span class="hero-dot">Scikit-learn · Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Overview ─────────────────────────────────────────────────────────────────
def page_overview() -> None:
    st.markdown("""
    <div class="module-grid">
        <div class="module-card" data-num="01">
            <span class="module-icon">⬡</span>
            <div class="module-name">KNN Classification</div>
            <div class="module-desc">Iris dataset · tunable k · PCA decision boundary · class probabilities</div>
            <span class="module-tag tag-knn">KNN</span>
            <span class="module-tag tag-cls">Classification</span>
        </div>
        <div class="module-card" data-num="02">
            <span class="module-icon">◈</span>
            <div class="module-name">KNN Regression</div>
            <div class="module-desc">House prices · custom feature set · scatter & residual plots</div>
            <span class="module-tag tag-knn">KNN</span>
            <span class="module-tag tag-reg">Regression</span>
        </div>
        <div class="module-card" data-num="03">
            <span class="module-icon">◇</span>
            <div class="module-name">SVM Classification</div>
            <div class="module-desc">Iris · RBF/linear/poly kernels · C & gamma tuning · confusion matrix</div>
            <span class="module-tag tag-svm">SVM</span>
            <span class="module-tag tag-cls">Classification</span>
        </div>
        <div class="module-card" data-num="04">
            <span class="module-icon">○</span>
            <div class="module-name">SVM Regression</div>
            <div class="module-desc">Gen Z social media · mental health score · SVR with epsilon tube</div>
            <span class="module-tag tag-svm">SVM</span>
            <span class="module-tag tag-reg">Regression</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── KNN Classification ───────────────────────────────────────────────────────
def knn_classification_page() -> None:
    st.markdown('<div class="panel"><div class="panel-title">KNN Classification</div>Iris dataset · scaler + KNN pipeline · PCA decision boundary</div>', unsafe_allow_html=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    feature_columns = iris.feature_names

    with st.sidebar:
        st.markdown('<div class="sidebar-section">KNN Controls</div>', unsafe_allow_html=True)
        k_value     = st.slider("Neighbors (k)", 1, 25, 5, 2)
        test_size   = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)

    X = df[feature_columns]; y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y)
    model = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k_value))])
    model.fit(X_train, y_train)
    tr_pred = model.predict(X_train); te_pred = model.predict(X_test)

    render_metric_row([
        ("Train Accuracy", accuracy_score(y_train, tr_pred), ""),
        ("Test Accuracy",  accuracy_score(y_test, te_pred), ""),
        ("Dataset Rows",   float(len(df)), ""),
    ])

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Classification Report</div>', unsafe_allow_html=True)
        st.code(classification_report(y_test, te_pred, target_names=iris.target_names), language=None)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Confusion Matrix</div>', unsafe_allow_html=True)
        st.pyplot(plot_confusion_matrix(confusion_matrix(y_test, te_pred), list(iris.target_names)))
        st.markdown('</div>', unsafe_allow_html=True)

    # PCA boundary
    pca = PCA(n_components=2, random_state=int(random_state))
    pca.fit(model.named_steps["scale"].transform(X_train))
    boundary_fig = plot_decision_boundary(model, X_train.values, y_train.values, pca, iris.target_names)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">PCA Decision Boundary</div>', unsafe_allow_html=True)
    st.pyplot(boundary_fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    input_frame = build_prediction_inputs(X, feature_columns, "knn_class")
    if input_frame is not None:
        pred = model.predict(input_frame)[0]
        st.success(f"✦ Predicted class → **{iris.target_names[int(pred)]}**")
        if hasattr(model.named_steps["knn"], "predict_proba"):
            probs = model.predict_proba(input_frame)[0]
            st.caption("  ·  ".join(f"{iris.target_names[i]}: {p:.3f}" for i, p in enumerate(probs)))
    st.markdown('</div>', unsafe_allow_html=True)


# ─── KNN Regression ───────────────────────────────────────────────────────────
def knn_regression_page() -> None:
    st.markdown('<div class="panel"><div class="panel-title">KNN Regression</div>House price prediction using location, area, furnishing, and amenity features.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-section">KNN Regression</div>', unsafe_allow_html=True)
        sample_frac = st.slider("Sample fraction", 0.05, 1.0, 0.25, 0.05)
        neighbors   = st.slider("Neighbors", 1, 35, 7, 2)
        weights     = st.selectbox("Weights", ["uniform", "distance"], index=1)
        p_value     = st.selectbox("Distance metric p", [1, 2], index=1)

    # ── File existence guard ──────────────────────────────────────────────────
    if not KNN_REGRESSION_CSV.exists():
        st.markdown(f"""
        <div class="panel" style="border-color:rgba(196,122,80,0.4);text-align:center;padding:2.5rem;">
            <div style="font-size:2rem;margin-bottom:0.8rem;">⚠</div>
            <div class="panel-title" style="justify-content:center;color:#D4946A;">Dataset Not Found</div>
            <p style="color:var(--text-dim);font-size:0.82rem;margin-top:0.5rem;">
                Expected file at:<br>
                <code style="color:var(--gold);background:rgba(200,169,110,0.1);
                             padding:0.2rem 0.5rem;border-radius:5px;font-size:0.78rem;">
                    {KNN_REGRESSION_CSV}
                </code>
            </p>
            <p style="color:var(--text-dim);font-size:0.78rem;margin-top:0.8rem;">
                Place <code style="color:var(--cream1)">house_prices.csv</code> or
                <code style="color:var(--cream1)">house_prices_deploy.csv</code>
                inside <code style="color:var(--cream1)">KNN/regrssion/</code> and restart.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = load_csv(str(KNN_REGRESSION_CSV))

    # ── Empty / missing target guard ─────────────────────────────────────────
    try:
        target_col = detect_house_price_target(df)
    except ValueError:
        st.error("✦ Could not detect a price/target column in the CSV. Check your column names.")
        st.dataframe(df.head(5))
        return

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    if len(df) < 10:
        st.markdown(f"""
        <div class="panel" style="border-color:rgba(196,122,80,0.4);padding:1rem 1.25rem;">
            <div class="panel-title" style="color:#D4946A;font-weight:700;">Too Few Rows</div>
            <p style="color:var(--text-dim);font-size:0.9rem;">
                Only <strong style="color:var(--cream-0)">{len(df)}</strong> usable rows found after dropping nulls
                in <code style="color:var(--gold)">{target_col}</code>.
            </p>
            <p style="color:var(--text-dim);font-size:0.85rem;">You can either upload a different CSV or use a small sample dataset to try the UI.</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload a CSV with house price data (optional)", type=["csv"])
        if uploaded is not None:
            try:
                uploaded_df = pd.read_csv(uploaded).drop_duplicates().reset_index(drop=True)
            except Exception as exc:  # fallback parsing error
                st.error(f"Failed to read uploaded CSV: {exc}")
                return
            # try to detect target in uploaded file
            try:
                tcol = detect_house_price_target(uploaded_df)
            except ValueError:
                st.error("Uploaded CSV does not contain a recognizable price column. Rename your price column and try again.")
                return
            uploaded_df[tcol] = pd.to_numeric(uploaded_df[tcol], errors="coerce")
            uploaded_df = uploaded_df.dropna(subset=[tcol]).reset_index(drop=True)
            if len(uploaded_df) < 10:
                st.error(f"Uploaded file has only {len(uploaded_df)} usable rows after cleaning. Need at least 10.")
                return
            # replace df with uploaded and continue
            df = uploaded_df
            target_col = tcol
        else:
            if st.button("Use generated sample data to try the UI"):
                # create a small synthetic sample dataset
                sample_size = 40
                import random

                locs = ["Area A", "Area B", "Area C", "Area D"]
                furnish = ["Furnished", "Semi-Furnished", "Unfurnished"]
                facing = ["North", "South", "East", "West"]
                ownership = ["Owner", "Builder"]

                sample = pd.DataFrame({
                    "location": [random.choice(locs) for _ in range(sample_size)],
                    "Carpet Area": [round(random.uniform(350, 3000), 1) for _ in range(sample_size)],
                    "Furnishing": [random.choice(furnish) for _ in range(sample_size)],
                    "facing": [random.choice(facing) for _ in range(sample_size)],
                    "Bathroom": [random.randint(1, 4) for _ in range(sample_size)],
                    "Balcony": [random.randint(0, 2) for _ in range(sample_size)],
                    "Car Parking": [random.randint(0, 2) for _ in range(sample_size)],
                    "Ownership": [random.choice(ownership) for _ in range(sample_size)],
                    target_col: [round(random.uniform(500000, 15000000), 2) for _ in range(sample_size)],
                })
                df = sample
            else:
                return

    sampled = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True) if sample_frac < 1.0 else df.copy()

    # Ensure sample is large enough after fractional sub-sampling
    if len(sampled) < 10:
        st.warning(f"✦ Sample fraction {sample_frac} yields only {len(sampled)} rows — increase the slider.")
        return

    feat_cols = [c for c in ["location","Carpet Area","Furnishing","facing","Bathroom","Balcony","Car Parking","Ownership"] if c in sampled.columns]
    house = sampled[feat_cols + [target_col]].copy()
    if "Carpet Area" in house.columns: house["Carpet Area"] = house["Carpet Area"].apply(parse_sqft)
    for c in ["Bathroom","Balcony","Car Parking"]:
        if c in house.columns: house[c] = pd.to_numeric(house[c], errors="coerce")

    X = house.drop(columns=[target_col]); y = house[target_col]
    preprocessor, _, _ = build_preprocessor(X)
    model = Pipeline([("preprocess", preprocessor), ("knn", KNeighborsRegressor(n_neighbors=neighbors, weights=weights, p=p_value))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train); preds = model.predict(X_test)

    render_metric_row([
        ("MAE",  mean_absolute_error(y_test, preds), ""),
        ("RMSE", float(np.sqrt(mean_squared_error(y_test, preds))), ""),
        ("R²",   r2_score(y_test, preds), ""),
    ])

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Predicted vs Actual</div>', unsafe_allow_html=True)
        st.pyplot(plot_regression_scatter(y_test, preds))
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Residual Distribution</div>', unsafe_allow_html=True)
        st.pyplot(plot_residuals(y_test, preds))
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Preview dataset"):
        st.dataframe(df.head(12), use_container_width=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    free_text = {"location","Furnishing","facing","Ownership"}
    input_frame = build_prediction_inputs(house, feat_cols, "knn_reg", free_text_columns=free_text)
    if input_frame is not None:
        pf = input_frame.copy()
        if "Carpet Area" in pf.columns: pf["Carpet Area"] = pf["Carpet Area"].apply(parse_sqft)
        for c in ["Bathroom","Balcony","Car Parking"]:
            if c in pf.columns: pf[c] = pd.to_numeric(pf[c], errors="coerce")
        pf = pf.reindex(columns=X.columns, fill_value=np.nan)
        try:
            pred_val = model.predict(pf)[0]
            st.success(f"✦ Predicted price → **{pred_val:,.2f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.caption("Prediction input (for debugging):")
            st.dataframe(pf.T, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── SVM Classification ───────────────────────────────────────────────────────
def svm_classification_page() -> None:
    st.markdown('<div class="panel"><div class="panel-title">SVM Classification</div>Iris classification via SVC pipeline with kernel selection and probability output.</div>', unsafe_allow_html=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    feature_columns = iris.feature_names

    with st.sidebar:
        st.markdown('<div class="sidebar-section">SVM Controls</div>', unsafe_allow_html=True)
        kernel    = st.selectbox("Kernel", ["rbf","linear","poly","sigmoid"], index=0)
        c_value   = st.slider("C", 0.1, 20.0, 10.0, 0.1)
        gamma     = st.selectbox("Gamma", ["scale","auto"], index=0)
        test_size = st.slider("Test size", 0.1, 0.4, 0.3, 0.05)

    X = df[feature_columns]; y = df["target"]
    preprocessor, _, _ = build_preprocessor(X)
    model = Pipeline([
        ("preprocess", preprocessor),
        ("svm", SVC(kernel=kernel, C=c_value, gamma=gamma, probability=True, class_weight="balanced")),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100, stratify=y)
    model.fit(X_train, y_train)
    tr_pred = model.predict(X_train); te_pred = model.predict(X_test)

    render_metric_row([
        ("Train Accuracy", accuracy_score(y_train, tr_pred), ""),
        ("Test Accuracy",  accuracy_score(y_test, te_pred), ""),
        ("Dataset Rows",   float(len(df)), ""),
    ])

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Classification Report</div>', unsafe_allow_html=True)
        st.code(classification_report(y_test, te_pred, target_names=iris.target_names), language=None)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Confusion Matrix</div>', unsafe_allow_html=True)
        st.pyplot(plot_confusion_matrix(confusion_matrix(y_test, te_pred), list(iris.target_names)))
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Preview Iris dataset"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    input_frame = build_prediction_inputs(df, feature_columns, "svm_class")
    if input_frame is not None:
        pred = model.predict(input_frame)[0]
        st.success(f"✦ Predicted Iris class → **{iris.target_names[int(pred)]}**")
        if hasattr(model.named_steps["svm"], "predict_proba"):
            probs = model.predict_proba(input_frame)[0]
            st.caption("  ·  ".join(f"{iris.target_names[i]}: {p:.3f}" for i, p in enumerate(probs)))
    st.markdown('</div>', unsafe_allow_html=True)


# ─── SVM Regression ───────────────────────────────────────────────────────────
def svm_regression_page() -> None:
    st.markdown('<div class="panel"><div class="panel-title">SVM Regression</div>Predict <code style="color:var(--gold);background:rgba(200,169,110,0.1);padding:0 4px;border-radius:4px">mental_health_score</code> from Gen Z social media dataset via SVR.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-section">SVR Controls</div>', unsafe_allow_html=True)
        sample_frac = st.slider("Sample fraction", 0.002, 0.05, 0.01, 0.002)
        kernel      = st.selectbox("Kernel", ["rbf","linear","poly","sigmoid"], index=0)
        c_value     = st.slider("C", 0.1, 50.0, 10.0, 0.1)
        epsilon     = st.slider("Epsilon", 0.01, 1.0, 0.1, 0.01)
        gamma       = st.selectbox("Gamma", ["scale","auto"], index=0)

    # ── File existence guard ──────────────────────────────────────────────────
    if not SVM_REGRESSION_CSV.exists():
        st.markdown(f"""
        <div class="panel" style="border-color:rgba(196,122,80,0.4);text-align:center;padding:2.5rem;">
            <div style="font-size:2rem;margin-bottom:0.8rem;">&#9888;</div>
            <div class="panel-title" style="justify-content:center;color:#D4946A;">Dataset Not Found</div>
            <p style="color:var(--text-dim);font-size:0.82rem;margin-top:0.5rem;">
                Expected at: <code style="color:var(--gold)">{SVM_REGRESSION_CSV}</code>
            </p>
            <p style="color:var(--text-dim);font-size:0.78rem;margin-top:0.8rem;">
                Place <code style="color:var(--cream1)">genz_social_media_usage_200k.csv</code>
                inside <code style="color:var(--cream1)">SVM/Regression/</code> and restart.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = load_csv(str(SVM_REGRESSION_CSV))
    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True) if sample_frac < 1.0 else df.copy()
    target_col = "mental_health_score"

    if target_col not in df.columns:
        st.error(f"Column '{target_col}' not found. Available: {', '.join(df.columns[:10])}")
        return
    if len(df) < 10:
        st.warning(f"Only {len(df)} rows after sampling (fraction={sample_frac}). Increase the slider.")
        return

    feat_cols  = [c for c in SVM_REGRESSION_FEATURES if c in df.columns]
    reg_frame = df[feat_cols + [target_col]].copy()
    X = reg_frame[feat_cols]; y = df[target_col]
    preprocessor, _, _ = build_preprocessor(X)
    model = Pipeline([
        ("preprocess", preprocessor),
        ("svm", SVR(kernel=kernel, C=c_value, epsilon=epsilon, gamma=gamma)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train); preds = model.predict(X_test)

    render_metric_row([
        ("MAE",  mean_absolute_error(y_test, preds), ""),
        ("RMSE", float(np.sqrt(mean_squared_error(y_test, preds))), ""),
        ("R²",   r2_score(y_test, preds), ""),
    ])

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Predicted vs Actual</div>', unsafe_allow_html=True)
        st.pyplot(plot_regression_scatter(y_test, preds))
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Residual Distribution</div>', unsafe_allow_html=True)
        st.pyplot(plot_residuals(y_test, preds))
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Preview sampled data"):
        st.dataframe(reg_frame.head(10), use_container_width=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    input_frame = build_prediction_inputs(reg_frame, feat_cols, "svm_reg")
    if input_frame is not None:
        st.success(f"✦ Predicted mental_health_score → **{model.predict(input_frame)[0]:.3f}**")
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    app_header()

    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;">
            <span style="font-size:1.2rem">✦</span>
            <span style="font-family:'Playfair Display',serif;font-size:1rem;color:#F5E6C8;">ML Studio</span>
        </div>
        <div style="font-size:0.68rem;color:#5A4A32;letter-spacing:0.1em;margin-bottom:1rem;">CONCEPT LAB · v3</div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

        page = st.radio(
            "Module",
            ["Overview", "KNN Classification", "KNN Regression", "SVM Classification", "SVM Regression"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Datasets</div>', unsafe_allow_html=True)

        try:
            knn_p = str(KNN_REGRESSION_CSV.relative_to(ROOT))
        except Exception:
            knn_p = str(KNN_REGRESSION_CSV)
        try:
            svm_p = str(SVM_REGRESSION_CSV.relative_to(ROOT))
        except Exception:
            svm_p = str(SVM_REGRESSION_CSV)

        st.caption(f"KNN: {knn_p}")
        st.caption("SVM cls: iris (built-in)")
        st.caption(f"SVR: {svm_p}")

    if page == "Overview":
        page_overview()
    elif page == "KNN Classification":
        knn_classification_page()
    elif page == "KNN Regression":
        knn_regression_page()
    elif page == "SVM Classification":
        svm_classification_page()
    else:
        svm_regression_page()


if __name__ == "__main__":
    main()