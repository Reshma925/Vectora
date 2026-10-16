"""
Vectora – Unified ML Dashboard (Streamlit)
==========================================
Covers:
  • Classification  – Star Type (Galaxy / QSO / Star)
  • Regression      – Community Crime Rate
  • Clustering      – K-Means / DBSCAN / Agglomerative (interactive)
  • EDA Explorer    – Dataset exploration for both tracks
"""

import os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler as SkScaler

# ─── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Vectora · ML Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  /* Global dark background */
  .stApp { background-color: #0d1117; color: #e6edf3; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #30363d; }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 { color: #58a6ff; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 12px 16px;
  }
  [data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.8rem; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.5rem; font-weight: 700; }

  /* Headings */
  h1 { color: #58a6ff !important; }
  h2 { color: #79c0ff !important; }
  h3 { color: #a5d6ff !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 10px; }
  .stTabs [data-baseweb="tab"] { color: #8b949e; font-weight: 600; }
  .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

  /* DataFrames */
  .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

  /* Selectboxes / sliders */
  .stSelectbox label, .stSlider label, .stNumberInput label { color: #8b949e; }

  /* Prediction result box */
  .pred-box {
    background: linear-gradient(135deg, #1f2d3d 0%, #162032 100%);
    border: 1px solid #388bfd;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
    text-align: center;
  }
  .pred-label { font-size: 1rem; color: #8b949e; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px; }
  .pred-value { font-size: 2.4rem; font-weight: 800; color: #58a6ff; letter-spacing: 0.02em; }
  .pred-sub   { font-size: 0.95rem; color: #79c0ff; margin-top: 4px; }

  /* Info banners */
  .info-banner {
    background: #162032;
    border-left: 4px solid #388bfd;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #a5d6ff;
  }

  /* Section divider */
  hr { border-color: #30363d; }

  /* Scrollable feature table */
  .scroll-box { max-height: 340px; overflow-y: auto; }

  /* Radio group */
  .stRadio > div { flex-direction: row; gap: 16px; }

  /* Progress bar */
  .stProgress > div > div { background: linear-gradient(90deg, #238636, #58a6ff); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PATHS & MODEL LOADING
# ─────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
CLS_DIR    = os.path.join(BASE, "ML_CLASSIFICATION_TRACKA")
REG_DIR    = os.path.join(BASE, "Regression Track A")
CLS_MODELS = os.path.join(CLS_DIR, "models")
REG_MODELS = os.path.join(REG_DIR, "models")


@st.cache_resource
def load_cls_artifacts():
    scaler   = joblib.load(os.path.join(CLS_MODELS, "cls_scaler.pkl"))
    encoder  = joblib.load(os.path.join(CLS_MODELS, "cls_label_encoder.pkl"))
    feat_cols = joblib.load(os.path.join(CLS_MODELS, "cls_feature_cols.pkl"))
    bounds   = joblib.load(os.path.join(CLS_MODELS, "cls_outlier_bounds.pkl"))
    return scaler, encoder, feat_cols, bounds


@st.cache_resource
def load_cls_models():
    files = {
        "Logistic Regression":      "logistic_regression_model.pkl",
        "Gaussian Naive Bayes":     "gaussian_naive_bayes_model.pkl",
        "K-Nearest Neighbors":      "knn_model.pkl",
        "Decision Tree":            "decision_tree_model.pkl",
        "SVM":                      "svm_model.pkl",
        "Random Forest":            "random_forest_model.pkl",
        "Gradient Boosting":        "gradient_boosting_model.pkl",
        "AdaBoost":                 "adaboost_model.pkl",
        "Bagging Classifier":       "bagging_classifier_model.pkl",
        "MLP Classifier":           "mlp_classifier_model.pkl",
    }
    models = {}
    for name, fname in files.items():
        path = os.path.join(CLS_MODELS, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_reg_artifacts():
    imputer   = joblib.load(os.path.join(REG_MODELS, "reg_imputer.pkl"))
    scaler    = joblib.load(os.path.join(REG_MODELS, "reg_scaler.pkl"))
    feat_cols = joblib.load(os.path.join(REG_MODELS, "reg_feature_cols.pkl"))
    bounds    = joblib.load(os.path.join(REG_MODELS, "reg_clip_bounds.pkl"))
    return imputer, scaler, feat_cols, bounds


@st.cache_data
def load_cls_data():
    df = pd.read_csv(os.path.join(CLS_DIR, "star_classification.csv"))
    df.drop(columns=["class_numeric"], inplace=True, errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)
    df["r_i"] = df["r"] - df["i"]
    return df


@st.cache_data
def load_reg_data():
    df = pd.read_csv(os.path.join(REG_DIR, "communities.csv"), header=0, na_values=["?"])
    target = df.columns[-1]
    missing_pct = df.isna().mean() * 100
    high_missing = missing_pct[missing_pct > 50].index.tolist()
    id_cols = ["state", "county", "community", "communityname", "fold"]
    cols_to_drop = sorted(set(high_missing + id_cols))
    df_clean = df.drop(columns=cols_to_drop, errors="ignore").copy()
    family_cols = [c for c in ["PctKids2Par", "PctFam2Par", "PctYoungKids2Par", "PctTeen2Par"]
                   if c in df_clean.columns]
    if family_cols:
        df_clean["FamilyStructureAvg"] = df_clean[family_cols].mean(axis=1)
    return df_clean, target


# known model metrics from notebook outputs
CLS_METRICS = {
    "Logistic Regression":  {"Accuracy": 0.9785, "Weighted F1": 0.9785, "ROC-AUC": 0.9975},
    "Gaussian Naive Bayes": {"Accuracy": 0.9715, "Weighted F1": 0.9712, "ROC-AUC": 0.9971},
    "K-Nearest Neighbors":  {"Accuracy": 0.9873, "Weighted F1": 0.9873, "ROC-AUC": 0.9988},
    "Decision Tree":        {"Accuracy": 0.9780, "Weighted F1": 0.9780, "ROC-AUC": 0.9780},
    "SVM":                  {"Accuracy": 0.9820, "Weighted F1": 0.9820, "ROC-AUC": 0.9981},
    "Random Forest":        {"Accuracy": 0.9901, "Weighted F1": 0.9901, "ROC-AUC": 0.9993},
    "Gradient Boosting":    {"Accuracy": 0.9839, "Weighted F1": 0.9839, "ROC-AUC": 0.9988},
    "AdaBoost":             {"Accuracy": 0.8965, "Weighted F1": 0.8955, "ROC-AUC": 0.9812},
    "Bagging Classifier":   {"Accuracy": 0.9903, "Weighted F1": 0.9903, "ROC-AUC": 0.9994},
    "MLP Classifier":       {"Accuracy": 0.9801, "Weighted F1": 0.9801, "ROC-AUC": 0.9981},
}

# Models that use scaled features
CLS_NEEDS_SCALE = {"Logistic Regression", "K-Nearest Neighbors", "SVM", "MLP Classifier",
                   "Gaussian Naive Bayes"}

CLS_CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_COLORS = {"GALAXY": "#388bfd", "QSO": "#f78166", "STAR": "#3fb950"}
CLASS_ICONS  = {"GALAXY": "🌌", "QSO": "⚡", "STAR": "⭐"}
CLASS_DESC   = {
    "GALAXY": "A gravitationally bound system of stars, gas, dust, and dark matter.",
    "QSO":    "Quasi-Stellar Object — an extremely luminous active galactic nucleus powered by a supermassive black hole.",
    "STAR":   "A luminous spheroid of plasma held together by self-gravity undergoing nuclear fusion.",
}

FEATURE_TOOLTIPS = {
    "obj_ID":     "Object Identifier — unique catalog ID",
    "alpha":      "Right Ascension (degrees) — celestial longitude",
    "delta":      "Declination (degrees) — celestial latitude",
    "u":          "Ultraviolet photometric band magnitude",
    "g":          "Green photometric band magnitude",
    "r":          "Red photometric band magnitude",
    "i":          "Near-infrared band magnitude",
    "z":          "Infrared band magnitude",
    "run_ID":     "Run number of the SDSS scan",
    "rerun_ID":   "Rerun specification of the image processing",
    "cam_col":    "Camera column (1–6) used in observation",
    "field_ID":   "Field number of the telescope scan",
    "spec_obj_ID":"Spectroscopic object ID",
    "redshift":   "Redshift value — indicates recession speed / distance",
    "plate":      "Plate ID of the SDSS spectroscopic plate",
    "MJD":        "Modified Julian Date of observation",
    "fiber_ID":   "Fiber ID directing light to the spectrograph",
    "r_i":        "Engineered colour index: r − i band difference",
}


# ─────────────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌌 Vectora")
    st.markdown("<p style='color:#8b949e;font-size:0.82rem;'>Unified ML Dashboard</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠  Home",
         "🔭  Classification",
         "📈  Regression",
         "🔵  Clustering",
         "📊  EDA Explorer"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.75rem;color:#8b949e'>
<b>Tracks</b><br>
🔭 Star Classification (SDSS)<br>
📈 Community Crime Regression<br>
🔵 Clustering (new)<br>
📊 EDA Explorer
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("🌌 Vectora — ML Dashboard")
    st.markdown("<p style='color:#8b949e;font-size:1.05rem;'>An end-to-end machine learning platform covering Classification, Regression, and Clustering.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div style='background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;text-align:center'>
  <div style='font-size:2.5rem'>🔭</div>
  <h3 style='color:#58a6ff'>Classification</h3>
  <p style='color:#8b949e;font-size:0.88rem'>Star-type prediction using SDSS photometric data.<br>10 models · 3 classes</p>
  <hr style='border-color:#30363d'>
  <p style='color:#3fb950;font-size:1.2rem;font-weight:700'>≥ 99% Accuracy</p>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div style='background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;text-align:center'>
  <div style='font-size:2.5rem'>📈</div>
  <h3 style='color:#58a6ff'>Regression</h3>
  <p style='color:#8b949e;font-size:0.88rem'>Community crime-rate prediction from socioeconomic features.<br>9 models · continuous target</p>
  <hr style='border-color:#30363d'>
  <p style='color:#3fb950;font-size:1.2rem;font-weight:700'>R² up to 0.74</p>
</div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div style='background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;text-align:center'>
  <div style='font-size:2.5rem'>🔵</div>
  <h3 style='color:#58a6ff'>Clustering</h3>
  <p style='color:#8b949e;font-size:0.88rem'>Exploratory unsupervised clustering with interactive PCA visualisation.<br>K-Means · DBSCAN · Agglomerative</p>
  <hr style='border-color:#30363d'>
  <p style='color:#3fb950;font-size:1.2rem;font-weight:700'>Interactive</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Datasets at a Glance")
    d1, d2 = st.columns(2)
    with d1:
        try:
            df_cls = load_cls_data()
            st.markdown("**🔭 Star Classification (SDSS)**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Rows", f"{len(df_cls):,}")
            mc2.metric("Features", f"{df_cls.shape[1]-1}")
            mc3.metric("Classes", "3")
            counts = df_cls["class"].value_counts()
            fig = px.pie(
                names=counts.index, values=counts.values,
                color=counts.index,
                color_discrete_map=CLASS_COLORS,
                hole=0.55,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3", showlegend=True,
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True, key="home_cls_pie")
        except Exception as e:
            st.warning(f"Could not load classification data: {e}")

    with d2:
        try:
            df_reg, target_reg = load_reg_data()
            st.markdown(f"**📈 Communities & Crime**")
            mr1, mr2, mr3 = st.columns(3)
            mr1.metric("Rows", f"{len(df_reg):,}")
            mr2.metric("Features", f"{df_reg.shape[1]-1}")
            mr3.metric("Target", target_reg[:15])
            fig2 = px.histogram(
                df_reg, x=target_reg, nbins=40,
                color_discrete_sequence=["#388bfd"],
                labels={target_reg: "Crime Rate (normalised 0–1)"},
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3", bargap=0.05,
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
                yaxis=dict(gridcolor="#21262d"),
                xaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig2, use_container_width=True, key="home_reg_hist")
        except Exception as e:
            st.warning(f"Could not load regression data: {e}")

    st.markdown("---")
    st.markdown("### 🏆 Model Leaderboard — Classification")
    metrics_df = pd.DataFrame(CLS_METRICS).T.reset_index().rename(columns={"index": "Model"})
    metrics_df = metrics_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    metrics_df.index += 1
    metrics_df_display = metrics_df.copy()
    for col in ["Accuracy", "Weighted F1", "ROC-AUC"]:
        metrics_df_display[col] = metrics_df_display[col].map(lambda x: f"{x:.4f}")
    st.dataframe(
        metrics_df_display.style.apply(
            lambda s: ["background-color:#1a3a2a; color:#3fb950; font-weight:bold"
                       if i == 0 else "" for i in range(len(s))],
            axis=0
        ),
        use_container_width=True,
        hide_index=False,
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE: CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
elif page == "🔭  Classification":
    st.title("🔭 Star Classification")
    st.markdown("<p style='color:#8b949e'>Classify an astronomical observation as <b>Galaxy</b>, <b>QSO</b>, or <b>Star</b> using Sloan Digital Sky Survey (SDSS) photometric features.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # --- load artifacts ---
    try:
        cls_scaler, cls_encoder, cls_feat_cols, cls_bounds = load_cls_artifacts()
        cls_models = load_cls_models()
    except FileNotFoundError:
        st.error("⚠️ Artifacts not found. Please run `python save_artifacts.py` first.")
        st.stop()

    tab_pred, tab_compare, tab_info = st.tabs(["🎯 Predict", "📊 Model Comparison", "ℹ️ Feature Guide"])

    # ── TAB: PREDICT ──────────────────────────────────────────
    with tab_pred:
        st.markdown("### Select Model")
        chosen_model_name = st.selectbox(
            "Classifier", list(cls_models.keys()),
            help="Pick any of the 10 trained classifiers"
        )
        m = CLS_METRICS.get(chosen_model_name, {})
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Accuracy",   f"{m.get('Accuracy',0):.4f}")
        mc2.metric("Weighted F1", f"{m.get('Weighted F1',0):.4f}")
        mc3.metric("ROC-AUC",   f"{m.get('ROC-AUC',0):.4f}")

        st.markdown("---")
        st.markdown("### Enter Observation Features")
        st.markdown("<div class='info-banner'>💡 Hover the <b>?</b> icons for feature descriptions. Default values are dataset medians.</div>", unsafe_allow_html=True)

        # Load data to get medians for defaults
        df_cls = load_cls_data()
        num_df = df_cls.drop(columns=["class"])

        cols_per_row = 3
        feat_names = cls_feat_cols
        input_vals = {}
        rows = [feat_names[i:i+cols_per_row] for i in range(0, len(feat_names), cols_per_row)]

        for row_feats in rows:
            cols = st.columns(cols_per_row)
            for col, feat in zip(cols, row_feats):
                med_val = float(num_df[feat].median()) if feat in num_df else 0.0
                min_val = float(num_df[feat].min()) if feat in num_df else 0.0
                max_val = float(num_df[feat].max()) if feat in num_df else 1.0
                tooltip = FEATURE_TOOLTIPS.get(feat, feat)
                input_vals[feat] = col.number_input(
                    f"{feat}", value=med_val,
                    min_value=min_val - abs(min_val)*0.5,
                    max_value=max_val + abs(max_val)*0.5,
                    step=(max_val - min_val) / 200 if max_val != min_val else 0.01,
                    help=tooltip,
                    key=f"cls_{feat}"
                )

        st.markdown("")
        predict_btn = st.button("🚀 Predict Star Type", use_container_width=True)

        if predict_btn:
            model = cls_models[chosen_model_name]

            # preprocess
            raw = pd.DataFrame([input_vals])[feat_names]
            for feat, bounds in cls_bounds.items():
                if feat in raw.columns:
                    raw[feat] = raw[feat].clip(bounds["lower"], bounds["upper"])

            if chosen_model_name in CLS_NEEDS_SCALE:
                X_in = cls_scaler.transform(raw)
            else:
                X_in = raw.values

            pred_enc = model.predict(X_in)[0]
            pred_class = cls_encoder.inverse_transform([pred_enc])[0]

            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_in)[0]

            color = CLASS_COLORS.get(pred_class, "#58a6ff")
            icon  = CLASS_ICONS.get(pred_class, "🔭")

            st.markdown(f"""
<div class='pred-box' style='border-color:{color}'>
  <div class='pred-label'>Predicted Class</div>
  <div class='pred-value' style='color:{color}'>{icon} {pred_class}</div>
  <div class='pred-sub'>{CLASS_DESC.get(pred_class,"")}</div>
</div>""", unsafe_allow_html=True)

            if proba is not None:
                st.markdown("#### Confidence per Class")
                classes = cls_encoder.classes_
                prob_fig = go.Figure(go.Bar(
                    x=classes,
                    y=proba,
                    marker_color=[CLASS_COLORS.get(c, "#58a6ff") for c in classes],
                    text=[f"{p:.1%}" for p in proba],
                    textposition="outside",
                ))
                prob_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e6edf3", height=320,
                    yaxis=dict(range=[0, 1.15], gridcolor="#21262d", tickformat=".0%"),
                    xaxis=dict(gridcolor="#21262d"),
                    margin=dict(t=30, b=10),
                )
                st.plotly_chart(prob_fig, use_container_width=True, key="cls_proba")

    # ── TAB: MODEL COMPARISON ────────────────────────────────
    with tab_compare:
        st.markdown("### 📊 All 10 Models — Performance Comparison")
        metrics_df = pd.DataFrame(CLS_METRICS).T.reset_index().rename(columns={"index": "Model"})

        metric_choice = st.selectbox("Metric to highlight", ["Accuracy", "Weighted F1", "ROC-AUC"], key="cls_metric_choice")
        fig_bar = px.bar(
            metrics_df.sort_values(metric_choice, ascending=True),
            x=metric_choice, y="Model", orientation="h",
            color=metric_choice,
            color_continuous_scale=[[0, "#1a3a2a"], [0.5, "#238636"], [1, "#3fb950"]],
            text=metric_choice,
        )
        fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", height=430,
            xaxis=dict(range=[0.85, 1.01], gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="cls_bar")

        st.markdown("#### Radar Chart — All Metrics")
        categories = ["Accuracy", "Weighted F1", "ROC-AUC"]
        radar_fig = go.Figure()
        palette = px.colors.qualitative.Plotly
        for idx, (model_name, row) in enumerate(CLS_METRICS.items()):
            vals = [row[c] for c in categories]
            vals_closed = vals + [vals[0]]
            cats_closed  = categories + [categories[0]]
            radar_fig.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                name=model_name, fill="toself",
                line_color=palette[idx % len(palette)],
                opacity=0.6,
            ))
        radar_fig.update_layout(
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(visible=True, range=[0.85, 1.0], gridcolor="#30363d", color="#8b949e"),
                angularaxis=dict(gridcolor="#30363d", color="#8b949e"),
            ),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
            height=440, legend=dict(font_size=11),
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(radar_fig, use_container_width=True, key="cls_radar")

    # ── TAB: FEATURE GUIDE ───────────────────────────────────
    with tab_info:
        st.markdown("### ℹ️ Feature Reference")
        rows = [(f, d) for f, d in FEATURE_TOOLTIPS.items()]
        ftable = pd.DataFrame(rows, columns=["Feature", "Description"])
        st.dataframe(ftable, use_container_width=True, hide_index=True, height=520)

        st.markdown("---")
        st.markdown("### 🔭 Dataset Overview")
        df_cls = load_cls_data()
        a, b = st.columns(2)
        with a:
            st.markdown("**Class Distribution**")
            vc = df_cls["class"].value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            fig_pie = px.pie(vc, names="Class", values="Count",
                             color="Class", color_discrete_map=CLASS_COLORS, hole=0.5)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", height=300,
                                  margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True, key="cls_info_pie")
        with b:
            st.markdown("**Summary Statistics**")
            st.dataframe(df_cls.drop(columns=["class"]).describe().T.round(3), use_container_width=True, height=300)


# ═══════════════════════════════════════════════════════════════
#  PAGE: REGRESSION
# ═══════════════════════════════════════════════════════════════
elif page == "📈  Regression":
    st.title("📈 Community Crime Rate Predictor")
    st.markdown("<p style='color:#8b949e'>Predict the violent crime rate per capita for a US community using socioeconomic and demographic features (Communities & Crime dataset).</p>", unsafe_allow_html=True)
    st.markdown("---")

    # load
    try:
        reg_imputer, reg_scaler, reg_feat_cols, reg_bounds = load_reg_artifacts()
    except FileNotFoundError:
        st.error("⚠️ Artifacts not found. Please run `python save_artifacts.py` first.")
        st.stop()

    df_reg, target_reg = load_reg_data()

    REG_METRICS = {
        "Linear Regression":     {"R²": 0.6231, "RMSE": 0.1094, "MAE": 0.0795},
        "Ridge Regression":      {"R²": 0.6231, "RMSE": 0.1094, "MAE": 0.0795},
        "Lasso Regression":      {"R²": 0.6218, "RMSE": 0.1096, "MAE": 0.0795},
        "ElasticNet Regression": {"R²": 0.6108, "RMSE": 0.1108, "MAE": 0.0805},
        "Decision Tree":         {"R²": 0.5402, "RMSE": 0.1206, "MAE": 0.0839},
        "Random Forest":         {"R²": 0.7022, "RMSE": 0.0973, "MAE": 0.0678},
        "Gradient Boosting":     {"R²": 0.7143, "RMSE": 0.0954, "MAE": 0.0669},
        "SVR":                   {"R²": 0.6590, "RMSE": 0.1042, "MAE": 0.0718},
        "KNN":                   {"R²": 0.5871, "RMSE": 0.1144, "MAE": 0.0797},
    }

    # Models in the regression track that need scaled data
    REG_NEEDS_SCALE = {"Linear Regression", "Ridge Regression", "Lasso Regression",
                       "ElasticNet Regression", "SVR", "KNN"}

    tab_pred_r, tab_compare_r, tab_data_r = st.tabs(["🎯 Predict", "📊 Model Comparison", "📋 Dataset"])

    with tab_pred_r:
        st.markdown("### Select Model")
        reg_model_name = st.selectbox("Regressor", list(REG_METRICS.keys()), key="reg_model_select")

        m_r = REG_METRICS[reg_model_name]
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("R²",   f"{m_r['R²']:.4f}")
        rc2.metric("RMSE", f"{m_r['RMSE']:.4f}")
        rc3.metric("MAE",  f"{m_r['MAE']:.4f}")

        st.markdown("---")
        st.markdown("### Enter Community Features")
        st.markdown("<div class='info-banner'>💡 All features are normalised to [0, 1] in the original dataset. Sliders reflect actual data range.</div>", unsafe_allow_html=True)

        # top 20 most correlated features for a simpler UX
        feature_subset = [f for f in reg_feat_cols if f in df_reg.columns]
        corr_with_tgt = df_reg[feature_subset + [target_reg]].corr()[target_reg].drop(target_reg).abs()
        top_feats = corr_with_tgt.sort_values(ascending=False).head(20).index.tolist()

        st.markdown(f"*Showing top 20 most predictive features (of {len(reg_feat_cols)} total). Remaining features use dataset medians.*")

        input_r = {}
        cols_r = st.columns(2)
        for idx, feat in enumerate(top_feats):
            col = cols_r[idx % 2]
            med_v = float(df_reg[feat].median())
            min_v = float(df_reg[feat].min())
            max_v = float(df_reg[feat].max())
            step_v = round((max_v - min_v) / 100, 4) if max_v != min_v else 0.01
            input_r[feat] = col.slider(
                feat, min_value=min_v, max_value=max_v,
                value=med_v, step=step_v, key=f"reg_{feat}"
            )

        predict_r_btn = st.button("🚀 Predict Crime Rate", use_container_width=True)

        if predict_r_btn:
            # build full feature vector (missing → median)
            full_input = {}
            for feat in reg_feat_cols:
                if feat in input_r:
                    full_input[feat] = input_r[feat]
                elif feat in df_reg.columns:
                    full_input[feat] = float(df_reg[feat].median())
                else:
                    full_input[feat] = 0.0

            raw_r = pd.DataFrame([full_input])[reg_feat_cols]

            # clip
            for feat in reg_feat_cols:
                if feat in reg_bounds["lower"].index:
                    raw_r[feat] = raw_r[feat].clip(reg_bounds["lower"][feat], reg_bounds["upper"][feat])

            # impute
            raw_r_imp = pd.DataFrame(
                reg_imputer.transform(raw_r), columns=reg_feat_cols
            )

            # scale if needed
            if reg_model_name in REG_NEEDS_SCALE:
                X_r_in = reg_scaler.transform(raw_r_imp)
            else:
                X_r_in = raw_r_imp.values

            # predict using saved model if available else show demo
            try:
                import glob
                # try to load any regression model pkl matching name
                name_map = {
                    "Linear Regression":     "linear",
                    "Ridge Regression":      "ridge",
                    "Lasso Regression":      "lasso",
                    "ElasticNet Regression": "elasticnet",
                    "Decision Tree":         "decision_tree",
                    "Random Forest":         "random_forest",
                    "Gradient Boosting":     "gradient_boosting",
                    "SVR":                   "svr",
                    "KNN":                   "knn",
                }
                key = name_map.get(reg_model_name, "").lower()
                pkl_files = glob.glob(os.path.join(REG_MODELS, f"*{key}*.pkl"))
                # filter out artifact files
                pkl_files = [p for p in pkl_files if not any(x in p for x in ["scaler", "imputer", "feature", "clip"])]
                if pkl_files:
                    reg_model_obj = joblib.load(pkl_files[0])
                    pred_val = float(reg_model_obj.predict(X_r_in)[0])
                else:
                    raise FileNotFoundError
            except Exception:
                # fall back: estimate from model metrics
                pred_val = float(df_reg[target_reg].mean())
                st.info("ℹ️ Model .pkl not found for regression — showing dataset mean as demo. Re-run the regression notebook and save models to enable live predictions.")

            pred_pct = pred_val * 100
            level = "🟢 Low" if pred_val < 0.25 else ("🟡 Moderate" if pred_val < 0.5 else "🔴 High")

            st.markdown(f"""
<div class='pred-box'>
  <div class='pred-label'>Predicted Violent Crimes Per Population</div>
  <div class='pred-value'>{pred_val:.4f}</div>
  <div class='pred-sub'>≈ {pred_pct:.2f}% · Risk Level: {level}</div>
</div>""", unsafe_allow_html=True)

            # gauge chart
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_val,
                number={"valueformat": ".4f", "font": {"color": "#58a6ff", "size": 36}},
                gauge={
                    "axis": {"range": [0, 1], "tickcolor": "#8b949e"},
                    "bar": {"color": "#388bfd"},
                    "steps": [
                        {"range": [0, 0.25],  "color": "#1a3a2a"},
                        {"range": [0.25, 0.5], "color": "#3a2a10"},
                        {"range": [0.5, 1.0],  "color": "#3a1010"},
                    ],
                    "threshold": {"line": {"color": "#f78166", "width": 3}, "thickness": 0.75, "value": pred_val},
                    "bgcolor": "#161b22",
                    "bordercolor": "#30363d",
                },
            ))
            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                height=280, margin=dict(t=20, b=20),
            )
            st.plotly_chart(gauge, use_container_width=True, key="reg_gauge")

    with tab_compare_r:
        st.markdown("### 📊 Regression Model Comparison")
        reg_m_df = pd.DataFrame(REG_METRICS).T.reset_index().rename(columns={"index": "Model"})

        reg_metric_c = st.selectbox("Sort by metric", ["R²", "RMSE", "MAE"], key="reg_metric_c")
        asc = reg_metric_c in ["RMSE", "MAE"]
        fig_reg_bar = px.bar(
            reg_m_df.sort_values(reg_metric_c, ascending=asc),
            x=reg_metric_c, y="Model", orientation="h",
            color=reg_metric_c,
            color_continuous_scale=[[0, "#1a3a2a"], [0.5, "#238636"], [1, "#3fb950"]] if not asc
                                   else [[0, "#3fb950"], [0.5, "#238636"], [1, "#1a3a2a"]],
            text=reg_metric_c,
        )
        fig_reg_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_reg_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", height=400,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_reg_bar, use_container_width=True, key="reg_bar")

        st.markdown("#### Full Metrics Table")
        st.dataframe(reg_m_df.sort_values("R²", ascending=False).set_index("Model").round(4),
                     use_container_width=True)

    with tab_data_r:
        st.markdown("### 📋 Regression Dataset Explorer")
        st.markdown(f"**Target:** `{target_reg}` — violent crimes per 100K population (normalised 0–1)")
        r1, r2, r3 = st.columns(3)
        r1.metric("Rows", f"{len(df_reg):,}")
        r2.metric("Features (after cleaning)", f"{df_reg.shape[1]-1}")
        r3.metric("Missing Values", "0 (imputed)")

        st.markdown("**Top 10 Features Correlated with Crime Rate**")
        feat_cols_reg_valid = [f for f in reg_feat_cols if f in df_reg.columns]
        corr = df_reg[feat_cols_reg_valid + [target_reg]].corr()[target_reg].drop(target_reg)
        top_corr = corr.abs().sort_values(ascending=False).head(10)

        fig_corr = px.bar(
            x=corr.loc[top_corr.index].values,
            y=top_corr.index,
            orientation="h",
            color=corr.loc[top_corr.index].values,
            color_continuous_scale="RdBu_r",
            labels={"x": f"Correlation with {target_reg}", "y": "Feature"},
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", height=380,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_corr, use_container_width=True, key="reg_corr_bar")


# ═══════════════════════════════════════════════════════════════
#  PAGE: CLUSTERING
# ═══════════════════════════════════════════════════════════════
elif page == "🔵  Clustering":
    st.title("🔵 Interactive Clustering")
    st.markdown("<p style='color:#8b949e'>Run unsupervised clustering on either dataset and explore the results interactively with PCA visualisation, silhouette scores, and elbow plots.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar-style controls inside main area
    c_left, c_right = st.columns([1, 2])

    with c_left:
        st.markdown("#### ⚙️ Configuration")
        dataset_choice = st.selectbox("Dataset", ["Star Classification (SDSS)", "Community Crime"], key="clust_dataset")
        algo_choice    = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Agglomerative"], key="clust_algo")

        if algo_choice == "K-Means":
            k_val       = st.slider("Number of clusters (k)", 2, 15, 3, key="clust_k")
            init_method = st.selectbox("Init method", ["k-means++", "random"], key="clust_init")
            max_iter    = st.slider("Max iterations", 100, 500, 300, step=50, key="clust_maxiter")

        elif algo_choice == "DBSCAN":
            eps_val    = st.slider("ε (epsilon) — neighbourhood radius", 0.1, 5.0, 0.5, step=0.05, key="clust_eps")
            min_samp   = st.slider("min_samples", 2, 30, 5, key="clust_minsamp")
            st.markdown("<div class='info-banner'>Points not assigned to any cluster are labelled <b>Noise (-1)</b>.</div>", unsafe_allow_html=True)

        elif algo_choice == "Agglomerative":
            agg_k      = st.slider("Number of clusters", 2, 15, 3, key="clust_agg_k")
            linkage_m  = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="clust_link")

        st.markdown("---")
        n_pca_comp = st.slider("PCA components for visualisation", 2, 3, 2, key="clust_pca")
        sample_frac = st.slider("Data sample fraction (for speed)", 0.05, 1.0, 0.30, step=0.05, key="clust_frac")
        run_btn = st.button("▶ Run Clustering", use_container_width=True)

        st.markdown("---")
        st.markdown("**Show elbow / silhouette scan**")
        show_elbow = st.checkbox("Elbow curve (K-Means range 2–12)", value=False, key="clust_elbow")

    with c_right:
        if run_btn or show_elbow:
            with st.spinner("Loading & preprocessing data…"):
                if "Star" in dataset_choice:
                    df_raw = load_cls_data()
                    X_raw = df_raw.drop(columns=["class"], errors="ignore")
                    y_true_labels = df_raw["class"].values if "class" in df_raw.columns else None
                else:
                    df_raw, tgt_r = load_reg_data()
                    y_true_labels = None
                    X_raw = df_raw.drop(columns=[tgt_r], errors="ignore")

                # sample
                X_raw = X_raw.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
                if y_true_labels is not None and "Star" in dataset_choice:
                    y_true_labels = load_cls_data()["class"].sample(frac=sample_frac, random_state=42).values

                # drop non-numeric, fill NaN
                X_num = X_raw.select_dtypes(include=[np.number]).fillna(X_raw.select_dtypes(include=[np.number]).median())

                # scale
                sc = SkScaler()
                X_scaled = sc.fit_transform(X_num)

                # PCA for visualisation
                pca = PCA(n_components=n_pca_comp, random_state=42)
                X_pca = pca.fit_transform(X_scaled)

            if show_elbow and algo_choice == "K-Means":
                with st.spinner("Computing elbow curve…"):
                    inertias, sil_scores = [], []
                    k_range = range(2, 13)
                    for k in k_range:
                        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=42)
                        labels_e = km.fit_predict(X_scaled)
                        inertias.append(km.inertia_)
                        sil_scores.append(silhouette_score(X_scaled, labels_e, sample_size=min(1000, len(X_scaled))))

                    fig_el = make_subplots(rows=1, cols=2,
                                           subplot_titles=["Inertia (Elbow)", "Silhouette Score"])
                    fig_el.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers",
                                                 marker_color="#388bfd", name="Inertia"), row=1, col=1)
                    fig_el.add_trace(go.Scatter(x=list(k_range), y=sil_scores, mode="lines+markers",
                                                 marker_color="#3fb950", name="Silhouette"), row=1, col=2)
                    fig_el.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font_color="#e6edf3", height=340,
                                          xaxis=dict(gridcolor="#21262d", title="k"),
                                          xaxis2=dict(gridcolor="#21262d", title="k"),
                                          yaxis=dict(gridcolor="#21262d"),
                                          yaxis2=dict(gridcolor="#21262d"),
                                          margin=dict(t=40, b=20), showlegend=False)
                    st.plotly_chart(fig_el, use_container_width=True, key="clust_elbow_fig")

            if run_btn:
                with st.spinner(f"Running {algo_choice}…"):
                    if algo_choice == "K-Means":
                        model_c = KMeans(n_clusters=k_val, init=init_method, max_iter=max_iter,
                                         n_init=10, random_state=42)
                        labels_c = model_c.fit_predict(X_scaled)

                    elif algo_choice == "DBSCAN":
                        model_c = DBSCAN(eps=eps_val, min_samples=min_samp, n_jobs=-1)
                        labels_c = model_c.fit_predict(X_scaled)

                    elif algo_choice == "Agglomerative":
                        model_c = AgglomerativeClustering(n_clusters=agg_k, linkage=linkage_m)
                        labels_c = model_c.fit_predict(X_scaled)

                n_clusters_found = len(set(labels_c)) - (1 if -1 in labels_c else 0)
                n_noise          = (labels_c == -1).sum()

                # metrics
                try:
                    sil  = silhouette_score(X_scaled, labels_c, sample_size=min(2000, len(X_scaled))) if n_clusters_found > 1 else float("nan")
                    db   = davies_bouldin_score(X_scaled, labels_c) if n_clusters_found > 1 else float("nan")
                except Exception:
                    sil, db = float("nan"), float("nan")

                km1, km2, km3, km4 = st.columns(4)
                km1.metric("Clusters Found",  str(n_clusters_found))
                km2.metric("Noise Points",    str(n_noise))
                km3.metric("Silhouette ↑",    f"{sil:.4f}" if not np.isnan(sil) else "N/A",
                            help="Range [−1, 1]; higher is better")
                km4.metric("Davies-Bouldin ↓", f"{db:.4f}" if not np.isnan(db) else "N/A",
                            help="Lower is better")

                st.markdown("---")
                # ── PCA scatter ──────────────────────────────
                labels_str = [str(l) for l in labels_c]
                if n_pca_comp == 2:
                    scatter_df = pd.DataFrame({
                        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                        "Cluster": labels_str,
                    })
                    if y_true_labels is not None and "Star" in dataset_choice:
                        scatter_df["True Class"] = y_true_labels
                        hover = ["True Class"]
                    else:
                        hover = None

                    fig_sc = px.scatter(
                        scatter_df, x="PC1", y="PC2", color="Cluster",
                        hover_data=hover,
                        title=f"{algo_choice} Clusters (PCA 2D) — {n_clusters_found} clusters",
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                    )
                else:
                    scatter_df = pd.DataFrame({
                        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "PC3": X_pca[:, 2],
                        "Cluster": labels_str,
                    })
                    fig_sc = px.scatter_3d(
                        scatter_df, x="PC1", y="PC2", z="PC3", color="Cluster",
                        title=f"{algo_choice} Clusters (PCA 3D) — {n_clusters_found} clusters",
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        opacity=0.7,
                    )

                fig_sc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e6edf3", height=520,
                    legend_title_text="Cluster",
                    margin=dict(t=50, b=20),
                )
                st.plotly_chart(fig_sc, use_container_width=True, key="clust_scatter")

                # ── Cluster size bar ─────────────────────────
                st.markdown("#### Cluster Size Distribution")
                clust_sizes = pd.Series(labels_c).value_counts().sort_index().reset_index()
                clust_sizes.columns = ["Cluster", "Count"]
                clust_sizes["Cluster"] = clust_sizes["Cluster"].astype(str)
                fig_sz = px.bar(clust_sizes, x="Cluster", y="Count",
                                color="Cluster",
                                color_discrete_sequence=px.colors.qualitative.Plotly,
                                text="Count")
                fig_sz.update_traces(textposition="outside")
                fig_sz.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e6edf3", height=320,
                    showlegend=False,
                    xaxis=dict(gridcolor="#21262d"),
                    yaxis=dict(gridcolor="#21262d"),
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig_sz, use_container_width=True, key="clust_size_bar")

                # PCA explained variance
                exp_var = pca.explained_variance_ratio_ * 100
                st.markdown(f"**PCA Explained Variance:** {' + '.join([f'PC{i+1}: {v:.1f}%' for i,v in enumerate(exp_var)])} = **{sum(exp_var):.1f}%** total")

        else:
            st.markdown("""
<div style='background:#161b22;border:1px dashed #30363d;border-radius:14px;
            padding:60px 40px;text-align:center;color:#8b949e'>
  <div style='font-size:3rem'>🔵</div>
  <h3 style='color:#58a6ff'>Configure & Run Clustering</h3>
  <p>Choose a dataset, algorithm, and parameters on the left,<br>then click <b>▶ Run Clustering</b>.</p>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: EDA EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == "📊  EDA Explorer":
    st.title("📊 EDA Explorer")
    st.markdown("<p style='color:#8b949e'>Interactively explore both datasets — distributions, correlations, scatter plots, and summary statistics.</p>", unsafe_allow_html=True)
    st.markdown("---")

    dataset_eda = st.radio("Select Dataset", ["🔭 Star Classification", "📈 Community Crime"], horizontal=True, key="eda_ds")

    if "Star" in dataset_eda:
        df_e = load_cls_data()
        num_cols = df_e.select_dtypes(include=[np.number]).columns.tolist()
        cat_col  = "class"
        color_map = CLASS_COLORS
        has_target_cat = True
        target_e = None
    else:
        df_e, target_e = load_reg_data()
        num_cols = df_e.select_dtypes(include=[np.number]).columns.tolist()
        cat_col  = None
        color_map = None
        has_target_cat = False

    tab_dist, tab_corr, tab_scatter, tab_raw = st.tabs(["📉 Distributions", "🌡️ Correlation", "🔀 Scatter", "🗃️ Raw Data"])

    with tab_dist:
        st.markdown("### Feature Distributions")
        feat_sel = st.multiselect("Select features to plot", num_cols,
                                  default=num_cols[:min(6, len(num_cols))], key="eda_dist_feats")
        if feat_sel:
            n_col_d = 3
            rows_d  = [feat_sel[i:i+n_col_d] for i in range(0, len(feat_sel), n_col_d)]
            for row_f in rows_d:
                c_d = st.columns(n_col_d)
                for col_d, feat in zip(c_d, row_f):
                    with col_d:
                        fig_d = px.histogram(df_e, x=feat, nbins=40,
                                             color=cat_col if has_target_cat else None,
                                             color_discrete_map=color_map,
                                             marginal="box",
                                             title=feat)
                        fig_d.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e6edf3", height=300,
                            margin=dict(t=30, b=10, l=10, r=10),
                            showlegend=False,
                            xaxis=dict(gridcolor="#21262d"),
                            yaxis=dict(gridcolor="#21262d"),
                        )
                        st.plotly_chart(fig_d, use_container_width=True, key=f"eda_dist_{feat}")

    with tab_corr:
        st.markdown("### Correlation Heatmap")
        max_feats = st.slider("Max features in heatmap", 5, min(40, len(num_cols)), 15, key="eda_corr_n")

        # sort by variance for most-informative subset
        top_var_feats = df_e[num_cols].var().sort_values(ascending=False).head(max_feats).index.tolist()
        corr_m = df_e[top_var_feats].corr()

        fig_hm = px.imshow(
            corr_m, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", height=600,
            coloraxis_colorbar=dict(tickcolor="#e6edf3", title="r"),
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True, key="eda_heatmap")

    with tab_scatter:
        st.markdown("### Feature Scatter Plot")
        sc1, sc2 = st.columns(2)
        x_feat = sc1.selectbox("X axis", num_cols, index=0, key="eda_sc_x")
        y_feat = sc2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="eda_sc_y")
        sample_n = st.slider("Sample size", 500, min(10000, len(df_e)), 3000, step=500, key="eda_sc_n")

        df_sample = df_e.sample(n=min(sample_n, len(df_e)), random_state=42)
        fig_sc2 = px.scatter(
            df_sample, x=x_feat, y=y_feat,
            color=cat_col if has_target_cat else target_e,
            color_discrete_map=color_map,
            color_continuous_scale="Viridis" if not has_target_cat else None,
            opacity=0.6,
            title=f"{x_feat} vs {y_feat}",
            hover_data=[cat_col] if has_target_cat else [target_e],
        )
        fig_sc2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", height=500,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_sc2, use_container_width=True, key="eda_scatter2")

    with tab_raw:
        st.markdown("### Raw Dataset Preview")
        n_rows = st.slider("Rows to display", 10, 200, 50, step=10, key="eda_raw_n")
        search_col = st.selectbox("Filter column", [None] + df_e.columns.tolist(), key="eda_raw_search")
        df_disp = df_e.head(n_rows)
        if search_col:
            st.dataframe(df_disp[[search_col] + [c for c in df_disp.columns if c != search_col]],
                         use_container_width=True, height=440)
        else:
            st.dataframe(df_disp, use_container_width=True, height=440)

        st.markdown("**Summary Statistics**")
        st.dataframe(df_e.describe().T.round(4), use_container_width=True)
