"""
Vectora – Unified ML Dashboard (Streamlit)
==========================================
Covers:
  • Classification  – Star Type (Galaxy / QSO / Star)
  • Regression      – Community Crime Rate
  • Clustering      – K-Means / DBSCAN / Agglomerative (interactive)
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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  :root { --espresso:#17130F; --panel:#211B15; --panel-raised:#2B2118; --border:#493B2D; --text:#F4EEE4; --muted:#B7A895; --gold:#D6AD55; --gold-soft:#E6C879; }
  .stApp { background:var(--espresso); color:var(--text); }
  [data-testid="stHeader"] { background:rgba(23,19,15,.96); }
  [data-testid="stSidebar"] { background:linear-gradient(180deg,#211B15 0%,#17130F 100%); border-right:1px solid var(--border); }
  [data-testid="stSidebar"] .stMarkdown h1,[data-testid="stSidebar"] .stMarkdown h2,[data-testid="stSidebar"] .stMarkdown h3 { color:var(--gold); }
  h1,h2,h3 { color:var(--gold-soft) !important; letter-spacing:-.02em; }
  h1 { font-weight:750 !important; }
  p,li,label,[data-testid="stMarkdownContainer"] { color:inherit; }
  [data-testid="metric-container"] { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:14px 16px; box-shadow:0 3px 12px rgba(0,0,0,.12); }
  [data-testid="metric-container"] label { color:var(--muted) !important; font-size:.8rem; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color:var(--gold) !important; font-size:1.55rem; font-weight:700; }
  .stButton > button { background:linear-gradient(135deg,#8C6A32,#A9843A); color:#FFF9EE; border:1px solid #B78A3F; border-radius:8px; font-weight:650; padding:.52rem 1.2rem; transition:all .15s ease; }
  .stButton > button:hover { background:#B78A3F; border-color:#E6C879; color:#17130F; }
  .stButton > button:focus { box-shadow:0 0 0 .15rem rgba(214,173,85,.3); }
  .stTabs [data-baseweb="tab-list"] { background:var(--panel); border:1px solid var(--border); border-radius:9px; padding:4px; gap:4px; }
  .stTabs [data-baseweb="tab"] { color:var(--muted); font-weight:600; border-radius:6px; }
  .stTabs [aria-selected="true"] { color:var(--gold-soft) !important; border-bottom:2px solid var(--gold) !important; background:#30251A; }
  .stDataFrame,[data-testid="stTable"] { border:1px solid var(--border); border-radius:9px; overflow:hidden; }
  .stSelectbox label,.stSlider label,.stNumberInput label,.stTextInput label { color:var(--muted) !important; }
  [data-baseweb="select"] > div,[data-baseweb="input"] > div,[data-testid="stNumberInput"] input { background:var(--panel) !important; border-color:var(--border) !important; border-radius:7px !important; }
  .pred-box { background:linear-gradient(135deg,#30251A 0%,#211B15 100%); border:1px solid var(--gold); border-radius:12px; padding:22px 24px; margin:12px 0; text-align:center; }
  .pred-label { font-size:.82rem; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; margin-bottom:6px; }
  .pred-value { font-size:2.2rem; font-weight:750; color:var(--gold-soft); letter-spacing:.01em; }
  .pred-sub { font-size:.92rem; color:var(--muted); margin-top:4px; }
  .info-banner { background:#2B2118; border:1px solid var(--border); border-left:4px solid var(--gold); border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0; font-size:.9rem; color:#E6D8C3; }
  hr { border-color:var(--border); }
  .scroll-box { max-height:340px; overflow-y:auto; }
  .stRadio > div { flex-direction:row; gap:16px; }
  .stProgress > div > div { background:linear-gradient(90deg,#8C6A32,#D6AD55); }
  [data-testid="stExpander"] { border:1px solid var(--border); border-radius:9px; background:var(--panel); }
  [data-testid="stAlert"] { border-radius:8px; }
  a { color:var(--gold-soft) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PATHS & MODEL LOADING
# ─────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
CLS_DIR     = os.path.join(BASE, "ML_CLASSIFICATION_TRACKA")
REG_DIR     = os.path.join(BASE, "Regression Track A")
CLUST_DIR   = os.path.join(BASE, "Clustering")
CLS_MODELS  = os.path.join(CLS_DIR,   "models")
REG_MODELS  = os.path.join(REG_DIR,   "models")
CLUST_MODELS = os.path.join(CLUST_DIR, "models")


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


@st.cache_resource
def load_cluster_artifacts():
    import json, sys
    # clustering_preprocessing.py lives at project root; ensure it's importable
    # so joblib can unpickle the FunctionTransformer inside the preprocessor pipeline.
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    preprocessor = joblib.load(os.path.join(CLUST_MODELS, "cluster_preprocessor.pkl"))
    kmeans       = joblib.load(os.path.join(CLUST_MODELS, "kmeans_k4.pkl"))
    centroids    = pd.read_csv(os.path.join(CLUST_MODELS, "kmeans_k4_centroids.csv"), index_col=0)
    labels_df    = pd.read_csv(os.path.join(CLUST_MODELS, "kmeans_k4_labels.csv"), index_col=0)
    with open(os.path.join(CLUST_MODELS, "cluster_meta.json")) as f:
        meta = json.load(f)
    return preprocessor, kmeans, centroids, labels_df, meta


@st.cache_data
def load_cc_data():
    csv_path = os.path.join(CLUST_DIR, "CC GENERAL.csv")
    df = pd.read_csv(csv_path)
    if "CUST_ID" in df.columns:
        df = df.drop(columns=["CUST_ID"])
    return df


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


@st.cache_data
def load_cls_metrics():
    """Load the recorded evaluation leaderboard exported by the classification notebook."""
    metrics_path = os.path.join(CLS_DIR, "classification_metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            "Classification metrics not found. Run classification.ipynb and export "
            "classification_metrics.csv from its model-comparison cell."
        )

    metrics = pd.read_csv(metrics_path)
    required = {"Model", "Accuracy", "Weighted F1", "ROC-AUC (OvR)"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            "Classification metrics file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Match notebook model names to the labels used in the GUI.
    metrics["Model"] = metrics["Model"].replace({"KNN": "K-Nearest Neighbors"})
    metrics = metrics.rename(columns={"ROC-AUC (OvR)": "ROC-AUC"})
    metrics = metrics.drop_duplicates(subset=["Model"], keep="last")
    # Keep the GUI's existing three-metric layout; precision and recall remain
    # available in the exported notebook CSV for deeper evaluation.
    return metrics[["Model", "Accuracy", "Weighted F1", "ROC-AUC"]]

CLS_CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_COLORS = {"GALAXY": "#B78A3F", "QSO": "#C77D52", "STAR": "#C6A04A"}
CLASS_ICONS  = {"GALAXY": "", "QSO": "", "STAR": "⭐"}
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
    st.markdown("##  Vectora")
    st.markdown("<p style='color:#B7A895;font-size:0.82rem;'>Unified ML Dashboard</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── nav bar buttons ──────────────────────────────────
    st.markdown("""
<style>
div[data-testid="stSidebar"] button {
    width: 100% !important;
    text-align: left !important;
    background: #211B15 !important;
    color: #F4EEE4 !important;
    border: 1px solid #493B2D !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    transition: background 0.15s, border-color 0.15s !important;
}
div[data-testid="stSidebar"] button:hover {
    background: #30261B !important;
    border-color: #D6AD55 !important;
    color: #E6C879 !important;
}
div[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg,#4A3824,#614A2C) !important;
    border-color: #D6AD55 !important;
    color: #E6C879 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    nav_items = [
        ("Home",           "Home"),
        ("Classification", "Classification"),
        ("Regression",     "Regression"),
        ("Clustering",     "Clustering"),
    ]
    for label, key in nav_items:
        kind = "primary" if st.session_state["page"] == key else "secondary"
        if st.button(label, key=f"nav_{key}", type=kind, use_container_width=True):
            st.session_state["page"] = key
            st.rerun()

    page = st.session_state["page"]

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.75rem;color:#B7A895;line-height:1.8'>
<b style='color:#D6AD55'>Tracks</b><br>
<span style='color:#C6A04A'></span> Star Classification (SDSS)<br>
<span style='color:#C6A04A'></span> Community Crime Regression<br>
<span style='color:#C6A04A'></span> Credit Card Clustering (k=4)<br>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════
if page == "Home":
    st.title(" Vectora — ML Dashboard")
    st.markdown("<p style='color:#B7A895;font-size:1.05rem;'>An end-to-end machine learning platform covering Classification, Regression, and Clustering.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    try:
        home_cls_metrics = load_cls_metrics()
        best_cls_accuracy = float(home_cls_metrics["Accuracy"].max())
        cls_accuracy_label = f"Best recorded accuracy: {best_cls_accuracy:.2%}"
    except (FileNotFoundError, ValueError, KeyError):
        cls_accuracy_label = "Notebook metrics unavailable"
    with col1:
        st.markdown(f"""
<div style='background:#211B15;border:1px solid #493B2D;border-radius:14px;padding:20px;text-align:center'>
  <div style='font-size:2.5rem'></div>
  <h3 style='color:#D6AD55'>Classification</h3>
  <p style='color:#B7A895;font-size:0.88rem'>Star-type prediction using SDSS photometric data.<br>10 models · 3 classes</p>
  <hr style='border-color:#493B2D'>
  <p style='color:#C6A04A;font-size:1.05rem;font-weight:700'>{cls_accuracy_label}</p>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div style='background:#211B15;border:1px solid #493B2D;border-radius:14px;padding:20px;text-align:center'>
  <div style='font-size:2.5rem'></div>
  <h3 style='color:#D6AD55'>Regression</h3>
  <p style='color:#B7A895;font-size:0.88rem'>Community crime-rate prediction from socioeconomic features.<br>11 models · continuous target</p>
  <hr style='border-color:#493B2D'>
  <p style='color:#C6A04A;font-size:1.2rem;font-weight:700'>Best R²: 0.6895</p>
</div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div style='background:#211B15;border:1px solid #493B2D;border-radius:14px;padding:20px;text-align:center'>
  <div style='font-size:2.5rem'></div>
  <h3 style='color:#D6AD55'>Clustering</h3>
  <p style='color:#B7A895;font-size:0.88rem'>Exploratory unsupervised clustering with interactive PCA visualisation.<br>K-Means · DBSCAN · Agglomerative</p>
  <hr style='border-color:#493B2D'>
  <p style='color:#C6A04A;font-size:1.2rem;font-weight:700'>Interactive</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("###  Datasets at a Glance")
    d1, d2, d3 = st.columns(3)
    with d1:
        try:
            df_cls = load_cls_data()
            st.markdown("Star Classification (SDSS)")
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
                font_color="#F4EEE4", showlegend=True,
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True, key="home_cls_pie")
        except Exception as e:
            st.warning(f"Could not load classification data: {e}")

    with d2:
        try:
            df_reg, target_reg = load_reg_data()
            st.markdown(f"Communities & Crime")
            mr1, mr2, mr3 = st.columns(3)
            mr1.metric("Rows", f"{len(df_reg):,}")
            mr2.metric("Features", f"{df_reg.shape[1]-1}")
            mr3.metric("Target", target_reg[:15])
            fig2 = px.histogram(
                df_reg, x=target_reg, nbins=40,
                color_discrete_sequence=["#B78A3F"],
                labels={target_reg: "Crime Rate (normalised 0–1)"},
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F4EEE4", bargap=0.05,
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
                yaxis=dict(gridcolor="#30261B"),
                xaxis=dict(gridcolor="#30261B"),
            )
            st.plotly_chart(fig2, use_container_width=True, key="home_reg_hist")
        except Exception as e:
            st.warning(f"Could not load regression data: {e}")

    
    with d3:
        try:
            df_cluster = load_cc_data()

            st.markdown("Credit Card Clustering")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Rows", f"{len(df_cluster):,}")
            mc2.metric("Features", f"{df_cluster.shape[1]}")
            mc3.metric("Clusters (k)", "4")

            st.caption("Customer segmentation using K-Means clustering")

        # Cluster evaluation metrics from saved metadata
            try:
                with open(CLUST_DIR / "models" / "cluster_meta.json", "r") as f:
                    cluster_meta = json.load(f)

                st.metric(
                "Silhouette Score",
                f"{cluster_meta['metrics']['Silhouette']:.4f}"
                 )
                st.metric(
                "Davies-Bouldin Index",
                f"{cluster_meta['metrics']['Davies-Bouldin']:.4f}"
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                st.caption("Saved clustering metrics are unavailable.")

        # Feature distribution
            feature = (
            "BALANCE"
            if "BALANCE" in df_cluster.columns
            else df_cluster.select_dtypes(include="number").columns[0]
        )

            fig3 = px.histogram(
            df_cluster,
            x=feature,
            nbins=40,
            color_discrete_sequence=["#C9A24D"],
            labels={feature: feature.replace("_", " ").title()},
        )

            fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F4EEE4",
            bargap=0.05,
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
            yaxis=dict(gridcolor="#30261B"),
            xaxis=dict(gridcolor="#30261B"),
        )

            st.plotly_chart(
            fig3,
            use_container_width=True,
            key="home_cluster_hist",
        )

        except Exception as e:
            st.warning(f"Could not load clustering data: {e}")
    st.markdown("---")


# ═══════════════════════════════════════════════════════════════
#  PAGE: CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
elif page == "Classification":
    st.title(" Star Classification")
    st.markdown("<p style='color:#B7A895'>Classify an astronomical observation as <b>Galaxy</b>, <b>QSO</b>, or <b>Star</b> using Sloan Digital Sky Survey (SDSS) photometric features.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # --- load artifacts ---
    try:
        cls_scaler, cls_encoder, cls_feat_cols, cls_bounds = load_cls_artifacts()
        cls_models = load_cls_models()
    except FileNotFoundError:
        st.error(" Artifacts not found. Please run `python save_artifacts.py` first.")
        st.stop()

    tab_pred, tab_compare, tab_info = st.tabs([" Predict", " Model Comparison", "Feature Guide"])
    try:
        cls_metrics_df = load_cls_metrics()
        cls_metrics_by_model = cls_metrics_df.set_index("Model").to_dict(orient="index")
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Could not load classification evaluation results: {e}")
        st.stop()

    # ── TAB: PREDICT ──────────────────────────────────────────
    with tab_pred:
        st.markdown("### Select Model")
        chosen_model_name = st.selectbox(
            "Classifier", list(cls_models.keys()),
            help="Pick any of the 10 trained classifiers"
        )
        m = cls_metrics_by_model.get(chosen_model_name, {})
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Accuracy",   f"{m.get('Accuracy',0):.4f}")
        mc2.metric("Weighted F1", f"{m.get('Weighted F1',0):.4f}")
        mc3.metric("ROC-AUC",   f"{m.get('ROC-AUC',0):.4f}")

        st.markdown("---")
        st.markdown("### Enter Observation Features")
        st.markdown("<div class='info-banner'> Hover the <b>?</b> icons for feature descriptions. Default values are dataset medians.</div>", unsafe_allow_html=True)

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
        predict_btn = st.button("Predict Star Type", use_container_width=True)

        if predict_btn:
            model = cls_models[chosen_model_name]

            # preprocess
            raw = pd.DataFrame([input_vals])[feat_names]
            for feat, bounds in cls_bounds.items():
                if feat in raw.columns:
                    raw[feat] = raw[feat].clip(bounds["lower"], bounds["upper"])

            # The notebook fits every saved classifier on X_train_scaled, including
            # tree-based and ensemble models. Use the same scaler for every model.
            X_in = cls_scaler.transform(raw)

            pred_enc = model.predict(X_in)[0]
            pred_class = cls_encoder.inverse_transform([pred_enc])[0]

            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_in)[0]

            color = CLASS_COLORS.get(pred_class, "#D6AD55")
            icon  = CLASS_ICONS.get(pred_class, "")

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
                    marker_color=[CLASS_COLORS.get(c, "#D6AD55") for c in classes],
                    text=[f"{p:.1%}" for p in proba],
                    textposition="outside",
                ))
                prob_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F4EEE4", height=320,
                    yaxis=dict(range=[0, 1.15], gridcolor="#30261B", tickformat=".0%"),
                    xaxis=dict(gridcolor="#30261B"),
                    margin=dict(t=30, b=10),
                )
                st.plotly_chart(prob_fig, use_container_width=True, key="cls_proba")

    # ── TAB: MODEL COMPARISON ────────────────────────────────
    with tab_compare:
        st.markdown("###  All 10 Models — Performance Comparison")
        metrics_df = cls_metrics_df.copy()

        metric_choice = st.selectbox("Metric to highlight", ["Accuracy", "Weighted F1", "ROC-AUC"], key="cls_metric_choice")
        fig_bar = px.bar(
            metrics_df.sort_values(metric_choice, ascending=True),
            x=metric_choice, y="Model", orientation="h",
            color=metric_choice,
            color_continuous_scale=[[0, "#3B2D1D"], [0.5, "#8C6A32"], [1, "#C6A04A"]],
            text=metric_choice,
        )
        fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F4EEE4", height=430,
            xaxis=dict(range=[0.85, 1.01], gridcolor="#30261B"),
            yaxis=dict(gridcolor="#30261B"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="cls_bar")

        st.markdown("#### Radar Chart — All Metrics")
        categories = ["Accuracy", "Weighted F1", "ROC-AUC"]
        radar_fig = go.Figure()
        palette = px.colors.qualitative.Plotly
        for idx, (_, row) in enumerate(metrics_df.iterrows()):
            model_name = row["Model"]
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
                bgcolor="#211B15",
                radialaxis=dict(visible=True, range=[0.85, 1.0], gridcolor="#493B2D", color="#B7A895"),
                angularaxis=dict(gridcolor="#493B2D", color="#B7A895"),
            ),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#F4EEE4",
            height=440, legend=dict(font_size=11),
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(radar_fig, use_container_width=True, key="cls_radar")

    # ── TAB: FEATURE GUIDE ───────────────────────────────────
    with tab_info:
        st.markdown("### ℹ Feature Reference")
        rows = [(f, d) for f, d in FEATURE_TOOLTIPS.items()]
        ftable = pd.DataFrame(rows, columns=["Feature", "Description"])
        st.dataframe(ftable, use_container_width=True, hide_index=True, height=520)

        st.markdown("---")
        st.markdown("###  Dataset Overview")
        df_cls = load_cls_data()
        a, b = st.columns(2)
        with a:
            st.markdown("**Class Distribution**")
            vc = df_cls["class"].value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            fig_pie = px.pie(vc, names="Class", values="Count",
                             color="Class", color_discrete_map=CLASS_COLORS, hole=0.5)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#F4EEE4", height=300,
                                  margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True, key="cls_info_pie")
        with b:
            st.markdown("**Summary Statistics**")
            st.dataframe(df_cls.drop(columns=["class"]).describe().T.round(3), use_container_width=True, height=300)


# ═══════════════════════════════════════════════════════════════
#  PAGE: REGRESSION
# ═══════════════════════════════════════════════════════════════
elif page == "Regression":
    st.title(" Community Crime Rate Predictor")
    st.markdown("<p style='color:#B7A895'>Predict the violent crime rate per capita for a US community using socioeconomic and demographic features (Communities & Crime dataset).</p>", unsafe_allow_html=True)
    st.markdown("---")

    # load
    try:
        reg_imputer, reg_scaler, reg_feat_cols, reg_bounds = load_reg_artifacts()
    except FileNotFoundError:
        st.error(" Artifacts not found. Please run `python save_artifacts.py` first.")
        st.stop()

    df_reg, target_reg = load_reg_data()

    # Read the complete evaluation leaderboard generated from the same split
    # and preprocessing as the regression notebook / saved model artifacts.
    reg_metrics_path = os.path.join(REG_DIR, "regression_metrics.csv")
    if not os.path.exists(reg_metrics_path):
        st.error("Regression metrics file is missing. Ensure Regression Track A/regression_metrics.csv is included in the project.")
        st.stop()
    reg_metrics_df = pd.read_csv(reg_metrics_path)
    required_reg_columns = {"Model", "R²", "RMSE", "MAE"}
    if not required_reg_columns.issubset(reg_metrics_df.columns):
        st.error("Regression metrics file must contain Model, R², RMSE, and MAE columns.")
        st.stop()
    reg_metrics_df = reg_metrics_df.drop_duplicates(subset=["Model"], keep="last")
    REG_METRICS = {
        row["Model"]: {"R²": float(row["R²"]), "RMSE": float(row["RMSE"]), "MAE": float(row["MAE"])}
        for _, row in reg_metrics_df.iterrows()
    }

    # Models trained on standardized features in save_reg_models.py.
    REG_NEEDS_SCALE = {"Linear Regression", "Ridge Regression", "Lasso Regression",
                       "ElasticNet Regression", "Polynomial Regression (PCA=10)",
                       "SVR", "KNN"}

    tab_pred_r, tab_compare_r, tab_data_r = st.tabs([" Predict", " Model Comparison", " Dataset"])

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
        st.markdown("<div class='info-banner'> All features are normalised to [0, 1] in the original dataset. Sliders reflect actual data range.</div>", unsafe_allow_html=True)

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

        predict_r_btn = st.button("Predict Crime Rate", use_container_width=True)

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
                model_files = {
                    "Linear Regression": "linear_regression.pkl",
                    "Ridge Regression": "ridge_regression.pkl",
                    "Lasso Regression": "lasso_regression.pkl",
                    "ElasticNet Regression": "elasticnet_regression.pkl",
                    "Polynomial Regression (PCA=10)": "polynomial_regression_pca10.pkl",
                    "Decision Tree": "decision_tree.pkl",
                    "Random Forest": "random_forest.pkl",
                    "Gradient Boosting": "gradient_boosting.pkl",
                    "SVR": "svr.pkl",
                    "KNN": "knn.pkl",
                }
                model_path = os.path.join(REG_MODELS, model_files[reg_model_name])
                if not os.path.isfile(model_path):
                    raise FileNotFoundError(f"Required model file not found: {model_path}")
                reg_model_obj = joblib.load(model_path)
                pred_val = float(reg_model_obj.predict(X_r_in)[0])
            except Exception as exc:
                st.error(f"Could not generate a real prediction for {reg_model_name}: {exc}. Check that the matching model artifact is present and compatible. No dataset-mean fallback was used.")
                st.stop()

            pred_pct = pred_val * 100
            level = " Low" if pred_val < 0.25 else (" Moderate" if pred_val < 0.5 else " High")

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
                number={"valueformat": ".4f", "font": {"color": "#D6AD55", "size": 36}},
                gauge={
                    "axis": {"range": [0, 1], "tickcolor": "#B7A895"},
                    "bar": {"color": "#B78A3F"},
                    "steps": [
                        {"range": [0, 0.25],  "color": "#3B2D1D"},
                        {"range": [0.25, 0.5], "color": "#3a2a10"},
                        {"range": [0.5, 1.0],  "color": "#3a1010"},
                    ],
                    "threshold": {"line": {"color": "#C77D52", "width": 3}, "thickness": 0.75, "value": pred_val},
                    "bgcolor": "#211B15",
                    "bordercolor": "#493B2D",
                },
            ))
            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#F4EEE4",
                height=280, margin=dict(t=20, b=20),
            )
            st.plotly_chart(gauge, use_container_width=True, key="reg_gauge")

    with tab_compare_r:
        st.markdown("###  Regression Model Comparison")
        reg_m_df = pd.DataFrame(REG_METRICS).T.reset_index().rename(columns={"index": "Model"})

        reg_metric_c = st.selectbox("Sort by metric", ["R²", "RMSE", "MAE"], key="reg_metric_c")
        asc = reg_metric_c in ["RMSE", "MAE"]
        fig_reg_bar = px.bar(
            reg_m_df.sort_values(reg_metric_c, ascending=asc),
            x=reg_metric_c, y="Model", orientation="h",
            color=reg_metric_c,
            color_continuous_scale=[[0, "#3B2D1D"], [0.5, "#8C6A32"], [1, "#C6A04A"]] if not asc
                                   else [[0, "#C6A04A"], [0.5, "#8C6A32"], [1, "#3B2D1D"]],
            text=reg_metric_c,
        )
        fig_reg_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_reg_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F4EEE4", height=400,
            xaxis=dict(gridcolor="#30261B"),
            yaxis=dict(gridcolor="#30261B"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_reg_bar, use_container_width=True, key="reg_bar")

        st.markdown("#### Full Metrics Table")
        st.dataframe(reg_m_df.sort_values("R²", ascending=False).set_index("Model").round(4),
                     use_container_width=True)

    with tab_data_r:
        st.markdown("###  Regression Dataset Explorer")
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
            font_color="#F4EEE4", height=380,
            xaxis=dict(gridcolor="#30261B"),
            yaxis=dict(gridcolor="#30261B"),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_corr, use_container_width=True, key="reg_corr_bar")


# ═══════════════════════════════════════════════════════════════
#  PAGE: CLUSTERING
# ═══════════════════════════════════════════════════════════════
elif page == "Clustering":
    st.title(" Clustering")
    st.markdown("<p style='color:#B7A895'>Credit Card Customer Segmentation (notebook model) · Interactive Explorer on any dataset.</p>", unsafe_allow_html=True)
    st.markdown("---")

    tab_cc, tab_explore = st.tabs([" Credit Card Segmentation", " Interactive Explorer"])

    # ────────────────────────────────────────────────────────────
    #  TAB 1 — CREDIT CARD SEGMENTATION (saved notebook model)
    # ────────────────────────────────────────────────────────────
    with tab_cc:
        try:
            clust_prep, clust_km, clust_centroids, clust_labels_df, clust_meta = load_cluster_artifacts()
            cc_data = load_cc_data()
        except FileNotFoundError as e:
            st.error(f"Clustering artifacts not found: {e}. Run the last cell of clustering.ipynb first.")
            st.stop()

        # ── Extract metadata with fallbacks ───────────────────
        RAW_FEATURES_CC = clust_meta.get("raw_features")
        if not RAW_FEATURES_CC:
            RAW_FEATURES_CC = [c for c in cc_data.columns if c != "CUST_ID"]
        ALL_FEATURES_CC = clust_meta.get("final_features") or clust_meta.get("feature_names")
        if not ALL_FEATURES_CC:
            ALL_FEATURES_CC = [c for c in clust_centroids.columns if c != "cluster"]
        K = clust_meta.get("k_final") or clust_meta.get("k", getattr(clust_km, "n_clusters", 4))
        n_customers = clust_meta.get("n_customers") or clust_meta.get("n_samples", len(cc_data))

        # metrics may be nested under "kmeans" sub-key or flat
        _raw_metrics = clust_meta.get("metrics", {})
        if "kmeans" in _raw_metrics:
            _km_metrics = _raw_metrics["kmeans"]
        else:
            _km_metrics = _raw_metrics
        sil_score = _km_metrics.get("silhouette") or _km_metrics.get("Silhouette", float("nan"))
        db_score  = _km_metrics.get("davies_bouldin") or _km_metrics.get("Davies-Bouldin", float("nan"))

        # Segment names and colours from the notebook metadata
        _segments = clust_meta.get("segments", [])
        CLUSTER_COLORS = [s.get("color", px.colors.qualitative.Plotly[i % 10]) for i, s in enumerate(_segments)] \
                         if _segments else px.colors.qualitative.Plotly
        CLUSTER_NAMES = {s["id"]: s.get("name", f"Cluster {s['id']}") for s in _segments} \
                        if _segments else {i: f"Cluster {i}" for i in range(K)}

        # ── Top metrics row ──────────────────────────────────
        st.markdown("###  Model Overview")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Algorithm",         "K-Means")
        m2.metric("Clusters (k)",      str(K))
        m3.metric("Customers",         f"{n_customers:,}")
        m4.metric("Silhouette ↑",      f"{sil_score:.4f}")
        m5.metric("Davies-Bouldin ↓",  f"{db_score:.4f}")

        st.markdown("---")

        # ── Tabs within the CC tab ───────────────────────────
        st_cc1, st_cc2, st_cc3, st_cc4 = st.tabs([" Cluster Profiles", " PCA Scatter", " Cluster Sizes", " Predict Cluster"])

        # ── Cluster Profiles (centroid radar / bar) ──────────
        with st_cc1:
            st.markdown("### Cluster Feature Profiles (Scaled Centroids)")
            st.markdown("<div class='info-banner'>Values are in the scaled (standardised) feature space. Positive = above average; Negative = below average.</div>", unsafe_allow_html=True)

            # Select features to display
            feat_show = st.multiselect(
                "Features to compare",
                ALL_FEATURES_CC,
                default=["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
                         "PAYMENTS", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY"],
                key="cc_profile_feats"
            )
            if feat_show:
                prof_df = clust_centroids[feat_show].copy()
                prof_df.index = [f"Cluster {i}" for i in prof_df.index]
                prof_melt = prof_df.reset_index().melt(id_vars="index", var_name="Feature", value_name="Scaled Value")
                prof_melt.rename(columns={"index": "Cluster"}, inplace=True)

                fig_prof = px.bar(
                    prof_melt, x="Feature", y="Scaled Value", color="Cluster",
                    barmode="group",
                    color_discrete_sequence=CLUSTER_COLORS,
                    title="Centroid Profiles — Scaled Feature Values by Cluster",
                )
                fig_prof.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F4EEE4", height=480,
                    xaxis=dict(gridcolor="#30261B", tickangle=-30),
                    yaxis=dict(gridcolor="#30261B", title="Scaled value (std devs)"),
                    legend_title_text="Cluster",
                    margin=dict(t=50, b=60),
                )
                st.plotly_chart(fig_prof, use_container_width=True, key="cc_profile_bar")

                st.markdown("**Centroid table (scaled)**")
                st.dataframe(clust_centroids[feat_show].round(4), use_container_width=True)

        # ── PCA Scatter ──────────────────────────────────────
        with st_cc2:
            st.markdown("### PCA 2D Projection")
            pca_n = st.radio("Dimensions", [2, 3], horizontal=True, key="cc_pca_n")
            sample_n_cc = st.slider("Sample size for plot", 500, len(cc_data), min(3000, len(cc_data)), step=500, key="cc_pca_sample")

            with st.spinner("Computing PCA…"):
                # use the fitted preprocessor to transform the raw data
                X_cc_scaled = clust_prep.transform(cc_data[RAW_FEATURES_CC])
                pca_viz = PCA(n_components=pca_n, random_state=42)
                X_pca_cc = pca_viz.fit_transform(X_cc_scaled)

            labels_all = clust_labels_df.iloc[:, 0].values
            idx_sample = np.random.default_rng(42).choice(len(labels_all), size=min(sample_n_cc, len(labels_all)), replace=False)

            if pca_n == 2:
                sc_df = pd.DataFrame({
                    "PC1": X_pca_cc[idx_sample, 0],
                    "PC2": X_pca_cc[idx_sample, 1],
                    "Cluster": [f"Cluster {l}" for l in labels_all[idx_sample]],
                })
                fig_pca = px.scatter(
                    sc_df, x="PC1", y="PC2", color="Cluster",
                    color_discrete_sequence=CLUSTER_COLORS,
                    title=f"K-Means k={K} — PCA 2D (sample n={len(idx_sample):,})",
                    opacity=0.65,
                )
            else:
                sc_df = pd.DataFrame({
                    "PC1": X_pca_cc[idx_sample, 0],
                    "PC2": X_pca_cc[idx_sample, 1],
                    "PC3": X_pca_cc[idx_sample, 2],
                    "Cluster": [f"Cluster {l}" for l in labels_all[idx_sample]],
                })
                fig_pca = px.scatter_3d(
                    sc_df, x="PC1", y="PC2", z="PC3", color="Cluster",
                    color_discrete_sequence=CLUSTER_COLORS,
                    title=f"K-Means k={K} — PCA 3D (sample n={len(idx_sample):,})",
                    opacity=0.65,
                )

            evr = pca_viz.explained_variance_ratio_ * 100
            fig_pca.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F4EEE4", height=540,
                legend_title_text="Cluster",
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_pca, use_container_width=True, key="cc_pca_scatter")
            st.markdown(f"**PCA Explained Variance:** {' + '.join([f'PC{i+1}: {v:.1f}%' for i, v in enumerate(evr)])} = **{sum(evr):.1f}%** total")

        # ── Cluster Sizes ────────────────────────────────────
        with st_cc3:
            st.markdown("### Cluster Size Distribution")
            # Build sizes from segments metadata or fall back to label counts
            if _segments:
                size_df = pd.DataFrame({
                    "Cluster": [CLUSTER_NAMES.get(s["id"], f"Cluster {s['id']}") for s in _segments],
                    "Customers": [s["size"] for s in _segments],
                })
            else:
                _lbl_counts = clust_labels_df.iloc[:, 0].value_counts().sort_index()
                size_df = pd.DataFrame({
                    "Cluster": [f"Cluster {i}" for i in _lbl_counts.index],
                    "Customers": _lbl_counts.values,
                })
            size_df["Percent"] = (size_df["Customers"] / size_df["Customers"].sum() * 100).round(1)

            fig_sz2 = px.bar(
                size_df, x="Cluster", y="Customers",
                color="Cluster",
                color_discrete_sequence=CLUSTER_COLORS,
                text=size_df["Percent"].apply(lambda x: f"{x}%"),
                title="Number of Customers per Cluster",
            )
            fig_sz2.update_traces(textposition="outside")
            fig_sz2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F4EEE4", height=380,
                showlegend=False,
                xaxis=dict(gridcolor="#30261B"),
                yaxis=dict(gridcolor="#30261B"),
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_sz2, use_container_width=True, key="cc_size_bar")
            st.dataframe(size_df.set_index("Cluster"), use_container_width=True)

        # ── Predict Cluster for New Customer ─────────────────
        with st_cc4:
            st.markdown("###  Predict Cluster for a New Customer")
            st.markdown("<div class='info-banner'>Enter raw credit card usage values. The notebook's fitted preprocessor (engineer → impute → scale) will be applied automatically before prediction.</div>", unsafe_allow_html=True)

            med_vals = cc_data[RAW_FEATURES_CC].median()
            input_cc = {}
            cols_cc = st.columns(3)
            for idx, feat in enumerate(RAW_FEATURES_CC):
                col_w = cols_cc[idx % 3]
                min_v = float(cc_data[feat].min())
                max_v = float(cc_data[feat].max())
                med_v = float(med_vals[feat])
                step_v = round((max_v - min_v) / 100, 4) if max_v != min_v else 0.01
                input_cc[feat] = col_w.number_input(
                    feat, min_value=min_v, max_value=max_v,
                    value=med_v, step=step_v, key=f"cc_inp_{feat}"
                )

            predict_cc_btn = st.button(" Predict Customer Segment", use_container_width=True, key="cc_predict_btn")

            if predict_cc_btn:
                row_df = pd.DataFrame([input_cc])[RAW_FEATURES_CC]
                X_in   = clust_prep.transform(row_df)
                cluster_id = int(clust_km.predict(X_in)[0])

                # distance to all centroids (in scaled space)
                dists = np.linalg.norm(clust_km.cluster_centers_ - X_in, axis=1)
                confidence = 1.0 - (dists[cluster_id] / dists.sum())

                st.markdown(f"""
<div class='pred-box'>
  <div class='pred-label'>Predicted Customer Segment</div>
  <div class='pred-value' style='color:{CLUSTER_COLORS[cluster_id]}'>Cluster {cluster_id}</div>
  <div class='pred-sub'>Relative distance score (heuristic): {confidence:.1%} &nbsp;|&nbsp; Distance to centroid: {dists[cluster_id]:.4f}</div>
</div>""", unsafe_allow_html=True)

                st.markdown("#### Centroid Profile of Predicted Cluster")
                cent_row = clust_centroids.loc[cluster_id]
                top10 = cent_row.abs().sort_values(ascending=False).head(10).index.tolist()
                cent_plot = cent_row[top10].reset_index()
                cent_plot.columns = ["Feature", "Scaled Value"]
                cent_plot["Direction"] = cent_plot["Scaled Value"].apply(lambda x: "Above Avg" if x >= 0 else "Below Avg")

                fig_cent = px.bar(
                    cent_plot, x="Feature", y="Scaled Value", color="Direction",
                    color_discrete_map={"Above Avg": "#C6A04A", "Below Avg": "#C77D52"},
                    title=f"Top 10 Distinguishing Features — Cluster {cluster_id}",
                )
                fig_cent.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F4EEE4", height=380,
                    xaxis=dict(gridcolor="#30261B", tickangle=-25),
                    yaxis=dict(gridcolor="#30261B"),
                    margin=dict(t=50, b=60),
                )
                st.plotly_chart(fig_cent, use_container_width=True, key="cc_cent_bar")

    # ────────────────────────────────────────────────────────────
    #  TAB 2 — INTERACTIVE EXPLORER
    # ────────────────────────────────────────────────────────────
    with tab_explore:
        st.markdown("###  Run clustering on any dataset interactively")
        c_left, c_right = st.columns([1, 2])

        with c_left:
            st.markdown("####  Configuration")
            dataset_choice = st.selectbox(
                "Dataset",
                ["Credit Card Clustering", "Star Classification (SDSS)", "Community Crime"],
                key="clust_dataset",
            )
            algo_choice    = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Agglomerative"], key="clust_algo")

            if algo_choice == "K-Means":
                k_val       = st.slider("Number of clusters (k)", 2, 15, 3, key="clust_k")
                init_method = st.selectbox("Init method", ["k-means++", "random"], key="clust_init")
                max_iter    = st.slider("Max iterations", 100, 500, 300, step=50, key="clust_maxiter")
            elif algo_choice == "DBSCAN":
                eps_val  = st.slider("ε (epsilon)", 0.1, 5.0, 0.5, step=0.05, key="clust_eps")
                min_samp = st.slider("min_samples", 2, 30, 5, key="clust_minsamp")
                st.markdown("<div class='info-banner'>Noise points labelled <b>-1</b>.</div>", unsafe_allow_html=True)
            elif algo_choice == "Agglomerative":
                agg_k     = st.slider("Number of clusters", 2, 15, 3, key="clust_agg_k")
                linkage_m = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="clust_link")

            st.markdown("---")
            n_pca_comp  = st.slider("PCA components for vis", 2, 3, 2, key="clust_pca")
            sample_frac = st.slider("Data sample fraction", 0.05, 1.0, 0.30, step=0.05, key="clust_frac")
            run_btn     = st.button("▶ Run Clustering", use_container_width=True, key="clust_run_btn")
            st.markdown("---")
            show_elbow  = st.checkbox("Elbow curve (K-Means 2–12)", value=False, key="clust_elbow")

        with c_right:
            if run_btn or show_elbow:
                with st.spinner("Loading & preprocessing data…"):
                    if dataset_choice == "Credit Card Clustering":
                        df_raw_ex = load_cc_data()
                        X_raw_ex = df_raw_ex.copy()
                        y_true_ex = None
                    elif "Star" in dataset_choice:
                        df_raw_ex = load_cls_data()
                        X_raw_ex = df_raw_ex.drop(columns=["class"], errors="ignore")
                        y_true_ex = df_raw_ex["class"] if "class" in df_raw_ex.columns else None
                    else:
                        df_raw_ex, tgt_r_ex = load_reg_data()
                        X_raw_ex = df_raw_ex.drop(columns=[tgt_r_ex], errors="ignore")
                        y_true_ex = None

                    # Use the same sampled row indices for features and optional true labels.
                    sample_n_ex = max(3, min(len(X_raw_ex), int(round(len(X_raw_ex) * sample_frac))))
                    sample_idx_ex = X_raw_ex.sample(n=sample_n_ex, random_state=42).index
                    X_raw_ex = X_raw_ex.loc[sample_idx_ex].reset_index(drop=True)
                    if y_true_ex is not None:
                        y_true_ex = y_true_ex.loc[sample_idx_ex].reset_index(drop=True)

                    X_num_ex = X_raw_ex.select_dtypes(include=[np.number]).copy()
                    X_num_ex = X_num_ex.replace([np.inf, -np.inf], np.nan)
                    X_num_ex = X_num_ex.fillna(X_num_ex.median()).fillna(0)
                    if X_num_ex.shape[1] < n_pca_comp:
                        st.error(f"This dataset has only {X_num_ex.shape[1]} numeric features; choose fewer PCA components.")
                        st.stop()
                    sc_ex = SkScaler()
                    X_sc_ex = sc_ex.fit_transform(X_num_ex)
                    pca_ex = PCA(n_components=n_pca_comp, random_state=42)
                    X_pca_ex = pca_ex.fit_transform(X_sc_ex)

                if show_elbow and algo_choice == "K-Means":
                    with st.spinner("Computing elbow curve…"):
                        inertias_ex, sil_ex = [], []
                        k_range_ex = range(2, 13)
                        for k in k_range_ex:
                            km_e = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=42)
                            lb_e = km_e.fit_predict(X_sc_ex)
                            inertias_ex.append(km_e.inertia_)
                            n_labels_e = len(np.unique(lb_e))
                            if 1 < n_labels_e < len(lb_e):
                                sil_ex.append(silhouette_score(X_sc_ex, lb_e, sample_size=min(1000, len(X_sc_ex)), random_state=42))
                            else:
                                sil_ex.append(float("nan"))

                        fig_el = make_subplots(rows=1, cols=2, subplot_titles=["Inertia (Elbow)", "Silhouette"])
                        fig_el.add_trace(go.Scatter(x=list(k_range_ex), y=inertias_ex, mode="lines+markers",
                                                     marker_color="#B78A3F", name="Inertia"), row=1, col=1)
                        fig_el.add_trace(go.Scatter(x=list(k_range_ex), y=sil_ex, mode="lines+markers",
                                                     marker_color="#C6A04A", name="Silhouette"), row=1, col=2)
                        fig_el.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                              font_color="#F4EEE4", height=340,
                                              xaxis=dict(gridcolor="#30261B", title="k"),
                                              xaxis2=dict(gridcolor="#30261B", title="k"),
                                              yaxis=dict(gridcolor="#30261B"),
                                              yaxis2=dict(gridcolor="#30261B"),
                                              margin=dict(t=40, b=20), showlegend=False)
                        st.plotly_chart(fig_el, use_container_width=True, key="clust_elbow_fig")

                if run_btn:
                    with st.spinner(f"Running {algo_choice}…"):
                        if algo_choice == "K-Means":
                            model_c  = KMeans(n_clusters=k_val, init=init_method, max_iter=max_iter,
                                              n_init=10, random_state=42)
                            labels_c = model_c.fit_predict(X_sc_ex)
                        elif algo_choice == "DBSCAN":
                            model_c  = DBSCAN(eps=eps_val, min_samples=min_samp, n_jobs=-1)
                            labels_c = model_c.fit_predict(X_sc_ex)
                        elif algo_choice == "Agglomerative":
                            model_c  = AgglomerativeClustering(n_clusters=agg_k, linkage=linkage_m)
                            labels_c = model_c.fit_predict(X_sc_ex)

                    n_found = len(set(labels_c)) - (1 if -1 in labels_c else 0)
                    n_noise = (labels_c == -1).sum()
                    try:
                        valid_mask = labels_c != -1 if algo_choice == "DBSCAN" else np.ones(len(labels_c), dtype=bool)
                        valid_labels = labels_c[valid_mask]
                        valid_X = X_sc_ex[valid_mask]
                        n_valid_clusters = len(np.unique(valid_labels))
                        if n_valid_clusters > 1 and n_valid_clusters < len(valid_labels):
                            sil = silhouette_score(valid_X, valid_labels,
                                                   sample_size=min(2000, len(valid_X)), random_state=42)
                            db = davies_bouldin_score(valid_X, valid_labels)
                        else:
                            sil, db = float("nan"), float("nan")
                    except Exception:
                        sil, db = float("nan"), float("nan")

                    em1, em2, em3, em4 = st.columns(4)
                    em1.metric("Clusters Found",  str(n_found))
                    em2.metric("Noise Points",    str(n_noise))
                    em3.metric("Silhouette ↑",    f"{sil:.4f}" if not np.isnan(sil) else "N/A")
                    em4.metric("Davies-Bouldin ↓", f"{db:.4f}" if not np.isnan(db) else "N/A")

                    st.markdown("---")
                    labels_str_ex = [str(l) for l in labels_c]
                    if n_pca_comp == 2:
                        sc_df_ex = pd.DataFrame({"PC1": X_pca_ex[:, 0], "PC2": X_pca_ex[:, 1], "Cluster": labels_str_ex})
                        if y_true_ex is not None and "Star" in dataset_choice:
                            sc_df_ex["True Class"] = y_true_ex
                        fig_sc_ex = px.scatter(
                            sc_df_ex, x="PC1", y="PC2", color="Cluster",
                            hover_data=["True Class"] if "True Class" in sc_df_ex else None,
                            title=f"{algo_choice} (PCA 2D) — {n_found} clusters",
                            color_discrete_sequence=px.colors.qualitative.Plotly,
                        )
                    else:
                        sc_df_ex = pd.DataFrame({"PC1": X_pca_ex[:, 0], "PC2": X_pca_ex[:, 1],
                                                  "PC3": X_pca_ex[:, 2], "Cluster": labels_str_ex})
                        fig_sc_ex = px.scatter_3d(
                            sc_df_ex, x="PC1", y="PC2", z="PC3", color="Cluster",
                            title=f"{algo_choice} (PCA 3D) — {n_found} clusters",
                            color_discrete_sequence=px.colors.qualitative.Plotly, opacity=0.7,
                        )

                    fig_sc_ex.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#F4EEE4", height=520, legend_title_text="Cluster",
                        margin=dict(t=50, b=20),
                    )
                    st.plotly_chart(fig_sc_ex, use_container_width=True, key="clust_scatter_ex")

                    st.markdown("#### Cluster Size Distribution")
                    sz_ex = pd.Series(labels_c).value_counts().sort_index().reset_index()
                    sz_ex.columns = ["Cluster", "Count"]
                    sz_ex["Cluster"] = sz_ex["Cluster"].astype(str)
                    fig_sz_ex = px.bar(sz_ex, x="Cluster", y="Count",
                                       color="Cluster", text="Count",
                                       color_discrete_sequence=px.colors.qualitative.Plotly)
                    fig_sz_ex.update_traces(textposition="outside")
                    fig_sz_ex.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#F4EEE4", height=320, showlegend=False,
                        xaxis=dict(gridcolor="#30261B"), yaxis=dict(gridcolor="#30261B"),
                        margin=dict(t=20, b=20),
                    )
                    st.plotly_chart(fig_sz_ex, use_container_width=True, key="clust_size_ex")
                    evr_ex = pca_ex.explained_variance_ratio_ * 100
                    st.markdown(f"**PCA Explained Variance:** {' + '.join([f'PC{i+1}: {v:.1f}%' for i,v in enumerate(evr_ex)])} = **{sum(evr_ex):.1f}%** total")

            else:
                st.markdown("""
<div style='background:#211B15;border:1px dashed #493B2D;border-radius:14px;
            padding:60px 40px;text-align:center;color:#B7A895'>
  <div style='font-size:3rem'></div>
  <h3 style='color:#D6AD55'>Configure & Run Clustering</h3>
  <p>Choose a dataset, algorithm, and parameters on the left,<br>then click <b>▶ Run Clustering</b>.</p>
</div>""", unsafe_allow_html=True)
