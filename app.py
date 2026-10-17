# """
# Vectora – Unified ML Dashboard (Streamlit)
# ==========================================
# Covers:
#   • Classification  – Star Type (Galaxy / QSO / Star)
#   • Regression      – Community Crime Rate
#   • Clustering      – K-Means / DBSCAN / Agglomerative (interactive)
#   • EDA Explorer    – Dataset exploration for both tracks
# """

# import os, warnings
# warnings.filterwarnings("ignore")

# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.metrics import silhouette_score, davies_bouldin_score
# from sklearn.preprocessing import StandardScaler as SkScaler

# # ─── page config ──────────────────────────────────────────────
# st.set_page_config(
#     page_title="Vectora · ML Dashboard",
#     page_icon="🌌",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─── custom CSS ───────────────────────────────────────────────
# st.markdown("""
# <style>
#   /* Global dark background */
#   .stApp { background-color: #0d1117; color: #e6edf3; }

#   /* Sidebar */
#   [data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #30363d; }
#   [data-testid="stSidebar"] .stMarkdown h1,
#   [data-testid="stSidebar"] .stMarkdown h2,
#   [data-testid="stSidebar"] .stMarkdown h3 { color: #58a6ff; }

#   /* Metric cards */
#   [data-testid="metric-container"] {
#     background: #161b22;
#     border: 1px solid #30363d;
#     border-radius: 12px;
#     padding: 12px 16px;
#   }
#   [data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.8rem; }
#   [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.5rem; font-weight: 700; }

#   /* Headings */
#   h1 { color: #58a6ff !important; }
#   h2 { color: #79c0ff !important; }
#   h3 { color: #a5d6ff !important; }

#   /* Buttons */
#   .stButton > button {
#     background: linear-gradient(135deg, #238636, #2ea043);
#     color: white;
#     border: none;
#     border-radius: 8px;
#     font-weight: 600;
#     padding: 0.5rem 1.5rem;
#     transition: opacity 0.2s;
#   }
#   .stButton > button:hover { opacity: 0.85; }

#   /* Tabs */
#   .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 10px; }
#   .stTabs [data-baseweb="tab"] { color: #8b949e; font-weight: 600; }
#   .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

#   /* DataFrames */
#   .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

#   /* Selectboxes / sliders */
#   .stSelectbox label, .stSlider label, .stNumberInput label { color: #8b949e; }

#   /* Prediction result box */
#   .pred-box {
#     background: linear-gradient(135deg, #1f2d3d 0%, #162032 100%);
#     border: 1px solid #388bfd;
#     border-radius: 14px;
#     padding: 20px 24px;
#     margin: 12px 0;
#     text-align: center;
#   }
#   .pred-label { font-size: 1rem; color: #8b949e; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px; }
#   .pred-value { font-size: 2.4rem; font-weight: 800; color: #58a6ff; letter-spacing: 0.02em; }
#   .pred-sub   { font-size: 0.95rem; color: #79c0ff; margin-top: 4px; }

#   /* Info banners */
#   .info-banner {
#     background: #162032;
#     border-left: 4px solid #388bfd;
#     border-radius: 0 8px 8px 0;
#     padding: 10px 16px;
#     margin: 8px 0;
#     font-size: 0.88rem;
#     color: #a5d6ff;
#   }

#   /* Section divider */
#   hr { border-color: #30363d; }

#   /* Scrollable feature table */
#   .scroll-box { max-height: 340px; overflow-y: auto; }

#   /* Radio group */
#   .stRadio > div { flex-direction: row; gap: 16px; }

#   /* Progress bar */
#   .stProgress > div > div { background: linear-gradient(90deg, #238636, #58a6ff); }
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────────────────────
# #  PATHS & MODEL LOADING
# # ─────────────────────────────────────────────────────────────
# BASE        = os.path.dirname(os.path.abspath(__file__))
# CLS_DIR     = os.path.join(BASE, "ML_CLASSIFICATION_TRACKA")
# REG_DIR     = os.path.join(BASE, "Regression Track A")
# CLUST_DIR   = os.path.join(BASE, "Clustering")
# CLS_MODELS  = os.path.join(CLS_DIR,   "models")
# REG_MODELS  = os.path.join(REG_DIR,   "models")
# CLUST_MODELS = os.path.join(CLUST_DIR, "models")


# @st.cache_resource
# def load_cls_artifacts():
#     scaler   = joblib.load(os.path.join(CLS_MODELS, "cls_scaler.pkl"))
#     encoder  = joblib.load(os.path.join(CLS_MODELS, "cls_label_encoder.pkl"))
#     feat_cols = joblib.load(os.path.join(CLS_MODELS, "cls_feature_cols.pkl"))
#     bounds   = joblib.load(os.path.join(CLS_MODELS, "cls_outlier_bounds.pkl"))
#     return scaler, encoder, feat_cols, bounds


# @st.cache_resource
# def load_cls_models():
#     files = {
#         "Logistic Regression":      "logistic_regression_model.pkl",
#         "Gaussian Naive Bayes":     "gaussian_naive_bayes_model.pkl",
#         "K-Nearest Neighbors":      "knn_model.pkl",
#         "Decision Tree":            "decision_tree_model.pkl",
#         "SVM":                      "svm_model.pkl",
#         "Random Forest":            "random_forest_model.pkl",
#         "Gradient Boosting":        "gradient_boosting_model.pkl",
#         "AdaBoost":                 "adaboost_model.pkl",
#         "Bagging Classifier":       "bagging_classifier_model.pkl",
#         "MLP Classifier":           "mlp_classifier_model.pkl",
#     }
#     models = {}
#     for name, fname in files.items():
#         path = os.path.join(CLS_MODELS, fname)
#         if os.path.exists(path):
#             models[name] = joblib.load(path)
#     return models


# @st.cache_resource
# def load_reg_artifacts():
#     imputer   = joblib.load(os.path.join(REG_MODELS, "reg_imputer.pkl"))
#     scaler    = joblib.load(os.path.join(REG_MODELS, "reg_scaler.pkl"))
#     feat_cols = joblib.load(os.path.join(REG_MODELS, "reg_feature_cols.pkl"))
#     bounds    = joblib.load(os.path.join(REG_MODELS, "reg_clip_bounds.pkl"))
#     return imputer, scaler, feat_cols, bounds


# @st.cache_resource
# def load_cluster_artifacts():
#     import json, sys
#     # Make clustering_preprocessing importable when unpickling the preprocessor pipeline
#     app_dir = os.path.join(CLUST_DIR, "app")
#     if app_dir not in sys.path:
#         sys.path.insert(0, app_dir)
#     preprocessor = joblib.load(os.path.join(CLUST_MODELS, "cluster_preprocessor.pkl"))
#     kmeans       = joblib.load(os.path.join(CLUST_MODELS, "kmeans_k4.pkl"))
#     centroids    = pd.read_csv(os.path.join(CLUST_MODELS, "kmeans_k4_centroids.csv"), index_col=0)
#     labels_df    = pd.read_csv(os.path.join(CLUST_MODELS, "kmeans_k4_labels.csv"), index_col=0)
#     with open(os.path.join(CLUST_MODELS, "cluster_meta.json")) as f:
#         meta = json.load(f)
#     return preprocessor, kmeans, centroids, labels_df, meta


# @st.cache_data
# def load_cc_data():
#     csv_path = os.path.join(CLUST_DIR, "CC GENERAL.csv")
#     df = pd.read_csv(csv_path)
#     if "CUST_ID" in df.columns:
#         df = df.drop(columns=["CUST_ID"])
#     return df


# @st.cache_data
# def load_cls_data():
#     df = pd.read_csv(os.path.join(CLS_DIR, "star_classification.csv"))
#     df.drop(columns=["class_numeric"], inplace=True, errors="ignore")
#     df = df.drop_duplicates().reset_index(drop=True)
#     df["r_i"] = df["r"] - df["i"]
#     return df


# @st.cache_data
# def load_reg_data():
#     df = pd.read_csv(os.path.join(REG_DIR, "communities.csv"), header=0, na_values=["?"])
#     target = df.columns[-1]
#     missing_pct = df.isna().mean() * 100
#     high_missing = missing_pct[missing_pct > 50].index.tolist()
#     id_cols = ["state", "county", "community", "communityname", "fold"]
#     cols_to_drop = sorted(set(high_missing + id_cols))
#     df_clean = df.drop(columns=cols_to_drop, errors="ignore").copy()
#     family_cols = [c for c in ["PctKids2Par", "PctFam2Par", "PctYoungKids2Par", "PctTeen2Par"]
#                    if c in df_clean.columns]
#     if family_cols:
#         df_clean["FamilyStructureAvg"] = df_clean[family_cols].mean(axis=1)
#     return df_clean, target


# # known model metrics from notebook outputs
# CLS_METRICS = {
#     "Logistic Regression":  {"Accuracy": 0.9785, "Weighted F1": 0.9785, "ROC-AUC": 0.9975},
#     "Gaussian Naive Bayes": {"Accuracy": 0.9715, "Weighted F1": 0.9712, "ROC-AUC": 0.9971},
#     "K-Nearest Neighbors":  {"Accuracy": 0.9873, "Weighted F1": 0.9873, "ROC-AUC": 0.9988},
#     "Decision Tree":        {"Accuracy": 0.9780, "Weighted F1": 0.9780, "ROC-AUC": 0.9780},
#     "SVM":                  {"Accuracy": 0.9820, "Weighted F1": 0.9820, "ROC-AUC": 0.9981},
#     "Random Forest":        {"Accuracy": 0.9901, "Weighted F1": 0.9901, "ROC-AUC": 0.9993},
#     "Gradient Boosting":    {"Accuracy": 0.9839, "Weighted F1": 0.9839, "ROC-AUC": 0.9988},
#     "AdaBoost":             {"Accuracy": 0.8965, "Weighted F1": 0.8955, "ROC-AUC": 0.9812},
#     "Bagging Classifier":   {"Accuracy": 0.9903, "Weighted F1": 0.9903, "ROC-AUC": 0.9994},
#     "MLP Classifier":       {"Accuracy": 0.9801, "Weighted F1": 0.9801, "ROC-AUC": 0.9981},
# }

# # Models that use scaled features
# CLS_NEEDS_SCALE = {"Logistic Regression", "K-Nearest Neighbors", "SVM", "MLP Classifier",
#                    "Gaussian Naive Bayes"}

# CLS_CLASSES = ["GALAXY", "QSO", "STAR"]
# CLASS_COLORS = {"GALAXY": "#388bfd", "QSO": "#f78166", "STAR": "#3fb950"}
# CLASS_ICONS  = {"GALAXY": "🌌", "QSO": "⚡", "STAR": "⭐"}
# CLASS_DESC   = {
#     "GALAXY": "A gravitationally bound system of stars, gas, dust, and dark matter.",
#     "QSO":    "Quasi-Stellar Object — an extremely luminous active galactic nucleus powered by a supermassive black hole.",
#     "STAR":   "A luminous spheroid of plasma held together by self-gravity undergoing nuclear fusion.",
# }

# FEATURE_TOOLTIPS = {
#     "obj_ID":     "Object Identifier — unique catalog ID",
#     "alpha":      "Right Ascension (degrees) — celestial longitude",
#     "delta":      "Declination (degrees) — celestial latitude",
#     "u":          "Ultraviolet photometric band magnitude",
#     "g":          "Green photometric band magnitude",
#     "r":          "Red photometric band magnitude",
#     "i":          "Near-infrared band magnitude",
#     "z":          "Infrared band magnitude",
#     "run_ID":     "Run number of the SDSS scan",
#     "rerun_ID":   "Rerun specification of the image processing",
#     "cam_col":    "Camera column (1–6) used in observation",
#     "field_ID":   "Field number of the telescope scan",
#     "spec_obj_ID":"Spectroscopic object ID",
#     "redshift":   "Redshift value — indicates recession speed / distance",
#     "plate":      "Plate ID of the SDSS spectroscopic plate",
#     "MJD":        "Modified Julian Date of observation",
#     "fiber_ID":   "Fiber ID directing light to the spectrograph",
#     "r_i":        "Engineered colour index: r − i band difference",
# }


# # ─────────────────────────────────────────────────────────────
# #  SIDEBAR NAVIGATION
# # ─────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 🌌 Vectora")
#     st.markdown("<p style='color:#8b949e;font-size:0.82rem;'>Unified ML Dashboard</p>", unsafe_allow_html=True)
#     st.markdown("---")

#     # ── nav bar buttons ──────────────────────────────────
#     st.markdown("""
# <style>
# div[data-testid="stSidebar"] button {
#     width: 100% !important;
#     text-align: left !important;
#     background: #161b22 !important;
#     color: #e6edf3 !important;
#     border: 1px solid #30363d !important;
#     border-radius: 8px !important;
#     padding: 10px 14px !important;
#     margin-bottom: 6px !important;
#     font-size: 0.95rem !important;
#     font-weight: 500 !important;
#     transition: background 0.15s, border-color 0.15s !important;
# }
# div[data-testid="stSidebar"] button:hover {
#     background: #21262d !important;
#     border-color: #58a6ff !important;
#     color: #58a6ff !important;
# }
# div[data-testid="stSidebar"] button[kind="primary"] {
#     background: linear-gradient(135deg,#1a3a5c,#1f4a7a) !important;
#     border-color: #58a6ff !important;
#     color: #58a6ff !important;
#     font-weight: 700 !important;
# }
# </style>
# """, unsafe_allow_html=True)

#     if "page" not in st.session_state:
#         st.session_state["page"] = "🏠  Home"

#     nav_items = [
#         ("🏠  Home",           "🏠  Home"),
#         ("🔭  Classification", "🔭  Classification"),
#         ("📈  Regression",     "📈  Regression"),
#         ("🔵  Clustering",     "🔵  Clustering"),
#         ("📊  EDA Explorer",   "📊  EDA Explorer"),
#     ]
#     for label, key in nav_items:
#         kind = "primary" if st.session_state["page"] == key else "secondary"
#         if st.button(label, key=f"nav_{key}", type=kind, use_container_width=True):
#             st.session_state["page"] = key
#             st.rerun()

#     page = st.session_state["page"]

#     st.markdown("---")
#     st.markdown("""
# <div style='font-size:0.75rem;color:#8b949e;line-height:1.8'>
# <b style='color:#58a6ff'>Tracks</b><br>
# <span style='color:#3fb950'>🔭</span> Star Classification (SDSS)<br>
# <span style='color:#3fb950'>📈</span> Community Crime Regression<br>
# <span style='color:#3fb950'>💳</span> Credit Card Clustering (k=4)<br>
# <span style='color:#3fb950'>📊</span> EDA Explorer
# </div>
# """, unsafe_allow_html=True)


# # ═══════════════════════════════════════════════════════════════
# #  PAGE: HOME
# # ═══════════════════════════════════════════════════════════════
# if page == "🏠  Home":
#     st.title("🌌 Vectora — ML Dashboard")
#     st.markdown("<p style='color:#8b949e;font-size:1.05rem;'>An end-to-end machine learning platform covering Classification, Regression, and Clustering.</p>", unsafe_allow_html=True)
#     st.markdown("---")

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown("""
# <div style='background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;text-align:center'>
#   <div style='font-size:2.5rem'>🔭</div>
#   <h3 style='color:#58a6ff'>Classification</h3>
#   <p style='color:#8b949e;font-size:0.88rem'>Star-type prediction using SDSS photometric data.<br>10 models · 3 classes</p>
#   <hr style='border-color:#30363d'>
#   <p style='color:#3fb950;font-size:1.2rem;font-weight:700'>≥ 99% Accuracy</p>
# </div>""", unsafe_allow_html=True)

#     with col2:
#         st.markdown("""
# <div style='background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;text-align:center'>
#   <div style='font-size:2.5rem'>📈</div>
#   <h3 style='color:#58a6ff'>Regression</h3>
#   <p style='color:#8b949e;font-size:0.88rem'>Community crime-rate prediction from socioeconomic features.<br>9 models · continuous target</p>
#   <hr style='border-color:#30363d'>
#   <p style='color:#3fb950;font-size:1.2rem;font-weight:700'>R² up to 0.74</p>
# </div>""", unsafe_allow_html=True)

#     with col3:
#         st.markdown("""
# <div style='background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;text-align:center'>
#   <div style='font-size:2.5rem'>🔵</div>
#   <h3 style='color:#58a6ff'>Clustering</h3>
#   <p style='color:#8b949e;font-size:0.88rem'>Exploratory unsupervised clustering with interactive PCA visualisation.<br>K-Means · DBSCAN · Agglomerative</p>
#   <hr style='border-color:#30363d'>
#   <p style='color:#3fb950;font-size:1.2rem;font-weight:700'>Interactive</p>
# </div>""", unsafe_allow_html=True)

#     st.markdown("---")
#     st.markdown("### 📦 Datasets at a Glance")
#     d1, d2 = st.columns(2)
#     with d1:
#         try:
#             df_cls = load_cls_data()
#             st.markdown("**🔭 Star Classification (SDSS)**")
#             mc1, mc2, mc3 = st.columns(3)
#             mc1.metric("Rows", f"{len(df_cls):,}")
#             mc2.metric("Features", f"{df_cls.shape[1]-1}")
#             mc3.metric("Classes", "3")
#             counts = df_cls["class"].value_counts()
#             fig = px.pie(
#                 names=counts.index, values=counts.values,
#                 color=counts.index,
#                 color_discrete_map=CLASS_COLORS,
#                 hole=0.55,
#             )
#             fig.update_layout(
#                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                 font_color="#e6edf3", showlegend=True,
#                 margin=dict(t=10, b=10, l=10, r=10),
#                 height=260,
#             )
#             st.plotly_chart(fig, use_container_width=True, key="home_cls_pie")
#         except Exception as e:
#             st.warning(f"Could not load classification data: {e}")

#     with d2:
#         try:
#             df_reg, target_reg = load_reg_data()
#             st.markdown(f"**📈 Communities & Crime**")
#             mr1, mr2, mr3 = st.columns(3)
#             mr1.metric("Rows", f"{len(df_reg):,}")
#             mr2.metric("Features", f"{df_reg.shape[1]-1}")
#             mr3.metric("Target", target_reg[:15])
#             fig2 = px.histogram(
#                 df_reg, x=target_reg, nbins=40,
#                 color_discrete_sequence=["#388bfd"],
#                 labels={target_reg: "Crime Rate (normalised 0–1)"},
#             )
#             fig2.update_layout(
#                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                 font_color="#e6edf3", bargap=0.05,
#                 margin=dict(t=10, b=10, l=10, r=10),
#                 height=260,
#                 yaxis=dict(gridcolor="#21262d"),
#                 xaxis=dict(gridcolor="#21262d"),
#             )
#             st.plotly_chart(fig2, use_container_width=True, key="home_reg_hist")
#         except Exception as e:
#             st.warning(f"Could not load regression data: {e}")

#     st.markdown("---")
#     st.markdown("### 🏆 Model Leaderboard — Classification")
#     metrics_df = pd.DataFrame(CLS_METRICS).T.reset_index().rename(columns={"index": "Model"})
#     metrics_df = metrics_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
#     metrics_df.index += 1
#     metrics_df_display = metrics_df.copy()
#     for col in ["Accuracy", "Weighted F1", "ROC-AUC"]:
#         metrics_df_display[col] = metrics_df_display[col].map(lambda x: f"{x:.4f}")
#     st.dataframe(
#         metrics_df_display.style.apply(
#             lambda s: ["background-color:#1a3a2a; color:#3fb950; font-weight:bold"
#                        if i == 0 else "" for i in range(len(s))],
#             axis=0
#         ),
#         use_container_width=True,
#         hide_index=False,
#     )


# # ═══════════════════════════════════════════════════════════════
# #  PAGE: CLASSIFICATION
# # ═══════════════════════════════════════════════════════════════
# elif page == "🔭  Classification":
#     st.title("🔭 Star Classification")
#     st.markdown("<p style='color:#8b949e'>Classify an astronomical observation as <b>Galaxy</b>, <b>QSO</b>, or <b>Star</b> using Sloan Digital Sky Survey (SDSS) photometric features.</p>", unsafe_allow_html=True)
#     st.markdown("---")

#     # --- load artifacts ---
#     try:
#         cls_scaler, cls_encoder, cls_feat_cols, cls_bounds = load_cls_artifacts()
#         cls_models = load_cls_models()
#     except FileNotFoundError:
#         st.error("⚠️ Artifacts not found. Please run `python save_artifacts.py` first.")
#         st.stop()

#     tab_pred, tab_compare, tab_info = st.tabs(["🎯 Predict", "📊 Model Comparison", "ℹ️ Feature Guide"])

#     # ── TAB: PREDICT ──────────────────────────────────────────
#     with tab_pred:
#         st.markdown("### Select Model")
#         chosen_model_name = st.selectbox(
#             "Classifier", list(cls_models.keys()),
#             help="Pick any of the 10 trained classifiers"
#         )
#         m = CLS_METRICS.get(chosen_model_name, {})
#         mc1, mc2, mc3 = st.columns(3)
#         mc1.metric("Accuracy",   f"{m.get('Accuracy',0):.4f}")
#         mc2.metric("Weighted F1", f"{m.get('Weighted F1',0):.4f}")
#         mc3.metric("ROC-AUC",   f"{m.get('ROC-AUC',0):.4f}")

#         st.markdown("---")
#         st.markdown("### Enter Observation Features")
#         st.markdown("<div class='info-banner'>💡 Hover the <b>?</b> icons for feature descriptions. Default values are dataset medians.</div>", unsafe_allow_html=True)

#         # Load data to get medians for defaults
#         df_cls = load_cls_data()
#         num_df = df_cls.drop(columns=["class"])

#         cols_per_row = 3
#         feat_names = cls_feat_cols
#         input_vals = {}
#         rows = [feat_names[i:i+cols_per_row] for i in range(0, len(feat_names), cols_per_row)]

#         for row_feats in rows:
#             cols = st.columns(cols_per_row)
#             for col, feat in zip(cols, row_feats):
#                 med_val = float(num_df[feat].median()) if feat in num_df else 0.0
#                 min_val = float(num_df[feat].min()) if feat in num_df else 0.0
#                 max_val = float(num_df[feat].max()) if feat in num_df else 1.0
#                 tooltip = FEATURE_TOOLTIPS.get(feat, feat)
#                 input_vals[feat] = col.number_input(
#                     f"{feat}", value=med_val,
#                     min_value=min_val - abs(min_val)*0.5,
#                     max_value=max_val + abs(max_val)*0.5,
#                     step=(max_val - min_val) / 200 if max_val != min_val else 0.01,
#                     help=tooltip,
#                     key=f"cls_{feat}"
#                 )

#         st.markdown("")
#         predict_btn = st.button("Predict Star Type", use_container_width=True)

#         if predict_btn:
#             model = cls_models[chosen_model_name]

#             # preprocess
#             raw = pd.DataFrame([input_vals])[feat_names]
#             for feat, bounds in cls_bounds.items():
#                 if feat in raw.columns:
#                     raw[feat] = raw[feat].clip(bounds["lower"], bounds["upper"])

#             if chosen_model_name in CLS_NEEDS_SCALE:
#                 X_in = cls_scaler.transform(raw)
#             else:
#                 X_in = raw.values

#             pred_enc = model.predict(X_in)[0]
#             pred_class = cls_encoder.inverse_transform([pred_enc])[0]

#             proba = None
#             if hasattr(model, "predict_proba"):
#                 proba = model.predict_proba(X_in)[0]

#             color = CLASS_COLORS.get(pred_class, "#58a6ff")
#             icon  = CLASS_ICONS.get(pred_class, "🔭")

#             st.markdown(f"""
# <div class='pred-box' style='border-color:{color}'>
#   <div class='pred-label'>Predicted Class</div>
#   <div class='pred-value' style='color:{color}'>{icon} {pred_class}</div>
#   <div class='pred-sub'>{CLASS_DESC.get(pred_class,"")}</div>
# </div>""", unsafe_allow_html=True)

#             if proba is not None:
#                 st.markdown("#### Confidence per Class")
#                 classes = cls_encoder.classes_
#                 prob_fig = go.Figure(go.Bar(
#                     x=classes,
#                     y=proba,
#                     marker_color=[CLASS_COLORS.get(c, "#58a6ff") for c in classes],
#                     text=[f"{p:.1%}" for p in proba],
#                     textposition="outside",
#                 ))
#                 prob_fig.update_layout(
#                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                     font_color="#e6edf3", height=320,
#                     yaxis=dict(range=[0, 1.15], gridcolor="#21262d", tickformat=".0%"),
#                     xaxis=dict(gridcolor="#21262d"),
#                     margin=dict(t=30, b=10),
#                 )
#                 st.plotly_chart(prob_fig, use_container_width=True, key="cls_proba")

#     # ── TAB: MODEL COMPARISON ────────────────────────────────
#     with tab_compare:
#         st.markdown("### 📊 All 10 Models — Performance Comparison")
#         metrics_df = pd.DataFrame(CLS_METRICS).T.reset_index().rename(columns={"index": "Model"})

#         metric_choice = st.selectbox("Metric to highlight", ["Accuracy", "Weighted F1", "ROC-AUC"], key="cls_metric_choice")
#         fig_bar = px.bar(
#             metrics_df.sort_values(metric_choice, ascending=True),
#             x=metric_choice, y="Model", orientation="h",
#             color=metric_choice,
#             color_continuous_scale=[[0, "#1a3a2a"], [0.5, "#238636"], [1, "#3fb950"]],
#             text=metric_choice,
#         )
#         fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
#         fig_bar.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#e6edf3", height=430,
#             xaxis=dict(range=[0.85, 1.01], gridcolor="#21262d"),
#             yaxis=dict(gridcolor="#21262d"),
#             coloraxis_showscale=False,
#             margin=dict(t=20, b=20),
#         )
#         st.plotly_chart(fig_bar, use_container_width=True, key="cls_bar")

#         st.markdown("#### Radar Chart — All Metrics")
#         categories = ["Accuracy", "Weighted F1", "ROC-AUC"]
#         radar_fig = go.Figure()
#         palette = px.colors.qualitative.Plotly
#         for idx, (model_name, row) in enumerate(CLS_METRICS.items()):
#             vals = [row[c] for c in categories]
#             vals_closed = vals + [vals[0]]
#             cats_closed  = categories + [categories[0]]
#             radar_fig.add_trace(go.Scatterpolar(
#                 r=vals_closed, theta=cats_closed,
#                 name=model_name, fill="toself",
#                 line_color=palette[idx % len(palette)],
#                 opacity=0.6,
#             ))
#         radar_fig.update_layout(
#             polar=dict(
#                 bgcolor="#161b22",
#                 radialaxis=dict(visible=True, range=[0.85, 1.0], gridcolor="#30363d", color="#8b949e"),
#                 angularaxis=dict(gridcolor="#30363d", color="#8b949e"),
#             ),
#             paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
#             height=440, legend=dict(font_size=11),
#             margin=dict(t=30, b=20),
#         )
#         st.plotly_chart(radar_fig, use_container_width=True, key="cls_radar")

#     # ── TAB: FEATURE GUIDE ───────────────────────────────────
#     with tab_info:
#         st.markdown("### ℹ️ Feature Reference")
#         rows = [(f, d) for f, d in FEATURE_TOOLTIPS.items()]
#         ftable = pd.DataFrame(rows, columns=["Feature", "Description"])
#         st.dataframe(ftable, use_container_width=True, hide_index=True, height=520)

#         st.markdown("---")
#         st.markdown("### 🔭 Dataset Overview")
#         df_cls = load_cls_data()
#         a, b = st.columns(2)
#         with a:
#             st.markdown("**Class Distribution**")
#             vc = df_cls["class"].value_counts().reset_index()
#             vc.columns = ["Class", "Count"]
#             fig_pie = px.pie(vc, names="Class", values="Count",
#                              color="Class", color_discrete_map=CLASS_COLORS, hole=0.5)
#             fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", height=300,
#                                   margin=dict(t=10, b=10))
#             st.plotly_chart(fig_pie, use_container_width=True, key="cls_info_pie")
#         with b:
#             st.markdown("**Summary Statistics**")
#             st.dataframe(df_cls.drop(columns=["class"]).describe().T.round(3), use_container_width=True, height=300)


# # ═══════════════════════════════════════════════════════════════
# #  PAGE: REGRESSION
# # ═══════════════════════════════════════════════════════════════
# elif page == "📈  Regression":
#     st.title("📈 Community Crime Rate Predictor")
#     st.markdown("<p style='color:#8b949e'>Predict the violent crime rate per capita for a US community using socioeconomic and demographic features (Communities & Crime dataset).</p>", unsafe_allow_html=True)
#     st.markdown("---")

#     # load
#     try:
#         reg_imputer, reg_scaler, reg_feat_cols, reg_bounds = load_reg_artifacts()
#     except FileNotFoundError:
#         st.error("⚠️ Artifacts not found. Please run `python save_artifacts.py` first.")
#         st.stop()

#     df_reg, target_reg = load_reg_data()

#     REG_METRICS = {
#         "Linear Regression":     {"R²": 0.6830, "RMSE": 0.1307, "MAE": 0.0949},
#         "Ridge Regression":      {"R²": 0.6820, "RMSE": 0.1309, "MAE": 0.0950},
#         "Lasso Regression":      {"R²": 0.6793, "RMSE": 0.1314, "MAE": 0.0939},
#         "ElasticNet Regression": {"R²": 0.6811, "RMSE": 0.1310, "MAE": 0.0945},
#         "Decision Tree":         {"R²": 0.3388, "RMSE": 0.1887, "MAE": 0.1289},
#         "Random Forest":         {"R²": 0.6672, "RMSE": 0.1339, "MAE": 0.0957},
#         "Gradient Boosting":     {"R²": 0.6830, "RMSE": 0.1307, "MAE": 0.0902},
#         "SVR":                   {"R²": 0.6261, "RMSE": 0.1419, "MAE": 0.1034},
#         "KNN":                   {"R²": 0.5983, "RMSE": 0.1471, "MAE": 0.0994},
#     }

#     # Models in the regression track that need scaled data
#     REG_NEEDS_SCALE = {"Linear Regression", "Ridge Regression", "Lasso Regression",
#                        "ElasticNet Regression", "SVR", "KNN"}

#     tab_pred_r, tab_compare_r, tab_data_r = st.tabs(["🎯 Predict", "📊 Model Comparison", "📋 Dataset"])

#     with tab_pred_r:
#         st.markdown("### Select Model")
#         reg_model_name = st.selectbox("Regressor", list(REG_METRICS.keys()), key="reg_model_select")

#         m_r = REG_METRICS[reg_model_name]
#         rc1, rc2, rc3 = st.columns(3)
#         rc1.metric("R²",   f"{m_r['R²']:.4f}")
#         rc2.metric("RMSE", f"{m_r['RMSE']:.4f}")
#         rc3.metric("MAE",  f"{m_r['MAE']:.4f}")

#         st.markdown("---")
#         st.markdown("### Enter Community Features")
#         st.markdown("<div class='info-banner'>💡 All features are normalised to [0, 1] in the original dataset. Sliders reflect actual data range.</div>", unsafe_allow_html=True)

#         # top 20 most correlated features for a simpler UX
#         feature_subset = [f for f in reg_feat_cols if f in df_reg.columns]
#         corr_with_tgt = df_reg[feature_subset + [target_reg]].corr()[target_reg].drop(target_reg).abs()
#         top_feats = corr_with_tgt.sort_values(ascending=False).head(20).index.tolist()

#         st.markdown(f"*Showing top 20 most predictive features (of {len(reg_feat_cols)} total). Remaining features use dataset medians.*")

#         input_r = {}
#         cols_r = st.columns(2)
#         for idx, feat in enumerate(top_feats):
#             col = cols_r[idx % 2]
#             med_v = float(df_reg[feat].median())
#             min_v = float(df_reg[feat].min())
#             max_v = float(df_reg[feat].max())
#             step_v = round((max_v - min_v) / 100, 4) if max_v != min_v else 0.01
#             input_r[feat] = col.slider(
#                 feat, min_value=min_v, max_value=max_v,
#                 value=med_v, step=step_v, key=f"reg_{feat}"
#             )

#         predict_r_btn = st.button("Predict Crime Rate", use_container_width=True)

#         if predict_r_btn:
#             # build full feature vector (missing → median)
#             full_input = {}
#             for feat in reg_feat_cols:
#                 if feat in input_r:
#                     full_input[feat] = input_r[feat]
#                 elif feat in df_reg.columns:
#                     full_input[feat] = float(df_reg[feat].median())
#                 else:
#                     full_input[feat] = 0.0

#             raw_r = pd.DataFrame([full_input])[reg_feat_cols]

#             # clip
#             for feat in reg_feat_cols:
#                 if feat in reg_bounds["lower"].index:
#                     raw_r[feat] = raw_r[feat].clip(reg_bounds["lower"][feat], reg_bounds["upper"][feat])

#             # impute
#             raw_r_imp = pd.DataFrame(
#                 reg_imputer.transform(raw_r), columns=reg_feat_cols
#             )

#             # scale if needed
#             if reg_model_name in REG_NEEDS_SCALE:
#                 X_r_in = reg_scaler.transform(raw_r_imp)
#             else:
#                 X_r_in = raw_r_imp.values

#             # predict using saved model if available else show demo
#             try:
#                 import glob
#                 # try to load any regression model pkl matching name
#                 name_map = {
#                     "Linear Regression":     "linear",
#                     "Ridge Regression":      "ridge",
#                     "Lasso Regression":      "lasso",
#                     "ElasticNet Regression": "elasticnet",
#                     "Decision Tree":         "decision_tree",
#                     "Random Forest":         "random_forest",
#                     "Gradient Boosting":     "gradient_boosting",
#                     "SVR":                   "svr",
#                     "KNN":                   "knn",
#                 }
#                 key = name_map.get(reg_model_name, "").lower()
#                 pkl_files = glob.glob(os.path.join(REG_MODELS, f"*{key}*.pkl"))
#                 # filter out artifact files
#                 pkl_files = [p for p in pkl_files if not any(x in p for x in ["scaler", "imputer", "feature", "clip"])]
#                 if pkl_files:
#                     reg_model_obj = joblib.load(pkl_files[0])
#                     pred_val = float(reg_model_obj.predict(X_r_in)[0])
#                 else:
#                     raise FileNotFoundError
#             except Exception:
#                 # fall back: estimate from model metrics
#                 pred_val = float(df_reg[target_reg].mean())
#                 st.info("ℹ️ Model .pkl not found for regression — showing dataset mean as demo. Re-run the regression notebook and save models to enable live predictions.")

#             pred_pct = pred_val * 100
#             level = "🟢 Low" if pred_val < 0.25 else ("🟡 Moderate" if pred_val < 0.5 else "🔴 High")

#             st.markdown(f"""
# <div class='pred-box'>
#   <div class='pred-label'>Predicted Violent Crimes Per Population</div>
#   <div class='pred-value'>{pred_val:.4f}</div>
#   <div class='pred-sub'>≈ {pred_pct:.2f}% · Risk Level: {level}</div>
# </div>""", unsafe_allow_html=True)

#             # gauge chart
#             gauge = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=pred_val,
#                 number={"valueformat": ".4f", "font": {"color": "#58a6ff", "size": 36}},
#                 gauge={
#                     "axis": {"range": [0, 1], "tickcolor": "#8b949e"},
#                     "bar": {"color": "#388bfd"},
#                     "steps": [
#                         {"range": [0, 0.25],  "color": "#1a3a2a"},
#                         {"range": [0.25, 0.5], "color": "#3a2a10"},
#                         {"range": [0.5, 1.0],  "color": "#3a1010"},
#                     ],
#                     "threshold": {"line": {"color": "#f78166", "width": 3}, "thickness": 0.75, "value": pred_val},
#                     "bgcolor": "#161b22",
#                     "bordercolor": "#30363d",
#                 },
#             ))
#             gauge.update_layout(
#                 paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
#                 height=280, margin=dict(t=20, b=20),
#             )
#             st.plotly_chart(gauge, use_container_width=True, key="reg_gauge")

#     with tab_compare_r:
#         st.markdown("### 📊 Regression Model Comparison")
#         reg_m_df = pd.DataFrame(REG_METRICS).T.reset_index().rename(columns={"index": "Model"})

#         reg_metric_c = st.selectbox("Sort by metric", ["R²", "RMSE", "MAE"], key="reg_metric_c")
#         asc = reg_metric_c in ["RMSE", "MAE"]
#         fig_reg_bar = px.bar(
#             reg_m_df.sort_values(reg_metric_c, ascending=asc),
#             x=reg_metric_c, y="Model", orientation="h",
#             color=reg_metric_c,
#             color_continuous_scale=[[0, "#1a3a2a"], [0.5, "#238636"], [1, "#3fb950"]] if not asc
#                                    else [[0, "#3fb950"], [0.5, "#238636"], [1, "#1a3a2a"]],
#             text=reg_metric_c,
#         )
#         fig_reg_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
#         fig_reg_bar.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#e6edf3", height=400,
#             xaxis=dict(gridcolor="#21262d"),
#             yaxis=dict(gridcolor="#21262d"),
#             coloraxis_showscale=False,
#             margin=dict(t=20, b=20),
#         )
#         st.plotly_chart(fig_reg_bar, use_container_width=True, key="reg_bar")

#         st.markdown("#### Full Metrics Table")
#         st.dataframe(reg_m_df.sort_values("R²", ascending=False).set_index("Model").round(4),
#                      use_container_width=True)

#     with tab_data_r:
#         st.markdown("### 📋 Regression Dataset Explorer")
#         st.markdown(f"**Target:** `{target_reg}` — violent crimes per 100K population (normalised 0–1)")
#         r1, r2, r3 = st.columns(3)
#         r1.metric("Rows", f"{len(df_reg):,}")
#         r2.metric("Features (after cleaning)", f"{df_reg.shape[1]-1}")
#         r3.metric("Missing Values", "0 (imputed)")

#         st.markdown("**Top 10 Features Correlated with Crime Rate**")
#         feat_cols_reg_valid = [f for f in reg_feat_cols if f in df_reg.columns]
#         corr = df_reg[feat_cols_reg_valid + [target_reg]].corr()[target_reg].drop(target_reg)
#         top_corr = corr.abs().sort_values(ascending=False).head(10)

#         fig_corr = px.bar(
#             x=corr.loc[top_corr.index].values,
#             y=top_corr.index,
#             orientation="h",
#             color=corr.loc[top_corr.index].values,
#             color_continuous_scale="RdBu_r",
#             labels={"x": f"Correlation with {target_reg}", "y": "Feature"},
#         )
#         fig_corr.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#e6edf3", height=380,
#             xaxis=dict(gridcolor="#21262d"),
#             yaxis=dict(gridcolor="#21262d"),
#             coloraxis_showscale=False,
#             margin=dict(t=20, b=20),
#         )
#         st.plotly_chart(fig_corr, use_container_width=True, key="reg_corr_bar")


# # ═══════════════════════════════════════════════════════════════
# #  PAGE: CLUSTERING
# # ═══════════════════════════════════════════════════════════════
# elif page == "🔵  Clustering":
#     st.title("🔵 Clustering")
#     st.markdown("<p style='color:#8b949e'>Credit Card Customer Segmentation (notebook model) · Interactive Explorer on any dataset.</p>", unsafe_allow_html=True)
#     st.markdown("---")

#     tab_cc, tab_explore = st.tabs(["💳 Credit Card Segmentation", "🔬 Interactive Explorer"])

#     # ────────────────────────────────────────────────────────────
#     #  TAB 1 — CREDIT CARD SEGMENTATION (saved notebook model)
#     # ────────────────────────────────────────────────────────────
#     with tab_cc:
#         try:
#             clust_prep, clust_km, clust_centroids, clust_labels_df, clust_meta = load_cluster_artifacts()
#             cc_data = load_cc_data()
#         except FileNotFoundError as e:
#             st.error(f"Clustering artifacts not found: {e}. Run the last cell of clustering.ipynb first.")
#             st.stop()

#         RAW_FEATURES_CC = clust_meta["raw_features"]
#         ALL_FEATURES_CC = clust_meta["feature_names"]
#         K = clust_meta["k"]
#         CLUSTER_COLORS  = px.colors.qualitative.Plotly
#         CLUSTER_NAMES   = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2", 3: "Cluster 3"}

#         # ── Top metrics row ──────────────────────────────────
#         st.markdown("### 📊 Model Overview")
#         m1, m2, m3, m4, m5 = st.columns(5)
#         m1.metric("Algorithm",         "K-Means")
#         m2.metric("Clusters (k)",      str(K))
#         m3.metric("Customers",         f"{clust_meta['n_samples']:,}")
#         m4.metric("Silhouette ↑",      f"{clust_meta['metrics']['Silhouette']:.4f}")
#         m5.metric("Davies-Bouldin ↓",  f"{clust_meta['metrics']['Davies-Bouldin']:.4f}")

#         st.markdown("---")

#         # ── Tabs within the CC tab ───────────────────────────
#         st_cc1, st_cc2, st_cc3, st_cc4 = st.tabs(["📈 Cluster Profiles", "🗺️ PCA Scatter", "📋 Cluster Sizes", "🎯 Predict Cluster"])

#         # ── Cluster Profiles (centroid radar / bar) ──────────
#         with st_cc1:
#             st.markdown("### Cluster Feature Profiles (Scaled Centroids)")
#             st.markdown("<div class='info-banner'>Values are in the scaled (standardised) feature space. Positive = above average; Negative = below average.</div>", unsafe_allow_html=True)

#             # Select features to display
#             feat_show = st.multiselect(
#                 "Features to compare",
#                 ALL_FEATURES_CC,
#                 default=["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
#                          "PAYMENTS", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY"],
#                 key="cc_profile_feats"
#             )
#             if feat_show:
#                 prof_df = clust_centroids[feat_show].copy()
#                 prof_df.index = [f"Cluster {i}" for i in prof_df.index]
#                 prof_melt = prof_df.reset_index().melt(id_vars="index", var_name="Feature", value_name="Scaled Value")
#                 prof_melt.rename(columns={"index": "Cluster"}, inplace=True)

#                 fig_prof = px.bar(
#                     prof_melt, x="Feature", y="Scaled Value", color="Cluster",
#                     barmode="group",
#                     color_discrete_sequence=CLUSTER_COLORS,
#                     title="Centroid Profiles — Scaled Feature Values by Cluster",
#                 )
#                 fig_prof.update_layout(
#                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                     font_color="#e6edf3", height=480,
#                     xaxis=dict(gridcolor="#21262d", tickangle=-30),
#                     yaxis=dict(gridcolor="#21262d", title="Scaled value (std devs)"),
#                     legend_title_text="Cluster",
#                     margin=dict(t=50, b=60),
#                 )
#                 st.plotly_chart(fig_prof, use_container_width=True, key="cc_profile_bar")

#                 st.markdown("**Centroid table (scaled)**")
#                 st.dataframe(clust_centroids[feat_show].round(4), use_container_width=True)

#         # ── PCA Scatter ──────────────────────────────────────
#         with st_cc2:
#             st.markdown("### PCA 2D Projection")
#             pca_n = st.radio("Dimensions", [2, 3], horizontal=True, key="cc_pca_n")
#             sample_n_cc = st.slider("Sample size for plot", 500, len(cc_data), min(3000, len(cc_data)), step=500, key="cc_pca_sample")

#             with st.spinner("Computing PCA…"):
#                 # use the fitted preprocessor to transform the raw data
#                 X_cc_scaled = clust_prep.transform(cc_data[RAW_FEATURES_CC])
#                 pca_viz = PCA(n_components=pca_n, random_state=42)
#                 X_pca_cc = pca_viz.fit_transform(X_cc_scaled)

#             labels_all = clust_labels_df.iloc[:, 0].values
#             idx_sample = np.random.default_rng(42).choice(len(labels_all), size=min(sample_n_cc, len(labels_all)), replace=False)

#             if pca_n == 2:
#                 sc_df = pd.DataFrame({
#                     "PC1": X_pca_cc[idx_sample, 0],
#                     "PC2": X_pca_cc[idx_sample, 1],
#                     "Cluster": [f"Cluster {l}" for l in labels_all[idx_sample]],
#                 })
#                 fig_pca = px.scatter(
#                     sc_df, x="PC1", y="PC2", color="Cluster",
#                     color_discrete_sequence=CLUSTER_COLORS,
#                     title=f"K-Means k={K} — PCA 2D (sample n={len(idx_sample):,})",
#                     opacity=0.65,
#                 )
#             else:
#                 sc_df = pd.DataFrame({
#                     "PC1": X_pca_cc[idx_sample, 0],
#                     "PC2": X_pca_cc[idx_sample, 1],
#                     "PC3": X_pca_cc[idx_sample, 2],
#                     "Cluster": [f"Cluster {l}" for l in labels_all[idx_sample]],
#                 })
#                 fig_pca = px.scatter_3d(
#                     sc_df, x="PC1", y="PC2", z="PC3", color="Cluster",
#                     color_discrete_sequence=CLUSTER_COLORS,
#                     title=f"K-Means k={K} — PCA 3D (sample n={len(idx_sample):,})",
#                     opacity=0.65,
#                 )

#             evr = pca_viz.explained_variance_ratio_ * 100
#             fig_pca.update_layout(
#                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                 font_color="#e6edf3", height=540,
#                 legend_title_text="Cluster",
#                 margin=dict(t=50, b=20),
#             )
#             st.plotly_chart(fig_pca, use_container_width=True, key="cc_pca_scatter")
#             st.markdown(f"**PCA Explained Variance:** {' + '.join([f'PC{i+1}: {v:.1f}%' for i, v in enumerate(evr)])} = **{sum(evr):.1f}%** total")

#         # ── Cluster Sizes ────────────────────────────────────
#         with st_cc3:
#             st.markdown("### Cluster Size Distribution")
#             sizes = clust_meta["cluster_sizes"]
#             size_df = pd.DataFrame({
#                 "Cluster": [f"Cluster {k}" for k in sizes],
#                 "Customers": list(sizes.values()),
#             })
#             size_df["Percent"] = (size_df["Customers"] / size_df["Customers"].sum() * 100).round(1)

#             fig_sz2 = px.bar(
#                 size_df, x="Cluster", y="Customers",
#                 color="Cluster",
#                 color_discrete_sequence=CLUSTER_COLORS,
#                 text=size_df["Percent"].apply(lambda x: f"{x}%"),
#                 title="Number of Customers per Cluster",
#             )
#             fig_sz2.update_traces(textposition="outside")
#             fig_sz2.update_layout(
#                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                 font_color="#e6edf3", height=380,
#                 showlegend=False,
#                 xaxis=dict(gridcolor="#21262d"),
#                 yaxis=dict(gridcolor="#21262d"),
#                 margin=dict(t=50, b=20),
#             )
#             st.plotly_chart(fig_sz2, use_container_width=True, key="cc_size_bar")
#             st.dataframe(size_df.set_index("Cluster"), use_container_width=True)

#         # ── Predict Cluster for New Customer ─────────────────
#         with st_cc4:
#             st.markdown("### 🎯 Predict Cluster for a New Customer")
#             st.markdown("<div class='info-banner'>Enter raw credit card usage values. The notebook's fitted preprocessor (engineer → impute → scale) will be applied automatically before prediction.</div>", unsafe_allow_html=True)

#             med_vals = cc_data[RAW_FEATURES_CC].median()
#             input_cc = {}
#             cols_cc = st.columns(3)
#             for idx, feat in enumerate(RAW_FEATURES_CC):
#                 col_w = cols_cc[idx % 3]
#                 min_v = float(cc_data[feat].min())
#                 max_v = float(cc_data[feat].max())
#                 med_v = float(med_vals[feat])
#                 step_v = round((max_v - min_v) / 100, 4) if max_v != min_v else 0.01
#                 input_cc[feat] = col_w.number_input(
#                     feat, min_value=min_v, max_value=max_v,
#                     value=med_v, step=step_v, key=f"cc_inp_{feat}"
#                 )

#             predict_cc_btn = st.button("🔍 Predict Customer Segment", use_container_width=True, key="cc_predict_btn")

#             if predict_cc_btn:
#                 row_df = pd.DataFrame([input_cc])[RAW_FEATURES_CC]
#                 X_in   = clust_prep.transform(row_df)
#                 cluster_id = int(clust_km.predict(X_in)[0])

#                 # distance to all centroids (in scaled space)
#                 dists = np.linalg.norm(clust_km.cluster_centers_ - X_in, axis=1)
#                 confidence = 1.0 - (dists[cluster_id] / dists.sum())

#                 st.markdown(f"""
# <div class='pred-box'>
#   <div class='pred-label'>Predicted Customer Segment</div>
#   <div class='pred-value' style='color:{CLUSTER_COLORS[cluster_id]}'>Cluster {cluster_id}</div>
#   <div class='pred-sub'>Confidence score: {confidence:.1%} &nbsp;|&nbsp; Distance to centroid: {dists[cluster_id]:.4f}</div>
# </div>""", unsafe_allow_html=True)

#                 st.markdown("#### Centroid Profile of Predicted Cluster")
#                 cent_row = clust_centroids.loc[cluster_id]
#                 top10 = cent_row.abs().sort_values(ascending=False).head(10).index.tolist()
#                 cent_plot = cent_row[top10].reset_index()
#                 cent_plot.columns = ["Feature", "Scaled Value"]
#                 cent_plot["Direction"] = cent_plot["Scaled Value"].apply(lambda x: "Above Avg" if x >= 0 else "Below Avg")

#                 fig_cent = px.bar(
#                     cent_plot, x="Feature", y="Scaled Value", color="Direction",
#                     color_discrete_map={"Above Avg": "#3fb950", "Below Avg": "#f78166"},
#                     title=f"Top 10 Distinguishing Features — Cluster {cluster_id}",
#                 )
#                 fig_cent.update_layout(
#                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                     font_color="#e6edf3", height=380,
#                     xaxis=dict(gridcolor="#21262d", tickangle=-25),
#                     yaxis=dict(gridcolor="#21262d"),
#                     margin=dict(t=50, b=60),
#                 )
#                 st.plotly_chart(fig_cent, use_container_width=True, key="cc_cent_bar")

#     # ────────────────────────────────────────────────────────────
#     #  TAB 2 — INTERACTIVE EXPLORER
#     # ────────────────────────────────────────────────────────────
#     with tab_explore:
#         st.markdown("### 🔬 Run clustering on any dataset interactively")
#         c_left, c_right = st.columns([1, 2])

#         with c_left:
#             st.markdown("#### ⚙️ Configuration")
#             dataset_choice = st.selectbox("Dataset", ["Star Classification (SDSS)", "Community Crime"], key="clust_dataset")
#             algo_choice    = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Agglomerative"], key="clust_algo")

#             if algo_choice == "K-Means":
#                 k_val       = st.slider("Number of clusters (k)", 2, 15, 3, key="clust_k")
#                 init_method = st.selectbox("Init method", ["k-means++", "random"], key="clust_init")
#                 max_iter    = st.slider("Max iterations", 100, 500, 300, step=50, key="clust_maxiter")
#             elif algo_choice == "DBSCAN":
#                 eps_val  = st.slider("ε (epsilon)", 0.1, 5.0, 0.5, step=0.05, key="clust_eps")
#                 min_samp = st.slider("min_samples", 2, 30, 5, key="clust_minsamp")
#                 st.markdown("<div class='info-banner'>Noise points labelled <b>-1</b>.</div>", unsafe_allow_html=True)
#             elif algo_choice == "Agglomerative":
#                 agg_k     = st.slider("Number of clusters", 2, 15, 3, key="clust_agg_k")
#                 linkage_m = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="clust_link")

#             st.markdown("---")
#             n_pca_comp  = st.slider("PCA components for vis", 2, 3, 2, key="clust_pca")
#             sample_frac = st.slider("Data sample fraction", 0.05, 1.0, 0.30, step=0.05, key="clust_frac")
#             run_btn     = st.button("▶ Run Clustering", use_container_width=True, key="clust_run_btn")
#             st.markdown("---")
#             show_elbow  = st.checkbox("Elbow curve (K-Means 2–12)", value=False, key="clust_elbow")

#         with c_right:
#             if run_btn or show_elbow:
#                 with st.spinner("Loading & preprocessing data…"):
#                     if "Star" in dataset_choice:
#                         df_raw_ex = load_cls_data()
#                         X_raw_ex  = df_raw_ex.drop(columns=["class"], errors="ignore")
#                         y_true_ex = df_raw_ex["class"].values if "class" in df_raw_ex.columns else None
#                     else:
#                         df_raw_ex, tgt_r_ex = load_reg_data()
#                         X_raw_ex  = df_raw_ex.drop(columns=[tgt_r_ex], errors="ignore")
#                         y_true_ex = None

#                     X_raw_ex = X_raw_ex.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
#                     if y_true_ex is not None and "Star" in dataset_choice:
#                         y_true_ex = load_cls_data()["class"].sample(frac=sample_frac, random_state=42).values

#                     X_num_ex = X_raw_ex.select_dtypes(include=[np.number]).fillna(
#                         X_raw_ex.select_dtypes(include=[np.number]).median())
#                     sc_ex     = SkScaler()
#                     X_sc_ex   = sc_ex.fit_transform(X_num_ex)
#                     pca_ex    = PCA(n_components=n_pca_comp, random_state=42)
#                     X_pca_ex  = pca_ex.fit_transform(X_sc_ex)

#                 if show_elbow and algo_choice == "K-Means":
#                     with st.spinner("Computing elbow curve…"):
#                         inertias_ex, sil_ex = [], []
#                         k_range_ex = range(2, 13)
#                         for k in k_range_ex:
#                             km_e = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=42)
#                             lb_e = km_e.fit_predict(X_sc_ex)
#                             inertias_ex.append(km_e.inertia_)
#                             sil_ex.append(silhouette_score(X_sc_ex, lb_e, sample_size=min(1000, len(X_sc_ex))))

#                         fig_el = make_subplots(rows=1, cols=2, subplot_titles=["Inertia (Elbow)", "Silhouette"])
#                         fig_el.add_trace(go.Scatter(x=list(k_range_ex), y=inertias_ex, mode="lines+markers",
#                                                      marker_color="#388bfd", name="Inertia"), row=1, col=1)
#                         fig_el.add_trace(go.Scatter(x=list(k_range_ex), y=sil_ex, mode="lines+markers",
#                                                      marker_color="#3fb950", name="Silhouette"), row=1, col=2)
#                         fig_el.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                                               font_color="#e6edf3", height=340,
#                                               xaxis=dict(gridcolor="#21262d", title="k"),
#                                               xaxis2=dict(gridcolor="#21262d", title="k"),
#                                               yaxis=dict(gridcolor="#21262d"),
#                                               yaxis2=dict(gridcolor="#21262d"),
#                                               margin=dict(t=40, b=20), showlegend=False)
#                         st.plotly_chart(fig_el, use_container_width=True, key="clust_elbow_fig")

#                 if run_btn:
#                     with st.spinner(f"Running {algo_choice}…"):
#                         if algo_choice == "K-Means":
#                             model_c  = KMeans(n_clusters=k_val, init=init_method, max_iter=max_iter,
#                                               n_init=10, random_state=42)
#                             labels_c = model_c.fit_predict(X_sc_ex)
#                         elif algo_choice == "DBSCAN":
#                             model_c  = DBSCAN(eps=eps_val, min_samples=min_samp, n_jobs=-1)
#                             labels_c = model_c.fit_predict(X_sc_ex)
#                         elif algo_choice == "Agglomerative":
#                             model_c  = AgglomerativeClustering(n_clusters=agg_k, linkage=linkage_m)
#                             labels_c = model_c.fit_predict(X_sc_ex)

#                     n_found = len(set(labels_c)) - (1 if -1 in labels_c else 0)
#                     n_noise = (labels_c == -1).sum()
#                     try:
#                         sil = silhouette_score(X_sc_ex, labels_c,
#                                                sample_size=min(2000, len(X_sc_ex))) if n_found > 1 else float("nan")
#                         db  = davies_bouldin_score(X_sc_ex, labels_c) if n_found > 1 else float("nan")
#                     except Exception:
#                         sil, db = float("nan"), float("nan")

#                     em1, em2, em3, em4 = st.columns(4)
#                     em1.metric("Clusters Found",  str(n_found))
#                     em2.metric("Noise Points",    str(n_noise))
#                     em3.metric("Silhouette ↑",    f"{sil:.4f}" if not np.isnan(sil) else "N/A")
#                     em4.metric("Davies-Bouldin ↓", f"{db:.4f}" if not np.isnan(db) else "N/A")

#                     st.markdown("---")
#                     labels_str_ex = [str(l) for l in labels_c]
#                     if n_pca_comp == 2:
#                         sc_df_ex = pd.DataFrame({"PC1": X_pca_ex[:, 0], "PC2": X_pca_ex[:, 1], "Cluster": labels_str_ex})
#                         if y_true_ex is not None and "Star" in dataset_choice:
#                             sc_df_ex["True Class"] = y_true_ex
#                         fig_sc_ex = px.scatter(
#                             sc_df_ex, x="PC1", y="PC2", color="Cluster",
#                             hover_data=["True Class"] if "True Class" in sc_df_ex else None,
#                             title=f"{algo_choice} (PCA 2D) — {n_found} clusters",
#                             color_discrete_sequence=px.colors.qualitative.Plotly,
#                         )
#                     else:
#                         sc_df_ex = pd.DataFrame({"PC1": X_pca_ex[:, 0], "PC2": X_pca_ex[:, 1],
#                                                   "PC3": X_pca_ex[:, 2], "Cluster": labels_str_ex})
#                         fig_sc_ex = px.scatter_3d(
#                             sc_df_ex, x="PC1", y="PC2", z="PC3", color="Cluster",
#                             title=f"{algo_choice} (PCA 3D) — {n_found} clusters",
#                             color_discrete_sequence=px.colors.qualitative.Plotly, opacity=0.7,
#                         )

#                     fig_sc_ex.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                         font_color="#e6edf3", height=520, legend_title_text="Cluster",
#                         margin=dict(t=50, b=20),
#                     )
#                     st.plotly_chart(fig_sc_ex, use_container_width=True, key="clust_scatter_ex")

#                     st.markdown("#### Cluster Size Distribution")
#                     sz_ex = pd.Series(labels_c).value_counts().sort_index().reset_index()
#                     sz_ex.columns = ["Cluster", "Count"]
#                     sz_ex["Cluster"] = sz_ex["Cluster"].astype(str)
#                     fig_sz_ex = px.bar(sz_ex, x="Cluster", y="Count",
#                                        color="Cluster", text="Count",
#                                        color_discrete_sequence=px.colors.qualitative.Plotly)
#                     fig_sz_ex.update_traces(textposition="outside")
#                     fig_sz_ex.update_layout(
#                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                         font_color="#e6edf3", height=320, showlegend=False,
#                         xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
#                         margin=dict(t=20, b=20),
#                     )
#                     st.plotly_chart(fig_sz_ex, use_container_width=True, key="clust_size_ex")
#                     evr_ex = pca_ex.explained_variance_ratio_ * 100
#                     st.markdown(f"**PCA Explained Variance:** {' + '.join([f'PC{i+1}: {v:.1f}%' for i,v in enumerate(evr_ex)])} = **{sum(evr_ex):.1f}%** total")

#             else:
#                 st.markdown("""
# <div style='background:#161b22;border:1px dashed #30363d;border-radius:14px;
#             padding:60px 40px;text-align:center;color:#8b949e'>
#   <div style='font-size:3rem'>🔬</div>
#   <h3 style='color:#58a6ff'>Configure & Run Clustering</h3>
#   <p>Choose a dataset, algorithm, and parameters on the left,<br>then click <b>▶ Run Clustering</b>.</p>
# </div>""", unsafe_allow_html=True)


# # ═══════════════════════════════════════════════════════════════
# #  PAGE: EDA EXPLORER
# # ═══════════════════════════════════════════════════════════════
# elif page == "📊  EDA Explorer":
#     st.title("📊 EDA Explorer")
#     st.markdown("<p style='color:#8b949e'>Interactively explore both datasets — distributions, correlations, scatter plots, and summary statistics.</p>", unsafe_allow_html=True)
#     st.markdown("---")

#     dataset_eda = st.radio("Select Dataset", ["🔭 Star Classification", "📈 Community Crime"], horizontal=True, key="eda_ds")

#     if "Star" in dataset_eda:
#         df_e = load_cls_data()
#         num_cols = df_e.select_dtypes(include=[np.number]).columns.tolist()
#         cat_col  = "class"
#         color_map = CLASS_COLORS
#         has_target_cat = True
#         target_e = None
#     else:
#         df_e, target_e = load_reg_data()
#         num_cols = df_e.select_dtypes(include=[np.number]).columns.tolist()
#         cat_col  = None
#         color_map = None
#         has_target_cat = False

#     tab_dist, tab_corr, tab_scatter, tab_raw = st.tabs(["📉 Distributions", "🌡️ Correlation", "🔀 Scatter", "🗃️ Raw Data"])

#     with tab_dist:
#         st.markdown("### Feature Distributions")
#         feat_sel = st.multiselect("Select features to plot", num_cols,
#                                   default=num_cols[:min(6, len(num_cols))], key="eda_dist_feats")
#         if feat_sel:
#             n_col_d = 3
#             rows_d  = [feat_sel[i:i+n_col_d] for i in range(0, len(feat_sel), n_col_d)]
#             for row_f in rows_d:
#                 c_d = st.columns(n_col_d)
#                 for col_d, feat in zip(c_d, row_f):
#                     with col_d:
#                         fig_d = px.histogram(df_e, x=feat, nbins=40,
#                                              color=cat_col if has_target_cat else None,
#                                              color_discrete_map=color_map,
#                                              marginal="box",
#                                              title=feat)
#                         fig_d.update_layout(
#                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                             font_color="#e6edf3", height=300,
#                             margin=dict(t=30, b=10, l=10, r=10),
#                             showlegend=False,
#                             xaxis=dict(gridcolor="#21262d"),
#                             yaxis=dict(gridcolor="#21262d"),
#                         )
#                         st.plotly_chart(fig_d, use_container_width=True, key=f"eda_dist_{feat}")

#     with tab_corr:
#         st.markdown("### Correlation Heatmap")
#         max_feats = st.slider("Max features in heatmap", 5, min(40, len(num_cols)), 15, key="eda_corr_n")

#         # sort by variance for most-informative subset
#         top_var_feats = df_e[num_cols].var().sort_values(ascending=False).head(max_feats).index.tolist()
#         corr_m = df_e[top_var_feats].corr()

#         fig_hm = px.imshow(
#             corr_m, text_auto=".2f", aspect="auto",
#             color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
#         )
#         fig_hm.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#e6edf3", height=600,
#             coloraxis_colorbar=dict(tickcolor="#e6edf3", title="r"),
#             margin=dict(t=30, b=20),
#         )
#         st.plotly_chart(fig_hm, use_container_width=True, key="eda_heatmap")

#     with tab_scatter:
#         st.markdown("### Feature Scatter Plot")
#         sc1, sc2 = st.columns(2)
#         x_feat = sc1.selectbox("X axis", num_cols, index=0, key="eda_sc_x")
#         y_feat = sc2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="eda_sc_y")
#         sample_n = st.slider("Sample size", 500, min(10000, len(df_e)), 3000, step=500, key="eda_sc_n")

#         df_sample = df_e.sample(n=min(sample_n, len(df_e)), random_state=42)
#         fig_sc2 = px.scatter(
#             df_sample, x=x_feat, y=y_feat,
#             color=cat_col if has_target_cat else target_e,
#             color_discrete_map=color_map,
#             color_continuous_scale="Viridis" if not has_target_cat else None,
#             opacity=0.6,
#             title=f"{x_feat} vs {y_feat}",
#             hover_data=[cat_col] if has_target_cat else [target_e],
#         )
#         fig_sc2.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             font_color="#e6edf3", height=500,
#             xaxis=dict(gridcolor="#21262d"),
#             yaxis=dict(gridcolor="#21262d"),
#             margin=dict(t=40, b=20),
#         )
#         st.plotly_chart(fig_sc2, use_container_width=True, key="eda_scatter2")

#     with tab_raw:
#         st.markdown("### Raw Dataset Preview")
#         n_rows = st.slider("Rows to display", 10, 200, 50, step=10, key="eda_raw_n")
#         search_col = st.selectbox("Filter column", [None] + df_e.columns.tolist(), key="eda_raw_search")
#         df_disp = df_e.head(n_rows)
#         if search_col:
#             st.dataframe(df_disp[[search_col] + [c for c in df_disp.columns if c != search_col]],
#                          use_container_width=True, height=440)
#         else:
#             st.dataframe(df_disp, use_container_width=True, height=440)

#         st.markdown("**Summary Statistics**")
#         st.dataframe(df_e.describe().T.round(4), use_container_width=True)



"""
Vectora – Unified ML Dashboard (Streamlit)
==========================================
Covers:
  • Classification  – Star Type (Galaxy / QSO / Star)
  • Regression      – Community Crime Rate
  • Clustering      – Credit-card customer segmentation (K-Means / Ward) from saved models
  • EDA Explorer    – Dataset exploration for all three tracks
"""

import os, sys, json, warnings
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
st.markdown(r"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
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

  /* ═════════ Global polish ═════════ */
  .stApp { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
  .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1280px; }
  h1 { font-weight: 800 !important; letter-spacing: -0.01em; }
  hr { margin: 1.1rem 0; }
  [data-testid="stHeader"] { background: transparent; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; padding: 4px; border: 1px solid #21262d; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; }
  .stTabs [data-baseweb="tab-highlight"] { background-color: #58a6ff !important; }

  /* ═════════ Sidebar: brand, track bars, status ═════════ */
  [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
  .nav-brand { display: flex; align-items: center; gap: 12px; padding: 2px 4px 6px 4px; }
  .nav-logo {
    width: 40px; height: 40px; border-radius: 11px; display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; font-weight: 800; color: #fff;
    background: linear-gradient(135deg, #388bfd 0%, #a371f7 100%);
    box-shadow: 0 6px 18px rgba(88, 101, 242, 0.35);
  }
  .nav-title { font-size: 1.2rem; font-weight: 800; color: #e6edf3; letter-spacing: 0.02em; line-height: 1.1; }
  .nav-sub   { font-size: 0.66rem; font-weight: 600; color: #8b949e; letter-spacing: 0.16em; margin-top: 3px; }
  .nav-section {
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
    color: #6e7681; padding: 18px 4px 8px 4px;
  }

  /* the radio becomes a stack of compact bars */
  [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] { flex-direction: column !important; gap: 7px !important; }
  [data-testid="stSidebar"] div[role="radiogroup"] > label {
    display: flex !important; flex-wrap: wrap; align-items: center; column-gap: 12px; row-gap: 1px;
    width: 100%; margin: 0 !important; padding: 9px 14px 9px 12px !important; box-sizing: border-box;
    border-radius: 10px; background: #12171e; border: 1px solid #21262d; border-left: 3px solid #30363d;
    cursor: pointer; transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
  }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: #1a212b; border-color: #30363d; border-left-color: var(--acc, #58a6ff); transform: translateX(2px);
  }
  /* hide the native radio dot */
  [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none !important; }
  /* icon chip */
  [data-testid="stSidebar"] div[role="radiogroup"] > label::before {
    width: 32px; height: 32px; border-radius: 9px; display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; font-weight: 700; color: var(--acc, #58a6ff); background: var(--accbg, rgba(88,166,255,0.14));
  }
  /* title */
  [data-testid="stSidebar"] div[role="radiogroup"] > label > div:last-child p {
    margin: 0; font-size: 0.92rem; font-weight: 650; color: #c9d1d9; letter-spacing: 0.01em;
  }
  /* subtitle on a second line, aligned with the title */
  [data-testid="stSidebar"] div[role="radiogroup"] > label::after {
    flex-basis: 100%; padding-left: 44px; margin-top: -9px; font-size: 0.71rem; color: #8b949e; line-height: 1.2;
  }
  /* active bar */
  [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, var(--accbg, rgba(88,166,255,0.14)) 0%, rgba(18,23,30,0) 100%), #12171e;
    border-color: #30363d; border-left-color: var(--acc, #58a6ff);
    box-shadow: 0 2px 14px rgba(0, 0, 0, 0.35);
  }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) > div:last-child p { color: #ffffff; }

  /* per-item accent colour, glyph and subtitle */
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1) { --acc: #58a6ff; --accbg: rgba(88,166,255,0.16); }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1)::before { content: "\2302"; }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1)::after  { content: "Overview and datasets"; }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2) { --acc: #388bfd; --accbg: rgba(56,139,253,0.16); }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2)::before { content: "\25C8"; }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2)::after  { content: "Star type \B7  10 models"; }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3) { --acc: #3fb950; --accbg: rgba(63,185,80,0.16); }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3)::before { content: "\2197"; }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3)::after  { content: "Crime rate \B7  9 models"; }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4) { --acc: #a371f7; --accbg: rgba(163,113,247,0.16); }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4)::before { content: "\2B21"; }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4)::after  { content: "Customer segments \B7  4 groups"; }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(5) { --acc: #e3b341; --accbg: rgba(227,179,65,0.16); }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(5)::before { content: "\25A4"; }
  [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(5)::after  { content: "Explore all three datasets"; }

  /* status footer */
  .status-row { display: flex; align-items: center; gap: 9px; padding: 3px 4px; font-size: 0.78rem; color: #c9d1d9; }
  .status-row .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
  .status-row .dot.ok  { background: #3fb950; box-shadow: 0 0 8px rgba(63,185,80,0.6); }
  .status-row .dot.bad { background: #f85149; box-shadow: 0 0 8px rgba(248,81,73,0.6); }
  .status-row .st { margin-left: auto; font-size: 0.7rem; color: #8b949e; }

  /* ═════════ Cards used by the clustering page ═════════ */
  .seg-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 14px 16px; height: 100%;
  }
  .seg-name  { font-size: 0.98rem; font-weight: 700; color: #e6edf3; margin-bottom: 2px; }
  .seg-count { font-size: 1.55rem; font-weight: 800; color: #e6edf3; line-height: 1.15; }
  .seg-pct   { font-size: 0.78rem; color: #8b949e; margin-bottom: 8px; }
  .seg-stat  { display: flex; justify-content: space-between; font-size: 0.78rem; color: #8b949e; padding: 2px 0; border-top: 1px solid #21262d; }
  .seg-stat b { color: #c9d1d9; font-weight: 600; }
  .section-lead { color: #8b949e; font-size: 1.0rem; margin-top: -6px; }
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
CLU_DIR    = os.path.join(BASE, "Clustering")
CLU_MODELS = os.path.join(CLU_DIR, "models")
CLU_APP    = os.path.join(CLU_DIR, "app")
if CLU_APP not in sys.path:
    sys.path.insert(0, CLU_APP)   # saved clustering pipeline imports `clustering_preprocessing`


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


# ─────────────────────────────────────────────────────────────
#  CLUSTERING TRACK: artifacts, data and helpers
# ─────────────────────────────────────────────────────────────
CLU_NEEDED = ["cluster_preprocessor.pkl", "kmeans_final.pkl", "hierarchical_model.pkl",
              "cluster_pca.pkl", "cluster_meta.json"]


def clu_artifacts_ready():
    return all(os.path.exists(os.path.join(CLU_MODELS, f)) for f in CLU_NEEDED)


@st.cache_resource
def load_clu_artifacts():
    # The saved preprocessing pipeline pickles a reference to `clustering_preprocessing`,
    # so that module must be importable (CLU_APP was added to sys.path above).
    import clustering_preprocessing  # noqa: F401
    pre  = joblib.load(os.path.join(CLU_MODELS, "cluster_preprocessor.pkl"))
    km   = joblib.load(os.path.join(CLU_MODELS, "kmeans_final.pkl"))
    hier = joblib.load(os.path.join(CLU_MODELS, "hierarchical_model.pkl"))
    pca  = joblib.load(os.path.join(CLU_MODELS, "cluster_pca.pkl"))
    with open(os.path.join(CLU_MODELS, "cluster_meta.json"), "r") as fh:
        meta = json.load(fh)
    return {"pre": pre, "km": km, "hier": hier, "pca": pca, "meta": meta}


@st.cache_data
def load_clu_data():
    path = os.path.join(CLU_DIR, "data", "CC GENERAL.csv")
    if not os.path.exists(path):
        path = os.path.join(CLU_DIR, "CC GENERAL.csv")
    return pd.read_csv(path)


@st.cache_resource
def load_clu_bundle():
    # Everything the clustering page needs for the full 8,950-customer dataset.
    art  = load_clu_artifacts()
    meta = art["meta"]
    df   = load_clu_data()
    X      = art["pre"].transform(df[meta["raw_features"]])
    coords = art["pca"].transform(X)
    km_ids = np.array(meta["km_labels"], dtype=int)
    h2k    = {int(k): int(v) for k, v in meta["hier_to_km"].items()}   # JSON keys are strings
    hc_ids = np.array([h2k[int(v)] for v in meta["hier_labels"]], dtype=int)
    if len(km_ids) != len(df):
        raise ValueError("Saved cluster labels do not match the CSV - re-run save_clustering_models.py")
    proj = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "PC3": coords[:, 2]})
    proj["CUST_ID"] = df["CUST_ID"].astype(str).values if "CUST_ID" in df.columns else np.arange(len(df)).astype(str)
    for c in ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT"]:
        proj[c] = df[c].values
    return {"df": df, "X": X, "coords": coords, "km_ids": km_ids, "hc_ids": hc_ids, "proj": proj}


def clu_predict(art, rows_df, algo):
    # Assign customers (a DataFrame with the 17 raw columns) to a segment.
    # Returns (segment ids, distance of each row to every segment centroid, scaled features).
    meta = art["meta"]
    X = art["pre"].transform(rows_df[meta["raw_features"]])
    if algo == "K-Means":
        ids  = art["km"].predict(X)
        dist = art["km"].transform(X)                          # columns follow cluster id 0..k-1
    else:
        h = art["hier"]
        h2k = {int(k): int(v) for k, v in meta["hier_to_km"].items()}
        raw_ids = h.predict(X)
        d_h = np.linalg.norm(X[:, None, :] - h.centroids_[None, :, :], axis=2)
        ids = np.array([h2k[int(i)] for i in raw_ids], dtype=int)
        dist = np.zeros_like(d_h)
        for col, cls in enumerate(h.classes_):                 # reorder columns to segment ids
            dist[:, h2k[int(cls)]] = d_h[:, col]
    return np.asarray(ids, dtype=int), dist, X


def style_fig(fig, height=420, **kw):
    # shared dark theme for all clustering plots
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e6edf3", height=height, margin=dict(t=44, b=20, l=10, r=10), **kw)
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    return fig


def hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def show_image(path, caption=None):
    # st.image's width argument was renamed across Streamlit versions
    try:
        st.image(path, caption=caption, use_container_width=True)
    except TypeError:
        st.image(path, caption=caption, use_column_width=True)


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
    st.markdown("""
<div class='nav-brand'>
  <div class='nav-logo'>V</div>
  <div>
    <div class='nav-title'>Vectora</div>
    <div class='nav-sub'>ML WORKBENCH</div>
  </div>
</div>
<div class='nav-section'>Navigation</div>
""", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Home", "Classification", "Regression", "Clustering", "EDA Explorer"],
        label_visibility="collapsed",
        key="nav_page",
    )

    _ok_cls = os.path.exists(os.path.join(CLS_MODELS, "cls_scaler.pkl"))
    _ok_reg = os.path.exists(os.path.join(REG_MODELS, "reg_scaler.pkl"))
    _ok_clu = clu_artifacts_ready()

    def _status_row(label, ok):
        dot = "ok" if ok else "bad"
        txt = "Ready" if ok else "Missing"
        return f"<div class='status-row'><span class='dot {dot}'></span>{label}<span class='st'>{txt}</span></div>"

    st.markdown(
        "<div class='nav-section'>Model status</div>"
        + _status_row("Classification", _ok_cls)
        + _status_row("Regression", _ok_reg)
        + _status_row("Clustering", _ok_clu),
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════
if page == "Home":
    st.title("Vectora — ML Dashboard")
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
  <p style='color:#8b949e;font-size:0.88rem'>Credit-card customer segmentation from behavioural features.<br>K-Means · Ward hierarchical · 8,950 customers</p>
  <hr style='border-color:#30363d'>
  <p style='color:#3fb950;font-size:1.2rem;font-weight:700'>4 customer segments</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Datasets at a Glance")
    d1, d2, d3 = st.columns(3)
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


    with d3:
        try:
            df_clu = load_clu_data()
            st.markdown("**Credit-Card Customers**")
            mk1, mk2, mk3 = st.columns(3)
            mk1.metric("Rows", f"{len(df_clu):,}")
            mk2.metric("Features", f"{df_clu.shape[1]-1}")
            mk3.metric("Segments", "4")
            if clu_artifacts_ready():
                _ma = load_clu_artifacts()["meta"]
                fig3 = px.pie(names=[s["name"] for s in _ma["segments"]], values=[s["size"] for s in _ma["segments"]],
                              color=[s["name"] for s in _ma["segments"]],
                              color_discrete_map={s["name"]: s["color"] for s in _ma["segments"]}, hole=0.55)
                fig3.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e6edf3", showlegend=True,
                    margin=dict(t=10, b=10, l=10, r=10), height=260,
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(fig3, use_container_width=True, key="home_clu_pie")
            else:
                st.info("Run `python save_clustering_models.py` to enable the segment chart.")
        except Exception as e:
            st.warning(f"Could not load clustering data: {e}")

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
elif page == "Classification":
    st.title("Star Classification")
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
elif page == "Regression":
    st.title("Community Crime Rate Predictor")
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
        "Linear Regression":     {"R²": 0.6830, "RMSE": 0.1307, "MAE": 0.0949},
        "Ridge Regression":      {"R²": 0.6820, "RMSE": 0.1309, "MAE": 0.0950},
        "Lasso Regression":      {"R²": 0.6793, "RMSE": 0.1314, "MAE": 0.0939},
        "ElasticNet Regression": {"R²": 0.6811, "RMSE": 0.1310, "MAE": 0.0945},
        "Decision Tree":         {"R²": 0.3388, "RMSE": 0.1887, "MAE": 0.1289},
        "Random Forest":         {"R²": 0.6672, "RMSE": 0.1339, "MAE": 0.0957},
        "Gradient Boosting":     {"R²": 0.6830, "RMSE": 0.1307, "MAE": 0.0902},
        "SVR":                   {"R²": 0.6261, "RMSE": 0.1419, "MAE": 0.1034},
        "KNN":                   {"R²": 0.5983, "RMSE": 0.1471, "MAE": 0.0994},
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
elif page == "Clustering":
    st.title("Customer Segmentation")
    st.markdown("<p class='section-lead'>Credit-card behaviour segments from K-Means and Ward hierarchical clustering "
                "on the CC GENERAL dataset (8,950 cardholders, 17 behavioural features).</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── make sure the saved models are available and loadable ──────────
    if not clu_artifacts_ready():
        st.error("The clustering models were not found in `Clustering/models/`.")
        st.markdown("Generate them once from the project root, then refresh this page:")
        st.code("python save_clustering_models.py", language="bash")
        st.stop()

    try:
        art = load_clu_artifacts()
        bun = load_clu_bundle()
    except Exception as e:
        st.error(f"Could not load the clustering models: {e}")
        st.markdown("Re-create them with the same scikit-learn version the app is running:")
        st.code("python save_clustering_models.py", language="bash")
        st.stop()

    meta     = art["meta"]
    segs     = meta["segments"]
    by_id    = {s["id"]: s for s in segs}
    seg_names = [s["name"] for s in segs]
    seg_color = {s["name"]: s["color"] for s in segs}
    feats    = meta["raw_features"]
    ranges   = meta["feature_ranges"]
    n_seg    = len(segs)

    try:
        import sklearn
        if meta.get("sklearn_version") and meta["sklearn_version"] != sklearn.__version__:
            st.warning(f"Models were saved with scikit-learn {meta['sklearn_version']} but this app runs "
                       f"{sklearn.__version__}. If anything looks off, re-run `python save_clustering_models.py`.")
    except Exception:
        pass

    tab_pred, tab_exp, tab_prof, tab_q, tab_play = st.tabs(
        ["Segment Predictor", "Segment Explorer", "Segment Profiles", "Model Quality", "Playground"])

    # ═════════════ TAB 1 — SEGMENT PREDICTOR ═════════════
    with tab_pred:
        FREQ  = {f for f in feats if "FREQUENCY" in f} | {"PRC_FULL_PAYMENT"}
        COUNT = {"CASH_ADVANCE_TRX", "PURCHASES_TRX"}
        T_MIN, T_MAX = int(ranges["TENURE"]["min"]), int(ranges["TENURE"]["max"])

        def _cast(f, v):
            v = 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            if f == "TENURE":
                return int(min(T_MAX, max(T_MIN, round(v))))
            if f in FREQ:
                return float(min(1.0, max(0.0, v)))
            if f in COUNT:
                return float(round(max(0.0, v)))
            return max(0.0, v)

        PRESET_MED = "Dataset median customer"
        presets = [PRESET_MED] + [f"Typical · {n}" for n in seg_names]

        def _apply_preset():
            ch = st.session_state["clu_preset"]
            if ch == PRESET_MED:
                src = {f: ranges[f]["median"] for f in feats}
            else:
                nm = ch.split("· ", 1)[1]
                src = next(s for s in segs if s["name"] == nm)["median_raw"]
            for f in feats:
                st.session_state[f"clu_in_{f}"] = _cast(f, src.get(f))

        for f in feats:                                   # initial values = dataset medians
            if f"clu_in_{f}" not in st.session_state:
                st.session_state[f"clu_in_{f}"] = _cast(f, ranges[f]["median"])

        st.markdown("#### Describe a customer")
        st.markdown("<p style='color:#8b949e;margin-top:-6px'>Edit the values or start from a preset. "
                    "The segment updates instantly.</p>", unsafe_allow_html=True)

        ctl_a, ctl_b = st.columns(2)
        algo_p = ctl_a.radio("Model", ["K-Means", "Hierarchical (Ward)"], horizontal=True, key="clu_algo_pred")
        ctl_b.selectbox("Start from a preset", presets, key="clu_preset", on_change=_apply_preset)

        groups = [
            ("Balance and credit", [
                ("BALANCE", "Balance ($)"), ("CREDIT_LIMIT", "Credit limit ($)"),
                ("BALANCE_FREQUENCY", "Balance update frequency (0–1)"), ("TENURE", "Tenure (months)")]),
            ("Purchases", [
                ("PURCHASES", "Total purchases ($)"), ("ONEOFF_PURCHASES", "One-off purchases ($)"),
                ("INSTALLMENTS_PURCHASES", "Instalment purchases ($)"), ("PURCHASES_TRX", "Purchase transactions"),
                ("PURCHASES_FREQUENCY", "Purchase frequency (0–1)"),
                ("ONEOFF_PURCHASES_FREQUENCY", "One-off frequency (0–1)"),
                ("PURCHASES_INSTALLMENTS_FREQUENCY", "Instalment frequency (0–1)")]),
            ("Cash advance and payments", [
                ("CASH_ADVANCE", "Cash advance ($)"), ("CASH_ADVANCE_TRX", "Cash-advance transactions"),
                ("CASH_ADVANCE_FREQUENCY", "Cash-advance frequency (0–1)"), ("PAYMENTS", "Payments ($)"),
                ("MINIMUM_PAYMENTS", "Minimum payments ($)"), ("PRC_FULL_PAYMENT", "Share of full payments (0–1)")]),
        ]
        gcols = st.columns(3)
        for gcol, (gname, items) in zip(gcols, groups):
            with gcol:
                st.markdown(f"**{gname}**")
                for f, label in items:
                    k = f"clu_in_{f}"
                    if f == "TENURE":
                        st.slider(label, T_MIN, T_MAX, step=1, key=k)
                    elif f in FREQ:
                        st.slider(label, 0.0, 1.0, step=0.01, key=k)
                    elif f in COUNT:
                        st.number_input(label, min_value=0.0, step=1.0, format="%.0f", key=k)
                    else:
                        st.number_input(label, min_value=0.0, step=50.0, format="%.2f", key=k)

        row = {f: st.session_state[f"clu_in_{f}"] for f in feats}
        ids_p, dist_p, X_p = clu_predict(art, pd.DataFrame([row]), algo_p)
        sid = int(ids_p[0])
        seg = by_id[sid]
        inv = 1.0 / (dist_p[0] + 1e-9)
        share = inv / inv.sum()

        st.markdown("---")
        r_left, r_right = st.columns([1, 1.25])
        with r_left:
            st.markdown(f"""
<div class='pred-box' style='border-color:{seg["color"]}'>
  <div class='pred-label'>Assigned segment</div>
  <div class='pred-value' style='color:{seg["color"]}'>{seg["name"]}</div>
  <div class='pred-sub'>{seg["summary"]}</div>
</div>""", unsafe_allow_html=True)
            st.markdown(f"<div class='info-banner'><b>Suggested action:</b> {seg['action']}</div>",
                        unsafe_allow_html=True)

            prox = pd.DataFrame({"Segment": [by_id[i]["name"] for i in range(n_seg)],
                                 "Share": share * 100}).sort_values("Share")
            fig_pr = px.bar(prox, x="Share", y="Segment", orientation="h", color="Segment",
                            color_discrete_map=seg_color, text=prox["Share"].map(lambda v: f"{v:.0f}%"))
            fig_pr.update_traces(textposition="outside", cliponaxis=False)
            style_fig(fig_pr, height=230, showlegend=False, title="Relative proximity to each segment")
            fig_pr.update_xaxes(range=[0, max(100, prox["Share"].max() + 15)], title=None)
            fig_pr.update_yaxes(title=None)
            st.plotly_chart(fig_pr, use_container_width=True, key="clu_pred_prox")
            st.caption("Proximity is derived from the distance to each segment centre. "
                       "It shows how clear-cut the assignment is; it is not a probability.")

        with r_right:
            pt = art["pca"].transform(X_p)[0]
            pop = bun["proj"].sample(n=min(3000, len(bun["proj"])), random_state=42)
            ids_pop = (bun["km_ids"] if algo_p == "K-Means" else bun["hc_ids"])[pop.index.values]
            pop = pop.assign(Segment=[by_id[int(i)]["name"] for i in ids_pop])
            fig_pt = px.scatter(pop, x="PC1", y="PC2", color="Segment", color_discrete_map=seg_color,
                                category_orders={"Segment": seg_names}, opacity=0.35,
                                title="Where this customer sits among all cardholders")
            fig_pt.update_traces(marker=dict(size=4))
            fig_pt.add_trace(go.Scatter(x=[pt[0]], y=[pt[1]], mode="markers", name="This customer",
                                        marker=dict(size=18, symbol="diamond", color="#ffffff",
                                                    line=dict(width=2, color=seg["color"]))))
            style_fig(fig_pt, height=400, legend_title_text="")
            st.plotly_chart(fig_pt, use_container_width=True, key="clu_pred_scatter")

        # ── batch scoring ──
        st.markdown("---")
        st.markdown("#### Score a file")
        st.markdown("<p style='color:#8b949e;margin-top:-6px'>Upload a CSV with the 17 raw feature columns "
                    "(a CUST_ID column is kept if present) to segment many customers at once.</p>",
                    unsafe_allow_html=True)
        up = st.file_uploader("CSV file", type=["csv"], key="clu_upload", label_visibility="collapsed")
        if up is not None:
            try:
                up_df = pd.read_csv(up)
                missing = [f for f in feats if f not in up_df.columns]
                if missing:
                    st.error("Missing columns: " + ", ".join(missing))
                else:
                    ids_b, _, _ = clu_predict(art, up_df, algo_p)
                    out = up_df.copy()
                    out["Segment"] = [by_id[int(i)]["name"] for i in ids_b]
                    cnt = out["Segment"].value_counts().reindex(seg_names).fillna(0).astype(int)
                    mcols = st.columns(n_seg)
                    for mc, nm in zip(mcols, seg_names):
                        mc.metric(nm, f"{int(cnt[nm]):,}")
                    st.dataframe(out.head(200), use_container_width=True, height=320)
                    st.download_button("Download segmented CSV", data=out.to_csv(index=False).encode("utf-8"),
                                       file_name="customers_segmented.csv", mime="text/csv", key="clu_dl")
            except Exception as e:
                st.error(f"Could not score this file: {e}")

    # ═════════════ TAB 2 — SEGMENT EXPLORER ═════════════
    with tab_exp:
        e1, e2 = st.columns(2)
        algo_e = e1.radio("Clustering", ["K-Means", "Hierarchical (Ward)"], horizontal=True, key="clu_algo_exp")
        view_e = e2.radio("View", ["2D", "3D"], horizontal=True, key="clu_view_exp")

        ids_e = bun["km_ids"] if algo_e == "K-Means" else bun["hc_ids"]
        name_lookup = np.array([by_id[i]["name"] for i in range(n_seg)])
        sizes = np.bincount(ids_e, minlength=n_seg)
        med = bun["df"][["BALANCE", "PURCHASES", "CASH_ADVANCE"]].groupby(ids_e).median()

        card_cols = st.columns(n_seg)
        for cc, s in zip(card_cols, segs):
            i = s["id"]
            m = med.loc[i] if i in med.index else None
            stats = ""
            if m is not None:
                stats = (f"<div class='seg-stat'><span>Median balance</span><b>${m['BALANCE']:,.0f}</b></div>"
                         f"<div class='seg-stat'><span>Median purchases</span><b>${m['PURCHASES']:,.0f}</b></div>"
                         f"<div class='seg-stat'><span>Median cash advance</span><b>${m['CASH_ADVANCE']:,.0f}</b></div>")
            cc.markdown(f"""
<div class='seg-card' style='border-top:3px solid {s["color"]}'>
  <div class='seg-name'>{s["name"]}</div>
  <div class='seg-count'>{int(sizes[i]):,}</div>
  <div class='seg-pct'>{sizes[i] / len(ids_e) * 100:.1f}% of customers</div>
  {stats}
</div>""", unsafe_allow_html=True)

        st.markdown("")
        p_left, p_right = st.columns([2.2, 1])
        proj = bun["proj"].assign(Segment=name_lookup[ids_e])
        with p_left:
            hover = {"CUST_ID": True, "BALANCE": ":,.0f", "PURCHASES": ":,.0f", "CASH_ADVANCE": ":,.0f",
                     "PC1": False, "PC2": False}
            if view_e == "2D":
                fig_e = px.scatter(proj, x="PC1", y="PC2", color="Segment", color_discrete_map=seg_color,
                                   category_orders={"Segment": seg_names}, hover_data=hover, opacity=0.6,
                                   title=f"{algo_e}: customers projected on the first two principal components")
                fig_e.update_traces(marker=dict(size=4))
            else:
                hover["PC3"] = False
                fig_e = px.scatter_3d(proj, x="PC1", y="PC2", z="PC3", color="Segment",
                                      color_discrete_map=seg_color, category_orders={"Segment": seg_names},
                                      hover_data=hover, opacity=0.7,
                                      title=f"{algo_e}: first three principal components")
                fig_e.update_traces(marker=dict(size=2.5))
            style_fig(fig_e, height=540, legend_title_text="")
            st.plotly_chart(fig_e, use_container_width=True, key="clu_exp_scatter")
        with p_right:
            fig_d = px.pie(names=seg_names, values=[int(sizes[s["id"]]) for s in segs], hole=0.6,
                           color=seg_names, color_discrete_map=seg_color, title="Segment mix")
            fig_d.update_traces(textinfo="percent", textfont_size=13)
            style_fig(fig_d, height=330, showlegend=False)
            st.plotly_chart(fig_d, use_container_width=True, key="clu_exp_donut")
            evr = meta["pca_explained_variance"]
            st.markdown(f"**Variance shown**  \nPC1 {evr[0]*100:.1f}% · PC2 {evr[1]*100:.1f}% · PC3 {evr[2]*100:.1f}%  \n"
                        f"2D total **{sum(evr[:2])*100:.1f}%** · 3D total **{sum(evr[:3])*100:.1f}%**")
            st.caption("A 2D/3D projection of a 21-feature space: overlap in the plot does not mean "
                       "the segments overlap in the full space.")

    # ═════════════ TAB 3 — SEGMENT PROFILES ═════════════
    with tab_prof:
        z_mat = pd.DataFrame({s["name"]: s["z"] for s in segs}).T[meta["final_features"]]
        st.markdown("#### What makes each segment different")
        st.markdown("<p style='color:#8b949e;margin-top:-6px'>Average standardised value of every feature "
                    "per segment. Red = above the customer average, blue = below.</p>", unsafe_allow_html=True)
        fig_h = px.imshow(z_mat, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1.5, zmax=1.5,
                          text_auto=".1f")
        fig_h.update_layout(coloraxis_colorbar=dict(title="z-score", tickcolor="#e6edf3"))
        fig_h.update_xaxes(tickangle=-45, side="bottom")
        style_fig(fig_h, height=380)
        st.plotly_chart(fig_h, use_container_width=True, key="clu_prof_heat")

        pr_left, pr_right = st.columns([1.1, 1])
        with pr_left:
            radar_f = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY",
                       "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT", "BALANCE_TO_LIMIT"]
            radar_f = [f for f in radar_f if f in z_mat.columns]
            rm = z_mat[radar_f]
            rm = (rm - rm.min()) / (rm.max() - rm.min() + 1e-9)          # 0-1 per feature across segments
            theta = [f.replace("_", " ").title() for f in radar_f]
            fig_r = go.Figure()
            for nm in seg_names:
                vals = rm.loc[nm].tolist()
                fig_r.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=theta + theta[:1], name=nm,
                                                fill="toself", line=dict(color=seg_color[nm], width=2),
                                                fillcolor=hex_rgba(seg_color[nm], 0.15)))
            fig_r.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                           radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d",
                                                           tickfont=dict(size=9, color="#8b949e")),
                                           angularaxis=dict(gridcolor="#30363d")),
                                legend=dict(orientation="h", y=-0.12), title="Segment fingerprint")
            style_fig(fig_r, height=470)
            st.plotly_chart(fig_r, use_container_width=True, key="clu_prof_radar")
        with pr_right:
            st.markdown("**Median raw values**")
            med_tbl = pd.DataFrame({s["name"]: s["median_raw"] for s in segs}).round(2)
            st.dataframe(med_tbl, use_container_width=True, height=440)

        st.markdown("#### Segment guide")
        g_cols = st.columns(n_seg)
        for gc, s in zip(g_cols, segs):
            gc.markdown(f"""
<div class='seg-card' style='border-top:3px solid {s["color"]}'>
  <div class='seg-name'>{s["name"]}</div>
  <p style='color:#8b949e;font-size:0.84rem;margin:6px 0 10px 0'>{s["summary"]}</p>
  <div style='font-size:0.78rem;color:#c9d1d9'><b>Action:</b> {s["action"]}</div>
</div>""", unsafe_allow_html=True)

    # ═════════════ TAB 4 — MODEL QUALITY ═════════════
    with tab_q:
        mq = meta["metrics"]
        st.markdown("#### Internal validity indices")
        q1, q2 = st.columns(2)
        for qcol, title, key in [(q1, "K-Means (k = %d)" % meta["k_final"], "kmeans"),
                                 (q2, "Hierarchical · Ward (k = %d)" % meta["k_final"], "hierarchical")]:
            with qcol:
                st.markdown(f"**{title}**")
                a, b, c = st.columns(3)
                a.metric("Silhouette ↑", f"{mq[key]['silhouette']:.3f}", help="Range −1 to 1; higher is better")
                b.metric("Davies-Bouldin ↓", f"{mq[key]['davies_bouldin']:.3f}", help="Lower is better")
                c.metric("Calinski-Harabasz ↑", f"{mq[key]['calinski_harabasz']:,.0f}", help="Higher is better")
        st.markdown(f"<div class='info-banner'>The two algorithms agree on most customers "
                    f"(Adjusted Rand Index = <b>{mq['ari']:.3f}</b>). K-Means is used for predictions "
                    f"because it scores slightly better on all three indices.</div>", unsafe_allow_html=True)

        st.markdown("#### Choosing k")
        sw = pd.DataFrame(meta["sweep"])
        fig_s = make_subplots(rows=1, cols=4, subplot_titles=["Inertia (elbow)", "Silhouette ↑",
                                                              "Davies-Bouldin ↓", "Calinski-Harabasz ↑"])
        for j, (col_name, colr) in enumerate([("inertia", "#388bfd"), ("silhouette", "#3fb950"),
                                              ("davies_bouldin", "#f78166"), ("calinski_harabasz", "#d2a8ff")], start=1):
            fig_s.add_trace(go.Scatter(x=sw["k"], y=sw[col_name], mode="lines+markers",
                                       line=dict(color=colr, width=2), name=col_name), row=1, col=j)
            fig_s.add_vline(x=meta["k_final"], line_dash="dash", line_color="#8b949e", row=1, col=j)
            fig_s.update_xaxes(title_text="k", dtick=1, row=1, col=j)
        style_fig(fig_s, height=330, showlegend=False)
        st.plotly_chart(fig_s, use_container_width=True, key="clu_q_sweep")
        st.caption(f"Dashed line = chosen k = {meta['k_final']}. Silhouette and Calinski-Harabasz favour fewer "
                   "clusters, so the final k balances them against the elbow and interpretability.")

        figs_dir = os.path.join(CLU_DIR, "figures", "clustering")
        if os.path.isdir(figs_dir):
            files = sorted(f for f in os.listdir(figs_dir) if f.lower().endswith(".png"))
            if files:
                st.markdown("#### Notebook figures")
                pick = st.selectbox("Figure", files, key="clu_fig_pick",
                                    format_func=lambda f: f[:-4].partition("_")[2].replace("_", " ").capitalize())
                show_image(os.path.join(figs_dir, pick))

    # ═════════════ TAB 5 — PLAYGROUND ═════════════
    with tab_play:
        st.markdown("<div class='info-banner'>Experiment with other algorithms and settings on the same "
                    "preprocessed data. This is exploratory; the saved production model is K-Means (k = %d)."
                    "</div>" % meta["k_final"], unsafe_allow_html=True)
        pg_l, pg_r = st.columns([1, 2.2])
        with pg_l:
            algo_pg = st.selectbox("Algorithm", ["K-Means", "Agglomerative", "DBSCAN"], key="clu_pg_algo")
            if algo_pg == "K-Means":
                k_pg = st.slider("Number of clusters (k)", 2, 12, meta["k_final"], key="clu_pg_k")
                init_pg = st.selectbox("Init method", ["k-means++", "random"], key="clu_pg_init")
            elif algo_pg == "Agglomerative":
                k_pg = st.slider("Number of clusters", 2, 12, meta["k_final"], key="clu_pg_aggk")
                link_pg = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="clu_pg_link")
            else:
                eps_pg = st.slider("ε (neighbourhood radius)", 0.5, 6.0, 2.0, step=0.1, key="clu_pg_eps")
                ms_pg = st.slider("min_samples", 3, 40, 10, key="clu_pg_ms")
                st.caption("Points that belong to no cluster are labelled −1 (noise).")
            frac_pg = st.slider("Share of customers used", 0.1, 1.0, 0.4, step=0.05, key="clu_pg_frac")
            dim_pg = st.radio("Plot", ["2D", "3D"], horizontal=True, key="clu_pg_dim")
            run_pg = st.button("Run clustering", use_container_width=True, key="clu_pg_run")
        with pg_r:
            if run_pg:
                rs = np.random.RandomState(42)
                idx = rs.choice(len(bun["X"]), size=max(200, int(len(bun["X"]) * frac_pg)), replace=False)
                Xs, Cs = bun["X"][idx], bun["coords"][idx]
                with st.spinner(f"Running {algo_pg}…"):
                    if algo_pg == "K-Means":
                        lab = KMeans(n_clusters=k_pg, init=init_pg, n_init=10, random_state=42).fit_predict(Xs)
                    elif algo_pg == "Agglomerative":
                        lab = AgglomerativeClustering(n_clusters=k_pg, linkage=link_pg).fit_predict(Xs)
                    else:
                        lab = DBSCAN(eps=eps_pg, min_samples=ms_pg).fit_predict(Xs)
                n_found = len(set(lab)) - (1 if -1 in lab else 0)
                n_noise = int((lab == -1).sum())
                try:
                    ok = n_found > 1
                    sil = silhouette_score(Xs, lab, sample_size=min(3000, len(Xs)), random_state=42) if ok else float("nan")
                    dbi = davies_bouldin_score(Xs, lab) if ok else float("nan")
                except Exception:
                    sil, dbi = float("nan"), float("nan")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Clusters found", str(n_found))
                m2.metric("Noise points", f"{n_noise:,}")
                m3.metric("Silhouette ↑", "N/A" if np.isnan(sil) else f"{sil:.3f}")
                m4.metric("Davies-Bouldin ↓", "N/A" if np.isnan(dbi) else f"{dbi:.3f}")

                pdf = pd.DataFrame({"PC1": Cs[:, 0], "PC2": Cs[:, 1], "PC3": Cs[:, 2],
                                    "Cluster": [("Noise" if v == -1 else f"Cluster {v}") for v in lab]})
                order = sorted(pdf["Cluster"].unique(), key=lambda s: (s == "Noise", s))
                if dim_pg == "2D":
                    fig_g = px.scatter(pdf, x="PC1", y="PC2", color="Cluster", category_orders={"Cluster": order},
                                       color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.65,
                                       title=f"{algo_pg}: {n_found} clusters")
                    fig_g.update_traces(marker=dict(size=4))
                else:
                    fig_g = px.scatter_3d(pdf, x="PC1", y="PC2", z="PC3", color="Cluster",
                                          category_orders={"Cluster": order},
                                          color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.7,
                                          title=f"{algo_pg}: {n_found} clusters")
                    fig_g.update_traces(marker=dict(size=2.5))
                style_fig(fig_g, height=500, legend_title_text="")
                st.plotly_chart(fig_g, use_container_width=True, key="clu_pg_scatter")

                sz = pdf["Cluster"].value_counts().reindex(order).reset_index()
                sz.columns = ["Cluster", "Customers"]
                fig_z = px.bar(sz, x="Cluster", y="Customers", color="Cluster", text="Customers",
                               color_discrete_sequence=px.colors.qualitative.Set2, title="Cluster sizes")
                fig_z.update_traces(textposition="outside")
                style_fig(fig_z, height=300, showlegend=False)
                st.plotly_chart(fig_z, use_container_width=True, key="clu_pg_sizes")
            else:
                st.markdown("""
<div style='background:#161b22;border:1px dashed #30363d;border-radius:14px;
            padding:70px 40px;text-align:center;color:#8b949e'>
  <div style='font-size:1.15rem;font-weight:700;color:#c9d1d9;margin-bottom:6px'>Configure and run</div>
  Choose an algorithm and its settings on the left, then press <b>Run clustering</b>.
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: EDA EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == "EDA Explorer":
    st.title("EDA Explorer")
    st.markdown("<p style='color:#8b949e'>Interactively explore all three datasets — distributions, correlations, scatter plots, and summary statistics.</p>", unsafe_allow_html=True)
    st.markdown("---")

    dataset_eda = st.radio("Select Dataset", ["Star Classification", "Community Crime", "Credit-Card Customers"], horizontal=True, key="eda_ds")

    if "Star" in dataset_eda:
        df_e = load_cls_data()
        num_cols = df_e.select_dtypes(include=[np.number]).columns.tolist()
        cat_col  = "class"
        color_map = CLASS_COLORS
        has_target_cat = True
        target_e = None
    elif "Crime" in dataset_eda:
        df_e, target_e = load_reg_data()
        num_cols = df_e.select_dtypes(include=[np.number]).columns.tolist()
        cat_col  = None
        color_map = None
        has_target_cat = False
    else:
        df_e = load_clu_data().drop(columns=["CUST_ID"], errors="ignore").copy()
        target_e = None
        cat_col, color_map, has_target_cat = None, None, False
        if clu_artifacts_ready():            # colour points by the saved K-Means segment
            try:
                _am = load_clu_artifacts()["meta"]
                _nm = np.array([s["name"] for s in sorted(_am["segments"], key=lambda s: s["id"])])
                df_e["Segment"] = _nm[np.array(_am["km_labels"], dtype=int)]
                cat_col, has_target_cat = "Segment", True
                color_map = {s["name"]: s["color"] for s in _am["segments"]}
            except Exception:
                pass
        num_cols = df_e.select_dtypes(include=[np.number]).columns.tolist()

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
            hover_data=[cat_col] if has_target_cat else ([target_e] if target_e else None),
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
