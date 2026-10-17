"""
save_clustering_models.py
-------------------------
Reproduces the final models of the Clustering track (credit-card customer
segmentation) exactly as configured in Clustering/clustering.ipynb and saves
everything the Streamlit GUI needs into  Clustering/models/ :

    cluster_preprocessor.pkl     engineer -> impute -> scale pipeline (fitted)
    kmeans_final.pkl             K-Means, k = 4, k-means++, n_init = 20
    hierarchical_model.pkl       NearestCentroid fitted on Ward-linkage labels
                                 (agglomerative clustering has no .predict(), so
                                 new customers are assigned to the nearest Ward centroid)
    cluster_pca.pkl              PCA(3) used for the 2D / 3D scatter plots
    cluster_meta.json            names, profiles, metrics, k-sweep, labels (plain JSON,
                                 so it is independent of the pandas version)

Run once (from the project root):
    python save_clustering_models.py
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.neighbors import NearestCentroid

BASE = os.path.dirname(os.path.abspath(__file__))
CL_DIR = os.path.join(BASE, "Clustering")
CL_MODELS = os.path.join(CL_DIR, "models")
os.makedirs(CL_MODELS, exist_ok=True)

# the pickled pipeline refers to this module by name, so import it the same way the GUI does
sys.path.insert(0, os.path.join(CL_DIR, "app"))
import clustering_preprocessing as cp  # noqa: E402

RANDOM_STATE = 42
K_FINAL = 4            # same choice as the notebook (section 6.4)
K_RANGE = list(range(2, 11))

# ─────────────────────────────────────────────────────────────
# DATA + PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("CLUSTERING TRACK  (CC GENERAL.csv)")
print("=" * 60)

df = pd.read_csv(os.path.join(CL_DIR, "data", "CC GENERAL.csv"))
raw = df[cp.RAW_FEATURES].copy()

preprocessor = cp.build_preprocessor()
X = preprocessor.fit_transform(raw)
print(f"  Customers: {len(df):,} | features after engineering: {X.shape[1]}")


def evaluate(X_, labels_):
    return {"silhouette": float(silhouette_score(X_, labels_)),
            "davies_bouldin": float(davies_bouldin_score(X_, labels_)),
            "calinski_harabasz": float(calinski_harabasz_score(X_, labels_))}


# ─────────────────────────────────────────────────────────────
# K-MEANS  (sweep + final model)
# ─────────────────────────────────────────────────────────────
sweep = []
for k in K_RANGE:
    km_k = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    sweep.append({"k": k, "inertia": float(km_k.inertia_), **evaluate(X, km_k.labels_)})
print("  K sweep done (k = 2 ... 10)")

kmeans = KMeans(n_clusters=K_FINAL, init="k-means++", n_init=20, random_state=RANDOM_STATE).fit(X)
km_labels = kmeans.labels_
km_metrics = evaluate(X, km_labels)
print(f"  K-Means final  : silhouette={km_metrics['silhouette']:.4f} "
      f"DB={km_metrics['davies_bouldin']:.4f} CH={km_metrics['calinski_harabasz']:.1f}")

# ─────────────────────────────────────────────────────────────
# HIERARCHICAL (Ward)  +  nearest-centroid wrapper for prediction
# ─────────────────────────────────────────────────────────────
hier_labels = AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward").fit_predict(X)
hier_metrics = evaluate(X, hier_labels)
hier_model = NearestCentroid().fit(X, hier_labels)
ari = float(adjusted_rand_score(km_labels, hier_labels))
print(f"  Ward (k={K_FINAL})   : silhouette={hier_metrics['silhouette']:.4f} "
      f"DB={hier_metrics['davies_bouldin']:.4f} CH={hier_metrics['calinski_harabasz']:.1f}")
print(f"  K-Means vs Ward agreement (ARI): {ari:.3f}")

# ─────────────────────────────────────────────────────────────
# PCA for visualisation
# ─────────────────────────────────────────────────────────────
pca = PCA(n_components=3, random_state=RANDOM_STATE).fit(X)

# ─────────────────────────────────────────────────────────────
# SEGMENT PROFILES + HUMAN-READABLE NAMES
# ─────────────────────────────────────────────────────────────
raw_prof = raw.assign(_c=km_labels)
median_raw = raw_prof.groupby("_c").median()
mean_raw = raw_prof.groupby("_c").mean()
X_df = pd.DataFrame(X, columns=cp.FINAL_FEATURES)
z_prof = X_df.assign(_c=km_labels).groupby("_c").mean()          # mean z-score per cluster

SEGMENTS = {
    "Active Spenders": {
        "color": "#58a6ff",
        "summary": "Frequent, high-value card use with little or no cash advance.",
        "action": "Reward loyalty: cashback, premium cards and credit-limit increases."},
    "Revolving Hybrid Users": {
        "color": "#f78166",
        "summary": "High balances, regular purchases and recurring cash advances.",
        "action": "Monitor credit risk; offer balance-transfer and instalment plans."},
    "Cash-Advance Reliant": {
        "color": "#d2a8ff",
        "summary": "Almost no purchases - the card is used mainly for cash advances.",
        "action": "Highest-risk group: early-warning outreach and repayment support."},
    "Low-Activity Accounts": {
        "color": "#3fb950",
        "summary": "Low balances, small purchases and infrequent account activity.",
        "action": "Re-engage with activation offers and low-friction usage incentives."},
}
FALLBACK_COLORS = ["#58a6ff", "#f78166", "#d2a8ff", "#3fb950", "#e3b341", "#79c0ff",
                   "#ff7b72", "#a5d6ff", "#7ee787", "#ffa657"]

names = {}
if K_FINAL == 4:
    remaining = set(range(K_FINAL))
    c = mean_raw.loc[list(remaining), "PURCHASES_FREQUENCY"].idxmin()
    names[c] = "Cash-Advance Reliant"; remaining.discard(c)
    c = mean_raw.loc[list(remaining), "PURCHASES_FREQUENCY"].idxmax()
    names[c] = "Active Spenders"; remaining.discard(c)
    c = mean_raw.loc[list(remaining), "CASH_ADVANCE"].idxmax()
    names[c] = "Revolving Hybrid Users"; remaining.discard(c)
    names[remaining.pop()] = "Low-Activity Accounts"
else:
    names = {i: f"Segment {i + 1}" for i in range(K_FINAL)}

segments = []
for i in range(K_FINAL):
    nm = names[i]
    info = SEGMENTS.get(nm, {})
    segments.append({
        "id": int(i), "name": nm,
        "color": info.get("color", FALLBACK_COLORS[i % len(FALLBACK_COLORS)]),
        "summary": info.get("summary", ""), "action": info.get("action", ""),
        "size": int((km_labels == i).sum()),
        "median_raw": {k: float(v) for k, v in median_raw.loc[i].items()},
        "mean_raw": {k: float(v) for k, v in mean_raw.loc[i].items()},
        "z": {k: float(v) for k, v in z_prof.loc[i].items()},
    })

# map every Ward cluster to the K-Means segment it overlaps most (Hungarian matching)
cont = pd.crosstab(hier_labels, km_labels).reindex(index=range(K_FINAL), columns=range(K_FINAL), fill_value=0)
rows, cols = linear_sum_assignment(-cont.values)
hier_to_km = {int(r): int(c) for r, c in zip(rows, cols)}

# slider / input ranges for the predictor (1st - 99th percentile) + dataset medians
rng = {}
for f in cp.RAW_FEATURES:
    s = raw[f].dropna()
    rng[f] = {"min": float(s.min()), "p01": float(s.quantile(0.01)), "median": float(s.median()),
              "p99": float(s.quantile(0.99)), "max": float(s.max())}

meta = {
    "sklearn_version": sklearn.__version__,
    "k_final": K_FINAL,
    "n_customers": int(len(df)),
    "raw_features": cp.RAW_FEATURES,
    "final_features": cp.FINAL_FEATURES,
    "segments": segments,
    "hier_to_km": hier_to_km,
    "metrics": {"kmeans": km_metrics, "hierarchical": hier_metrics, "ari": ari},
    "sweep": sweep,
    "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
    "feature_ranges": rng,
    "cust_id": df["CUST_ID"].astype(str).tolist() if "CUST_ID" in df.columns else [],
    "km_labels": [int(v) for v in km_labels],
    "hier_labels": [int(v) for v in hier_labels],
}

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
joblib.dump(preprocessor, os.path.join(CL_MODELS, "cluster_preprocessor.pkl"))
joblib.dump(kmeans, os.path.join(CL_MODELS, "kmeans_final.pkl"))
joblib.dump(hier_model, os.path.join(CL_MODELS, "hierarchical_model.pkl"))
joblib.dump(pca, os.path.join(CL_MODELS, "cluster_pca.pkl"))
with open(os.path.join(CL_MODELS, "cluster_meta.json"), "w") as fh:
    json.dump(meta, fh)

print()
for s in segments:
    print(f"  cluster {s['id']}: {s['name']:<24} {s['size']:>5} customers")
print(f"\n  Saved 4 model files + cluster_meta.json -> {CL_MODELS}")
print("\nAll clustering artifacts saved successfully!")
