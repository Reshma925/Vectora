"""
save_artifacts.py
-----------------
Runs the preprocessing pipelines for both the Classification and Regression
tracks and saves the fitted transformers (scaler, encoder, imputer) plus
feature-column lists as .pkl files so the Streamlit GUI can use them without
re-running the full notebooks.

Run once:
    python save_artifacts.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
CLS_DIR   = os.path.join(BASE, "ML_CLASSIFICATION_TRACKA")
REG_DIR   = os.path.join(BASE, "Regression Track A")

CLS_MODELS = os.path.join(CLS_DIR, "models")
REG_MODELS = os.path.join(REG_DIR, "models")
os.makedirs(REG_MODELS, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CLASSIFICATION TRACK  (star_classification.csv)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("CLASSIFICATION TRACK")
print("=" * 60)

cls_csv = os.path.join(CLS_DIR, "star_classification.csv")
df_cls = pd.read_csv(cls_csv)

# same cleaning as notebook
df_cls.drop(columns=["class_numeric"], inplace=True, errors="ignore")
df_cls = df_cls.drop_duplicates().reset_index(drop=True)

# feature engineering
df_cls["r_i"] = df_cls["r"] - df_cls["i"]

# separate X / y
target_col = "class"
X_cls = df_cls.drop(columns=[target_col])
y_cls = df_cls[target_col]

feature_cols_cls = X_cls.columns.tolist()

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.20, random_state=42, stratify=y_cls
)

# IQR capping (replicate notebook exactly)
outlier_bounds = {}
for feature in X_train_c.select_dtypes(include=["int64", "float64"]).columns:
    Q1 = X_train_c[feature].quantile(0.25)
    Q3 = X_train_c[feature].quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    if ((X_train_c[feature] < lb) | (X_train_c[feature] > ub)).sum() > 0:
        outlier_bounds[feature] = {"lower": lb, "upper": ub}
        X_train_c[feature] = X_train_c[feature].clip(lb, ub)
        X_test_c[feature]  = X_test_c[feature].clip(lb, ub)

# label encoder (fit on train only)
label_encoder = LabelEncoder()
label_encoder.fit_transform(y_train_c)

# scaler (fit on train only)
cls_scaler = StandardScaler()
cls_scaler.fit(X_train_c)

# save
joblib.dump(cls_scaler,       os.path.join(CLS_MODELS, "cls_scaler.pkl"))
joblib.dump(label_encoder,    os.path.join(CLS_MODELS, "cls_label_encoder.pkl"))
joblib.dump(feature_cols_cls, os.path.join(CLS_MODELS, "cls_feature_cols.pkl"))
joblib.dump(outlier_bounds,   os.path.join(CLS_MODELS, "cls_outlier_bounds.pkl"))

print(f"  Scaler saved        -> {CLS_MODELS}/cls_scaler.pkl")
print(f"  LabelEncoder saved  -> {CLS_MODELS}/cls_label_encoder.pkl")
print(f"  Feature cols saved  -> {CLS_MODELS}/cls_feature_cols.pkl")
print(f"  Outlier bounds saved-> {CLS_MODELS}/cls_outlier_bounds.pkl")
print(f"  Classes: {label_encoder.classes_}")
print(f"  Feature count: {len(feature_cols_cls)}")

# ─────────────────────────────────────────────────────────────
# REGRESSION TRACK  (communities.csv)
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("REGRESSION TRACK")
print("=" * 60)

reg_csv = os.path.join(REG_DIR, "communities.csv")
df_reg = pd.read_csv(reg_csv, header=0, na_values=["?"])

target_reg = df_reg.columns[-1]
print(f"  Target variable: {target_reg}")

# drop high-missing + id cols (same as notebook)
missing_pct = df_reg.isna().mean() * 100
high_missing = missing_pct[missing_pct > 50].index.tolist()
id_cols = ["state", "county", "community", "communityname", "fold"]
cols_to_drop = sorted(set(high_missing + id_cols))
df_clean = df_reg.drop(columns=cols_to_drop, errors="ignore").copy()

# feature engineering: FamilyStructureAvg
family_cols = [c for c in ["PctKids2Par", "PctFam2Par", "PctYoungKids2Par", "PctTeen2Par"]
               if c in df_clean.columns]
if family_cols:
    df_clean["FamilyStructureAvg"] = df_clean[family_cols].mean(axis=1)

X_reg = df_clean.drop(columns=[target_reg])
y_reg = df_clean[target_reg]

feature_cols_reg = X_reg.columns.tolist()

y_bins = pd.qcut(y_reg, q=5, labels=False, duplicates="drop")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42, stratify=y_bins
)

# percentile capping (1st-99th from train)
lower_r = X_train_r.quantile(0.01)
upper_r = X_train_r.quantile(0.99)
X_train_r = X_train_r.clip(lower_r, upper_r, axis=1)
X_test_r  = X_test_r.clip(lower_r, upper_r, axis=1)

# imputer + scaler (fit on train)
reg_imputer = SimpleImputer(strategy="median")
X_train_r_imp = pd.DataFrame(
    reg_imputer.fit_transform(X_train_r),
    columns=X_reg.columns, index=X_train_r.index
)

reg_scaler = StandardScaler()
reg_scaler.fit(X_train_r_imp)

# save
joblib.dump(reg_imputer,      os.path.join(REG_MODELS, "reg_imputer.pkl"))
joblib.dump(reg_scaler,       os.path.join(REG_MODELS, "reg_scaler.pkl"))
joblib.dump(feature_cols_reg, os.path.join(REG_MODELS, "reg_feature_cols.pkl"))
joblib.dump({"lower": lower_r, "upper": upper_r},
            os.path.join(REG_MODELS, "reg_clip_bounds.pkl"))

print(f"  Imputer saved       -> {REG_MODELS}/reg_imputer.pkl")
print(f"  Scaler saved        -> {REG_MODELS}/reg_scaler.pkl")
print(f"  Feature cols saved  -> {REG_MODELS}/reg_feature_cols.pkl")
print(f"  Clip bounds saved   -> {REG_MODELS}/reg_clip_bounds.pkl")
print(f"  Feature count: {len(feature_cols_reg)}")
print(f"  Target: {target_reg}")

print()
print("All artifacts saved successfully!")
