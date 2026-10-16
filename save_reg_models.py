"""
save_reg_models.py
------------------
Retrains all regression models exactly as done in Regression.ipynb
and saves them to  Regression Track A/models/
so that the Streamlit GUI can load them for live prediction.

Run once:
    python save_reg_models.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

# --- Paths ---
BASE       = os.path.dirname(os.path.abspath(__file__))
REG_DIR    = os.path.join(BASE, "Regression Track A")
REG_MODELS = os.path.join(REG_DIR, "models")
os.makedirs(REG_MODELS, exist_ok=True)

# --- Load & clean data (mirrors notebook exactly) ---
print("Loading data ...")
df = pd.read_csv(os.path.join(REG_DIR, "communities.csv"), header=0, na_values=["?"])
target = df.columns[-1]
print(f"  Target: {target}")

missing_pct  = df.isna().mean() * 100
high_missing = missing_pct[missing_pct > 50].index.tolist()
id_cols      = ["state", "county", "community", "communityname", "fold"]
cols_to_drop = sorted(set(high_missing + id_cols))
df_clean     = df.drop(columns=cols_to_drop, errors="ignore").copy()

# Feature engineering
family_cols = [c for c in ["PctKids2Par", "PctFam2Par", "PctYoungKids2Par", "PctTeen2Par"]
               if c in df_clean.columns]
if family_cols:
    df_clean["FamilyStructureAvg"] = df_clean[family_cols].mean(axis=1)

X = df_clean.drop(columns=[target])
y = df_clean[target]

# --- Train / test split ---
y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_bins
)

# --- 1st-99th percentile capping ---
lower = X_train.quantile(0.01)
upper = X_train.quantile(0.99)
X_train = X_train.clip(lower, upper, axis=1)
X_test  = X_test.clip(lower, upper, axis=1)

# Imputer
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X.columns, index=X_test.index)

# Scaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns, index=X_test.index)

# --- Train and save ---
saved = []

def train_and_save(name, model, use_scaled=True):
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled  if use_scaled else X_test
    print(f"  Training {name} ...", end=" ", flush=True)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    r2   = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae  = float(mean_absolute_error(y_test, pred))
    print(f"R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("/", "_") + ".pkl"
    path  = os.path.join(REG_MODELS, fname)
    joblib.dump(model, path)
    saved.append((name, fname, r2))

print("\n" + "="*60)
print("REGRESSION MODEL TRAINING")
print("="*60)

train_and_save("DummyRegressor (Baseline)",      DummyRegressor(strategy="mean"),              True)
train_and_save("Linear Regression",              LinearRegression(),                            True)
train_and_save("Ridge Regression",               Ridge(alpha=1.0),                             True)
train_and_save("Lasso Regression",               Lasso(alpha=0.001, max_iter=10000),           True)
train_and_save("ElasticNet Regression",          ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000), True)
train_and_save("Polynomial Regression (PCA=10)", Pipeline([
    ("pca",  PCA(n_components=10)),
    ("poly", PolynomialFeatures(2, include_bias=False)),
    ("lr",   LinearRegression()),
]),                                                                                           True)
train_and_save("Decision Tree",                  DecisionTreeRegressor(random_state=42),       False)
train_and_save("Random Forest",                  RandomForestRegressor(n_estimators=100, random_state=42), False)
train_and_save("Gradient Boosting",              GradientBoostingRegressor(n_estimators=100, random_state=42), False)
train_and_save("SVR",                            SVR(kernel="rbf"),                            True)
train_and_save("KNN",                            KNeighborsRegressor(n_neighbors=5),           True)

print("\n" + "="*60)
print("SAVED MODELS (sorted by R2)")
print("="*60)
for name, fname, r2 in sorted(saved, key=lambda x: -x[2]):
    print(f"  {fname:<60}  R2={r2:.4f}")

print(f"\nAll {len(saved)} models saved to:\n  {REG_MODELS}")
print("Done!")
