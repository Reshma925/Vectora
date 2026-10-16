# Reusable preprocessing for the credit-card customer segmentation project.
# Written by notebooks/clustering.ipynb - import this module before loading any saved model:
#     import sys; sys.path.append("app"); import clustering_preprocessing
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer

RAW_FEATURES = ["BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
                "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
                "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX",
                "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"]

# skewed monetary / count variables -> log1p
LOG_COLS = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
            "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]

# engineered behaviour ratios (also log1p-transformed because they are heavy-tailed)
ENGINEERED = ["BALANCE_TO_LIMIT", "PAYMENT_TO_MIN_RATIO", "AVG_PURCHASE_PER_TRX", "AVG_CASH_ADV_PER_TRX"]
FINAL_FEATURES = RAW_FEATURES + ENGINEERED


def _safe_div(a, b):
    # a / b where b > 0, NaN elsewhere (NaN is imputed later)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=(b > 0))
    return out


def add_engineered_features(df):
    # add the 4 engineered ratio features (raw scale) to a copy of df
    d = pd.DataFrame(df).copy()
    d["BALANCE_TO_LIMIT"] = _safe_div(d["BALANCE"], d["CREDIT_LIMIT"])
    d["PAYMENT_TO_MIN_RATIO"] = _safe_div(d["PAYMENTS"], d["MINIMUM_PAYMENTS"])
    d["AVG_PURCHASE_PER_TRX"] = np.where(d["PURCHASES_TRX"] > 0, _safe_div(d["PURCHASES"], d["PURCHASES_TRX"]), 0.0)
    d["AVG_CASH_ADV_PER_TRX"] = np.where(d["CASH_ADVANCE_TRX"] > 0, _safe_div(d["CASH_ADVANCE"], d["CASH_ADVANCE_TRX"]), 0.0)
    return d


def engineer_features(df):
    # raw customer table -> engineered + log1p-transformed feature table (NaNs kept for the imputer)
    d = add_engineered_features(df)
    for c in LOG_COLS + ENGINEERED:
        d[c] = np.log1p(d[c].astype(float).clip(lower=0))
    return d[FINAL_FEATURES].astype(float)


def build_preprocessor():
    # engineer -> median imputation -> standard scaling
    return Pipeline([("engineer", FunctionTransformer(engineer_features, validate=False)),
                     ("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
