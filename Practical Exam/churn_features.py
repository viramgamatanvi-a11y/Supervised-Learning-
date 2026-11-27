"""Feature engineering used by the saved pipeline (churn_model.pkl).

Kept in a separate module so that the pickled sklearn Pipeline can be loaded
from any script / notebook (a function defined inside a notebook cannot be
unpickled elsewhere).  Mirrors Step 3 of the notebook exactly.
"""
import numpy as np
import pandas as pd

YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
               "MultipleLines", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "num_services"]
ONEHOT = {
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)",
                      "Electronic check", "Mailed check"],
}
TENURE_BINS = [-1, 12, 36, 60, np.inf]          # New, Mid, Senior, Loyal
TENURE_LABELS = [0, 1, 2, 3]                    # ordinal label-encoding


def engineer(df: pd.DataFrame, totalcharges_median: float = 1397.475) -> pd.DataFrame:
    """Raw Telco columns -> fully numeric model matrix (unscaled)."""
    d = df.copy()
    d = d.drop(columns=[c for c in ("customerID", "Churn") if c in d.columns])

    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d["TotalCharges"] = d["TotalCharges"].fillna(totalcharges_median)

    # --- feature engineering
    d["tenure_group"] = pd.cut(d["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(int)
    d["num_services"] = (d[ADDON_COLS] == "Yes").sum(axis=1)
    d["AutoPay"] = d["PaymentMethod"].str.contains("automatic", case=False).astype(int)

    # --- binary encoding (anything that is not 'Yes' -> 0, incl. 'No internet service')
    for c in YES_NO_COLS:
        d[c] = (d[c] == "Yes").astype(int)
    d["gender"] = (d["gender"] == "Male").astype(int)

    # --- one-hot (fixed categories so a single customer still gets every column)
    for col, cats in ONEHOT.items():
        for cat in cats:
            d[f"{col}_{cat}"] = (d[col] == cat).astype(int)
        d = d.drop(columns=col)
    return d
