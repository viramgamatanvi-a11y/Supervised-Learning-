"""Re-train the final model (RBF-kernel SVR, tuned in the notebook) from the CSV and save model.joblib.
Run:  python train_model.py     (also used by app.py as a fallback if model.joblib cannot be loaded)"""
import json
import joblib
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA = "data/Advanced_Regression_HousePrice_Dataset_3800.csv"


def train(save=False):
    info = json.load(open("model_info.json"))
    p = info["best_params"]
    df = pd.read_csv(DATA, parse_dates=["sale_date"])
    df["sale_year"] = df["sale_date"].dt.year
    X, y = df[info["features"]], df[info["target"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()),
                            ("svr", SVR(kernel="rbf", C=p["C"], gamma=p["gamma"], epsilon=p["epsilon"], cache_size=500))]),
        transformer=StandardScaler()).fit(X_train, y_train)
    print(f"Test R2 = {model.score(X_test, y_test):.4f}")
    if save:
        joblib.dump(model, "model.joblib", compress=3)
    return model


if __name__ == "__main__":
    train(save=True)
