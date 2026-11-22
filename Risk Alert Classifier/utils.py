"""Helper functions shared by the Streamlit app (feature building, prediction, fallback training)."""
import json
import numpy as np
import pandas as pd
import joblib

CATEGORIES = {
    'gender': ['Female', 'Male', 'Other'],
    'region': ['Central', 'East', 'North', 'South', 'West', 'Unknown'],
    'employment_type': ['Salaried', 'Self-Employed', 'Unemployed', 'Retired', 'Student', 'Unknown'],
}
RS = 42


def build_feature_row(inputs: dict, columns: list) -> pd.DataFrame:
    """Turn the raw form inputs into the exact one-hot encoded column layout used in training."""
    row = {}
    for col in columns:
        matched = False
        for cat in CATEGORIES:
            if col.startswith(cat + '_'):
                row[col] = int(inputs[cat] == col[len(cat) + 1:])
                matched = True
                break
        if not matched:
            row[col] = inputs[col]
    return pd.DataFrame([row], columns=columns)


def predict_probability(artifact: dict, inputs: dict) -> float:
    X = build_feature_row(inputs, artifact['columns'])
    X = artifact['scaler'].transform(X)
    X = artifact['imputer'].transform(X)
    return float(artifact['model'].predict_proba(X)[0, 1])


def fit_fallback(csv_path='Risk_Alert_Classifier_Dataset_4600.csv', config_path='results/final_config.json') -> dict:
    """Re-train 'Logistic + SMOTE' from the CSV if model.pkl cannot be loaded (e.g. scikit-learn version mismatch)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import KNNImputer
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE

    cfg = json.load(open(config_path))
    df = pd.read_csv(csv_path)
    df['last_transaction_date'] = pd.to_datetime(df['last_transaction_date'])
    df['days_since_last_txn'] = (df['last_transaction_date'].max() - df['last_transaction_date']).dt.days
    y = df['risk_status']
    X = df.drop(columns=['risk_status', 'customer_id', 'last_transaction_date'])
    cats = list(CATEGORIES)
    X[cats] = X[cats].fillna('Unknown')
    X = pd.get_dummies(X, columns=cats, drop_first=True, dtype=int)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X_train)
    imputer = KNNImputer(n_neighbors=5); Xi = imputer.fit_transform(Xs)
    Xr, yr = SMOTE(random_state=RS).fit_resample(Xi, y_train.values)
    model = LogisticRegression(max_iter=1000, random_state=RS).fit(Xr, yr)
    return {'model': model, 'scaler': scaler, 'imputer': imputer, 'columns': X.columns.tolist(),
            'threshold': cfg['threshold'], 'ref_date': cfg['ref_date'], 'model_name': 'Logistic + SMOTE (re-trained)'}


def load_artifact() -> dict:
    try:
        return joblib.load('model.pkl')
    except Exception:
        return fit_fallback()
