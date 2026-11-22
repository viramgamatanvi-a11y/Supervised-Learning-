"""Streamlit app: House Price Prediction (best model from the Advanced Regression project)."""
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    try:
        return joblib.load("model.joblib")
    except Exception:                      # e.g. scikit-learn version mismatch -> retrain from the CSV
        from train_model import train
        return train()


@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)


model, info = load_model(), load_info()
FEATURES, RNG = info["features"], info["feature_ranges"]

st.title("🏠 House Price Prediction")
st.caption(f"Model: **{info['best_model']}** · trained on 3,800 properties · "
           f"test R² = {info['test_metrics']['R2']:.3f}")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model comparison", "ℹ️ About"])

with tab1:
    st.subheader("Enter property details")
    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.number_input("Area (sq.ft)", 300, 6000, 1800, 50)
        bed = st.slider("Bedrooms", 1, 8, 3)
        bath = st.slider("Bathrooms", 1, 7, 3)
    with c2:
        loc = st.slider("Location score (1-10)", 1.0, 10.0, 6.5, 0.1)
        dist = st.slider("Distance from city centre (km)", 0.5, 45.0, 13.0, 0.5)
        crime = st.slider("Crime rate index", 0.0, 15.0, 4.2, 0.1)
    with c3:
        age = st.slider("Property age (years)", 0, 100, 20)
        year = st.slider("Sale year", 2010, 2030, 2023)
        school = st.checkbox("Near a school", True)
        metro = st.checkbox("Near a metro station", False)

    row = {"area_sqft": area, "bedrooms": bed, "bathrooms": bath, "location_score": loc,
           "property_age": age, "distance_city_km": dist, "near_school": int(school),
           "near_metro": int(metro), "crime_rate_index": crime, "sale_year": year}
    X = pd.DataFrame([row])[FEATURES]

    if st.button("Predict price", type="primary"):
        price = float(model.predict(X)[0])
        mae = info["test_metrics"]["MAE"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated price", f"₹ {price:,.0f}")
        m2.metric("In crore", f"₹ {price / 1e7:,.2f} Cr")
        m3.metric("Typical error (test MAE)", f"± ₹ {mae / 1e5:,.1f} L")
        st.info(f"Likely range (± 1 MAE): ₹ {(price - mae) / 1e7:,.2f} Cr to ₹ {(price + mae) / 1e7:,.2f} Cr")
        if year > 2023 or any(row[c] < RNG[c]["min"] or row[c] > RNG[c]["max"] for c in FEATURES):
            st.warning("Some inputs are outside the range seen in the training data, so treat the estimate with care.")

    st.divider()
    st.subheader("Batch prediction (CSV)")
    up = st.file_uploader(f"Upload a CSV with columns: {', '.join(FEATURES)}", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["predicted_price_inr"] = model.predict(data[FEATURES]).round(0)
            st.dataframe(data)
            st.download_button("Download predictions", data.to_csv(index=False), "predictions.csv", "text/csv")

with tab2:
    st.subheader("All models (train / validation / test)")
    try:
        comp = pd.read_csv("outputs/model_comparison.csv", index_col=0)
        st.dataframe(comp[["Group", "Train_RMSE", "CV_RMSE", "Test_RMSE", "Test_MAE", "Train_R2", "Test_R2"]]
                     .sort_values("CV_RMSE").style.format({"Train_RMSE": "{:,.0f}", "CV_RMSE": "{:,.0f}",
                                                           "Test_RMSE": "{:,.0f}", "Test_MAE": "{:,.0f}",
                                                           "Train_R2": "{:.4f}", "Test_R2": "{:.4f}"}))
        st.bar_chart(comp["Test_RMSE"].sort_values())
    except FileNotFoundError:
        st.warning("outputs/model_comparison.csv not found.")
    for img, cap in [("images/13_model_comparison.png", "Model comparison"),
                     ("images/15_actual_vs_predicted.png", "Actual vs predicted")]:
        try:
            st.image(img, caption=cap)
        except Exception:
            pass

with tab3:
    st.markdown("""
**Project:** Advanced Regression Pipeline for House Price Prediction  
**Techniques:** Ridge/Lasso regularization, K-Fold / Stratified / LOOCV / Time-Series cross-validation,
Decision Tree, Random Forest, Support Vector Regression.  
The model was selected by cross-validated RMSE on the training data and confirmed on an unseen test set.
Predictions are estimates, not a professional valuation.
""")
