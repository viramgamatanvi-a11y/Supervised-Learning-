import json
import pandas as pd
import streamlit as st
from utils import CATEGORIES, load_artifact, predict_probability

st.set_page_config(page_title="Risk Alert Classifier", page_icon="🚨", layout="wide")


@st.cache_resource
def get_artifact():
    return load_artifact()


art = get_artifact()

st.title("🚨 Risk Alert Classifier")
st.caption("Early-warning system that flags high-risk digital-banking customers (possible default / fraud).")

tab1, tab2 = st.tabs(["🔍 Predict customer risk", "📊 Model performance"])

with tab1:
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider("Alert threshold (probability ≥ threshold → High Risk)", 0.05, 0.95,
                                  float(art['threshold']), 0.05)
    st.sidebar.info(f"Model: **{art['model_name']}**\n\nLower threshold = fewer missed risky customers "
                    f"(fewer false negatives) but more false alarms.")

    st.subheader("Customer details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 90, 36)
        gender = st.selectbox("Gender", CATEGORIES['gender'])
        region = st.selectbox("Region", CATEGORIES['region'])
        employment_type = st.selectbox("Employment type", CATEGORIES['employment_type'])
        annual_income_inr = st.number_input("Annual income (INR)", 0, 500000, 40000, step=1000)
        credit_score = st.number_input("Credit score", 300, 850, 680)
    with c2:
        credit_utilization_ratio = st.slider("Credit utilization ratio", 0.0, 1.0, 0.35, 0.01)
        missed_payments_12m = st.number_input("Missed payments (last 12 months)", 0, 20, 1)
        avg_late_payment_days = st.number_input("Average late-payment days", 0.0, 100.0, 4.0, step=0.5)
        monthly_transaction_count = st.number_input("Monthly transaction count", 0, 300, 60)
        monthly_spend_inr = st.number_input("Monthly spend (INR)", 0, 200000, 20000, step=500)
    with c3:
        cash_advance_count_6m = st.number_input("Cash advances (last 6 months)", 0, 20, 1)
        complaints_last_6m = st.number_input("Complaints (last 6 months)", 0, 20, 1)
        failed_login_attempts_3m = st.number_input("Failed login attempts (last 3 months)", 0, 30, 2)
        account_tenure_months = st.number_input("Account tenure (months)", 0, 400, 60)
        debt_balance_inr = st.number_input("Debt balance (INR)", 0, 500000, 40000, step=1000)
        days_since_last_txn = st.number_input("Days since last transaction", 0, 365, 30)

    if st.button("Predict risk", type="primary"):
        inputs = dict(age=age, gender=gender, region=region, employment_type=employment_type,
                      annual_income_inr=annual_income_inr, credit_score=credit_score,
                      credit_utilization_ratio=credit_utilization_ratio, missed_payments_12m=missed_payments_12m,
                      avg_late_payment_days=avg_late_payment_days, monthly_transaction_count=monthly_transaction_count,
                      monthly_spend_inr=monthly_spend_inr, cash_advance_count_6m=cash_advance_count_6m,
                      complaints_last_6m=complaints_last_6m, failed_login_attempts_3m=failed_login_attempts_3m,
                      account_tenure_months=account_tenure_months, debt_balance_inr=debt_balance_inr,
                      days_since_last_txn=days_since_last_txn)
        p = predict_probability(art, inputs)
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("High-risk probability", f"{p:.1%}")
        m2.metric("Alert threshold", f"{threshold:.0%}")
        st.progress(min(max(p, 0.0), 1.0))
        if p >= threshold:
            st.error("🚨 HIGH RISK – raise an alert and send this customer for manual review.")
        else:
            st.success("✅ LOW RISK – no alert needed.")

with tab2:
    st.subheader("Model comparison (test set)")
    try:
        st.dataframe(pd.read_csv("results/model_comparison.csv").round(3), use_container_width=True)
        cfg = json.load(open("results/final_config.json"))
        st.caption(f"Final model: **{cfg['model_name']}** | decision threshold = {cfg['threshold']}")
    except Exception:
        st.info("Run the notebook first to generate results/model_comparison.csv")
    for img, cap in [("images/09_roc_curves.png", "ROC curves – all models"),
                     ("images/05_balancing_comparison.png", "Before vs after balancing"),
                     ("images/10_cm_final_model.png", "Confusion matrix – final model")]:
        try:
            st.image(img, caption=cap)
        except Exception:
            pass
