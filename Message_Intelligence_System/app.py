import re, pickle
import numpy as np, pandas as pd
import streamlit as st

st.set_page_config(page_title="Message Intelligence System", page_icon="📩", layout="wide")

@st.cache_resource
def load():
    with open("models/message_models.pkl", "rb") as f:
        return pickle.load(f)

bundle = load(); models, meta = bundle["models"], bundle["meta"]
FEATURES = meta["features"]; D = meta["defaults"]; R = meta["ranges"]

st.title("📩 Message Intelligence System")
st.caption("Spam vs Legitimate message classifier – KNN · SVM · Naive Bayes")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model comparison", "ℹ️ About"])

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model", list(models.keys()), index=1)
    threshold = st.slider("Spam probability threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Lower → catch more spam (higher recall); higher → fewer false alarms (higher precision)")
    st.info("Tip: SVM gave the best precision/recall on the realistic stress-test.")

with tab1:
    st.subheader("1️⃣ Message text (optional helper – auto-fills the text features)")
    text = st.text_area("Paste a message", "Act fast! Your account will be suspended unless you pay ₹50,000 now! http://bit.ly/9897", height=90)
    auto = {
        "message_length": len(text),
        "word_count": len(text.split()),
        "num_urls": len(re.findall(r"https?://\S+|www\.\S+", text)),
        "num_digits": sum(ch.isdigit() for ch in text),
        "num_special_chars": len(re.findall(r"[^\w\s.,'?:/\-!₹]", text)),
    }
    st.subheader("2️⃣ Features sent to the model (edit freely)")
    vals = {}
    cols = st.columns(3)
    for i, f in enumerate(FEATURES):
        lo, hi = R[f]; default = auto.get(f, D[f])
        hi = max(hi, default)
        with cols[i % 3]:
            if f in ("num_urls", "num_digits", "num_special_chars", "spam_keyword_score", "legit_keyword_score",
                     "message_length", "word_count", "hour_of_day", "day_of_week"):
                vals[f] = st.number_input(f, min_value=int(lo), max_value=int(hi), value=int(min(max(default, lo), hi)), step=1, key=f + str(hash(text) if f in auto else ""))
            else:
                vals[f] = st.number_input(f, min_value=float(lo), max_value=float(hi * 1.5), value=float(default), key=f)
    if st.button("🚀 Classify message", type="primary"):
        X = pd.DataFrame([vals])[FEATURES]
        m = models[model_name]
        p_spam = float(m.predict_proba(X)[0, 1])
        label = int(p_spam >= threshold)
        c1, c2 = st.columns(2)
        if label:
            c1.error(f"🚨 SPAM  (probability {p_spam:.1%})")
        else:
            c1.success(f"✅ LEGITIMATE  (spam probability {p_spam:.1%})")
        c2.progress(min(max(p_spam, 0.0), 1.0), text=f"Spam probability – {model_name}")
        st.markdown("**All models:**")
        st.dataframe(pd.DataFrame({k: {"P(Spam)": round(float(v.predict_proba(X)[0, 1]), 4),
                                       "Prediction": "SPAM" if v.predict_proba(X)[0, 1] >= threshold else "LEGIT"} for k, v in models.items()}).T)

with tab2:
    st.subheader("Test-set metrics (original 12 features)")
    st.dataframe(pd.DataFrame(meta["metrics"]).T)
    st.caption("All models score ≈100% on the original data because `num_special_chars` separates the classes perfectly. "
               "On the realistic stress-test (no special-chars / keyword features): SVM-Linear F1 0.985 > SVM-RBF 0.982 > KNN 0.958 > Naive Bayes 0.919.")
    st.table(pd.DataFrame({"Model": ["SVM (Linear)", "SVM (RBF)", "KNN (K=3)", "Naive Bayes"],
                           "Precision": [0.995, 0.995, 0.984, 0.886], "Recall": [0.974, 0.969, 0.933, 0.954], "F1": [0.985, 0.982, 0.958, 0.919]}))
    import os
    for img in sorted(os.listdir("charts")) if os.path.isdir("charts") else []:
        if img.startswith(("13", "15", "17", "19")):
            st.image(f"charts/{img}", caption=img.replace("_", " ").replace(".png", ""))

with tab3:
    st.markdown("""
**Pipeline:** median imputation → StandardScaler → classifier (fitted on a stratified 80/20 split).  
**Models:** KNN (distance-based), SVM-RBF (margin-based), Gaussian Naive Bayes (probabilistic, Bayes' theorem).  
**Labels:** `0` = Legitimate, `1` = Spam.  
The keyword scores need a lexicon, so they are entered manually here; text-count features are auto-filled from the message.
""")
