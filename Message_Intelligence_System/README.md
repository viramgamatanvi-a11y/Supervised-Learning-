# 📩 Message Intelligence System
Spam vs Legitimate message classification using **KNN**, **SVM** and **Naive Bayes**, combining probability theory (conditional probability, Bayes' theorem) with distance-based, margin-based and probabilistic classifiers.

🔗 **Live Streamlit app:** `<paste your Streamlit Community Cloud URL here>`

## 🎯 Objective
Classify digital messages as `0 = Legitimate` or `1 = Spam` and explain model behaviour with probability concepts.

## 📁 Repository structure
```
├── Message_Intelligence_System.ipynb   # Full project: Parts A–G (code, plots, interpretations)
├── app.py                              # Streamlit app
├── requirements.txt
├── Theory_Concepts.pdf                 # Theory definitions (Part A)
├── data/Message_Intelligence_Dataset_5200.csv
├── models/message_models.pkl           # Fitted KNN / SVM / NB pipelines used by the app
├── models/final_metrics.csv
└── charts/                             # All exported plots
```

## 🧪 Dataset
5,200 messages, 12 numeric features (message length, word count, URLs, digits, special characters, spam/legit keyword scores, sender activity, account age, messages in last 24 h, hour, weekday) + target `spam_label` (18.7% spam). `message_id`, `message_text` and raw `timestamp` are not used as model inputs.

## ⚙️ Method
1. Stratified 80/20 train/test split (random_state=42) **before** any fitting → no leakage
2. Median imputation (3 columns with missing values) + `StandardScaler`
3. **KNN:** K sweep (CV), 5 distance metrics, scaling effect, error analysis
4. **SVM:** linear / RBF / polynomial kernels, GridSearch (C, γ), margin & support-vector analysis
5. **Naive Bayes:** GaussianNB + manual conditional probabilities and Bayes' theorem (matches `BernoulliNB` exactly)
6. Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, PR curves

## 📊 Results
| Setting | Model | Acc | Prec | Recall | F1 |
|---|---|---|---|---|---|
| Original 12 features | KNN / SVM / NB | 1.000 | 1.000 | 1.000 | 1.000 |
| Stress-test (no `num_special_chars`, no keyword scores) | KNN (K=3) | 0.985 | 0.984 | 0.933 | 0.958 |
| | SVM-RBF | 0.993 | 0.995 | 0.969 | 0.982 |
| | **SVM-Linear** | **0.994** | **0.995** | **0.974** | **0.985** |
| | Naive Bayes | 0.968 | 0.886 | 0.954 | 0.919 |

⚠️ **Important finding:** `num_special_chars` alone separates the classes perfectly in this dataset (and keyword scores are nearly deterministic), so all models score 100%. The stress-test removes these features to get a realistic comparison.

**Recommendation:** deploy the (linear) SVM as the main classifier, Naive Bayes as a fast explainable scorer / review-queue feeder, avoid KNN in production; use a two-threshold (block / review / allow) policy.

## ▶️ Run locally
```bash
pip install -r requirements.txt
jupyter notebook Message_Intelligence_System.ipynb   # re-run all cells
streamlit run app.py
```

## ☁️ Deploy on Streamlit Community Cloud
Push this folder to GitHub → https://share.streamlit.io → *New app* → select repo, branch `main`, file `app.py` → Deploy → paste the URL above.

Link :
https://drive.google.com/file/d/1usWoxmeQZ_3ZXfhCpjbHWWA_LTxbpe35/view?usp=sharing