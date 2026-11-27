# 🎯 Smart Outcome Predictor — Ensemble-Based ML System

An ensemble machine-learning system for an EdTech platform that predicts

* **Classification:** will a student **complete** the online course? (`completion_status`, 0/1)
* **Regression:** what will the student's **final performance score** be? (`final_score`, 0-100)

It compares **Bagging, Boosting (AdaBoost, Gradient Boosting, XGBoost), Voting (hard & soft) and Stacking** against single baseline models and ships a **Streamlit app** for live predictions.

> 🔗 **Live app:** `<paste your Streamlit Community Cloud URL here>`

---

## 📁 Repository structure

```
├── Smart_Outcome_Predictor.ipynb   # full analysis, Parts A-G (theory, EDA, models, evaluation, report)
├── app.py                          # Streamlit app
├── requirements.txt
├── Theory_Concepts.pdf             # definitions of all theory concepts
├── data/Smart_Outcome_Predictor_Dataset_5200.csv
├── models/                         # best classifier / regressor saved by the notebook (joblib)
├── outputs/                        # evaluation tables (test set + 5-fold CV) and feature importances
└── plots/                          # all charts used in the notebook
```

## 📊 Dataset
5 200 students · 19 columns: demographics (age, country, device, education), course info (level, category, start date, week), engagement
(sessions, time spent, videos, quiz attempts, assignments, forum posts), assessment (avg quiz score, attendance) and the two targets.
Missing values exist in `time_spent_hours`, `avg_quiz_score` and `attendance_rate` and are imputed inside the model pipeline.

## 🧪 Method (summary)
1. **Preprocessing** – median imputation + scaling for numeric, most-frequent imputation + one-hot for categorical, wrapped in a `Pipeline` (no leakage, also inside CV). `student_id` and the raw date are dropped; `start_month` is engineered; each target is excluded from the other task's features.
2. **Split** – 80/20 (stratified for classification).
3. **Models** – single Decision Tree & weak learner (baselines) · Bagging · AdaBoost · Gradient Boosting · XGBoost · Voting (hard/soft) · Stacking (Logistic / Ridge meta-learner) · plus Logistic/Ridge/Random Forest references.
4. **Evaluation** – Accuracy, Precision, Recall, F1, ROC-AUC (classification); MAE, RMSE, R² (regression); 5-fold cross-validation for stability; permutation importance.

## 🏆 Key results (test set)
| Task | Best ensemble | Result | Single decision tree |
|---|---|---|---|
| Completion | **Stacking Classifier** | ROC-AUC 0.792, F1 0.628, Acc 0.742 | ROC-AUC 0.606 |
| Final score | **Stacking Regressor** | RMSE 9.70, MAE 7.77, R² 0.496 | RMSE 14.05, R² −0.06 |

Main findings: ensembles remove the instability of single models; `avg_quiz_score` and `attendance_rate` are the strongest drivers; the data has a noise ceiling (~0.79 AUC / ~0.50 R²), so better features matter more than more complex models.
Full discussion, interpretations and the deployment recommendation are in the notebook (Parts F & G).

> The numbers above come from the run stored in the notebook. If you re-run with the real `xgboost` package the XGBoost rows (and tiny differences elsewhere) may change slightly.

## ▶️ Run it locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook Smart_Outcome_Predictor.ipynb         # Run All (≈ 5-6 min) – regenerates plots/, outputs/, models/
streamlit run app.py                                   # open http://localhost:8501
```
The app loads the models saved in `models/`; if they cannot be loaded (e.g. a different scikit-learn version) it automatically retrains quick Gradient-Boosting models from the CSV.

## 🌐 Deploy on Streamlit Community Cloud
1. Push this folder to a **public GitHub repository**.
2. Go to <https://share.streamlit.io> → *New app* → select the repo, branch `main`, main file `app.py` → *Deploy*.
3. Copy the app URL into the "Live app" line at the top of this README.

## 🖥️ App features
* 🔮 **Predict a student** – completion probability, predicted final score, risk level and suggested actions
* 📂 **Batch prediction** – upload a CSV, download predictions
* 📊 **Model comparison** – result tables, CV tables and charts
* 🔎 **Data insights** – completion rate / score by feature bucket and category

Link :
https://drive.google.com/file/d/1LXuWwJA9rG1bxteEW8KWg1ujn5AVpzoh/view?usp=sharing