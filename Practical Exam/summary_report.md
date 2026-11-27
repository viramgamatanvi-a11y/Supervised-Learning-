# Summary Report — Telco Customer Churn Prediction

## 1. Business problem & dataset
A Jio/Airtel-style telecom is losing paying subscribers every month and cannot tell who is about to leave. Acquiring a new customer (CAC) costs several times more than retaining one, and every churner also takes away their future lifetime value (CLV). We used the IBM Telco Customer Churn dataset (7,043 customers, 21 columns) to predict the binary target `Churn`. **26.5% of customers churned** (1,869 vs 5,174 retained), so the data is moderately imbalanced.

## 2. Preprocessing & imbalance strategy
- Dropped `customerID`; `TotalCharges` held blank strings for 11 brand-new customers (tenure = 0) → converted to float and filled with the median.
- Engineered `tenure_group` (New/Mid/Senior/Loyal, label-encoded in order), `num_services` (count of add-ons) and `AutoPay`. Yes/No columns mapped to 1/0; `InternetService`, `Contract`, `PaymentMethod` one-hot encoded.
- Stratified 80/20 split (`random_state=42`); `StandardScaler` fitted on train only.
- Imbalance: **SMOTE on the training set only** for KNN, Naive Bayes and SVM; `class_weight='balanced'` for the Decision Tree. For the tree, class_weight gave better churn recall than SMOTE (0.783 vs 0.754).

## 3. Model comparison (test set, churn class)
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| KNN (k=15) | 0.727 | 0.491 | **0.794** | 0.607 | 0.815 |
| Naive Bayes | 0.736 | 0.502 | 0.786 | 0.613 | 0.822 |
| Decision Tree (depth 3) | 0.746 | 0.514 | 0.783 | **0.621** | 0.816 |
| SVM (RBF, C=0.1) | 0.750 | 0.519 | 0.751 | 0.614 | **0.829** |

**Recommendation: Decision Tree (depth 3, balanced).** Its recall is within ~1 point of the best model while it has the best F1 and better precision than KNN/NB (fewer wasted retention calls), trains and predicts almost instantly, and gives rules the business can read. SVM has the best AUC but the lowest recall and is a black box; KNN is slow at prediction time. A False Negative (missed churner) costs far more than a False Positive (an unnecessary offer), which is why recall was the primary metric. The error analysis shows the tree still misses ~22% of churners — mostly *long-tenure customers with contracts and more services*, who do not fit the usual risk profile.

## 4. Top 3 churn signals & actions
1. **Month-to-month contract** — 42.7% churn vs 2.8% on two-year contracts. *Action:* offer discounts/free add-ons to move customers onto 1–2 year plans.
2. **Short tenure** — 47% of customers in their first 12 months churn. *Action:* onboarding journey and a proactive call in months 2–6.
3. **Fibre-optic service with high monthly bill** — fibre churns at 41.9% vs 19.0% for DSL; month-to-month + high bill + tenure ≤ 12 months churns at 70%. *Action:* review fibre pricing/quality, bundle security and tech-support add-ons, and put this segment first on the daily call list (rank by probability and call the top ~500).

Bonus lever: electronic-check payers churn at 45.3% vs ~15–17% on automatic payment → push AutoPay.

## 5. Next steps
- **More data:** usage trends, support tickets, contract-expiry dates and competitor offers — needed to catch the "unexpected" churners the tree misses.
- **Better models:** gradient boosting (XGBoost/LightGBM) with threshold tuning on a cost matrix (FN vs FP cost); deep learning only if sequential usage data becomes available.
- **Deployment:** wrap the pipeline in a real-time scoring API (FastAPI) / nightly batch scoring, and monitor drift and recall monthly.
