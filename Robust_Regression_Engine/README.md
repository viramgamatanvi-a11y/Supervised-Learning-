# 🏠 Advanced Regression Pipeline for House Price Prediction

Regularization • Cross-Validation • Tree-Based Models • Support Vector Regression • Streamlit app

**Live Streamlit app:** `<paste your Streamlit Cloud URL here>`

## 📌 Problem
A real-estate analytics company predicts house prices with a basic regression model that **overfits and is unstable** across datasets. This project builds a robust pipeline that applies **regularization**, uses proper **cross-validation**, compares **linear vs non-linear regressors** and selects the best model by validation performance, so that it generalizes to unseen data.

## 📊 Dataset
`data/Advanced_Regression_HousePrice_Dataset_3800.csv` — 3,800 properties, no missing values, no duplicates.

| Column | Meaning |
|---|---|
| `property_id` | identifier (dropped) |
| `sale_date` | monthly sale date 2010–2023 (→ `sale_year`, also used for Time-Series CV) |
| `area_sqft`, `bedrooms`, `bathrooms` | size & structure |
| `location_score`, `distance_city_km`, `near_school`, `near_metro`, `crime_rate_index` | location indicators |
| `property_age` | age of the property |
| **`house_price_inr`** | **target: price in ₹** |

## 🗂️ Repository structure
```
├── Advanced_Regression_HousePrice.ipynb   # full solution, Parts A–H, outputs + interpretations
├── app.py                                  # Streamlit app
├── train_model.py                          # re-trains the final model (also app fallback)
├── model.joblib / model_info.json          # saved best model + metadata
├── requirements.txt
├── Theory_Concepts.pdf                     # theory & definitions
├── data/                                   # dataset
├── images/                                 # all charts (17 plots)
└── outputs/model_comparison.csv            # metrics of every model
```

## 🔬 Methodology
1. **EDA & preparation** – target distribution, correlation heat-map, outlier check, `sale_year` feature, 80/20 split (`random_state=42`), scaling inside `Pipeline` (no leakage).
2. **Regularized linear models** – OLS, Ridge (best α ≈ 1.7), Lasso (best α ≈ 11,250); α tuned by 5-fold CV; coefficient paths compared.
3. **Cross-validation strategies** – K-Fold, Stratified K-Fold (binned target), LOOCV, Time Series Split compared on tuned Ridge.
4. **Tree models** – unrestricted vs tuned Decision Tree (`max_depth`, `min_samples_split`, `min_samples_leaf`) and tuned Random Forest.
5. **SVR** – linear and RBF kernels, tuned C, γ, ε (target also standardised).
6. **Comparison** – MSE, MAE, RMSE, R² for 8 models, over/under-fitting diagnosis, residual analysis, learning curve.

## 🏆 Results (errors in ₹ lakh; 1 lakh = 100,000)
| Model | Family | CV RMSE | Test RMSE | Test MAE | Train R² | Test R² |
|---|---|---|---|---|---|---|
| SVR (RBF kernel) | SVR | 22.1 | 22.1 | 16.1 | 0.9385 | 0.9392 |
| Random Forest (tuned) | Tree-based | 23.8 | 23.9 | 17.5 | 0.9895 | 0.9293 |
| Lasso (L1) | Linear | 24.9 | 25.4 | 19.5 | 0.9174 | 0.9198 |
| Ridge (L2) | Linear | 25.0 | 25.4 | 19.4 | 0.9174 | 0.9199 |
| Linear Regression (OLS) | Linear | 25.0 | 25.4 | 19.4 | 0.9174 | 0.9199 |
| SVR (linear kernel) | SVR | 25.0 | 25.4 | 19.6 | 0.9172 | 0.9197 |
| Decision Tree (tuned) | Tree-based | 27.0 | 26.9 | 19.8 | 0.9363 | 0.9103 |
| Decision Tree (unrestricted) | Tree-based | 34.2 | 31.9 | 23.4 | 1.0000 | 0.8740 |

**Best model: SVR with RBF kernel** (C = 100, γ = 0.003, ε = 0.02 on standardised target): lowest validation and test error, test R² ≈ 0.939, average error ≈ ₹16.1 lakh (≈ 8% of the mean price), and no train–test gap.

### Key findings
- **Best model:** RBF-SVR – chosen by cross-validated RMSE, confirmed on the untouched test set.
- **Regularization:** Ridge/Lasso ≈ OLS in accuracy (OLS was not badly overfitted) but give stable coefficients; Lasso with a stronger α removes `distance_city_km`, `near_school`, `near_metro`, `crime_rate_index`. Regularizing the tree (depth/leaf size) fixed the unrestricted tree (train R² 1.00 → test 0.87).
- **Cross-validation:** K-Fold, Stratified, LOOCV and Time-Series split agree (RMSE ≈ ₹24.9 lakh for Ridge), so estimates are reliable; LOOCV costs 3,040 fits.
- **Linear vs non-linear:** linear models R² ≈ 0.92; tuned non-linear models are better (RBF-SVR 0.939, Random Forest 0.929).
- **Business insight:** `area_sqft` is the dominant price driver, then `location_score`; older houses are cheaper; the model is a good valuation aid, but errors are larger for expensive houses.

## ▶️ Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # Python 3.11+
jupyter notebook Advanced_Regression_HousePrice.ipynb  # re-run all cells (~5–8 min)
streamlit run app.py
```

## ☁️ Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to share.streamlit.io → *New app* → select the repo, branch `main`, main file `app.py` → *Deploy*.
3. Copy the app URL into the **Live Streamlit app** line at the top of this README.

## 📄 Documents
- Theory & definitions: `Theory_Concepts.pdf`
- All charts with interpretations: the notebook and `images/`

## VIDEO LINK 
- https://drive.google.com/file/d/1Ycbql10aekGP-7ZondyOeVFrCA8IkQpg/view?usp=sharing
