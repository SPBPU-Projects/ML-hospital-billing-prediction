# Hospital Billing Amount Prediction (Machine Learning)

![Python](https://img.shields.io/badge/Python-3.x-yellow)
![Pandas](https://img.shields.io/badge/Library-pandas-blue)
![NumPy](https://img.shields.io/badge/Library-numpy-blue)
![Matplotlib](https://img.shields.io/badge/Visualization-matplotlib-orange)
![ScikitLearn](https://img.shields.io/badge/ML-scikit--learn-green)
![Regression](https://img.shields.io/badge/Task-Regression-informational)
![Healthcare](https://img.shields.io/badge/Domain-Healthcare-red)
![MedicalAI](https://img.shields.io/badge/Focus-Medical%20AI-purple)
![EDA](https://img.shields.io/badge/Step-EDA-lightgrey)
![FeatureEngineering](https://img.shields.io/badge/Step-Feature%20Engineering-lightgrey)
![OneHot](https://img.shields.io/badge/Encoding-One--Hot-blueviolet)
![StandardScaler](https://img.shields.io/badge/Scaling-StandardScaler-yellowgreen)
![Pipeline](https://img.shields.io/badge/Design-ML%20Pipeline-success)
![RandomForest](https://img.shields.io/badge/Model-Random%20Forest-darkgreen)
![LinearRegression](https://img.shields.io/badge/Model-Linear%20Regression-blue)
![Ridge](https://img.shields.io/badge/Model-Ridge-blue)
![CrossValidation](https://img.shields.io/badge/Method-Cross%20Validation-orange)
![HyperparameterTuning](https://img.shields.io/badge/Method-Hyperparameter%20Tuning-orange)
![Interpretability](https://img.shields.io/badge/Focus-Interpretability-important)
![Academic](https://img.shields.io/badge/Type-Academic%20Project-informational)

## Project Structure

```text
.
├── data/
│   └── healthcare_dataset.csv     # Structured healthcare dataset
├── outputs/
│   └── test_predictions.csv       # Model prediction outputs
├── pic/
│   ├── Figure_1.png
│   ├── Figure_2.png
│   ├── Figure_3.png
│   ├── Figure_4.png
│   ├── Figure_5.png
│   ├── Figure_6.png
│   ├── Figure_7.png
│   └── Figure_8.png
├── main.py                        # Main ML pipeline
├── Report.docx                    # Detailed report
└── README.md
```

---

## Overview

Healthcare cost prediction is a key problem in medical data analytics.
In this project, supervised learning techniques are applied to estimate
hospital billing amounts based on patient demographics, admission details,
medical conditions, and length of stay.

The project is learning-focused and research-oriented, forming part of a
longer-term interest in medical artificial intelligence and healthcare analytics.

---

## Objectives

- Apply supervised machine learning to a real healthcare dataset
- Practice data cleaning, preprocessing, and feature engineering
- Compare baseline and non-linear regression models
- Evaluate model performance using standard regression metrics
- Build a reproducible and research-ready ML workflow

---

## 📊 Dataset Description

- Size: ~55,000 patient records
- Format: CSV (tabular data)
- Target variable: Billing Amount
- Features include:
  - Demographics (Age, Gender)
  - Admission details
  - Medical condition
  - Insurance and room information
  - Engineered feature: Length of Stay

The dataset contains no missing values, but includes duplicated rows and
invalid billing entries, which are handled explicitly in preprocessing.

---

## 🔬 Methodology & Pipeline

The implementation follows a clear, step-by-step ML workflow:

###  Data Loading & Inspection
```python
df = pd.read_csv("data/healthcare_dataset.csv")
df.info()
df.describe(include="all")
```
- Shape inspection
- Column types and distributions
- Duplicate detection

---

###  Target Definition and Feature Selection

```python
TARGET = "Billing Amount" 
DROP_COLS = ["Name", "Doctor", "Hospital"] FEATURES = [c for c in df.columns if c not in DROP_COLS + [TARGET]] 
```

High-cardinality identifier columns are removed to reduce noise and prevent model memorization.

---

### Feature Engineering

```python
df["Length_of_Stay"] = ( 
	df["Discharge Date" df["Date of Admission"] 
).dt.days 
```

Date columns are converted to datetime objects and used to derive a numerical feature representing hospital stay duration.

---

### Data Cleaning

```python
df = df.drop_duplicates() 
df = df[df["Billing Amount"] >= 0] 
```

- Duplicate rows are removed
- Negative billing amounts are filtered out


---


### Preprocessing Pipeline

```python
preprocessor = ColumnTransformer( 
	transformers=[ 
		("num", StandardScaler(), numeric_features), 
		("cat", OneHotEncoder(handle_unknown="ignore"), +categorical_features), 
	] 
) 
```

A preprocessing pipeline is used to ensure consistent scaling of numerical features and encoding of categorical variables, while preventing data leakage.

- Numerical scaling
- Categorical encoding
- Leakage-free fitting via pipelines

---


### Model Training

```python
Baseline and non-linear models are trained:
LinearRegression() Ridge() RandomForestRegressor(n_estimators=300) 
```
---

### Model Evaluation

Metrics used:
- **MAE**
- **RMSE**
- **R²**

```python
mean_absolute_error(y_true, y_pred) 
mean_squared_error(y_true, y_pred, squared=False) 
r2_score(y_true, y_pred) 
```

Random Forest outperforms linear baselines on validation and test sets.


---

### Hyperparameter Tuning

```python
RandomizedSearchCV( estimator=rf_pipe, param_distributions=param_distributions, scoring="neg_root_mean_squared_error", cv=5 ) 
```
Cross-validation is used to explore model robustness.
---

### Final Evaluation & Diagnostics

- Test-set evaluation
- Residual distribution
- Predicted vs. true scatter plot
- Feature importance analysis

```python
rf_fitted.feature_importances_ 
```
---

###  Key Results (Test Set)
- **MAE ≈ 11,500**
- **RMSE ≈ 13,600**
- **R² ≈ 0.08**

While predictive performance is moderate, the project emphasizes:

- Correct methodology
- Reproducibility
- Interpretability
- Research readiness

## Academic Context

This project was developed as part of academic coursework and independent study
in Artificial Intelligence & Machine Learning at Peter the Great St. Petersburg
Polytechnic University (SPbPU).

---

## Author

Matin Dastanboo  
MSc Student – Artificial Intelligence & Machine Learning  
Peter the Great St. Petersburg Polytechnic University  

Email: matin.dastanboo@gmail.com  
LinkedIn: https://www.linkedin.com/in/matindastanboo  
