# HR Employee Attrition Prediction

A machine learning project to analyze and predict employee attrition, helping HR teams identify at-risk employees early and take proactive retention measures.

---

## Objective

Build a binary classification model to predict whether an employee will leave the company (`Attrition: Yes/No`), using HR survey data, employee records, and time-tracking information.

---

## Dataset

Five CSV files loaded from Google Drive:

| File | Description |
|---|---|
| `employee_survey_data` | Employee self-reported satisfaction (job, environment, work-life balance) |
| `general_data` | General employee info (age, salary, experience, etc.) |
| `manager_survey_data` | Manager evaluations |
| `in_time` | Daily clock-in timestamps per employee |
| `out_time` | Daily clock-out timestamps per employee |

All tables are merged on `EmployeeID` into a single DataFrame (`df_final`).

---

## Pipeline

### 1. Data Loading & Merging
- Detect encoding for each file
- Inner-merge `employee_survey_data`, `general_data`, and `manager_survey_data` on `EmployeeID`

### 2. Data Cleaning & Feature Engineering
- Rename index columns in `in_time` / `out_time`
- Parse date columns to `datetime`
- Engineer **`AvgWorkHours`** and **`AbsentDays`** from clock-in/out records
- Impute missing values with column means; drop remaining rows with NaN
- Remove duplicates and constant columns (single unique value)
- Encode categorical variables with `LabelEncoder`

### 3. Exploratory Data Analysis (EDA)
- Target variable distribution — bar chart & pie chart
- Boxplots of numerical features grouped by Stayed / Left
- Attrition rate by categorical feature
- Full feature correlation heatmap
- Per-feature correlation with `Attrition`

### 4. Feature Selection
Dropped columns before modeling:
`BusinessTravel`, `StockOptionLevel`, `DistanceFromHome`, `JobLevel`, `EmployeeID`, `Attrition`

### 5. Machine Learning
Train/test split: **80/20** with `stratify=y`.

| Model | Notes |
|---|---|
| Logistic Regression | Baseline — `class_weight='balanced'` |
| Decision Tree | `max_depth=6`, `class_weight='balanced'` |
| SVM (RBF kernel) | `class_weight='balanced'`, scaled input |
| Neural Network (MLP) | Hidden layers (64, 32), **SMOTE** oversampling |

### 6. Model Evaluation
- Classification report per model
- Confusion matrices (all models)
- ROC Curve & AUC-ROC comparison
- Metric comparison: Accuracy, Recall (Churn), F1-Score, AUC
- 5-fold Stratified Cross-Validation on the best model and SVM

---

## Results

The best model is selected by highest **AUC-ROC** across the four algorithms. Cross-validation confirms generalization stability.

> **Note:** The dataset is class-imbalanced — prioritize **Recall** and **AUC-ROC** over Accuracy when interpreting results.

---

## Requirements

```
pandas, numpy, matplotlib, seaborn
scikit-learn
imbalanced-learn
chardet
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn chardet
```

---

## How to Run

1. Open `DataAnalysis.ipynb` in Jupyter Notebook or Google Colab
2. Run cells sequentially from top to bottom
3. Ensure an active internet connection to download data from Google Drive

---

## Notebook Structure

```
DataAnalysis.ipynb
├── Import Library
├── Data Loading
│   └── Merge DataFrame
├── Data Cleaning & Processing
│   ├── Rename columns
│   ├── Convert date columns to datetime
│   ├── Handle missing data
│   ├── Drop duplicates
│   └── Variable encoding
├── Exploratory Data Analysis (EDA)
│   ├── 1. Attrition Distribution
│   ├── 2. Boxplots — Numerical Features vs Attrition
│   ├── 3. Attrition Rate by Categorical Features
│   ├── 4. Correlation Heatmap
│   ├── 5. Feature Correlation with Attrition
│   └── 6. Feature Selection / Target Split
├── Implementing Machine Learning Algorithms
└── Model Evaluation
    ├── Train & Evaluation
    ├── Model Comparison
    ├── Confusion Matrix
    ├── ROC Curve
    └── Cross-Validation
```
