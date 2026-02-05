# Predictive Advertising Insights Criteo Conversions & Campaign Analytics
End-to-end ML pipeline for Criteo attribution prediction on 16.4M records. Ensemble of XGBoost, LightGBM, CatBoost achieving 95% ROC-AUC. PostgreSQL + Python + Tableau. Production-ready model selection with precision-focused optimization.


# Predictive Advertising Insights: Criteo Attribution Modeling

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL-316192.svg)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Production-ready machine learning framework for advertising attribution prediction using 16.4M Criteo impressions**

End-to-end data science pipeline predicting conversion attribution with 95% ROC-AUC, deployed through interactive Tableau dashboards. Optimized for class imbalance with precision-focused ensemble modeling.

---

## Project Overview

**Business Problem:** Predict which advertising impressions lead to conversions within 30 days to optimize ad spend and bidding strategies.

**Solution:** Built a scalable ML pipeline handling 16.4 million records with PostgreSQL data management, statistical feature engineering, and ensemble modeling (XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression).

**Impact:**
- **95.1% ROC-AUC** (XGBoost) for attribution prediction
- **13.9% precision** with 91.3% recall on highly imbalanced data (2.7% attribution rate)
- **10× faster training** (34.9s) vs Random Forest (393.6s) with better performance
- **Production-ready** model selection framework with saved pipelines

---

## Key Results

| Model | ROC-AUC | Precision | Recall | Training Time | Dataset Size |
|-------|---------|-----------|--------|---------------|--------------|
| **XGBoost**  | **0.951** | **13.9%** | 91.3% | 34.9s | 16.4M rows |
| LightGBM | 0.949 | 13.6% | 91.1% | 36.4s | 16.4M rows |
| CatBoost | 0.947 | 12.8% | 91.3% | 325.2s | 16.4M rows |
| Random Forest | 0.948 | 13.2% | 91.2% | 393.6s | 16.4M rows |
| Logistic Regression | 0.933 | 10.6% | 93.0% | 19.2s | 16.4M rows |

**Selected Model:** XGBoost (best ROC-AUC + precision, 10× faster than RF)

---

## Repository Structure
├── Datasets/
│ └── Link.txt # Criteo dataset download link
│
├── Presentation/
│ └── CRITEO PPT 1.pptx # Project presentation slides
│
├── Report/
│ └── Criteo Attribution Modeling Report.pdf # Detailed analysis report
│
├── Source code/
│ ├── 1_Data_Loading.ipynb # PostgreSQL data loading
│ ├── 1_eda_analysis.ipynb # Exploratory Data Analysis
│ ├── 2_Data_Loading.ipynb # Additional data processing
│ ├── 2_feature_engineering.ipynb # Feature creation pipeline
│ │
│ ├── XGBOOST/ # XGBoost implementation (SELECTED)
│ │ ├── XGBoost.ipynb # Main training notebook
│ │ ├── XGBOOST_HYPER.ipynb # Hyperparameter optimization
│ │ ├── xgboost_trained_model.pkl # Production model
│ │ ├── xgboost_complete_summary.json # Performance metrics
│ │ └── xgboost_feature_importance.csv
│ │
│ ├── LightGBM/ # LightGBM implementation
│ │ ├── LightGBM.ipynb
│ │ ├── lightgbm_trained_model.pkl
│ │ └── lightgbm_model_info.json
│ │
│ ├── CatBoost/ # CatBoost implementation
│ │ ├── CatBoost.ipynb
│ │ ├── catboost_trained_model.cbm
│ │ └── catboost_model_info.json
│ │
│ ├── RF/ # Random Forest implementation
│ │ ├── RF.ipynb
│ │ ├── random_forest_complete_summary.json
│ │ └── random_forest_feature_importance.csv
│ │
│ └── Logistic/ # Logistic Regression baseline
│ ├── Logistic.ipynb
│ └── logistic_regression_complete_summary.json
│
├── .gitattributes
└── README.md

undefined
