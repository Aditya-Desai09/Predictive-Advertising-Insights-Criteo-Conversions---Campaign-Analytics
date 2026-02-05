# Predictive Advertising Insights: Criteo Conversions & Campaign Analytics

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL-316192.svg)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Production-ready machine learning framework for advertising attribution prediction using 16.4M Criteo impressions**

End-to-end data science pipeline predicting conversion attribution with 95% ROC-AUC, deployed through interactive Tableau dashboards. Optimized for class imbalance with precision-focused ensemble modeling.

### [View Interactive Tableau Dashboard](https://public.tableau.com/views/Criteo_Visualization_17701408628490/CampaignPerformanceAttributionOverview?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

##  Project Overview

**Business Problem:** Predict which advertising impressions lead to conversions within 30 days to optimize ad spend and bidding strategies.

**Solution:** Built a scalable ML pipeline handling 16.4 million records with PostgreSQL data management, statistical feature engineering, and ensemble modeling (XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression).

**Impact:**
- **95.1% ROC-AUC** (XGBoost) for attribution prediction
- **13.9% precision** with 91.3% recall on highly imbalanced data (2.7% attribution rate)
- **10× faster training** (34.9s) vs Random Forest (393.6s) with better performance
- **Production-ready** model selection framework with saved pipelines

---

##  Key Results

| Model | ROC-AUC | Precision | Recall | Training Time | Dataset Size |
|-------|---------|-----------|--------|---------------|--------------|
| **XGBoost**  | **0.951** | **13.9%** | 91.3% | 34.9s | 16.4M rows |
| LightGBM | 0.949 | 13.6% | 91.1% | 36.4s | 16.4M rows |
| CatBoost | 0.947 | 12.8% | 91.3% | 325.2s | 16.4M rows |
| Random Forest | 0.948 | 13.2% | 91.2% | 393.6s | 16.4M rows |
| Logistic Regression | 0.933 | 10.6% | 93.0% | 19.2s | 16.4M rows |

**Selected Model:** XGBoost (best ROC-AUC + precision, 10× faster than RF)

---

##  Repository Structure

```
├── Datasets/
│   └── Link.txt                                    # Criteo dataset download link
│
├── Presentation/
│   └── CRITEO PPT 1.pptx                          # Project presentation slides
│
├── Report/
│   └── Criteo Attribution Modeling Report.pdf     # Detailed analysis report
│
├── Source code/
│   ├── 1_Data_Loading.ipynb                       # PostgreSQL data loading
│   ├── 1_eda_analysis.ipynb                       # Exploratory Data Analysis
│   ├── 2_Data_Loading.ipynb                       # Additional data processing
│   ├── 2_feature_engineering.ipynb                # Feature creation pipeline
│   ├── app.py                                     # Gradio app
│   │
│   ├── XGBOOST/                                   # XGBoost implementation (SELECTED)
│   │   ├── XGBoost.ipynb                          # Main training notebook
│   │   ├── XGBOOST_HYPER.ipynb                    # Hyperparameter optimization
│   │   ├── xgboost_trained_model.pkl              # Production model
│   │   ├── xgboost_complete_summary.json          # Performance metrics
│   │   └── xgboost_feature_importance.csv
│   │
│   ├── LightGBM/                                  # LightGBM implementation
│   │   ├── LightGBM.ipynb
│   │   ├── lightgbm_trained_model.pkl
│   │   └── lightgbm_model_info.json
│   │
│   ├── CatBoost/                                  # CatBoost implementation
│   │   ├── CatBoost.ipynb
│   │   ├── catboost_trained_model.cbm
│   │   └── catboost_model_info.json
│   │
│   ├── RF/                                        # Random Forest implementation
│   │   ├── RF.ipynb
│   │   ├── random_forest_complete_summary.json
│   │   └── random_forest_feature_importance.csv
│   │
│   └── Logistic/                                  # Logistic Regression baseline
│       ├── Logistic.ipynb
│       └── logistic_regression_complete_summary.json
│
├── .gitattributes
└── README.md
```

---

##  Quick Start

### Prerequisites
```bash
Python 3.8+
PostgreSQL 12+
Jupyter Notebook
```

---

##  Technical Pipeline

### Data Management
- **Source:** Criteo Attribution Dataset (16,468,027 impressions)
- **Storage:** PostgreSQL database for scalable data handling
- **Target:** `attribution` (1 = conversion attributed to Criteo, 0 = not)

### Feature Engineering (14 features)
1. **Campaign Performance:** Historical attribution rate per campaign
2. **Cost Features:** Cost quartiles (1-4), CPO efficiency score
3. **Contextual Categories:** Attribution rates for cat1-cat9 (excluded cat7: 57k values)
4. **Click Features:** Click indicator, position, count
5. **Temporal Features:** Normalized timestamp, time periods

### Model Training
- **Class Imbalance Handling:** `scale_pos_weight` in XGBoost/LightGBM, `class_weight='balanced'` in RF/LR
- **Evaluation Metric:** ROC-AUC (appropriate for 2.7% attribution rate)
- **Validation:** 80-20 train-test split with stratification

### Production Deployment
- **Saved Models:** `.pkl` (XGBoost, LightGBM, RF, LR), `.cbm` (CatBoost)
- **Metadata:** `.json` files with feature lists, performance metrics
- **Inference Time:** <1ms per impression (real-time bidding compatible)

---

##  Model Selection Justification

### Why XGBoost?

1. **Best Performance:** Highest ROC-AUC (0.951) and precision (13.9%)
2. **Speed:** 34.9s training (11× faster than Random Forest at 393.6s)
3. **Business Impact:**
   - Correctly identifies **91.3% of attributed conversions** (recall)
   - **13.9% precision** reduces false positives → fewer wasted bids
   - Improves precision from 10.6% (Logistic Regression) to 13.9%

### Comparison Insights

- **LightGBM:** Nearly identical performance (ROC-AUC 0.949, 36.4s) - excellent alternative
- **Random Forest:** Strong but 11× slower (393.6s vs 34.9s)
- **CatBoost:** Good categorical handling but 9× slower (325.2s)
- **Logistic Regression:** Fast baseline but lower performance (ROC-AUC 0.933)

### Feature Importance (XGBoost Top 5)

1. **click** (96.9%) - Dominant predictor
2. **campaign_perf** (1.2%) - Historical campaign success
3. **cat1** (0.9%) - Primary contextual category
4. **cat4** (0.2%) - Secondary contextual signal
5. **cost** (0.2%) - Bidding cost information

---

##  Technology Stack

**Data Processing:**
- Python (Pandas, NumPy)
- PostgreSQL (16.4M records management)

**Machine Learning:**
- XGBoost 1.5+ (selected model)
- LightGBM 3.3+
- CatBoost 1.0+
- Scikit-learn (Random Forest, Logistic Regression, metrics)

**Development:**
- Jupyter Notebook (interactive analysis)
- Git/GitHub (version control)

**Visualization:**
- Tableau (interactive dashboards - separate deployment)
- Matplotlib, Seaborn (exploratory plots)

---

##  Key Learnings

1. **Class Imbalance:** ROC-AUC > Accuracy for 2.7% attribution rate
2. **Feature Engineering:** Campaign performance features critical for all models
3. **Model Trade-offs:** XGBoost/LightGBM balance performance + speed for production
4. **Scalability:** PostgreSQL + batching enables 16M+ row processing
5. **Hyperparameter Tuning:** Grid search improved XGBoost ROC-AUC by 0.5%

---

##  Author

**Aditya Desai**
- GitHub: [@Aditya-Desai09](https://github.com/Aditya-Desai09)
- Location: Pune, Maharashtra, India
- Project: PG-DBDA (Post Graduate Diploma in Big Data Analytics)

---

##  License

This project is licensed under the MIT License.

---

##  Acknowledgments

- **Criteo AI Lab** - Attribution dataset and research paper
- **Diemert et al.** - "Attribution Modeling Increases Efficiency of Bidding in Display Advertising" (AdKDD 2017)
- Open-source ML community (XGBoost, LightGBM, CatBoost)

---

##  Contact

For questions or collaboration:
- Gmail: [adidesai119@gmail.com](mailto:adidesai119@gmail.com)

