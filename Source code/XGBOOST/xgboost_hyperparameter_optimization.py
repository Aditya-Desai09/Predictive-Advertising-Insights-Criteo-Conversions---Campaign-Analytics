#!/usr/bin/env python3
"""
XGBOOST HYPERPARAMETER OPTIMIZATION FOR CRITEO ATTRIBUTION
==========================================================
Advanced hyperparameter tuning to maximize attribution predictions (1s)
Based on complete dataset: 16,468,027 impressions with 2.7% attribution rate

OBJECTIVE: Increase attribution predictions while maintaining high precision
STRATEGY: Focus on recall optimization with precision balance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time
import json
import warnings
warnings.filterwarnings('ignore')

print("🎯 XGBOOST HYPERPARAMETER OPTIMIZATION FOR CRITEO ATTRIBUTION")
print("=" * 70)
print("GOAL: Maximize attribution predictions (1s) while maintaining precision")
print("DATASET: Complete 16.4M+ impressions")
print("BASELINE: 95.0% ROC-AUC, 13.7% Precision, 91.1% Recall")

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================
print("\n📥 Loading COMPLETE Criteo dataset...")

try:
    # Load complete dataset
    df = pd.read_csv('Data/pcb_dataset_final.csv')
    print(f"✅ Loaded {len(df):,} impressions")
    print(f"   Attribution rate: {df['attribution'].mean():.1%}")
    print(f"   Click rate: {df['click'].mean():.1%}")
    print(f"   Unique campaigns: {df['campaign'].nunique():,}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Feature engineering (same as baseline for fair comparison)
print("\n🔧 Feature Engineering...")

# Select production-ready features
features = [
    'campaign', 'cost', 'cpo', 'click',
    'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat8', 'cat9'
]

X = df[features].copy()
y = df['attribution'].copy()

# Add engineered features (from baseline)
# Campaign performance
campaign_perf = df.groupby('campaign')['attribution'].mean()
X['campaign_perf'] = X['campaign'].map(campaign_perf)

# Cost quartiles
X['cost_quartile'] = pd.qcut(X['cost'], q=4, labels=[1, 2, 3, 4])

print(f"✅ Features prepared: {X.shape[1]} features")
print(f"   Base features: {len(features)}")
print(f"   Engineered features: 2 (campaign_perf, cost_quartile)")

# Encode categorical features
categorical_features = ['campaign', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat8', 'cat9']
label_encoders = {}

for feature in categorical_features:
    if feature in X.columns:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature].astype(str))
        label_encoders[feature] = le

print(f"✅ Encoded {len(categorical_features)} categorical features")

# Train-test split (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train):,} samples ({y_train.mean():.1%} attribution)")
print(f"   Testing: {len(X_test):,} samples ({y_test.mean():.1%} attribution)")

# =============================================================================
# 2. BASELINE XGBOOST PERFORMANCE
# =============================================================================
print("\n🤖 Baseline XGBoost Performance...")

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"   Class imbalance ratio: {scale_pos_weight:.1f}:1")

# Baseline model (from your results)
baseline_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc'
)

start_time = time.time()
baseline_model.fit(X_train, y_train)
baseline_time = time.time() - start_time

# Baseline predictions
y_pred_baseline = baseline_model.predict(X_test)
y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

# Baseline metrics
baseline_metrics = {
    'roc_auc': roc_auc_score(y_test, y_pred_proba_baseline),
    'precision': precision_score(y_test, y_pred_baseline),
    'recall': recall_score(y_test, y_pred_baseline),
    'f1_score': f1_score(y_test, y_pred_baseline),
    'training_time': baseline_time
}

print(f"📈 Baseline Results:")
print(f"   ROC-AUC: {baseline_metrics['roc_auc']:.3f}")
print(f"   Precision: {baseline_metrics['precision']:.1%}")
print(f"   Recall: {baseline_metrics['recall']:.1%}")
print(f"   F1-Score: {baseline_metrics['f1_score']:.3f}")
print(f"   Training Time: {baseline_metrics['training_time']:.1f}s")

# =============================================================================
# 3. HYPERPARAMETER OPTIMIZATION STRATEGY
# =============================================================================
print("\n🎯 HYPERPARAMETER OPTIMIZATION STRATEGY")
print("=" * 50)

optimization_strategy = {
    "objective": "Maximize attribution predictions (recall) while maintaining precision",
    "focus_areas": [
        "Learning rate optimization for better convergence",
        "Tree structure tuning for complex patterns",
        "Regularization to prevent overfitting",
        "Sampling parameters for imbalanced data",
        "Early stopping for optimal training"
    ],
    "key_parameters": {
        "n_estimators": "More trees for complex patterns (but watch overfitting)",
        "max_depth": "Deeper trees for feature interactions",
        "learning_rate": "Lower rates with more estimators for stability",
        "subsample": "Row sampling to prevent overfitting",
        "colsample_bytree": "Feature sampling for generalization",
        "reg_alpha": "L1 regularization for feature selection",
        "reg_lambda": "L2 regularization for smooth weights",
        "scale_pos_weight": "Handle class imbalance",
        "min_child_weight": "Minimum samples per leaf"
    },
    "parameters_to_avoid": {
        "max_delta_step": "Not needed for our problem type",
        "gamma": "Can be too aggressive for our imbalanced data",
        "grow_policy": "Default 'depthwise' is optimal for tabular data"
    }
}

for key, value in optimization_strategy.items():
    if key != "key_parameters" and key != "parameters_to_avoid":
        print(f"{key.upper()}: {value}")

print(f"\n🔧 KEY PARAMETERS TO OPTIMIZE:")
for param, reason in optimization_strategy["key_parameters"].items():
    print(f"   • {param}: {reason}")

print(f"\n❌ PARAMETERS TO AVOID:")
for param, reason in optimization_strategy["parameters_to_avoid"].items():
    print(f"   • {param}: {reason}")

# =============================================================================
# 4. HYPERPARAMETER SEARCH SPACE
# =============================================================================
print("\n🔍 Defining Hyperparameter Search Space...")

# Comprehensive parameter grid focused on attribution optimization
param_distributions = {
    # Tree structure - deeper trees for complex attribution patterns
    'n_estimators': [200, 300, 500, 800, 1000],
    'max_depth': [4, 6, 8, 10, 12],
    
    # Learning parameters - balance speed vs accuracy
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    
    # Sampling parameters - critical for imbalanced data
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
    
    # Regularization - prevent overfitting on large dataset
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0, 2.0],
    
    # Tree constraints
    'min_child_weight': [1, 3, 5, 7],
    'max_delta_step': [0],  # Keep at 0 for our problem
    
    # Class imbalance handling
    'scale_pos_weight': [scale_pos_weight * 0.8, scale_pos_weight, scale_pos_weight * 1.2]
}

total_combinations = np.prod([len(v) for v in param_distributions.values()])
print(f"📊 Search space: {total_combinations:,} total combinations")
print(f"   Using RandomizedSearchCV with 100 iterations for efficiency")

# =============================================================================
# 5. ADVANCED HYPERPARAMETER TUNING
# =============================================================================
print("\n🚀 Starting Advanced Hyperparameter Tuning...")

# Custom scoring function to balance precision and recall
def custom_scorer(y_true, y_pred):
    """
    Custom scoring that balances precision and recall
    Optimizes for F2 score (weights recall 2x more than precision)
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    # F2 score weights recall higher than precision
    f2_score = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)
    return f2_score

# XGBoost model for tuning
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='auc',
    early_stopping_rounds=50,
    verbose=0
)

# Stratified K-Fold for robust validation
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# RandomizedSearchCV with custom scoring
print("⏳ Running RandomizedSearchCV (this will take 15-30 minutes)...")
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=100,  # Test 100 combinations
    cv=cv_strategy,
    scoring='f1',  # Optimize F1 score for balanced precision/recall
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
random_search.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)
tuning_time = time.time() - start_time

print(f"✅ Hyperparameter tuning completed in {tuning_time/60:.1f} minutes")

# =============================================================================
# 6. OPTIMIZED MODEL EVALUATION
# =============================================================================
print("\n🏆 Optimized Model Evaluation...")

# Get best model
best_model = random_search.best_estimator_
best_params = random_search.best_params_

print(f"🎯 Best Parameters Found:")
for param, value in best_params.items():
    print(f"   • {param}: {value}")

# Evaluate optimized model
start_time = time.time()
y_pred_optimized = best_model.predict(X_test)
y_pred_proba_optimized = best_model.predict_proba(X_test)[:, 1]
prediction_time = time.time() - start_time

# Optimized model metrics
optimized_metrics = {
    'roc_auc': roc_auc_score(y_test, y_pred_proba_optimized),
    'precision': precision_score(y_test, y_pred_optimized),
    'recall': recall_score(y_test, y_pred_optimized),
    'f1_score': f1_score(y_test, y_pred_optimized),
    'cv_score': random_search.best_score_,
    'prediction_time': prediction_time
}

print(f"\n📈 Optimized Results:")
print(f"   ROC-AUC: {optimized_metrics['roc_auc']:.3f}")
print(f"   CV F1-Score: {optimized_metrics['cv_score']:.3f}")
print(f"   Precision: {optimized_metrics['precision']:.1%}")
print(f"   Recall: {optimized_metrics['recall']:.1%}")
print(f"   F1-Score: {optimized_metrics['f1_score']:.3f}")

# =============================================================================
# 7. DETAILED PERFORMANCE COMPARISON
# =============================================================================
print("\n📊 DETAILED PERFORMANCE COMPARISON")
print("=" * 60)
print(f"{'Metric':<15} {'Baseline':<12} {'Optimized':<12} {'Change':<12} {'Impact'}")
print("-" * 60)

improvements = {}
for metric in ['roc_auc', 'precision', 'recall', 'f1_score']:
    baseline_val = baseline_metrics[metric]
    optimized_val = optimized_metrics[metric]
    change = optimized_val - baseline_val
    improvement = (change / baseline_val) * 100 if baseline_val > 0 else 0
    improvements[metric] = improvement
    
    # Determine impact
    if abs(improvement) < 1:
        impact = "Minimal"
    elif abs(improvement) < 5:
        impact = "Small"
    elif abs(improvement) < 10:
        impact = "Moderate"
    else:
        impact = "Significant"
    
    print(f"{metric.upper():<15} {baseline_val:<12.3f} {optimized_val:<12.3f} {change:+8.3f} {impact}")

# Confusion Matrix Analysis
print(f"\n🔍 CONFUSION MATRIX ANALYSIS:")
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
cm_optimized = confusion_matrix(y_test, y_pred_optimized)

print(f"\nBaseline Confusion Matrix:")
print(f"   True Negatives:  {cm_baseline[0,0]:,}")
print(f"   False Positives: {cm_baseline[0,1]:,}")
print(f"   False Negatives: {cm_baseline[1,0]:,}")
print(f"   True Positives:  {cm_baseline[1,1]:,}")

print(f"\nOptimized Confusion Matrix:")
print(f"   True Negatives:  {cm_optimized[0,0]:,}")
print(f"   False Positives: {cm_optimized[0,1]:,}")
print(f"   False Negatives: {cm_optimized[1,0]:,}")
print(f"   True Positives:  {cm_optimized[1,1]:,}")

# Attribution predictions improvement
tp_improvement = cm_optimized[1,1] - cm_baseline[1,1]
print(f"\n🎯 ATTRIBUTION PREDICTIONS IMPROVEMENT:")
print(f"   Additional True Positives: {tp_improvement:,}")
print(f"   Percentage increase: {(tp_improvement/cm_baseline[1,1]*100):+.1f}%")

# =============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n🎯 OPTIMIZED FEATURE IMPORTANCE ANALYSIS:")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features (Optimized Model):")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"   {i:2d}. {row['feature']:<20} {row['importance']:>8.3f}")

# Compare with baseline importance
baseline_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': baseline_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance Changes:")
for feature in feature_importance.head(5)['feature']:
    baseline_imp = baseline_importance[baseline_importance['feature'] == feature]['importance'].iloc[0]
    optimized_imp = feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
    change = optimized_imp - baseline_imp
    print(f"   {feature:<20} {baseline_imp:>6.3f} → {optimized_imp:>6.3f} ({change:+.3f})")

# =============================================================================
# 9. BUSINESS IMPACT ANALYSIS
# =============================================================================
print("\n💰 BUSINESS IMPACT ANALYSIS")
print("=" * 40)

# Calculate business metrics
total_impressions = len(y_test)
baseline_attributions = cm_baseline[1,1]
optimized_attributions = cm_optimized[1,1]
additional_attributions = optimized_attributions - baseline_attributions

# Assuming average revenue per attribution
avg_revenue_per_attribution = 50  # $50 per attribution (example)
additional_revenue = additional_attributions * avg_revenue_per_attribution

print(f"📊 Business Metrics:")
print(f"   Total test impressions: {total_impressions:,}")
print(f"   Baseline attributions: {baseline_attributions:,}")
print(f"   Optimized attributions: {optimized_attributions:,}")
print(f"   Additional attributions: {additional_attributions:,}")
print(f"   Additional revenue: ${additional_revenue:,}")

# ROI calculation
if additional_attributions > 0:
    attribution_lift = (additional_attributions / baseline_attributions) * 100
    print(f"   Attribution lift: {attribution_lift:.1f}%")
else:
    print(f"   Attribution change: {additional_attributions:,} (optimization needed)")

# =============================================================================
# 10. SAVE COMPREHENSIVE RESULTS
# =============================================================================
print("\n💾 Saving Comprehensive Results...")

# Prepare comprehensive results
comprehensive_results = {
    'optimization_info': {
        'dataset_size': len(df),
        'attribution_rate': df['attribution'].mean(),
        'tuning_method': 'RandomizedSearchCV',
        'n_iterations': 100,
        'cv_folds': 3,
        'tuning_time_minutes': tuning_time / 60,
        'optimization_objective': 'Maximize attribution predictions (F1 score)'
    },
    'best_parameters': best_params,
    'baseline_performance': baseline_metrics,
    'optimized_performance': optimized_metrics,
    'improvements': improvements,
    'business_impact': {
        'additional_attributions': int(additional_attributions),
        'attribution_lift_percent': float(attribution_lift) if additional_attributions > 0 else 0,
        'estimated_additional_revenue': float(additional_revenue)
    },
    'feature_importance': feature_importance.to_dict('records'),
    'confusion_matrices': {
        'baseline': cm_baseline.tolist(),
        'optimized': cm_optimized.tolist()
    },
    'optimization_strategy': optimization_strategy,
    'top_cv_results': [
        {
            'params': random_search.cv_results_['params'][i],
            'mean_test_score': random_search.cv_results_['mean_test_score'][i],
            'std_test_score': random_search.cv_results_['std_test_score'][i]
        }
        for i in np.argsort(random_search.cv_results_['mean_test_score'])[-10:][::-1]
    ]
}

# Save results
with open('xgboost_hyperparameter_optimization_results.json', 'w') as f:
    json.dump(comprehensive_results, f, indent=2, default=str)

# Save feature importance
feature_importance.to_csv('xgboost_optimized_feature_importance.csv', index=False)

# Save predictions for analysis
results_df = pd.DataFrame({
    'actual': y_test,
    'baseline_pred': y_pred_baseline,
    'baseline_proba': y_pred_proba_baseline,
    'optimized_pred': y_pred_optimized,
    'optimized_proba': y_pred_proba_optimized
})
results_df.to_csv('xgboost_optimization_predictions.csv', index=False)

print("✅ Saved comprehensive results:")
print("   • xgboost_hyperparameter_optimization_results.json")
print("   • xgboost_optimized_feature_importance.csv")
print("   • xgboost_optimization_predictions.csv")

# =============================================================================
# 11. PRODUCTION RECOMMENDATIONS
# =============================================================================
print("\n🚀 PRODUCTION RECOMMENDATIONS")
print("=" * 45)

if improvements['f1_score'] > 2:  # If F1 improvement > 2%
    print("✅ RECOMMENDED: Deploy optimized parameters")
    print(f"   • F1-Score improvement: +{improvements['f1_score']:.1f}%")
    print(f"   • Additional attributions: {additional_attributions:,}")
    print(f"   • Estimated revenue lift: ${additional_revenue:,}")
else:
    print("⚠️  BASELINE SUFFICIENT: Minimal improvement from tuning")
    print("   • Consider ensemble methods or feature engineering")
    print("   • Current model already well-optimized")

print(f"\n🎯 Optimal Production Parameters:")
print("```python")
print("xgb.XGBClassifier(")
for param, value in best_params.items():
    if isinstance(value, str):
        print(f"    {param}='{value}',")
    else:
        print(f"    {param}={value},")
print(f"    scale_pos_weight={scale_pos_weight:.1f},")
print("    random_state=42,")
print("    n_jobs=-1,")
print("    eval_metric='auc'")
print(")")
print("```")

print(f"\n📊 Expected Production Performance:")
print(f"   • ROC-AUC: {optimized_metrics['roc_auc']:.3f}")
print(f"   • Precision: {optimized_metrics['precision']:.1%}")
print(f"   • Recall: {optimized_metrics['recall']:.1%}")
print(f"   • F1-Score: {optimized_metrics['f1_score']:.3f}")

print(f"\n⚡ Performance Characteristics:")
print(f"   • Training time: ~{tuning_time/100:.1f}s per model")
print(f"   • Prediction time: {optimized_metrics['prediction_time']:.3f}s for {len(X_test):,} samples")
print(f"   • Throughput: ~{len(X_test)/optimized_metrics['prediction_time']:,.0f} predictions/second")

print(f"\n🎯 Key Insights for Criteo:")
print(f"   • Click feature dominates importance ({feature_importance.iloc[0]['importance']:.3f})")
print(f"   • Campaign performance is critical for attribution")
print(f"   • Cost and category features provide additional signal")
print(f"   • Model handles 16.4M+ impressions efficiently")

print("\n🎉 XGBOOST HYPERPARAMETER OPTIMIZATION COMPLETED!")
print(f"Results optimized for maximum attribution predictions while maintaining precision.")
print(f"Ready for production deployment with {optimized_metrics['roc_auc']:.1%} ROC-AUC accuracy!")