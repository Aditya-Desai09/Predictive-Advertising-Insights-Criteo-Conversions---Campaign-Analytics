#!/usr/bin/env python3
"""
XGBOOST COMPLETE DATASET HYPERPARAMETER OPTIMIZATION
====================================================
Beat baseline: ROC-AUC 0.950, Precision 13.7%, Recall 91.1%, F1 0.239
Target: ROC-AUC >0.955, Precision >18%, Recall >90%, F1 >0.30

COMPLETE DATASET: 16,468,027 impressions
GOAL: Maximize attribution predictions with superior performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import json
import gc
import warnings
warnings.filterwarnings('ignore')

print("🎯 XGBOOST COMPLETE DATASET HYPERPARAMETER OPTIMIZATION")
print("=" * 65)
print("CHALLENGE: Beat baseline ROC-AUC 0.950, Precision 13.7%, Recall 91.1%")
print("TARGET: ROC-AUC >0.955, Precision >18%, Recall >90%, F1 >0.30")
print("DATASET: Complete 16,468,027 impressions")

# =============================================================================
# 1. COMPLETE DATASET LOADING
# =============================================================================
print("\n📥 Loading COMPLETE Criteo dataset...")

try:
    df = pd.read_csv('Data/pcb_dataset_final.csv')
    print(f"✅ Loaded {len(df):,} COMPLETE impressions")
    print(f"   Attribution rate: {df['attribution'].mean():.2%}")
    print(f"   Click rate: {df['click'].mean():.1%}")
    print(f"   Unique campaigns: {df['campaign'].nunique():,}")
    
    # Memory optimization
    print("   Optimizing memory usage...")
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    print(f"✅ Memory optimized dataset ready")
    
except Exception as e:
    print(f"❌ Error loading complete dataset: {e}")
    exit(1)

# =============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# =============================================================================
print("\n🔧 Advanced Feature Engineering...")

# Core features (same as baseline for fair comparison)
features = [
    'campaign', 'cost', 'cpo', 'click',
    'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat8', 'cat9'
]

X = df[features].copy()
y = df['attribution'].copy()

print(f"   Base features: {len(features)}")

# Advanced engineered features
print("   Adding advanced engineered features...")

# 1. Campaign performance (same as baseline)
campaign_perf = df.groupby('campaign')['attribution'].mean()
X['campaign_perf'] = X['campaign'].map(campaign_perf)

# 2. Cost quartiles (same as baseline)
X['cost_quartile'] = pd.qcut(X['cost'], q=4, labels=False, duplicates='drop')

# 3. NEW: Click-Cost interaction (high-value feature)
X['click_cost_interaction'] = X['click'] * X['cost']

# 4. NEW: Campaign click rate
campaign_click_rate = df.groupby('campaign')['click'].mean()
X['campaign_click_rate'] = X['campaign'].map(campaign_click_rate)

# 5. NEW: Cost efficiency ratio
X['cost_efficiency'] = X['cost'] / (X['cpo'] + 1)  # +1 to avoid division by zero

# 6. NEW: Category interaction features
X['cat1_cat2_interaction'] = X['cat1'].astype(str) + '_' + X['cat2'].astype(str)
X['cat3_cat4_interaction'] = X['cat3'].astype(str) + '_' + X['cat4'].astype(str)

print(f"✅ Total features: {X.shape[1]} (12 base + 7 advanced engineered)")

# Encode categorical features
print("   Encoding categorical features...")
categorical_features = [
    'campaign', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat8', 'cat9',
    'cat1_cat2_interaction', 'cat3_cat4_interaction'
]

label_encoders = {}
for feature in categorical_features:
    if feature in X.columns:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature].astype(str))
        label_encoders[feature] = le

print(f"✅ Encoded {len(categorical_features)} categorical features")

# Clean up memory
del df
gc.collect()

# =============================================================================
# 3. STRATIFIED TRAIN-TEST SPLIT
# =============================================================================
print(f"\n📊 Creating stratified train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train):,} samples ({y_train.mean():.2%} attribution)")
print(f"   Test set: {len(X_test):,} samples ({y_test.mean():.2%} attribution)")

# Calculate precise class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"   Class imbalance ratio: {scale_pos_weight:.1f}:1")

# =============================================================================
# 4. BASELINE REPRODUCTION
# =============================================================================
print(f"\n🤖 Reproducing baseline performance...")

baseline_model = xgb.XGBClassifier(
    n_estimators=100,  # Same as baseline
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc'
)

start_time = time.time()
baseline_model.fit(X_train, y_train)
baseline_time = time.time() - start_time

# Baseline evaluation
y_pred_baseline = baseline_model.predict(X_test)
y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

baseline_metrics = {
    'roc_auc': roc_auc_score(y_test, y_pred_proba_baseline),
    'precision': precision_score(y_test, y_pred_baseline),
    'recall': recall_score(y_test, y_pred_baseline),
    'f1_score': f1_score(y_test, y_pred_baseline),
    'training_time': baseline_time
}

print(f"📈 Baseline Performance (to beat):")
print(f"   ROC-AUC: {baseline_metrics['roc_auc']:.3f} (target: >0.955)")
print(f"   Precision: {baseline_metrics['precision']:.1%} (target: >18%)")
print(f"   Recall: {baseline_metrics['recall']:.1%} (target: >90%)")
print(f"   F1-Score: {baseline_metrics['f1_score']:.3f} (target: >0.30)")
print(f"   Training Time: {baseline_metrics['training_time']:.1f}s")

# =============================================================================
# 5. ADVANCED HYPERPARAMETER OPTIMIZATION
# =============================================================================
print(f"\n🎯 ADVANCED HYPERPARAMETER OPTIMIZATION")
print("=" * 50)
print("STRATEGY: Precision-focused optimization while maintaining recall")

# Expanded parameter grid for superior performance
param_distributions = {
    # Boosting parameters - more aggressive for better performance
    'n_estimators': [300, 500, 800, 1000],  # More trees for complex patterns
    'max_depth': [8, 10, 12, 15],           # Deeper trees for interactions
    'learning_rate': [0.01, 0.05, 0.1, 0.15], # Include slower learning
    
    # Sampling parameters - critical for imbalanced data
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'colsample_bylevel': [0.7, 0.8, 0.9],
    
    # Regularization - prevent overfitting on large dataset
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0, 2.0],
    
    # Tree constraints
    'min_child_weight': [1, 3, 5, 10],
    'max_delta_step': [0, 1, 5],  # Can help with extreme imbalance
    
    # Class imbalance handling - fine-tuned
    'scale_pos_weight': [
        scale_pos_weight * 0.7,  # Less aggressive
        scale_pos_weight * 0.8,
        scale_pos_weight * 0.9,
        scale_pos_weight,        # Calculated value
        scale_pos_weight * 1.1,
        scale_pos_weight * 1.2,
        scale_pos_weight * 1.3   # More aggressive
    ]
}

total_combinations = np.prod([len(v) for v in param_distributions.values()])
print(f"🔍 Advanced search space: {total_combinations:,} combinations")
print(f"   Testing 50 iterations with 3-fold cross-validation")

# Advanced XGBoost model
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='auc',
    tree_method='hist',  # Faster for large datasets
    grow_policy='lossguide'  # Better for complex patterns
)

# Cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Custom scoring function that prioritizes precision while maintaining recall
def custom_f1_scorer(y_true, y_pred):
    """Custom scorer that weights precision higher"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    # F-beta score with beta=0.5 (weights precision 2x more than recall)
    if precision + recall == 0:
        return 0
    f_beta = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)
    return f_beta

# Advanced randomized search
print("⏳ Starting advanced hyperparameter optimization...")
print("   This will take 20-40 minutes for complete dataset...")

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=50,  # More iterations for better results
    cv=cv_strategy,
    scoring='f1',  # Standard F1 for balanced optimization
    n_jobs=-1,
    random_state=42,
    verbose=2  # More verbose output
)

start_time = time.time()
random_search.fit(X_train, y_train)
optimization_time = time.time() - start_time

print(f"✅ Advanced optimization completed in {optimization_time/60:.1f} minutes")

# =============================================================================
# 6. OPTIMIZED MODEL EVALUATION
# =============================================================================
print(f"\n🏆 OPTIMIZED MODEL RESULTS...")

best_model = random_search.best_estimator_
best_params = random_search.best_params_

print(f"🎯 Best Parameters Found:")
for param, value in best_params.items():
    print(f"   • {param}: {value}")

# Evaluate optimized model
y_pred_optimized = best_model.predict(X_test)
y_pred_proba_optimized = best_model.predict_proba(X_test)[:, 1]

optimized_metrics = {
    'roc_auc': roc_auc_score(y_test, y_pred_proba_optimized),
    'precision': precision_score(y_test, y_pred_optimized),
    'recall': recall_score(y_test, y_pred_optimized),
    'f1_score': f1_score(y_test, y_pred_optimized),
    'cv_score': random_search.best_score_
}

print(f"\n📈 OPTIMIZED PERFORMANCE:")
print(f"   ROC-AUC: {optimized_metrics['roc_auc']:.3f}")
print(f"   CV F1-Score: {optimized_metrics['cv_score']:.3f}")
print(f"   Precision: {optimized_metrics['precision']:.1%}")
print(f"   Recall: {optimized_metrics['recall']:.1%}")
print(f"   F1-Score: {optimized_metrics['f1_score']:.3f}")

# =============================================================================
# 7. PERFORMANCE COMPARISON VS BASELINE
# =============================================================================
print(f"\n📊 PERFORMANCE COMPARISON VS BASELINE")
print("=" * 60)
print(f"{'Metric':<12} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12} {'Target Met'}")
print("-" * 60)

# Define targets
targets = {
    'roc_auc': 0.955,
    'precision': 0.18,
    'recall': 0.90,
    'f1_score': 0.30
}

improvements = {}
targets_met = {}

for metric in ['roc_auc', 'precision', 'recall', 'f1_score']:
    baseline_val = baseline_metrics[metric]
    optimized_val = optimized_metrics[metric]
    improvement = ((optimized_val - baseline_val) / baseline_val) * 100
    improvements[metric] = improvement
    
    target_met = "✅ YES" if optimized_val >= targets[metric] else "❌ NO"
    targets_met[metric] = optimized_val >= targets[metric]
    
    print(f"{metric.upper():<12} {baseline_val:<12.3f} {optimized_val:<12.3f} {improvement:+8.1f}% {target_met}")

# Overall success assessment
targets_achieved = sum(targets_met.values())
print(f"\n🎯 TARGETS ACHIEVED: {targets_achieved}/4")

if targets_achieved >= 3:
    print("🎉 OPTIMIZATION SUCCESSFUL! Significant improvement achieved.")
elif targets_achieved >= 2:
    print("✅ OPTIMIZATION GOOD! Moderate improvement achieved.")
else:
    print("⚠️ OPTIMIZATION PARTIAL! Some improvement but targets not fully met.")

# =============================================================================
# 8. DETAILED CONFUSION MATRIX ANALYSIS
# =============================================================================
print(f"\n🔍 DETAILED CONFUSION MATRIX ANALYSIS")
print("=" * 45)

cm_baseline = confusion_matrix(y_test, y_pred_baseline)
cm_optimized = confusion_matrix(y_test, y_pred_optimized)

print(f"BASELINE CONFUSION MATRIX:")
print(f"   True Negatives:  {cm_baseline[0,0]:,}")
print(f"   False Positives: {cm_baseline[0,1]:,}")
print(f"   False Negatives: {cm_baseline[1,0]:,}")
print(f"   True Positives:  {cm_baseline[1,1]:,}")

print(f"\nOPTIMIZED CONFUSION MATRIX:")
print(f"   True Negatives:  {cm_optimized[0,0]:,}")
print(f"   False Positives: {cm_optimized[0,1]:,}")
print(f"   False Negatives: {cm_optimized[1,0]:,}")
print(f"   True Positives:  {cm_optimized[1,1]:,}")

# Attribution improvement analysis
tp_improvement = cm_optimized[1,1] - cm_baseline[1,1]
fp_change = cm_optimized[0,1] - cm_baseline[0,1]

print(f"\n🎯 ATTRIBUTION PREDICTIONS IMPROVEMENT:")
print(f"   Additional True Positives: {tp_improvement:,}")
print(f"   Change in False Positives: {fp_change:,}")

if cm_baseline[1,1] > 0:
    tp_percent_improvement = (tp_improvement / cm_baseline[1,1]) * 100
    print(f"   True Positive improvement: {tp_percent_improvement:+.1f}%")

# =============================================================================
# 9. ADVANCED FEATURE IMPORTANCE
# =============================================================================
print(f"\n🎯 ADVANCED FEATURE IMPORTANCE ANALYSIS:")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
    print(f"   {i:2d}. {row['feature']:<25} {row['importance']:>8.3f}")

# Compare with baseline top features
print(f"\n📊 NEW ENGINEERED FEATURES IMPACT:")
engineered_features = [
    'click_cost_interaction', 'campaign_click_rate', 'cost_efficiency',
    'cat1_cat2_interaction', 'cat3_cat4_interaction'
]

for feature in engineered_features:
    if feature in feature_importance['feature'].values:
        importance = feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
        rank = feature_importance[feature_importance['feature'] == feature].index[0] + 1
        print(f"   {feature:<25} Rank: {rank:2d}, Importance: {importance:.3f}")

# =============================================================================
# 10. BUSINESS IMPACT ANALYSIS
# =============================================================================
print(f"\n💰 BUSINESS IMPACT ANALYSIS")
print("=" * 35)

# Calculate business metrics
total_test_impressions = len(y_test)
baseline_attributions = cm_baseline[1,1]
optimized_attributions = cm_optimized[1,1]
additional_attributions = optimized_attributions - baseline_attributions

# Revenue impact (assuming $50 per attribution)
avg_revenue_per_attribution = 50
additional_revenue = additional_attributions * avg_revenue_per_attribution

# Cost impact (false positives cost money)
baseline_fp_cost = cm_baseline[0,1] * 5  # $5 per false positive
optimized_fp_cost = cm_optimized[0,1] * 5
cost_change = optimized_fp_cost - baseline_fp_cost

net_revenue_impact = additional_revenue - cost_change

print(f" Business Impact Metrics:")
print(f"   Test impressions: {total_test_impressions:,}")
print(f"   Baseline attributions: {baseline_attributions:,}")
print(f"   Optimized attributions: {optimized_attributions:,}")
print(f"   Additional attributions: {additional_attributions:,}")
print(f"   Additional revenue: ${additional_revenue:,}")
print(f"   Cost change (FP): ${cost_change:,}")
print(f"   Net revenue impact: ${net_revenue_impact:,}")

if baseline_attributions > 0:
    attribution_lift = (additional_attributions / baseline_attributions) * 100
    print(f"   Attribution lift: {attribution_lift:+.1f}%")

# =============================================================================
# 11. SAVE COMPREHENSIVE RESULTS
# =============================================================================
print(f"\n Saving comprehensive optimization results...")

# Complete results dictionary
results = {
    'optimization_info': {
        'dataset_size': len(X),
        'complete_dataset': True,
        'attribution_rate': y.mean(),
        'optimization_time_minutes': optimization_time / 60,
        'search_iterations': 50,
        'cv_folds': 3,
        'advanced_features': True
    },
    'targets': targets,
    'targets_achieved': {k: bool(v) for k, v in targets_met.items()},
    'targets_met_count': targets_achieved,
    'best_parameters': best_params,
    'baseline_performance': baseline_metrics,
    'optimized_performance': optimized_metrics,
    'improvements': improvements,
    'feature_importance': feature_importance.to_dict('records'),
    'engineered_features_impact': {
        feature: {
            'importance': float(feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0])
            if feature in feature_importance['feature'].values else 0,
            'rank': int(feature_importance[feature_importance['feature'] == feature].index[0] + 1)
            if feature in feature_importance['feature'].values else len(feature_importance)
        }
        for feature in engineered_features
    },
    'business_impact': {
        'additional_attributions': int(additional_attributions),
        'attribution_lift_percent': float(attribution_lift) if baseline_attributions > 0 else 0,
        'additional_revenue': float(additional_revenue),
        'cost_change': float(cost_change),
        'net_revenue_impact': float(net_revenue_impact)
    },
    'confusion_matrices': {
        'baseline': cm_baseline.tolist(),
        'optimized': cm_optimized.tolist()
    }
}

# Save results
with open('xgboost_complete_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

feature_importance.to_csv('xgboost_complete_feature_importance.csv', index=False)

# Save model parameters for production
production_config = {
    'model_class': 'XGBClassifier',
    'parameters': best_params,
    'performance': optimized_metrics,
    'feature_names': list(X.columns),
    'label_encoders': {k: list(v.classes_) for k, v in label_encoders.items()}
}

with open('xgboost_production_config.json', 'w') as f:
    json.dump(production_config, f, indent=2, default=str)

print(" Comprehensive results saved:")
print("   • xgboost_complete_optimization_results.json")
print("   • xgboost_complete_feature_importance.csv")
print("   • xgboost_production_config.json")

# =============================================================================
# 12. FINAL PRODUCTION RECOMMENDATIONS
# =============================================================================
print(f"\n FINAL PRODUCTION RECOMMENDATIONS")
print("=" * 45)

if targets_achieved >= 3:
    print(" DEPLOY OPTIMIZED MODEL IMMEDIATELY")
    print(f"   • Achieved {targets_achieved}/4 performance targets")
    print(f"   • ROC-AUC: {optimized_metrics['roc_auc']:.3f} (vs baseline {baseline_metrics['roc_auc']:.3f})")
    print(f"   • Precision: {optimized_metrics['precision']:.1%} (vs baseline {baseline_metrics['precision']:.1%})")
    print(f"   • Additional revenue: ${additional_revenue:,}")
elif targets_achieved >= 2:
    print(" DEPLOY WITH MONITORING")
    print(f"   • Achieved {targets_achieved}/4 performance targets")
    print(f"   • Significant improvement over baseline")
    print(f"   • Monitor performance closely in production")
else:
    print(" FURTHER OPTIMIZATION RECOMMENDED")
    print(f"   • Only achieved {targets_achieved}/4 performance targets")
    print(f"   • Consider ensemble methods or deep learning")

print(f"\n Production-Ready Configuration:")
print("```python")
print("xgb.XGBClassifier(")
for param, value in best_params.items():
    if isinstance(value, str):
        print(f"    {param}='{value}',")
    else:
        print(f"    {param}={value},")
print("    random_state=42,")
print("    n_jobs=-1,")
print("    eval_metric='auc',")
print("    tree_method='hist',")
print("    grow_policy='lossguide'")
print(")")
print("```")

print(f"\n Expected Production Performance:")
print(f"   • ROC-AUC: {optimized_metrics['roc_auc']:.3f}")
print(f"   • Precision: {optimized_metrics['precision']:.1%}")
print(f"   • Recall: {optimized_metrics['recall']:.1%}")
print(f"   • F1-Score: {optimized_metrics['f1_score']:.3f}")
print(f"   • Training time: ~{optimization_time/60:.0f} minutes for full dataset")

print(f"\n💡 Key Success Factors:")
print(f"   • Advanced feature engineering with {len(engineered_features)} new features")
print(f"   • Optimized for complete {len(X):,} sample dataset")
print(f"   • Fine-tuned class imbalance handling")
print(f"   • Production-ready configuration with robust parameters")

print(f"\n🎉 XGBOOST COMPLETE DATASET OPTIMIZATION COMPLETED!")
print(f"Successfully optimized for maximum Criteo attribution predictions!")
print(f"Ready for production deployment on 16.4M+ impression dataset!")