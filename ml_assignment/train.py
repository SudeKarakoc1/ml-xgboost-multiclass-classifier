"""
train.py
========
Training script for the ML Assignment.

Pipeline:
  1. Load X_train.csv / y_train.csv
  2. StandardScaler normalization
  3. XGBoost classifier with balanced class weights
  4. 5-fold cross-validation evaluation
  5. Final model trained on full dataset, saved as model.ubj

Run:
    python train.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# =============================================================================
# 1. Load data
# =============================================================================
print("Loading data...")
X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv")["label"]
print(f"  X shape : {X.shape}")
print(f"  y dist  : {dict(y.value_counts().sort_index())}")

# =============================================================================
# 2. Preprocessing — StandardScaler
# =============================================================================
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Print fitted params for hard-coding in preprocess.py
print("\nFitted scaler params (for preprocess.py):")
print("  means:", scaler.mean_)
print("  stds :", scaler.scale_)

# =============================================================================
# 3. Model definition
# =============================================================================
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

# =============================================================================
# 4. Cross-validation
# =============================================================================
print("\nRunning 5-fold cross-validation...")
sample_weights = compute_sample_weight("balanced", y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    model,
    X_sc,
    y,
    cv=cv,
    scoring=["accuracy", "f1_macro"],
    params={"sample_weight": sample_weights},
)

acc  = cv_results["test_accuracy"].mean()
f1   = cv_results["test_f1_macro"].mean()
comp = 0.5 * acc + 0.5 * f1

print(f"  Accuracy   : {acc:.4f} ± {cv_results['test_accuracy'].std():.4f}")
print(f"  F1 macro   : {f1:.4f} ± {cv_results['test_f1_macro'].std():.4f}")
print(f"  Composite  : {comp:.4f}")

# =============================================================================
# 5. Train on full dataset & save
# =============================================================================
print("\nTraining on full dataset...")
model.fit(X_sc, y, sample_weight=sample_weights)

model.save_model("model.ubj")
print("Model saved → model.ubj")
