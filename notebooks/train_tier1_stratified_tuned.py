import os
import sys
import pickle
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Resolve directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")

X_train = np.load(os.path.join(model_dir, "X_train_sm.npy"))
y_train = np.load(os.path.join(model_dir, "y_train_sm.npy"))
X_val = np.load(os.path.join(model_dir, "X_val_sc.npy"))
y_val = np.load(os.path.join(model_dir, "y_val.npy"))
X_test = np.load(os.path.join(model_dir, "X_test_sc.npy"))
y_test = np.load(os.path.join(model_dir, "y_test.npy"))

with open(os.path.join(model_dir, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

print(f"Train (SMOTE) : {X_train.shape}  failures: {int(y_train.sum())}")
print(f"Val           : {X_val.shape}    failures: {int(y_val.sum())}")
print(f"Test          : {X_test.shape}   failures: {int(y_test.sum())}")

# Grid Search Parameters
max_depths = [4, 6, 8, 10]
num_leaves_list = [15, 31, 63, 127]
learning_rates = [0.01, 0.03, 0.05, 0.1]
n_estimators_list = [100, 200, 300, 500]
min_child_samples_list = [5, 10, 20, 50]
class_weights = [None, 'balanced']

import itertools
param_combinations = list(itertools.product(
    max_depths, num_leaves_list, learning_rates, n_estimators_list, min_child_samples_list, class_weights
))

print(f"Total parameter combinations: {len(param_combinations)}")

best_val_f1 = -1.0
best_params = None
best_threshold = 0.50
best_model = None

for i, (md, nl, lr, ne, mcs, cw) in enumerate(param_combinations):
    if nl >= 2**md:
        continue
        
    clf = LGBMClassifier(
        max_depth=md,
        num_leaves=nl,
        learning_rate=lr,
        n_estimators=ne,
        min_child_samples=mcs,
        class_weight=cw,
        random_state=42,
        verbose=-1
    )
    clf.fit(X_train, y_train)
    probs_val = clf.predict_proba(X_val)[:, 1]
    
    # Sweep thresholds to find the best F1 where Recall >= 0.85
    for th in np.arange(0.10, 0.70, 0.02):
        th = round(th, 2)
        preds_val = (probs_val >= th).astype(int)
        rec = recall_score(y_val, preds_val, zero_division=0)
        
        if rec >= 0.85:
            f1 = f1_score(y_val, preds_val, zero_division=0)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_threshold = th
                best_params = {
                    "max_depth": md,
                    "num_leaves": nl,
                    "learning_rate": lr,
                    "n_estimators": ne,
                    "min_child_samples": mcs,
                    "class_weight": cw
                }
                best_model = clf

print(f"\nTuning Complete!")
print(f"Best Validation F1: {best_val_f1:.4f} @ threshold {best_threshold:.2f}")
print(f"Best Parameters: {best_params}")

# Evaluate best model on test split
probs_test = best_model.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= best_threshold).astype(int)

test_rec = recall_score(y_test, preds_test, zero_division=0)
test_prec = precision_score(y_test, preds_test, zero_division=0)
test_f1 = f1_score(y_test, preds_test, zero_division=0)

print(f"\nFinal Test Split Metrics:")
print(f"  Threshold : {best_threshold:.2f}")
print(f"  Precision : {test_prec:.4f}")
print(f"  Recall    : {test_rec:.4f} (Target: >0.85)")
print(f"  F1 Score  : {test_f1:.4f} (Target: >0.85)")

if test_f1 >= 0.85 and test_rec >= 0.85:
    print("\n✅ SUCCESS: BOTH F1 AND RECALL TARGETS MET!")
    # Save the updated model
    clf_bundle = {
        "model": best_model,
        "threshold": float(best_threshold),
        "name": "LightGBM",
        "features": feature_names
    }
    with open(os.path.join(model_dir, "best_failure_predictor.pkl"), "wb") as f:
        pickle.dump(clf_bundle, f)
    print("Saved tuned model to models/best_failure_predictor.pkl")
else:
    print("\n⚠️ WARNING: F1 or Recall did not meet the >0.85 target. Model was not updated.")
