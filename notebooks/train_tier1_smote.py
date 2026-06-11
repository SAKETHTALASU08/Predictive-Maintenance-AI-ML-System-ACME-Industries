import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Resolve directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
print("Loading dataset...")
df_raw = None
try:
    from ucimlrepo import fetch_ucirepo
    ai4i = fetch_ucirepo(id=601)
    X_raw = ai4i.data.features
    y_raw = ai4i.data.targets
    ids_raw = ai4i.data.ids
    df_raw = pd.concat([ids_raw, X_raw, y_raw], axis=1)
except Exception as e:
    print(f"Warning: Failed to load dataset from UCI ({e})")

if df_raw is None:
    try:
        from spark_session import get_spark
        spark = get_spark()
        df_spark = spark.table("acme_pm.ai4i_cleaned").toPandas()
        if "UID" not in df_spark.columns:
            df_spark["UID"] = range(1, len(df_spark) + 1)
        df_raw = df_spark
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)

# Rename columns
rename_map = {
    "Type"                    : "product_type",
    "Air temperature"         : "air_temp_k",
    "Process temperature"     : "process_temp_k",
    "Rotational speed"        : "rotational_speed_rpm",
    "Torque"                  : "torque_nm",
    "Tool wear"               : "tool_wear_min",
    "Machine failure"         : "machine_failure",
}
rename_map = {k: v for k, v in rename_map.items() if k in df_raw.columns}
df = df_raw.rename(columns=rename_map)

# Compute basic features
df["product_type_enc"] = df["product_type"].map({"L": 0, "M": 1, "H": 2})
df["power_w"] = df["torque_nm"] * (df["rotational_speed_rpm"] * 2 * np.pi / 60)
df["machine_type"] = df["product_type"]
df["temp_delta_k"] = df["process_temp_k"] - df["air_temp_k"]

# Sort chronologically
df = df.sort_values("UID").reset_index(drop=True)

# Engineer features
import feature_engineering_common
X_engineered = feature_engineering_common.engineer_features(df)
y = df["machine_failure"]

feature_columns = feature_engineering_common.FEATURE_ORDER
X_old = X_engineered[feature_columns]

n_samples = len(df)
train_end = int(0.6 * n_samples)
val_end = int(0.8 * n_samples)

X_train = X_old.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X_old.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X_old.iloc[val_end:]
y_test = y.iloc[val_end:]

# Fit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to training split
print("Applying SMOTE to training split...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"SMOTE Train shape: {X_train_sm.shape} (failures: {int(y_train_sm.sum())})")

# Hyperparameter Grid Search
print("Starting grid search hyperparameter tuning with SMOTE...")
best_val_f1 = -1.0
best_params = None
best_threshold = 0.50
best_model = None

# Grid parameters
max_depths = [4, 6, 8]
num_leaves_list = [15, 31, 63]
learning_rates = [0.01, 0.05, 0.1]
n_estimators_list = [200, 300, 500]
min_child_samples_list = [5, 10, 20]
class_weights = [None, 'balanced']

import itertools
param_combinations = list(itertools.product(
    max_depths, num_leaves_list, learning_rates, n_estimators_list, min_child_samples_list, class_weights
))

print(f"Total combinations to try: {len(param_combinations)}")

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
    clf.fit(X_train_sm, y_train_sm)
    probs_val = clf.predict_proba(X_val_scaled)[:, 1]
    
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
probs_test = best_model.predict_proba(X_test_scaled)[:, 1]
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
    # Save the updated model and scaler
    clf_bundle = {
        "model": best_model,
        "threshold": float(best_threshold),
        "name": "LightGBM",
        "features": feature_columns
    }
    with open(os.path.join(base_dir, "models", "best_failure_predictor.pkl"), "wb") as f:
        pickle.dump(clf_bundle, f)
    print("Saved tuned model to models/best_failure_predictor.pkl")
    
    with open(os.path.join(base_dir, "models", "phase2_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("Saved scaler to models/phase2_scaler.pkl")
else:
    print("\n⚠️ WARNING: F1 or Recall did not meet the >0.85 target. Model was not updated.")
