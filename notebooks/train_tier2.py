import sys
sys.path.insert(0, "setup")

import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, classification_report
)

print("\n" + "="*70)
print("ACME Industries - Tier 2 Supervised Classifier Training")
print("="*70)

# ─── 1. FETCH & PROCESS DATA ────────────────────────────────────────────────
print("Downloading AI4I 2020 dataset from UCI...")
try:
    from ucimlrepo import fetch_ucirepo
    ai4i = fetch_ucirepo(id=601)
    X_raw = ai4i.data.features
    y_raw = ai4i.data.targets
    df_raw = pd.concat([X_raw, y_raw], axis=1)
    print(f"  -> Raw dataset loaded: {df_raw.shape}")
except Exception as e:
    print(f"  -> ERROR: Failed to load dataset from UCI: {e}")
    sys.exit(1)

# Rename columns to match feature engineering expectations
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

# Add derived helper columns
df["product_type_enc"] = df["product_type"].map({"L": 0, "M": 1, "H": 2})
df["power_w"] = df["torque_nm"] * (df["rotational_speed_rpm"] * 2 * np.pi / 60)
df["machine_type"] = df["product_type"]

# Run features through backend feature engineering pipeline
sys.path.insert(0, "notebooks")
import feature_engineering_common

print("Engineering base features...")
X_engineered = feature_engineering_common.engineer_features(df)

# Append encoded Machine Type
X_engineered["product_type_enc"] = df["product_type_enc"]
feature_columns = list(X_engineered.columns)
y = df["machine_failure"]

print(f"Features dimension: {X_engineered.shape}")
print(f"Target distribution: Normal={int(len(y) - y.sum())}, Failure={int(y.sum())}")

# ─── 2. SPLIT DATA (60/20/20 Stratified) ──────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X_engineered, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Data Splits:")
print(f"  Train: {X_train.shape[0]} samples (failures: {int(y_train.sum())})")
print(f"  Val  : {X_val.shape[0]} samples (failures: {int(y_val.sum())})")
print(f"  Test : {X_test.shape[0]} samples (failures: {int(y_test.sum())})")

# ─── 3. SCALE & APPLY SMOTE ────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("Applying SMOTE to training split...")
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"  -> Resampled Train shape: {X_train_res.shape} (failures: {int(y_train_res.sum())})")

# ─── 4. MODEL COMPARISON ────────────────────────────────────────────────────
models = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        use_label_encoder=False,
        eval_metric='aucpr',
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced',
        min_samples_leaf=2,
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
}

best_model = None
best_model_name = ""
best_val_f1 = 0.0
best_threshold = 0.5
model_results = {}

print("\nTraining and comparing models on Validation split (using threshold tuning)...")
for name, clf in models.items():
    clf.fit(X_train_res, y_train_res)
    probs_val = clf.predict_proba(X_val_scaled)[:, 1]
    
    # Tune threshold on validation split to maximize F1-score
    model_best_th = 0.5
    model_best_f1 = 0.0
    for th in np.arange(0.05, 0.95, 0.01):
        preds = (probs_val >= th).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > model_best_f1:
            model_best_f1 = f1
            model_best_th = th
            
    print(f"  {name:15} -> Val F1: {model_best_f1:.4f} (at threshold {model_best_th:.2f})")
    model_results[name] = {"model": clf, "best_th": model_best_th, "best_f1": model_best_f1}
    
    if model_best_f1 > best_val_f1:
        best_val_f1 = model_best_f1
        best_model = clf
        best_model_name = name
        best_threshold = model_best_th

print(f"\n✨ Selected Best Model: {best_model_name} (Val F1: {best_val_f1:.4f})")

# ─── 5. EVALUATION ON TEST SET ──────────────────────────────────────────────
print("\n" + "-"*50)
print(f"Evaluating {best_model_name} on Unseen Test Split...")
print("-"*50)

probs_test = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred_test = (probs_test >= best_threshold).astype(int)

test_acc = accuracy_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, probs_test)
test_pr_auc = average_precision_score(y_test, probs_test)
test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
test_precision = precision_score(y_test, y_pred_test, zero_division=0)
test_recall = recall_score(y_test, y_pred_test, zero_division=0)

print(f"✨ Test Accuracy  : {test_acc:.4f}")
print(f"✨ Test ROC-AUC   : {test_auc:.4f}")
print(f"✨ Test PR-AUC    : {test_pr_auc:.4f}")
print(f"✨ Test F1-Score  : {test_f1:.4f}")
print(f"✨ Test Precision : {test_precision:.4f}")
print(f"✨ Test Recall    : {test_recall:.4f}")
print(f"✨ Tuned Threshold: {best_threshold:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=["Normal", "Failure"]))

# ─── 6. SAVE MODEL BUNDLE ──────────────────────────────────────────────────
bundle = {
    "model": best_model,
    "threshold": float(best_threshold),
    "feature_names": feature_columns,
    "scaler": scaler,
    "encoding_map": {"L": 0, "M": 1, "H": 2},
    "test_metrics": {
        "auc_roc": float(test_auc),
        "pr_auc": float(test_pr_auc),
        "f1_score": float(test_f1),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "accuracy": float(test_acc)
    },
    "classification_report": classification_report(
        y_test, y_pred_test, target_names=["Normal", "Failure"],
        zero_division=0, output_dict=True
    )
}

os.makedirs("models", exist_ok=True)
joblib.dump(bundle, "models/tier2_classifier.pkl")
print(f"\n✅ Successfully saved model bundle to models/tier2_classifier.pkl")
print("="*70 + "\n")
