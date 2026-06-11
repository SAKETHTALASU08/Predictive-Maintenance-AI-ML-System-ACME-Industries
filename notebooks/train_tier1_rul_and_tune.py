import sys
sys.path.insert(0, "setup")
sys.path.insert(0, "notebooks")

import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import joblib
import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore")

print("\n" + "="*70)
print("ACME Industries - Tier 1 Recall Recovery & RUL Regression Pipeline")
print("="*70)

# ─── 1. FETCH & PROCESS DATA ────────────────────────────────────────────────
print("Loading dataset...")
df_raw = None
try:
    from ucimlrepo import fetch_ucirepo
    print("Fetching dataset from UCI Repository...")
    ai4i = fetch_ucirepo(id=601)
    X_raw = ai4i.data.features
    y_raw = ai4i.data.targets
    ids_raw = ai4i.data.ids
    df_raw = pd.concat([ids_raw, X_raw, y_raw], axis=1)
    print(f"  -> Raw dataset loaded from UCI: {df_raw.shape}")
except Exception as e:
    print(f"  -> Warning: Failed to load dataset from UCI ({e})")

if df_raw is None:
    try:
        print("Falling back to local Databricks Delta table...")
        from spark_session import get_spark
        spark = get_spark()
        df_spark = spark.table("acme_pm.ai4i_cleaned").toPandas()
        if "UID" not in df_spark.columns:
            df_spark["UID"] = range(1, len(df_spark) + 1)
        df_raw = df_spark
        print(f"  -> Raw dataset loaded from Delta: {df_raw.shape}")
    except Exception as e:
        print(f"  -> ERROR: Failed to load dataset from Delta: {e}")
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

# Compute basic derived features
df["product_type_enc"] = df["product_type"].map({"L": 0, "M": 1, "H": 2})
df["power_w"] = df["torque_nm"] * (df["rotational_speed_rpm"] * 2 * np.pi / 60)
df["machine_type"] = df["product_type"]
df["temp_delta_k"] = df["process_temp_k"] - df["air_temp_k"]

# Ensure chronological order
df = df.sort_values("UID").reset_index(drop=True)

# Calculate RUL labels (Option B: Tool wear run based)
df["reset"] = (df["tool_wear_min"].diff() < 0).astype(int)
df["run_id"] = df.groupby("product_type")["reset"].cumsum()
df["tool_run_id"] = df["product_type"] + "_Run_" + df["run_id"].astype(str)
df["max_cycle_B"] = df.groupby("tool_run_id")["tool_wear_min"].transform("max")
df["RUL"] = df["max_cycle_B"] - df["tool_wear_min"]

print("Engineering base features...")
import feature_engineering_common
X_engineered = feature_engineering_common.engineer_features(df)
y = df["machine_failure"]

# Ensure columns match FEATURE_ORDER exactly (20 features)
feature_columns_old = feature_engineering_common.FEATURE_ORDER
X_old = X_engineered[feature_columns_old]

# Chronological time-based split (60/20/20)
n_samples = len(df)
train_end = int(0.6 * n_samples)
val_end = int(0.8 * n_samples)

X_train_old = X_old.iloc[:train_end]
y_train = y.iloc[:train_end]
y_train_rul = df["RUL"].iloc[:train_end]

X_val_old = X_old.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]
y_val_rul = df["RUL"].iloc[train_end:val_end]

X_test_old = X_old.iloc[val_end:]
y_test = y.iloc[val_end:]
y_test_rul = df["RUL"].iloc[val_end:]

print(f"Chronological Splits:")
print(f"  Train: {X_train_old.shape[0]} samples (failures: {int(y_train.sum())})")
print(f"  Val  : {X_val_old.shape[0]} samples (failures: {int(y_val.sum())})")
print(f"  Test : {X_test_old.shape[0]} samples (failures: {int(y_test.sum())})")


# ─── TASK 1: THRESHOLD TUNING FOR RECALL RECOVERY ────────────────────────────
print("\n" + "-"*50)
print("TASK 1: Tuning Tier 1 Failure Predictor Threshold")
print("-"*50)

# Load existing failure predictor
if not os.path.exists("models/best_failure_predictor.pkl"):
    print("  -> ERROR: models/best_failure_predictor.pkl not found!")
    sys.exit(1)

with open("models/best_failure_predictor.pkl", "rb") as f:
    clf_bundle = pickle.load(f)

classifier = clf_bundle["model"]
default_threshold = 0.50

# Load scaler for Tier 1
if not os.path.exists("models/phase2_scaler.pkl"):
    print("  -> ERROR: models/phase2_scaler.pkl not found!")
    sys.exit(1)

with open("models/phase2_scaler.pkl", "rb") as f:
    clf_scaler = pickle.load(f)

# Recreate scaler fit on the unscaled training data to fix scaling mismatch
print("Fitting standard scaler on training split of 20 base features...")
old_scaler = StandardScaler()
X_train_scaled_old = old_scaler.fit_transform(X_train_old)
X_val_scaled_old = old_scaler.transform(X_val_old)
X_test_scaled_old = old_scaler.transform(X_test_old)

# Predict probabilities using the classifier on the correctly scaled validation set
probs_val = classifier.predict_proba(X_val_scaled_old)[:, 1]

# Sweep thresholds from 0.1 to 0.5 in steps of 0.05
thresholds = np.arange(0.1, 0.51, 0.05)
sweep_results = []

print(f"Threshold Sweep Results (Validation Split):")
print(f"  Threshold | Precision | Recall | F1 Score")
print(f"  ------------------------------------------")

optimal_threshold = None
max_precision_opt = -1.0
optimal_metrics = None

for th in thresholds:
    th = round(th, 2)
    preds = (probs_val >= th).astype(int)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    print(f"  {th:.2f}      | {prec:.4f}    | {rec:.4f} | {f1:.4f}")
    
    sweep_results.append({
        "threshold": th,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })
    
    # Condition: Recall >= 0.92 AND Precision is maximized
    if rec >= 0.92:
        if prec > max_precision_opt:
            max_precision_opt = prec
            optimal_threshold = th
            optimal_metrics = {"precision": prec, "recall": rec, "f1_score": f1}

if optimal_threshold is None:
    print("  -> Warning: No threshold met Recall >= 0.92. Falling back to 0.10.")
    optimal_threshold = 0.10
    preds_fallback = (probs_val >= 0.10).astype(int)
    optimal_metrics = {
        "precision": precision_score(y_val, preds_fallback, zero_division=0),
        "recall": recall_score(y_val, preds_fallback, zero_division=0),
        "f1_score": f1_score(y_val, preds_fallback, zero_division=0)
    }

print(f"\nOptimal Threshold Identified: {optimal_threshold:.2f}")

# Plot Precision-Recall curve with threshold annotations
plt.figure(figsize=(8, 6))
recalled = [res["recall"] for res in sweep_results]
precisions = [res["precision"] for res in sweep_results]
thresh_labels = [res["threshold"] for res in sweep_results]

plt.plot(recalled, precisions, 'o--', color='#3b82f6', linewidth=2, markersize=6, label='Threshold Sweep')
for i, th_val in enumerate(thresh_labels):
    plt.annotate(f"{th_val:.2f}", (recalled[i], precisions[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

opt_idx = thresh_labels.index(optimal_threshold)
plt.plot(recalled[opt_idx], precisions[opt_idx], 'ro', markersize=10, label=f'Optimal ({optimal_threshold:.2f})', fillstyle='none', markeredgewidth=2)

plt.xlabel('Recall', fontsize=11, fontweight='semibold')
plt.ylabel('Precision', fontsize=11, fontweight='semibold')
plt.title('Tier 1 Validation Precision-Recall Curve', fontsize=12, fontweight='bold', pad=15)
plt.xlim(0.7, 1.05)
plt.ylim(0.7, 1.05)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left')
plt.tight_layout()

os.makedirs("eda_plots", exist_ok=True)
plt.savefig("eda_plots/tier1_pr_curve.png", dpi=300)
plt.close()
print("Saved PR curve to eda_plots/tier1_pr_curve.png")

# Evaluate validation and test sets (Default vs Optimal)
probs_test = classifier.predict_proba(X_test_scaled_old)[:, 1]

val_default_preds = (probs_val >= default_threshold).astype(int)
test_default_preds = (probs_test >= default_threshold).astype(int)
test_opt_preds = (probs_test >= optimal_threshold).astype(int)

# Validation Metrics
val_def_rec = recall_score(y_val, val_default_preds, zero_division=0)
val_def_prec = precision_score(y_val, val_default_preds, zero_division=0)
val_def_f1 = f1_score(y_val, val_default_preds, zero_division=0)

# Test Metrics
test_def_rec = recall_score(y_test, test_default_preds, zero_division=0)
test_def_prec = precision_score(y_test, test_default_preds, zero_division=0)
test_def_f1 = f1_score(y_test, test_default_preds, zero_division=0)
test_def_auc = roc_auc_score(y_test, probs_test)

test_opt_rec = recall_score(y_test, test_opt_preds, zero_division=0)
test_opt_prec = precision_score(y_test, test_opt_preds, zero_division=0)
test_opt_f1 = f1_score(y_test, test_opt_preds, zero_division=0)
test_opt_auc = roc_auc_score(y_test, probs_test)

print("\nComparison Table: Default (0.50) vs Optimal Threshold Metrics")
print("="*80)
print(f" {'Split':12} | {'Threshold':9} | {'Precision':9} | {'Recall':9} | {'F1 Score':9}")
print("-"*80)
print(f" {'Validation':12} | {default_threshold:9.2f} | {val_def_prec:9.4f} | {val_def_rec:9.4f} | {val_def_f1:9.4f}")
print(f" {'Validation':12} | {optimal_threshold:9.2f} | {optimal_metrics['precision']:9.4f} | {optimal_metrics['recall']:9.4f} | {optimal_metrics['f1_score']:9.4f} (Optimal)")
print("-"*80)
print(f" {'Test':12} | {default_threshold:9.2f} | {test_def_prec:9.4f} | {test_def_rec:9.4f} | {test_def_f1:9.4f}")
print(f" {'Test':12} | {optimal_threshold:9.2f} | {test_opt_prec:9.4f} | {test_opt_rec:9.4f} | {test_opt_f1:9.4f} (Optimal)")
print("="*80)

# Update threshold in pkl file
clf_bundle["threshold"] = float(optimal_threshold)
with open("models/best_failure_predictor.pkl", "wb") as f:
    pickle.dump(clf_bundle, f)
print("Updated models/best_failure_predictor.pkl with optimal threshold.")

# Also update models/phase2_scaler.pkl with old_scaler to ensure scaling consistency at inference time
with open("models/phase2_scaler.pkl", "wb") as f:
    pickle.dump(old_scaler, f)
print("Updated models/phase2_scaler.pkl with corrected StandardScaler.")


# ─── TASK 2: REMAINING USEFUL LIFE (RUL) REGRESSION ──────────────────────────
print("\n" + "-"*50)
print("TASK 2: Training RUL Regressor")
print("-"*50)

# Fit scaler specifically for RUL regressor on training set
reg_scaler = StandardScaler()
X_train_scaled_reg = reg_scaler.fit_transform(X_train_old)
X_test_scaled_reg = reg_scaler.transform(X_test_old)

# Train LGBMRegressor
print("Training LGBMRegressor...")
regressor = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1
)
regressor.fit(X_train_scaled_reg, y_train_rul)

# Predict and evaluate on test split
preds_rul_test = regressor.predict(X_test_scaled_reg)
preds_rul_test = np.clip(preds_rul_test, 0, None)

rmse = np.sqrt(mean_squared_error(y_test_rul, preds_rul_test))
mae = mean_absolute_error(y_test_rul, preds_rul_test)
r2 = r2_score(y_test_rul, preds_rul_test)

print(f"RUL Regression Metrics (Test Split):")
print(f"  RMSE    : {rmse:.4f}")
print(f"  MAE     : {mae:.4f}")
print(f"  R² Score: {r2:.4f}")

# Plot actual vs predicted RUL
plt.figure(figsize=(8, 6))
plt.scatter(y_test_rul, preds_rul_test, alpha=0.4, color='#f59e0b', edgecolors='none', label='Test Samples')
min_val = min(y_test_rul.min(), preds_rul_test.min())
max_val = max(y_test_rul.max(), preds_rul_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual RUL (cycles)', fontsize=11, fontweight='semibold')
plt.ylabel('Predicted RUL (cycles)', fontsize=11, fontweight='semibold')
plt.title('Tier 1 Actual vs Predicted RUL Scatter Plot', fontsize=12, fontweight='bold', pad=15)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("eda_plots/tier1_rul_predictions.png", dpi=300)
plt.close()
print("Saved RUL scatter plot to eda_plots/tier1_rul_predictions.png")

# Maintenance urgency mapping function
def map_urgency(rul):
    if rul > 100:
        return "Healthy"
    elif rul >= 50:
        return "Monitor"
    elif rul >= 20:
        return "Plan Maintenance"
    else:
        return "Immediate Action"

# Save Tier 1 RUL regressor bundle
reg_bundle = {
    "model": regressor,
    "scaler": reg_scaler,
    "feature_names": feature_columns_old,
    "test_metrics": {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2)
    },
    "urgency_mapping": {
        "Healthy": "RUL > 100 cycles",
        "Monitor": "RUL 50-100 cycles",
        "Plan Maintenance": "RUL 20-50 cycles",
        "Immediate Action": "RUL < 20 cycles"
    }
}
with open("models/tier1_regressor.pkl", "wb") as f:
    pickle.dump(reg_bundle, f)
print("Saved models/tier1_regressor.pkl")


# ─── TASK 3: FEATURE ENGINEERING FOR TIER 1 UPLIFT ───────────────────────────
print("\n" + "-"*50)
print("TASK 3: Feature Engineering for Tier 1 Uplift")
print("-"*50)

# Copy base dataframe to engineer features
df_feat = df.copy()
sensor_cols = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min", "power_w", "temp_delta_k"]

# Compute rolling stats, lags, diff, and pct change per product_type group to prevent leakage
grouped = df_feat.groupby("product_type")
new_ts_features = {}

print("Engineering rolling features (windows: 5, 10, 20)...")
for col in sensor_cols:
    for window in [5, 10, 20]:
        roll = grouped[col].rolling(window, min_periods=1)
        new_ts_features[f"{col}_roll{window}_mean"] = roll.mean().reset_index(level=0, drop=True)
        new_ts_features[f"{col}_roll{window}_std"] = roll.std().fillna(0.0).reset_index(level=0, drop=True)
        new_ts_features[f"{col}_roll{window}_min"] = roll.min().reset_index(level=0, drop=True)
        new_ts_features[f"{col}_roll{window}_max"] = roll.max().reset_index(level=0, drop=True)

print("Engineering lag features (lags: 1, 3, 5)...")
for col in sensor_cols:
    for lag in [1, 3, 5]:
        lag_series = grouped[col].shift(lag)
        new_ts_features[f"{col}_lag{lag}"] = lag_series.groupby(df_feat["product_type"]).bfill().fillna(0.0)

print("Engineering rate-of-change features...")
for col in sensor_cols:
    new_ts_features[f"{col}_diff1"] = grouped[col].diff(1).fillna(0.0)
    new_ts_features[f"{col}_pct_change1"] = grouped[col].pct_change(1).fillna(0.0).replace([np.inf, -np.inf], 0.0)

# Concat engineered base features and the new rolling/lag/diff features
X_base_engineered = feature_engineering_common.engineer_features(df_feat)
df_ts = pd.DataFrame(new_ts_features)
X_new_features = pd.concat([X_base_engineered, df_ts], axis=1)

feature_columns_new = list(X_new_features.columns)
print(f"Enriched feature set shape: {X_new_features.shape}")

# Split new features chronologically (60/20/20)
X_train_new = X_new_features.iloc[:train_end]
X_val_new = X_new_features.iloc[train_end:val_end]
X_test_new = X_new_features.iloc[val_end:]

# Scale new features
new_scaler = StandardScaler()
X_train_new_sc = new_scaler.fit_transform(X_train_new)
X_val_new_sc = new_scaler.transform(X_val_new)
X_test_new_sc = new_scaler.transform(X_test_new)

# Train validation pass classifier on new features
print("Training validation pass classifier on new features...")
val_clf = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    class_weight="balanced",
    random_state=42,
    verbose=-1
)
val_clf.fit(X_train_new_sc, y_train)

# Tune threshold on validation set with new features (Recall >= 0.92)
probs_val_new = val_clf.predict_proba(X_val_new_sc)[:, 1]
optimal_threshold_new = 0.50
max_prec_new = -1.0
for th in thresholds:
    th = round(th, 2)
    preds = (probs_val_new >= th).astype(int)
    rec = recall_score(y_val, preds, zero_division=0)
    prec = precision_score(y_val, preds, zero_division=0)
    if rec >= 0.92:
        if prec > max_prec_new:
            max_prec_new = prec
            optimal_threshold_new = th

# Predict on test split with new features
probs_test_new = val_clf.predict_proba(X_test_new_sc)[:, 1]
preds_test_new = (probs_test_new >= optimal_threshold_new).astype(int)

test_new_rec = recall_score(y_test, preds_test_new, zero_division=0)
test_new_prec = precision_score(y_test, preds_test_new, zero_division=0)
test_new_f1 = f1_score(y_test, preds_test_new, zero_division=0)
test_new_auc = roc_auc_score(y_test, probs_test_new)

print("\nFeature Impact Summary (Test Split):")
print("="*60)
print(f" Metric    | Before New Features | After New Features")
print("-"*60)
print(f" AUC-ROC   | {test_opt_auc:19.4f} | {test_new_auc:18.4f}")
print(f" Precision | {test_opt_prec:19.4f} | {test_new_prec:18.4f}")
print(f" Recall    | {test_opt_rec:19.4f} | {test_new_rec:18.4f}")
print(f" F1 Score  | {test_opt_f1:19.4f} | {test_new_f1:18.4f}")
print("="*60)

# Check Recall improvement >= 0.03
recall_diff = test_new_rec - test_opt_rec
print(f"Recall Improvement: {recall_diff:+.4f}")
if recall_diff >= 0.03:
    print("\n[STATUS] NEW FEATURE SET APPROVED FOR FULL RETRAIN")
else:
    print("\n[STATUS] NEW FEATURE SET NOT APPROVED FOR FULL RETRAIN")

# Plot feature importance (top 20)
plt.figure(figsize=(10, 8))
importances = val_clf.feature_importances_
indices = np.argsort(importances)[::-1][:20]
top_importances = importances[indices]
top_features = [feature_columns_new[i] for i in indices]

plt.barh(range(20), top_importances[::-1], color='#3b82f6', align='center')
plt.yticks(range(20), top_features[::-1], fontsize=9)
plt.xlabel('Feature Importance', fontsize=11, fontweight='semibold')
plt.title('Top 20 Feature Importances (Time-Series Aware Enriched Features)', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("eda_plots/tier1_feature_importance.png", dpi=300)
plt.close()
print("Saved feature importance chart to eda_plots/tier1_feature_importance.png")


# ─── COMBINED OUTPUT ─────────────────────────────────────────────────────────
print("\n" + "-"*50)
print("COMBINED OUTPUT: Test Split Unified Dataframe")
print("-"*50)

asset_id_test = df["product_type"].iloc[val_end:].map({
    "L": "Asset_L",
    "M": "Asset_M",
    "H": "Asset_H"
}).values

urgency_labels = [map_urgency(r) for r in preds_rul_test]

combined_df = pd.DataFrame({
    "asset_id": asset_id_test,
    "failure_score": np.round(probs_test, 4),
    "is_failure (tuned)": test_opt_preds,
    "predicted_RUL": np.round(preds_rul_test, 1),
    "urgency_label": urgency_labels,
    "feature_version": "v1.0"
})

combined_df.to_csv("models/asset_failure_predictions.csv", index=False)
print("Saved combined predictions to models/asset_failure_predictions.csv")
print("\nFirst 10 sample rows of predictions:")
print(combined_df.head(10).to_string(index=False))


# ─── FINAL VALIDATION — TIER INVERSION CHECK ─────────────────────────────────
print("\n" + "-"*50)
print("FINAL VALIDATION — TIER INVERSION CHECK")
print("-"*50)

# Load Tier 2 metrics
tier2_recall = 0.8080
tier2_f1 = 0.8871
tier2_prec = 0.9821
tier2_auc = 0.9877

if os.path.exists("models/tier2_classifier.pkl"):
    try:
        t2_bundle = joblib.load("models/tier2_classifier.pkl")
        t2_metrics = t2_bundle.get("test_metrics", {})
        tier2_recall = t2_metrics.get("recall", tier2_recall)
        tier2_prec = t2_metrics.get("precision", tier2_prec)
        tier2_f1 = t2_metrics.get("f1_score", tier2_f1)
        tier2_auc = t2_metrics.get("auc_roc", tier2_auc)
        print(f"Loaded Tier 2 metrics from models/tier2_classifier.pkl successfully.")
    except Exception as e:
        print(f"Warning: Could not read Tier 2 metrics from file ({e}). Using constants.")

print("\nSide-by-Side Model Comparison (Test Split):")
print("="*60)
print(f" Metric    | Tier 1 Failure Predictor | Tier 2 Anomaly Detector")
print("-"*60)
print(f" AUC-ROC   | {test_opt_auc:24.4f} | {tier2_auc:23.4f}")
print(f" Precision | {test_opt_prec:24.4f} | {tier2_prec:23.4f}")
print(f" Recall    | {test_opt_rec:24.4f} | {tier2_recall:23.4f}")
print(f" F1 Score  | {test_opt_f1:24.4f} | {tier2_f1:23.4f}")
print("="*60)

# Assert: Tier 1 Recall > Tier 2 Recall (0.808)
assert_val = 0.8080
if test_opt_rec > assert_val:
    print(f"Assertion Passed: Tier 1 Recall ({test_opt_rec:.4f}) > Tier 2 Baseline Recall ({assert_val:.4f})")
    print("✅ TIER INVERSION RESOLVED")
else:
    print(f"Assertion Failed: Tier 1 Recall ({test_opt_rec:.4f}) <= Tier 2 Baseline Recall ({assert_val:.4f})")
    print("⚠️ TIER INVERSION STILL PRESENT")

print("\n" + "="*70 + "\n")
