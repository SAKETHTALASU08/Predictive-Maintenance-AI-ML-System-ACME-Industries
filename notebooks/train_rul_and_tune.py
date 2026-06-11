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
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore")

print("\n" + "="*70)
print("ACME Industries - Tier 2 Recall Recovery & RUL Regression Pipeline")
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
        # Add UID if not present to ensure proper chronological sorting
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

# Ensure chronological order
df = df.sort_values("UID").reset_index(drop=True)

# Calculate RUL labels (Option B: Tool wear run based)
# Reset signals tool replacement (where tool wear decreases)
df["reset"] = (df["tool_wear_min"].diff() < 0).astype(int)
df["run_id"] = df.groupby("product_type")["reset"].cumsum()
df["tool_run_id"] = df["product_type"] + "_Run_" + df["run_id"].astype(str)
df["max_cycle_B"] = df.groupby("tool_run_id")["tool_wear_min"].transform("max")
df["RUL"] = df["max_cycle_B"] - df["tool_wear_min"]

print("Engineering base features...")
import feature_engineering_common
X_engineered = feature_engineering_common.engineer_features(df)
X_engineered["product_type_enc"] = df["product_type_enc"]
y = df["machine_failure"]

feature_columns = list(X_engineered.columns)
print(f"Features dimension: {X_engineered.shape}")

# Chronological time-based split (60/20/20)
n_samples = len(df)
train_end = int(0.6 * n_samples)
val_end = int(0.8 * n_samples)

X_train = X_engineered.iloc[:train_end]
y_train = y.iloc[:train_end]
y_train_rul = df["RUL"].iloc[:train_end]

X_val = X_engineered.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]
y_val_rul = df["RUL"].iloc[train_end:val_end]

X_test = X_engineered.iloc[val_end:]
y_test = y.iloc[val_end:]
y_test_rul = df["RUL"].iloc[val_end:]

print(f"Chronological Splits:")
print(f"  Train: {X_train.shape[0]} samples (failures: {int(y_train.sum())})")
print(f"  Val  : {X_val.shape[0]} samples (failures: {int(y_val.sum())})")
print(f"  Test : {X_test.shape[0]} samples (failures: {int(y_test.sum())})")

# ─── TASK 1: THRESHOLD TUNING FOR RECALL RECOVERY ────────────────────────────
print("\n" + "-"*50)
print("TASK 1: Tuning Classifier Threshold on Validation Split")
print("-"*50)

# Load existing classifier bundle
if not os.path.exists("models/tier2_classifier.pkl"):
    print("  -> ERROR: models/tier2_classifier.pkl not found!")
    sys.exit(1)

clf_bundle = joblib.load("models/tier2_classifier.pkl")
classifier = clf_bundle["model"]
clf_scaler = clf_bundle["scaler"]
default_threshold = 0.50 # As per requirements

# Scale validation features using classifier's scaler
X_val_scaled_clf = clf_scaler.transform(X_val)
probs_val = classifier.predict_proba(X_val_scaled_clf)[:, 1]

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
    
    # Condition: Recall >= 0.90 AND Precision is maximized
    if rec >= 0.90:
        if prec > max_precision_opt:
            max_precision_opt = prec
            optimal_threshold = th
            optimal_metrics = {"precision": prec, "recall": rec, "f1_score": f1}

# If no threshold meets Recall >= 0.90, fall back to the lowest threshold (0.10)
if optimal_threshold is None:
    print("  -> Warning: No threshold met Recall >= 0.90 on Validation set. Selecting threshold 0.10 as fallback.")
    optimal_threshold = 0.10
    preds_fallback = (probs_val >= 0.10).astype(int)
    optimal_metrics = {
        "precision": precision_score(y_val, preds_fallback, zero_division=0),
        "recall": recall_score(y_val, preds_fallback, zero_division=0),
        "f1_score": f1_score(y_val, preds_fallback, zero_division=0)
    }

print(f"\nIdentified Optimal Threshold: {optimal_threshold:.2f}")

# Plot Precision-Recall curve with threshold annotations
plt.figure(figsize=(8, 6))
recalled = [res["recall"] for res in sweep_results]
precisions = [res["precision"] for res in sweep_results]
thresh_labels = [res["threshold"] for res in sweep_results]

plt.plot(recalled, precisions, 'o--', color='#6366f1', linewidth=2, markersize=6, label='Threshold Sweep')
for i, th_val in enumerate(thresh_labels):
    plt.annotate(f"{th_val:.2f}", (recalled[i], precisions[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='#374151')

# Highlight optimal threshold
opt_idx = thresh_labels.index(optimal_threshold)
plt.plot(recalled[opt_idx], precisions[opt_idx], 'ro', markersize=10, label=f'Optimal ({optimal_threshold:.2f})', fillstyle='none', markeredgewidth=2)

plt.xlabel('Recall', fontsize=11, fontweight='semibold')
plt.ylabel('Precision', fontsize=11, fontweight='semibold')
plt.title('Validation Precision-Recall Curve (Threshold Sweep)', fontsize=12, fontweight='bold', pad=15)
plt.xlim(0.7, 1.05)
plt.ylim(0.7, 1.05)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left')
plt.tight_layout()

os.makedirs("eda_plots", exist_ok=True)
pr_curve_path = "eda_plots/tier2_pr_curve.png"
plt.savefig(pr_curve_path, dpi=300)
plt.close()
print(f"Saved Precision-Recall curve plot to {pr_curve_path}")

# Evaluate on Validation set (Default vs Optimal)
default_preds_val = (probs_val >= default_threshold).astype(int)
default_prec_val = precision_score(y_val, default_preds_val, zero_division=0)
default_rec_val = recall_score(y_val, default_preds_val, zero_division=0)
default_f1_val = f1_score(y_val, default_preds_val, zero_division=0)

# Evaluate on Test set (Default vs Optimal)
X_test_scaled_clf = clf_scaler.transform(X_test)
probs_test = classifier.predict_proba(X_test_scaled_clf)[:, 1]

default_preds_test = (probs_test >= default_threshold).astype(int)
default_prec_test = precision_score(y_test, default_preds_test, zero_division=0)
default_rec_test = recall_score(y_test, default_preds_test, zero_division=0)
default_f1_test = f1_score(y_test, default_preds_test, zero_division=0)

opt_preds_test = (probs_test >= optimal_threshold).astype(int)
opt_prec_test = precision_score(y_test, opt_preds_test, zero_division=0)
opt_rec_test = recall_score(y_test, opt_preds_test, zero_division=0)
opt_f1_test = f1_score(y_test, opt_preds_test, zero_division=0)

print("\nComparison Table: Default (0.50) vs Optimal Threshold Metrics")
print("="*80)
print(f" {'Split':12} | {'Threshold':9} | {'Precision':9} | {'Recall':9} | {'F1 Score':9}")
print("-"*80)
print(f" {'Validation':12} | {default_threshold:9.2f} | {default_prec_val:9.4f} | {default_rec_val:9.4f} | {default_f1_val:9.4f}")
print(f" {'Validation':12} | {optimal_threshold:9.2f} | {optimal_metrics['precision']:9.4f} | {optimal_metrics['recall']:9.4f} | {optimal_metrics['f1_score']:9.4f} (Optimal)")
print("-"*80)
print(f" {'Test':12} | {default_threshold:9.2f} | {default_prec_test:9.4f} | {default_rec_test:9.4f} | {default_f1_test:9.4f}")
print(f" {'Test':12} | {optimal_threshold:9.2f} | {opt_prec_test:9.4f} | {opt_rec_test:9.4f} | {opt_f1_test:9.4f} (Optimal)")
print("="*80)

# Save updated classifier bundle with optimal threshold and performance report
clf_bundle["threshold"] = float(optimal_threshold)
clf_bundle["test_metrics"]["precision"] = float(opt_prec_test)
clf_bundle["test_metrics"]["recall"] = float(opt_rec_test)
clf_bundle["test_metrics"]["f1_score"] = float(opt_f1_test)
clf_bundle["classification_report"] = {
    "Normal": {
        "precision": float(precision_score(y_test, opt_preds_test, pos_label=0, zero_division=0)),
        "recall": float(recall_score(y_test, opt_preds_test, pos_label=0, zero_division=0)),
        "f1-score": float(f1_score(y_test, opt_preds_test, pos_label=0, zero_division=0)),
        "support": int(len(y_test) - y_test.sum())
    },
    "Failure": {
        "precision": float(opt_prec_test),
        "recall": float(opt_rec_test),
        "f1-score": float(opt_f1_test),
        "support": int(y_test.sum())
    },
    "macro avg": {
        "precision": float(precision_score(y_test, opt_preds_test, average='macro', zero_division=0)),
        "recall": float(recall_score(y_test, opt_preds_test, average='macro', zero_division=0)),
        "f1-score": float(f1_score(y_test, opt_preds_test, average='macro', zero_division=0)),
        "support": int(len(y_test))
    },
    "weighted avg": {
        "precision": float(precision_score(y_test, opt_preds_test, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_test, opt_preds_test, average='weighted', zero_division=0)),
        "f1-score": float(f1_score(y_test, opt_preds_test, average='weighted', zero_division=0)),
        "support": int(len(y_test))
    },
    "accuracy": float(np.mean(y_test == opt_preds_test))
}

joblib.dump(clf_bundle, "models/tier2_classifier.pkl")
print("Updated models/tier2_classifier.pkl with optimal threshold.")


# ─── TASK 2: REMAINING USEFUL LIFE (RUL) REGRESSION ──────────────────────────
print("\n" + "-"*50)
print("TASK 2: Training RUL Regressor")
print("-"*50)

# Scale features specifically for the regressor
reg_scaler = StandardScaler()
X_train_scaled_reg = reg_scaler.fit_transform(X_train)
X_test_scaled_reg = reg_scaler.transform(X_test)

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

# Evaluate on test split
preds_rul_test = regressor.predict(X_test_scaled_reg)
# Clip predictions at 0 (RUL cannot be negative)
preds_rul_test = np.clip(preds_rul_test, 0, None)

rmse = np.sqrt(mean_squared_error(y_test_rul, preds_rul_test))
mae = mean_absolute_error(y_test_rul, preds_rul_test)
r2 = r2_score(y_test_rul, preds_rul_test)

print(f"RUL Regression Metrics (Test Split):")
print(f"  RMSE    : {rmse:.4f}")
print(f"  MAE     : {mae:.4f}")
print(f"  R² Score: {r2:.4f}")

# Plot Actual RUL vs Predicted RUL scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_rul, preds_rul_test, alpha=0.4, color='#f59e0b', edgecolors='none', label='Test Samples')
# Draw identity line
min_val = min(y_test_rul.min(), preds_rul_test.min())
max_val = max(y_test_rul.max(), preds_rul_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual RUL (cycles)', fontsize=11, fontweight='semibold')
plt.ylabel('Predicted RUL (cycles)', fontsize=11, fontweight='semibold')
plt.title('Actual vs Predicted RUL Scatter Plot', fontsize=12, fontweight='bold', pad=15)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()

rul_plot_path = "eda_plots/rul_predictions.png"
plt.savefig(rul_plot_path, dpi=300)
plt.close()
print(f"Saved RUL scatter plot to {rul_plot_path}")

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

# Save RUL model bundle
reg_bundle = {
    "model": regressor,
    "scaler": reg_scaler,
    "feature_names": feature_columns,
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
joblib.dump(reg_bundle, "models/tier2_regressor.pkl")
print("Saved models/tier2_regressor.pkl")


# ─── COMBINED OUTPUT ─────────────────────────────────────────────────────────
print("\n" + "-"*50)
print("COMBINED OUTPUT: Test Split Unified Dataframe")
print("-"*50)

# Create asset_id mapping
asset_id_test = df["product_type"].iloc[val_end:].map({
    "L": "Asset_L",
    "M": "Asset_M",
    "H": "Asset_H"
}).values

# Apply optimal threshold for is_anomaly
opt_is_anomaly = (probs_test >= optimal_threshold).astype(int)

# Map predicted RUL to maintenance urgency
urgency_labels = [map_urgency(r) for r in preds_rul_test]

# Construct combined dataframe
combined_df = pd.DataFrame({
    "asset_id": asset_id_test,
    "anomaly_score": np.round(probs_test, 4),
    "is_anomaly (tuned)": opt_is_anomaly,
    "predicted_RUL": np.round(preds_rul_test, 1),
    "urgency_label": urgency_labels
})

combined_df.to_csv("models/asset_rul_predictions.csv", index=False)
print("Saved combined predictions to models/asset_rul_predictions.csv")
print("\nFirst 10 sample rows of predictions:")
print(combined_df.head(10).to_string(index=False))
print("\n" + "="*70 + "\n")
