import sys
sys.path.insert(0, "setup")

from spark_session import get_spark
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle, os

spark = get_spark("acme_pm")
os.makedirs("models", exist_ok=True)

# ── 1. Load features ──────────────────────────────────────────────────────────
df = spark.table("acme_pm.ai4i_features").toPandas()
print(f"Loaded acme_pm.ai4i_features: {df.shape}")

# ── 2. Select top 20 features (from EDA) + drop power_w (0.98 corr) ──────────
TOP_FEATURES = [
    "torque_nm_freq_dev",
    "power_w_freq_dev",
    "power_w_rms_dev",
    "torque_nm_rms_dev",
    "feat_stat_std",
    "rotational_speed_rpm_rms_dev",
    "feat_stat_range",
    "rotational_speed_rpm_freq_dev",
    "power_speed_ratio",
    "wear_torque_interact",
    "mech_stress",
    "feat_stat_mean",
    "temp_delta_k_rms_dev",
    "power_w_roll10_max",
    "temp_delta_k_roll10_mean",
    "temp_delta_k_roll10_max",
    "temp_delta_k",
    "torque_nm_roll10_max",
    "temp_delta_k_freq_dev",
    "tool_wear_ratio",
]

# Keep only features that exist in the table
TOP_FEATURES = [f for f in TOP_FEATURES if f in df.columns]
print(f"\nUsing {len(TOP_FEATURES)} features:")
for f in TOP_FEATURES:
    print(f"  {f}")

X = df[TOP_FEATURES].copy()
y = df["machine_failure"].copy()

print(f"\nClass distribution before SMOTE:")
print(y.value_counts().to_string())

# ── 3. Train / val / test split (60/20/20) ────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nSplit sizes:")
print(f"  Train : {X_train.shape[0]:,} rows  (failures: {y_train.sum()})")
print(f"  Val   : {X_val.shape[0]:,}  rows  (failures: {y_val.sum()})")
print(f"  Test  : {X_test.shape[0]:,}  rows  (failures: {y_test.sum()})")

# ── 4. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

with open("models/phase2_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("\nScaler saved → models/phase2_scaler.pkl")

# ── 5. SMOTE on training set only ─────────────────────────────────────────────
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)

print(f"\nClass distribution after SMOTE:")
print(pd.Series(y_train_sm).value_counts().to_string())
print(f"Train size after SMOTE: {X_train_sm.shape[0]:,}")

# ── 6. Anomaly detection set (normal samples only for unsupervised training) ──
normal_mask   = y_train == 0
X_train_normal = X_train_sc[normal_mask]
print(f"\nAnomaly detection training set (normal only): {X_train_normal.shape[0]:,} rows")

# ── 7. Save all splits as numpy arrays ───────────────────────────────────────
np.save("models/X_train_sm.npy",     X_train_sm)
np.save("models/y_train_sm.npy",     y_train_sm)
np.save("models/X_train_normal.npy", X_train_normal)
np.save("models/X_val_sc.npy",       X_val_sc)
np.save("models/y_val.npy",          y_val.values)
np.save("models/X_test_sc.npy",      X_test_sc)
np.save("models/y_test.npy",         y_test.values)

# Save feature names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(TOP_FEATURES, f)

print("\n✅  All splits saved to models/")
print("""
Summary:
  models/X_train_sm.npy      ← SMOTE balanced (failure prediction)
  models/X_train_normal.npy  ← Normal only    (anomaly detection)
  models/X_val_sc.npy        ← Validation set
  models/X_test_sc.npy       ← Test set
  models/phase2_scaler.pkl   ← Scaler
  models/feature_names.pkl   ← Feature list
""")