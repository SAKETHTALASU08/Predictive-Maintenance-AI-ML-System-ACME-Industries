import sys
sys.path.insert(0, "setup")

from spark_session import get_spark
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ── Connect ───────────────────────────────────────────────────────────────────
spark = get_spark("acme_pm")

# ── 1. Load cleaned table ─────────────────────────────────────────────────────
df = spark.table("acme_pm.ai4i_cleaned").toPandas()
print(f"Loaded acme_pm.ai4i_cleaned: {df.shape}")

# Add row index as udi if not present
if "udi" not in df.columns:
    df["udi"] = range(1, len(df) + 1)

# ── 2. Run feature engineering using unified single source of truth ───────────
import feature_engineering_common
feat_df_raw = feature_engineering_common.engineer_features(df)
feature_cols = feature_engineering_common.FEATURE_ORDER

# Combine engineered features with keys and targets
feat_df = pd.concat([
    feat_df_raw,
    df[["udi", "machine_failure", "failure_type"]]
], axis=1)

print(f"\nTotal engineered features: {len(feature_cols)}")

# ── 3. Scale + save ───────────────────────────────────────────────────────────
scaler  = StandardScaler()
feat_df[feature_cols] = scaler.fit_transform(feat_df[feature_cols])

scaler_params = pd.DataFrame({
    "feature": feature_cols,
    "mean"   : scaler.mean_,
    "scale"  : scaler.scale_
})
scaler_params.to_csv("/tmp/scaler_params.csv", index=False)
print("Scaler params saved → /tmp/scaler_params.csv")

# Save scaler back to models directory
import pickle
with open("models/phase2_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved → models/phase2_scaler.pkl")

spark_feat = spark.createDataFrame(feat_df)
spark_feat.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("acme_pm.ai4i_features")

print(f"\n✅  Saved → acme_pm.ai4i_features")
print(f"    Rows    : {spark_feat.count():,}")
print(f"    Features: {len(feature_cols)}")

# ── 4. Top features by correlation ───────────────────────────────────────────
corr = feat_df[feature_cols + ["machine_failure"]].corr()["machine_failure"] \
           .drop("machine_failure").abs().sort_values(ascending=False)
print("\n=== Top 10 features by |correlation| with failure ===")
print(corr.head(10).to_string())

