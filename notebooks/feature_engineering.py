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

SENSOR_COLS = [
    "air_temp_k", "process_temp_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min", "power_w", "temp_delta_k",
]

# ── 2. Statistical features ───────────────────────────────────────────────────
scaler_mm  = MinMaxScaler()
sensor_norm = pd.DataFrame(
    scaler_mm.fit_transform(df[SENSOR_COLS]),
    columns=[f"{c}_norm" for c in SENSOR_COLS]
)
df["feat_stat_mean"]  = sensor_norm.mean(axis=1)
df["feat_stat_std"]   = sensor_norm.std(axis=1)
df["feat_stat_skew"]  = sensor_norm.apply(lambda r: stats.skew(r),     axis=1)
df["feat_stat_kurt"]  = sensor_norm.apply(lambda r: stats.kurtosis(r), axis=1)
df["feat_stat_range"] = sensor_norm.max(axis=1) - sensor_norm.min(axis=1)
print("Statistical features added.")

# ── 3. Time-domain features ───────────────────────────────────────────────────
def rms(s):          return np.sqrt(np.mean(s ** 2))
def peak_to_peak(s): return s.max() - s.min()
def crest_factor(s): r = rms(s); return s.abs().max() / r if r != 0 else 0
def shape_factor(s): r = rms(s); m = s.abs().mean(); return r / m if m != 0 else 0

for col in SENSOR_COLS:
    s = df[col]
    df[f"{col}_rms_dev"]      = (s - rms(s)).abs()
    df[f"{col}_p2p"]          = peak_to_peak(s)
    df[f"{col}_crest"]        = crest_factor(s)
    df[f"{col}_shape_factor"] = shape_factor(s)
print("Time-domain features added.")

# ── 4. Frequency-domain features ─────────────────────────────────────────────
def spectral_features(signal, fs=1.0):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    psd_norm   = psd / (psd.sum() + 1e-10)
    spec_ent   = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    dom_freq   = freqs[np.argmax(psd)]
    return dom_freq, spec_ent

for col in SENSOR_COLS:
    signal = df[col].values.astype(float)
    dom_f, s_ent = spectral_features(signal)
    df[f"{col}_dom_freq"]     = dom_f
    df[f"{col}_spec_entropy"] = s_ent
    col_mean = df[col].mean()
    col_std  = df[col].std() + 1e-10
    df[f"{col}_freq_dev"] = ((df[col] - col_mean) / col_std).abs()
print("Frequency-domain features added.")

# ── 5. Rolling window features ────────────────────────────────────────────────
df = df.sort_values("udi").reset_index(drop=True)
WINDOWS   = [10, 50, 100]
ROLL_COLS = ["rotational_speed_rpm","torque_nm","tool_wear_min","power_w","temp_delta_k"]

for col in ROLL_COLS:
    for w in WINDOWS:
        df[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_roll{w}_std"]  = df[col].rolling(w, min_periods=1).std().fillna(0)
        df[f"{col}_roll{w}_max"]  = df[col].rolling(w, min_periods=1).max()
print(f"Rolling features added for windows {WINDOWS}.")

# ── 6. Composite features ─────────────────────────────────────────────────────
df["tool_wear_ratio"]      = df["tool_wear_min"] / 250
df["thermal_stress"]       = df["temp_delta_k"] * df["rotational_speed_rpm"] / 1000
df["mech_stress"]          = df["torque_nm"] * df["tool_wear_min"]
df["power_speed_ratio"]    = df["power_w"] / (df["rotational_speed_rpm"] + 1e-5)
df["wear_torque_interact"] = df["tool_wear_min"] * df["torque_nm"]
print("Composite features added.")

# ── 7. Feature list ───────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c.startswith("feat_")
                or any(x in c for x in ["_rms_dev","_p2p","_crest","_spec_",
                                         "_freq_dev","_roll","_shape_factor"])
                or c in ["tool_wear_ratio","thermal_stress","mech_stress",
                         "power_speed_ratio","wear_torque_interact",
                         "temp_delta_k","power_w","product_type_enc"]]
print(f"\nTotal engineered features: {len(feature_cols)}")

# ── 8. Scale + save ───────────────────────────────────────────────────────────
feat_df = df[feature_cols + ["udi","machine_failure","failure_type"]].copy()
scaler  = StandardScaler()
feat_df[feature_cols] = scaler.fit_transform(feat_df[feature_cols])

scaler_params = pd.DataFrame({
    "feature": feature_cols,
    "mean"   : scaler.mean_,
    "scale"  : scaler.scale_
})
scaler_params.to_csv("/tmp/scaler_params.csv", index=False)
print("Scaler params saved → /tmp/scaler_params.csv")

spark_feat = spark.createDataFrame(feat_df)
spark_feat.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("acme_pm.ai4i_features")

print(f"\n✅  Saved → acme_pm.ai4i_features")
print(f"    Rows    : {spark_feat.count():,}")
print(f"    Features: {len(feature_cols)}")

# ── 9. Top features by correlation ───────────────────────────────────────────
corr = feat_df[feature_cols + ["machine_failure"]].corr()["machine_failure"] \
           .drop("machine_failure").abs().sort_values(ascending=False)
print("\n=== Top 10 features by |correlation| with failure ===")
print(corr.head(10).to_string())
