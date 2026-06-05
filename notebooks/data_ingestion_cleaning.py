import sys
sys.path.insert(0, "setup")

from spark_session import get_spark
import pandas as pd
import numpy as np
import subprocess

# ── Connect ───────────────────────────────────────────────────────────────────
spark = get_spark("acme_pm")

# ── 1. Install + download dataset ─────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "ucimlrepo", "-q"])
from ucimlrepo import fetch_ucirepo

print("Downloading AI4I 2020 dataset from UCI...")
ai4i   = fetch_ucirepo(id=601)
X      = ai4i.data.features
y      = ai4i.data.targets
df_raw = pd.concat([X, y], axis=1)
print(f"Raw shape : {df_raw.shape}")
print(f"Columns   : {df_raw.columns.tolist()}")

# ── 2. Rename columns (matches actual UCI column names) ───────────────────────
rename_map = {
    "Type"                    : "product_type",
    "Air temperature"         : "air_temp_k",
    "Process temperature"     : "process_temp_k",
    "Rotational speed"        : "rotational_speed_rpm",
    "Torque"                  : "torque_nm",
    "Tool wear"               : "tool_wear_min",
    "Machine failure"         : "machine_failure",
    "TWF"                     : "tool_wear_failure",
    "HDF"                     : "heat_dissipation_failure",
    "PWF"                     : "power_failure",
    "OSF"                     : "overstrain_failure",
    "RNF"                     : "random_failure",
}

# Only rename columns that exist
rename_map = {k: v for k, v in rename_map.items() if k in df_raw.columns}
df = df_raw.rename(columns=rename_map)
print("\nRenamed columns:", df.columns.tolist())

# ── 3. Derived columns ────────────────────────────────────────────────────────
df["product_type_enc"] = df["product_type"].map({"L": 0, "M": 1, "H": 2})
df["air_temp_c"]       = df["air_temp_k"]     - 273.15
df["process_temp_c"]   = df["process_temp_k"] - 273.15
df["temp_delta_k"]     = df["process_temp_k"] - df["air_temp_k"]
df["power_w"]          = df["torque_nm"] * (df["rotational_speed_rpm"] * 2 * np.pi / 60)

def failure_type(row):
    if   row["tool_wear_failure"]        == 1: return "TWF"
    elif row["heat_dissipation_failure"] == 1: return "HDF"
    elif row["power_failure"]            == 1: return "PWF"
    elif row["overstrain_failure"]       == 1: return "OSF"
    elif row["random_failure"]           == 1: return "RNF"
    elif row["machine_failure"]          == 1: return "UNKNOWN"
    else:                                       return "NONE"

df["failure_type"] = df.apply(failure_type, axis=1)
print("Derived columns added: temp_delta_k, power_w, failure_type")

# ── 4. Null handling + outlier flagging ───────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

def flag_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr    = q3 - q1
    return ((series < q1 - 3*iqr) | (series > q3 + 3*iqr)).astype(int)

for col in ["air_temp_k","process_temp_k","rotational_speed_rpm",
            "torque_nm","tool_wear_min","power_w"]:
    df[f"{col}_outlier"] = flag_outliers(df[col])

print("Outlier flags added.")

# ── 5. Class distribution ─────────────────────────────────────────────────────
print("\n=== Failure distribution ===")
print(df["machine_failure"].value_counts(normalize=True).mul(100).round(2).to_string())
print("\n=== Failure type breakdown ===")
print(df["failure_type"].value_counts().to_string())

# ── 6. Save as Delta table ────────────────────────────────────────────────────
spark_df = spark.createDataFrame(df)
spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("acme_pm.ai4i_cleaned")

print(f"\n✅  Saved → acme_pm.ai4i_cleaned")
print(f"    Rows : {spark_df.count():,}")
print(f"    Cols : {len(spark_df.columns)}")

# ── 7. Validation ─────────────────────────────────────────────────────────────
df_check = spark.table("acme_pm.ai4i_cleaned").toPandas()
print("\n=== Sample (3 rows) ===")
print(df_check.head(3).to_string())
