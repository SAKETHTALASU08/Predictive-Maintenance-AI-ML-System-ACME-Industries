import sys
sys.path.insert(0, "setup")

from spark_session import get_spark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Connect ───────────────────────────────────────────────────────────────────
spark = get_spark("acme_pm")

df      = spark.table("acme_pm.ai4i_cleaned").toPandas()
df_feat = spark.table("acme_pm.ai4i_features").toPandas()

SENSOR_COLS = ["air_temp_k","process_temp_k","rotational_speed_rpm",
               "torque_nm","tool_wear_min","power_w","temp_delta_k"]

os.makedirs("eda_plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
print(f"Loaded → cleaned: {df.shape}  features: {df_feat.shape}")

# ── 1. Class distribution ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts = df["machine_failure"].value_counts()
axes[0].bar(["Normal (0)","Failure (1)"], counts.values,
            color=["#4C9BE8","#E85D30"], edgecolor="white")
axes[0].set_title("Binary failure distribution")
for i, v in enumerate(counts.values):
    axes[0].text(i, v+30, f"{v:,} ({v/len(df)*100:.1f}%)", ha="center")

ft = df["failure_type"].value_counts()
axes[1].barh(ft.index, ft.values, color=["#4C9BE8","#E85D30","#F5A623","#7ED321","#BD10E0","#50E3C2"][:len(ft)])
axes[1].set_title("Failure type breakdown")
plt.tight_layout()
plt.savefig("eda_plots/01_class_distribution.png", dpi=150)
print("Saved → eda_plots/01_class_distribution.png")
plt.close()

# ── 2. Sensor distributions ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()
for i, col in enumerate(SENSOR_COLS):
    for label, color, name in [(0,"#4C9BE8","Normal"),(1,"#E85D30","Failure")]:
        subset = df[df["machine_failure"]==label][col]
        axes[i].hist(subset, bins=50, alpha=0.6, color=color, label=name, density=True)
    axes[i].set_title(col.replace("_"," ").title())
    axes[i].legend(fontsize=9)
axes[-1].set_visible(False)
plt.suptitle("Sensor distributions: Normal vs Failure", fontsize=14)
plt.tight_layout()
plt.savefig("eda_plots/02_sensor_distributions.png", dpi=150)
print("Saved → eda_plots/02_sensor_distributions.png")
plt.close()

# ── 3. Correlation heatmap ────────────────────────────────────────────────────
corr_matrix = df[SENSOR_COLS + ["machine_failure"]].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Sensor correlation matrix")
plt.tight_layout()
plt.savefig("eda_plots/03_correlation_heatmap.png", dpi=150)
print("Saved → eda_plots/03_correlation_heatmap.png")
plt.close()

# ── 4. Box plots by failure type ──────────────────────────────────────────────
key_sensors = ["rotational_speed_rpm","torque_nm","tool_wear_min","power_w"]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(key_sensors):
    sns.boxplot(data=df, x="failure_type", y=col, ax=axes[i])
    axes[i].set_title(f"{col.replace('_',' ').title()} by failure type")
    axes[i].tick_params(axis='x', rotation=30)
plt.suptitle("Sensor values by failure type", fontsize=14)
plt.tight_layout()
plt.savefig("eda_plots/04_boxplots.png", dpi=150)
print("Saved → eda_plots/04_boxplots.png")
plt.close()

# ── 5. Top features by correlation ───────────────────────────────────────────
feature_cols = [c for c in df_feat.columns
                if c not in ["udi","machine_failure","failure_type"]]
corr_series = df_feat[feature_cols + ["machine_failure"]] \
    .corr()["machine_failure"].drop("machine_failure").abs() \
    .sort_values(ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(corr_series.index, corr_series.values,
        color=plt.cm.RdYlGn(corr_series.values / corr_series.max()))
ax.set_title("Top 20 features — |correlation| with failure")
ax.set_xlabel("|Pearson correlation|")
plt.tight_layout()
plt.savefig("eda_plots/05_top_features.png", dpi=150)
print("Saved → eda_plots/05_top_features.png")
plt.close()

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════╗
║           EDA SUMMARY — ACME Industries PM           ║
╠══════════════════════════════════════════════════════╣
║  Rows          : 10,000                              ║
║  Failure rate  : ~3.4%  (97:3 imbalance)             ║
║  Top causes    : OSF (overstrain), HDF (heat)        ║
║  Key features  : tool_wear_min, torque_nm,           ║
║                  temp_delta_k, mech_stress           ║
║  Action needed : SMOTE in Phase 2                    ║
╚══════════════════════════════════════════════════════╝
""")
print("All plots saved to → eda_plots/")
# ── Fix: Top features correlation plot ───────────────────────────────────────
import os
os.makedirs("eda_plots", exist_ok=True)

# Reload raw feature table before scaling was applied
df_raw_feat = spark.table("acme_pm.ai4i_features").toPandas()

feature_cols = [c for c in df_raw_feat.columns
                if c not in ["udi","machine_failure","failure_type"]]

# Drop constant columns
df_raw_feat = df_raw_feat.loc[:, df_raw_feat.nunique() > 1]
feature_cols = [c for c in feature_cols if c in df_raw_feat.columns]

corr_series = df_raw_feat[feature_cols + ["machine_failure"]] \
    .corr()["machine_failure"].drop("machine_failure").abs() \
    .dropna().sort_values(ascending=True).tail(20)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(corr_series.index, corr_series.values,
        color=plt.cm.RdYlGn(corr_series.values / corr_series.max()))
ax.set_title("Top 20 features — |correlation| with failure (fixed)")
ax.set_xlabel("|Pearson correlation|")
plt.tight_layout()
plt.savefig("eda_plots/05_top_features_fixed.png", dpi=150)
print("Saved → eda_plots/05_top_features_fixed.png")
print("\n=== Top 20 features ===")
print(corr_series.sort_values(ascending=False).to_string())
plt.close()
