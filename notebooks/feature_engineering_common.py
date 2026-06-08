"""
ACME Industries — Unified Feature Engineering Layer
===================================================
Single source of truth for engineering features from raw sensor data.
Used identically during model training, evaluation, and inference.
"""

import pandas as pd
import numpy as np

# ─── TRAINING DATASET STATISTICS (AI4I 2020) ─────────────────────────────────
# Computed directly from the original training split.
STATS = {
    "torque_nm": {
        "mean": 39.986910,
        "std": 9.968934,
        "rms": 41.210711,
        "min": 3.8,
        "max": 76.6
    },
    "power_w": {
        "mean": 6279.744953,
        "std": 1067.418295,
        "rms": 6369.808832,
        "min": 1148.440610,
        "max": 10469.923005
    },
    "rotational_speed_rpm": {
        "mean": 1538.776100,
        "std": 179.284096,
        "rms": 1549.184127,
        "min": 1168.0,
        "max": 2886.0
    },
    "temp_delta_k": {
        "mean": 10.000630,
        "std": 1.001094,
        "rms": 10.050606,
        "min": 7.6,
        "max": 12.1
    },
    "air_temp_k": {
        "min": 295.3,
        "max": 304.5
    },
    "process_temp_k": {
        "min": 305.7,
        "max": 313.8
    },
    "tool_wear_min": {
        "min": 0.0,
        "max": 253.0
    }
}

# Ordered features expected by the trained ML models (LightGBM & Isolation Forest)
FEATURE_ORDER = [
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
    "tool_wear_ratio"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a pandas DataFrame containing raw or pre-derived sensor columns.
    Computes all 20 engineered features and returns them in the exact order
    the ML model expects.
    
    Expected raw columns:
    - torque_nm
    - rotational_speed_rpm
    - tool_wear_pct (0-100) or tool_wear_min
    - power_w
    - temperature_k or process_temp_k
    """
    df = df.copy()

    # 1. Normalize/clean input raw column names & scale factors
    if "tool_wear_min" in df.columns:
        df["tool_wear_pct"] = df["tool_wear_min"] / 2.5
    elif "tool_wear_pct" in df.columns:
        df["tool_wear_min"] = df["tool_wear_pct"] * 2.5
    else:
        df["tool_wear_min"] = 0.0
        df["tool_wear_pct"] = 0.0


    if "temperature_k" in df.columns:
        # If single temperature is given, we treat it as process temp
        # and assume air temp is 10 K lower (delta = 10 K)
        df["process_temp_k"] = df["temperature_k"]
        df["air_temp_k"] = df["temperature_k"] - 10.0
        df["temp_delta_k"] = 10.0
    else:
        # Fallbacks/training dataset mappings
        if "process_temp_k" not in df.columns:
            df["process_temp_k"] = 310.0
        if "air_temp_k" not in df.columns:
            df["air_temp_k"] = 300.0
        if "temp_delta_k" not in df.columns:
            df["temp_delta_k"] = df["process_temp_k"] - df["air_temp_k"]

    # 2. Time-domain & Frequency-domain deviations
    df["torque_nm_freq_dev"] = ((df["torque_nm"] - STATS["torque_nm"]["mean"]) / STATS["torque_nm"]["std"]).abs()
    df["power_w_freq_dev"] = ((df["power_w"] - STATS["power_w"]["mean"]) / STATS["power_w"]["std"]).abs()
    df["temp_delta_k_freq_dev"] = ((df["temp_delta_k"] - STATS["temp_delta_k"]["mean"]) / STATS["temp_delta_k"]["std"]).abs()
    df["rotational_speed_rpm_freq_dev"] = ((df["rotational_speed_rpm"] - STATS["rotational_speed_rpm"]["mean"]) / STATS["rotational_speed_rpm"]["std"]).abs()

    df["torque_nm_rms_dev"] = (df["torque_nm"] - STATS["torque_nm"]["rms"]).abs()
    df["power_w_rms_dev"] = (df["power_w"] - STATS["power_w"]["rms"]).abs()
    df["temp_delta_k_rms_dev"] = (df["temp_delta_k"] - STATS["temp_delta_k"]["rms"]).abs()
    df["rotational_speed_rpm_rms_dev"] = (df["rotational_speed_rpm"] - STATS["rotational_speed_rpm"]["rms"]).abs()

    # 3. Composite & Interaction features
    df["tool_wear_ratio"] = df["tool_wear_min"] / 250.0
    df["mech_stress"] = df["torque_nm"] * df["tool_wear_min"]
    df["power_speed_ratio"] = df["power_w"] / (df["rotational_speed_rpm"] + 1e-5)
    
    # Compute wear_torque_interaction using pct if tool_wear_pct is present, else scale from min
    if "tool_wear_pct" in df.columns:
        df["wear_torque_interact"] = df["tool_wear_pct"] * df["torque_nm"]
    else:
        df["wear_torque_interact"] = (df["tool_wear_min"] / 2.5) * df["torque_nm"]

    # 4. Statistical features over MinMaxScaler normalized base sensors
    sensor_cols = [
        "air_temp_k", "process_temp_k", "rotational_speed_rpm",
        "torque_nm", "tool_wear_min", "power_w", "temp_delta_k"
    ]
    
    normalized_vals = {}
    for col in sensor_cols:
        col_min = STATS[col]["min"]
        col_max = STATS[col]["max"]
        normalized_vals[col] = np.clip((df[col] - col_min) / (col_max - col_min), 0.0, 1.0)
        
    sensor_norm = pd.DataFrame(normalized_vals, index=df.index)
    
    df["feat_stat_mean"] = sensor_norm.mean(axis=1)
    # df.std(axis=1) uses ddof=1 sample standard deviation
    df["feat_stat_std"] = sensor_norm.std(axis=1).fillna(0.0)
    df["feat_stat_range"] = sensor_norm.max(axis=1) - sensor_norm.min(axis=1)

    # 5. Rolling features
    # Support rolling max/mean if data has historical context, otherwise default to current value
    if len(df) > 1:
        df["power_w_roll10_max"] = df["power_w"].rolling(10, min_periods=1).max()
        df["temp_delta_k_roll10_mean"] = df["temp_delta_k"].rolling(10, min_periods=1).mean()
        df["temp_delta_k_roll10_max"] = df["temp_delta_k"].rolling(10, min_periods=1).max()
        df["torque_nm_roll10_max"] = df["torque_nm"].rolling(10, min_periods=1).max()
    else:
        df["power_w_roll10_max"] = df["power_w"]
        df["temp_delta_k_roll10_mean"] = df["temp_delta_k"]
        df["temp_delta_k_roll10_max"] = df["temp_delta_k"]
        df["torque_nm_roll10_max"] = df["torque_nm"]

    # Return only the requested features in the exact model registry order
    return df[FEATURE_ORDER]
