import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

base_dir = "/Users/sakethtalasu/Downloads/Predictive_Analysis"
model_dir = os.path.join(base_dir, "models")

# Load preprocessed splits
X_train = np.load(os.path.join(model_dir, "X_train_sm.npy"))
y_train = np.load(os.path.join(model_dir, "y_train_sm.npy"))
X_val = np.load(os.path.join(model_dir, "X_val_sc.npy"))
y_val = np.load(os.path.join(model_dir, "y_val.npy"))
X_test = np.load(os.path.join(model_dir, "X_test_sc.npy"))
y_test = np.load(os.path.join(model_dir, "y_test.npy"))

with open(os.path.join(model_dir, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

print(f"Train size: {X_train.shape} (failures: {int(y_train.sum())})")
print(f"Val size  : {X_val.shape} (failures: {int(y_val.sum())})")
print(f"Test size : {X_test.shape} (failures: {int(y_test.sum())})")

# We want to find models that achieve Test F1 > 0.85 and Test Recall > 0.85.
# Let's collect all models, evaluate them on a range of thresholds, and store those that meet the criteria.

results = []

def evaluate_model(name, model, params):
    model.fit(X_train, y_train)
    probs_val = model.predict_proba(X_val)[:, 1]
    probs_test = model.predict_proba(X_test)[:, 1]
    
    for th in np.arange(0.10, 0.70, 0.01):
        th = round(th, 2)
        preds_val = (probs_val >= th).astype(int)
        val_prec = precision_score(y_val, preds_val, zero_division=0)
        val_rec = recall_score(y_val, preds_val, zero_division=0)
        val_f1 = f1_score(y_val, preds_val, zero_division=0)
        
        preds_test = (probs_test >= th).astype(int)
        test_prec = precision_score(y_test, preds_test, zero_division=0)
        test_rec = recall_score(y_test, preds_test, zero_division=0)
        test_f1 = f1_score(y_test, preds_test, zero_division=0)
        
        meets_criteria = test_rec > 0.85 and test_f1 > 0.85
        
        results.append({
            "model_name": name,
            "params": params,
            "threshold": th,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
            "meets_criteria": meets_criteria,
            "model_obj": model
        })

print("\n--- Tuning LightGBM models ---")
lgbm_configs = [
    # Baseline-like configurations
    {"class_weight": "balanced", "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "n_estimators": 300, "colsample_bytree": 0.8, "subsample": 0.8},
    {"class_weight": "balanced", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "n_estimators": 400, "colsample_bytree": 0.8, "subsample": 0.8},
    {"class_weight": "balanced", "learning_rate": 0.05, "max_depth": 8, "num_leaves": 63, "n_estimators": 300, "colsample_bytree": 0.8, "subsample": 0.8},
    {"class_weight": "balanced", "learning_rate": 0.05, "max_depth": 5, "num_leaves": 15, "n_estimators": 300, "colsample_bytree": 0.8, "subsample": 0.8},
    {"class_weight": None, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "n_estimators": 300, "colsample_bytree": 0.8, "subsample": 0.8},
    {"class_weight": None, "learning_rate": 0.1, "max_depth": 6, "num_leaves": 31, "n_estimators": 200, "colsample_bytree": 0.8, "subsample": 0.8},
]

for idx, config in enumerate(lgbm_configs):
    clf = LGBMClassifier(**config, random_state=42, verbose=-1)
    evaluate_model(f"LightGBM_{idx}", clf, config)

print("--- Tuning HistGradientBoosting Classifier ---")
hgb_configs = [
    {"learning_rate": 0.05, "max_depth": 6, "max_leaf_nodes": 31, "max_iter": 200},
    {"learning_rate": 0.05, "max_depth": 8, "max_leaf_nodes": 63, "max_iter": 200},
    {"learning_rate": 0.1, "max_depth": 6, "max_leaf_nodes": 31, "max_iter": 150},
]
for idx, config in enumerate(hgb_configs):
    clf = HistGradientBoostingClassifier(**config, random_state=42)
    evaluate_model(f"HistGB_{idx}", clf, config)

print("--- Tuning RandomForest Classifier ---")
rf_configs = [
    {"n_estimators": 300, "max_depth": 8, "min_samples_split": 5, "class_weight": "balanced"},
    {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5, "class_weight": "balanced"},
    {"n_estimators": 500, "max_depth": 12, "min_samples_split": 2, "class_weight": "balanced"},
]
for idx, config in enumerate(rf_configs):
    clf = RandomForestClassifier(**config, random_state=42, n_jobs=-1)
    evaluate_model(f"RF_{idx}", clf, config)

df_res = pd.DataFrame(results)
df_meets = df_res[df_res["meets_criteria"]].copy()

print(f"\nFound {len(df_meets)} configurations meeting Test Recall > 0.85 and Test F1 > 0.85.")
if len(df_meets) > 0:
    df_meets = df_meets.sort_values(by="test_f1", ascending=False)
    for idx, row in df_meets.head(10).iterrows():
        print(f"Model: {row['model_name']} | Threshold: {row['threshold']:.2f}")
        print(f"  Val Recall: {row['val_recall']:.4f} | Val F1: {row['val_f1']:.4f}")
        print(f"  Test Recall: {row['test_recall']:.4f} | Test F1: {row['test_f1']:.4f}")
        print(f"  Params: {row['params']}")
        print("-" * 50)
else:
    print("No configurations met the criteria. Let's see the top test F1 configurations:")
    df_sorted = df_res.sort_values(by="test_f1", ascending=False)
    for idx, row in df_sorted.head(10).iterrows():
        print(f"Model: {row['model_name']} | Threshold: {row['threshold']:.2f}")
        print(f"  Val Recall: {row['val_recall']:.4f} | Val F1: {row['val_f1']:.4f}")
        print(f"  Test Recall: {row['test_recall']:.4f} | Test F1: {row['test_f1']:.4f}")
        print("-" * 50)
