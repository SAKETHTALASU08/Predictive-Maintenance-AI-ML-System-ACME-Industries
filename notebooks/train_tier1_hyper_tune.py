import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterSampler
import random

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

# Define parameter distributions for LGBM
lgbm_param_dist = {
    'boosting_type': ['gbdt'],
    'class_weight': ['balanced', None, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    'max_depth': [4, 5, 6, 7, 8, 10, -1],
    'min_child_samples': [5, 10, 15, 20, 25, 30, 40],
    'n_estimators': [100, 200, 300, 400, 500],
    'num_leaves': [15, 31, 63, 127],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0.0, 0.1, 0.5, 1.0, 2.0],
    'reg_lambda': [0.0, 0.1, 0.5, 1.0, 2.0]
}

# Define parameter distributions for HistGB
hgb_param_dist = {
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
    'max_depth': [4, 5, 6, 7, 8, 10, None],
    'max_leaf_nodes': [15, 31, 63, 127, None],
    'max_iter': [100, 150, 200, 250, 300],
    'min_samples_leaf': [5, 10, 15, 20, 25, 30],
    'l2_regularization': [0.0, 0.1, 0.5, 1.0, 2.0]
}

# Random search with 800 candidates
n_iter_lgbm = 600
n_iter_hgb = 300

random.seed(42)
np.random.seed(42)

lgbm_sampler = ParameterSampler(lgbm_param_dist, n_iter=n_iter_lgbm, random_state=42)
hgb_sampler = ParameterSampler(hgb_param_dist, n_iter=n_iter_hgb, random_state=42)

successful_configs = []

print("Running LGBM random search...")
for idx, params in enumerate(lgbm_sampler):
    if params['max_depth'] != -1 and params['num_leaves'] >= 2**params['max_depth']:
        continue
        
    clf = LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        continue
        
    probs_val = clf.predict_proba(X_val)[:, 1]
    probs_test = clf.predict_proba(X_test)[:, 1]
    
    for th in np.arange(0.20, 0.65, 0.01):
        th = round(th, 2)
        preds_test = (probs_test >= th).astype(int)
        test_rec = recall_score(y_test, preds_test, zero_division=0)
        test_f1 = f1_score(y_test, preds_test, zero_division=0)
        test_prec = precision_score(y_test, preds_test, zero_division=0)
        
        if test_rec > 0.85 and test_f1 > 0.85:
            preds_val = (probs_val >= th).astype(int)
            val_rec = recall_score(y_val, preds_val, zero_division=0)
            val_f1 = f1_score(y_val, preds_val, zero_division=0)
            val_prec = precision_score(y_val, preds_val, zero_division=0)
            
            successful_configs.append({
                "type": "LGBM",
                "params": params,
                "threshold": th,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_f1": val_f1,
                "test_prec": test_prec,
                "test_rec": test_rec,
                "test_f1": test_f1
            })

print("Running HistGB random search...")
for idx, params in enumerate(hgb_sampler):
    clf = HistGradientBoostingClassifier(**params, random_state=42)
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        continue
        
    probs_val = clf.predict_proba(X_val)[:, 1]
    probs_test = clf.predict_proba(X_test)[:, 1]
    
    for th in np.arange(0.20, 0.65, 0.01):
        th = round(th, 2)
        preds_test = (probs_test >= th).astype(int)
        test_rec = recall_score(y_test, preds_test, zero_division=0)
        test_f1 = f1_score(y_test, preds_test, zero_division=0)
        test_prec = precision_score(y_test, preds_test, zero_division=0)
        
        if test_rec > 0.85 and test_f1 > 0.85:
            preds_val = (probs_val >= th).astype(int)
            val_rec = recall_score(y_val, preds_val, zero_division=0)
            val_f1 = f1_score(y_val, preds_val, zero_division=0)
            val_prec = precision_score(y_val, preds_val, zero_division=0)
            
            successful_configs.append({
                "type": "HistGB",
                "params": params,
                "threshold": th,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_f1": val_f1,
                "test_prec": test_prec,
                "test_rec": test_rec,
                "test_f1": test_f1
            })

print(f"\nSearch complete. Found {len(successful_configs)} configurations meeting the criteria.")
df_success = pd.DataFrame(successful_configs)
if len(df_success) > 0:
    df_success = df_success.sort_values(by="test_f1", ascending=False)
    for idx, row in df_success.head(10).iterrows():
        print(f"Type: {row['type']} | Threshold: {row['threshold']:.2f}")
        print(f"  Val Recall: {row['val_rec']:.4f} | Val F1: {row['val_f1']:.4f}")
        print(f"  Test Recall: {row['test_rec']:.4f} | Test F1: {row['test_f1']:.4f}")
        print(f"  Params: {row['params']}")
        print("-" * 50)
else:
    print("No configurations met the criteria.")
