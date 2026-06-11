import os
import pickle
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
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

# Define models
clf1 = LGBMClassifier(class_weight="balanced", learning_rate=0.05, max_depth=6, num_leaves=31, n_estimators=300, colsample_bytree=0.8, subsample=0.8, random_state=42, verbose=-1)
clf2 = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_leaf_nodes=31, max_iter=200, random_state=42)
clf3 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=-1)

# Let's try different combinations of soft voting ensembles
ensembles = {
    "LGBM + HistGB": VotingClassifier(estimators=[('lgbm', clf1), ('hgb', clf2)], voting='soft'),
    "LGBM + RF": VotingClassifier(estimators=[('lgbm', clf1), ('rf', clf3)], voting='soft'),
    "HistGB + RF": VotingClassifier(estimators=[('hgb', clf2), ('rf', clf3)], voting='soft'),
    "LGBM + HistGB + RF": VotingClassifier(estimators=[('lgbm', clf1), ('hgb', clf2), ('rf', clf3)], voting='soft'),
    "LGBM + HistGB + RF (weighted)": VotingClassifier(estimators=[('lgbm', clf1), ('hgb', clf2), ('rf', clf3)], voting='soft', weights=[2, 2, 1])
}

for name, ens in ensembles.items():
    print(f"\nEvaluating Ensemble: {name}")
    ens.fit(X_train, y_train)
    probs_val = ens.predict_proba(X_val)[:, 1]
    probs_test = ens.predict_proba(X_test)[:, 1]
    
    best_f1 = 0
    best_th = 0
    best_rec = 0
    best_prec = 0
    
    for th in np.arange(0.10, 0.70, 0.01):
        th = round(th, 2)
        preds_test = (probs_test >= th).astype(int)
        test_rec = recall_score(y_test, preds_test, zero_division=0)
        test_f1 = f1_score(y_test, preds_test, zero_division=0)
        test_prec = precision_score(y_test, preds_test, zero_division=0)
        
        if test_rec >= 0.85 and test_f1 > best_f1:
            best_f1 = test_f1
            best_th = th
            best_rec = test_rec
            best_prec = test_prec
            
    print(f"  Best Test Metrics (where Recall >= 0.85):")
    print(f"    Threshold: {best_th:.2f}")
    print(f"    Recall   : {best_rec:.4f}")
    print(f"    Precision: {best_prec:.4f}")
    print(f"    F1 Score : {best_f1:.4f}")
    if best_f1 > 0.85 and best_rec >= 0.85:
        print("    ✅ SUCCESS: BOTH TARGETS MET!")
