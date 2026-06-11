import os
import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "best_failure_predictor.pkl")
X_test = np.load(os.path.join(base_dir, "models", "X_test_sc.npy"))
y_test = np.load(os.path.join(base_dir, "models", "y_test.npy"))

with open(model_path, "rb") as f:
    clf_bundle = pickle.load(f)

print("Loaded Model Bundle:")
print("  Name:", clf_bundle.get("name"))
print("  Current Threshold:", clf_bundle.get("threshold"))

model = clf_bundle["model"]
probs = model.predict_proba(X_test)[:, 1]

print("\nThreshold Sweep on Test Split:")
print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
print("-" * 50)
for th in np.arange(0.10, 0.60, 0.05):
    th = round(th, 2)
    preds = (probs >= th).astype(int)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"{th:<10.2f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")
