import sys
sys.path.insert(0, "setup")

from spark_session import get_spark
import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (f1_score, precision_score, recall_score,
                              classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

spark = get_spark("acme_pm")
os.makedirs("models", exist_ok=True)
os.makedirs("eda_plots", exist_ok=True)

# ── 1. Load preprocessed splits ───────────────────────────────────────────────
X_train = np.load("models/X_train_sm.npy")
y_train = np.load("models/y_train_sm.npy")
X_val   = np.load("models/X_val_sc.npy")
y_val   = np.load("models/y_val.npy")
X_test  = np.load("models/X_test_sc.npy")
y_test  = np.load("models/y_test.npy")

with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

print(f"Train (SMOTE) : {X_train.shape}  failures: {int(y_train.sum())}")
print(f"Val           : {X_val.shape}    failures: {int(y_val.sum())}")
print(f"Test          : {X_test.shape}   failures: {int(y_test.sum())}")

# ── 2. MLflow setup ───────────────────────────────────────────────────────────
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/acme_pm_failure_prediction")

# ── Helper ────────────────────────────────────────────────────────────────────
def tune_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  Best threshold: {best_t:.2f}  (val F1: {best_f1:.4f})")
    return best_t

def evaluate_model(model, X, y_true, threshold, model_name, split="test"):
    probs  = model.predict_proba(X)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    auc    = roc_auc_score(y_true, probs)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    print(f"\n── {model_name} ({split}) ──")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Normal","Failure"],
                                 zero_division=0))
    return {"auc": auc, "f1": f1, "precision": prec,
            "recall": rec, "probs": probs, "preds": y_pred}

# ── 3. XGBoost ────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Training XGBoost...")
try:
    from xgboost import XGBClassifier
    with mlflow.start_run(run_name="xgboost"):
        params_xgb = {
            "n_estimators"     : 300,
            "max_depth"        : 6,
            "learning_rate"    : 0.05,
            "subsample"        : 0.8,
            "colsample_bytree" : 0.8,
            "scale_pos_weight" : 10,
            "eval_metric"      : "logloss",
            "random_state"     : 42,
        }
        mlflow.log_params(params_xgb)

        xgb = XGBClassifier(**params_xgb)
        xgb.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50)

        best_t_xgb = tune_threshold(xgb, X_val, y_val)
        m_val_xgb  = evaluate_model(xgb, X_val,  y_val,  best_t_xgb, "XGBoost", "val")
        m_test_xgb = evaluate_model(xgb, X_test, y_test, best_t_xgb, "XGBoost", "test")

        mlflow.log_metrics({
            "val_auc"       : m_val_xgb["auc"],
            "val_f1"        : m_val_xgb["f1"],
            "test_auc"      : m_test_xgb["auc"],
            "test_f1"       : m_test_xgb["f1"],
            "test_precision": m_test_xgb["precision"],
            "test_recall"   : m_test_xgb["recall"],
            "best_threshold": best_t_xgb,
        })
        mlflow.sklearn.log_model(xgb, "xgboost_model")

        with open("models/xgboost.pkl", "wb") as f:
            pickle.dump({"model": xgb, "threshold": best_t_xgb}, f)
        print("Saved → models/xgboost.pkl")

        # Feature importance
        importance = pd.Series(
            xgb.feature_importances_, index=feature_names
        ).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(importance.index, importance.values, color="#4C9BE8")
        ax.set_title("XGBoost — top 15 feature importances")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig("eda_plots/07_xgb_feature_importance.png", dpi=150)
        mlflow.log_artifact("eda_plots/07_xgb_feature_importance.png")
        plt.close()
        print("Saved → eda_plots/07_xgb_feature_importance.png")

except ImportError:
    print("[ERROR] xgboost not installed. Run: pip install xgboost")
    m_test_xgb = None

# ── 4. LightGBM ───────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Training LightGBM...")
try:
    from lightgbm import LGBMClassifier
    with mlflow.start_run(run_name="lightgbm"):
        params_lgbm = {
            "n_estimators"    : 300,
            "max_depth"       : 6,
            "learning_rate"   : 0.05,
            "subsample"       : 0.8,
            "colsample_bytree": 0.8,
            "class_weight"    : "balanced",
            "random_state"    : 42,
            "verbose"         : -1,
        }
        mlflow.log_params(params_lgbm)

        lgbm = LGBMClassifier(**params_lgbm)
        lgbm.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)])

        best_t_lgbm = tune_threshold(lgbm, X_val, y_val)
        m_val_lgbm  = evaluate_model(lgbm, X_val,  y_val,  best_t_lgbm, "LightGBM", "val")
        m_test_lgbm = evaluate_model(lgbm, X_test, y_test, best_t_lgbm, "LightGBM", "test")

        mlflow.log_metrics({
            "val_auc"       : m_val_lgbm["auc"],
            "val_f1"        : m_val_lgbm["f1"],
            "test_auc"      : m_test_lgbm["auc"],
            "test_f1"       : m_test_lgbm["f1"],
            "test_precision": m_test_lgbm["precision"],
            "test_recall"   : m_test_lgbm["recall"],
            "best_threshold": best_t_lgbm,
        })
        mlflow.sklearn.log_model(lgbm, "lightgbm_model")

        with open("models/lightgbm.pkl", "wb") as f:
            pickle.dump({"model": lgbm, "threshold": best_t_lgbm}, f)
        print("Saved → models/lightgbm.pkl")

except ImportError:
    print("[ERROR] lightgbm not installed. Run: pip install lightgbm")
    m_test_lgbm = None

# ── 5. Comparison plots ───────────────────────────────────────────────────────
if m_test_xgb and m_test_lgbm:
    print("\nGenerating comparison plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC curves
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, m_test_xgb["probs"])
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, m_test_lgbm["probs"])
    axes[0].plot(fpr_xgb, tpr_xgb, color="#4C9BE8",
                 label=f"XGBoost  AUC={m_test_xgb['auc']:.3f}")
    axes[0].plot(fpr_lgb, tpr_lgb, color="#E85D30",
                 label=f"LightGBM AUC={m_test_lgbm['auc']:.3f}")
    axes[0].plot([0,1],[0,1],"--", color="gray")
    axes[0].set_title("ROC curve comparison")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()

    # Confusion matrix
    cm = confusion_matrix(y_test, m_test_xgb["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Normal","Failure"],
                yticklabels=["Normal","Failure"])
    axes[1].set_title("XGBoost — confusion matrix")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    # Metrics bar chart
    metrics_names = ["AUC", "F1", "Precision", "Recall"]
    xgb_vals  = [m_test_xgb["auc"],  m_test_xgb["f1"],
                 m_test_xgb["precision"],  m_test_xgb["recall"]]
    lgbm_vals = [m_test_lgbm["auc"], m_test_lgbm["f1"],
                 m_test_lgbm["precision"], m_test_lgbm["recall"]]
    x = np.arange(len(metrics_names))
    axes[2].bar(x - 0.2, xgb_vals,  0.35, label="XGBoost",  color="#4C9BE8")
    axes[2].bar(x + 0.2, lgbm_vals, 0.35, label="LightGBM", color="#E85D30")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics_names)
    axes[2].set_title("Model comparison — test metrics")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    for i, (v1, v2) in enumerate(zip(xgb_vals, lgbm_vals)):
        axes[2].text(i-0.2, v1+0.01, f"{v1:.2f}", ha="center", fontsize=9)
        axes[2].text(i+0.2, v2+0.01, f"{v2:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("eda_plots/08_model_comparison.png", dpi=150)
    print("Saved → eda_plots/08_model_comparison.png")
    plt.close()

    # ── 6. Best model ─────────────────────────────────────────────────────────
    xgb_f1  = m_test_xgb["f1"]
    lgbm_f1 = m_test_lgbm["f1"]
    best_name  = "XGBoost"  if xgb_f1 >= lgbm_f1 else "LightGBM"
    best_model = xgb        if xgb_f1 >= lgbm_f1 else lgbm
    best_t     = best_t_xgb if xgb_f1 >= lgbm_f1 else best_t_lgbm

    with open("models/best_failure_predictor.pkl", "wb") as f:
        pickle.dump({"model": best_model, "threshold": best_t,
                     "name": best_name, "features": feature_names}, f)

    print(f"""
╔══════════════════════════════════════════════════════╗
║       Failure Prediction — complete                  ║
╠══════════════════════════════════════════════════════╣
║  XGBoost  F1 : {xgb_f1:.4f}                              ║
║  LightGBM F1 : {lgbm_f1:.4f}                              ║
║  Best model  : {best_name:<10}                       ║
║  Saved       : models/best_failure_predictor.pkl     ║
║  MLflow exp  : /acme_pm_failure_prediction           ║
╚══════════════════════════════════════════════════════╝
""")