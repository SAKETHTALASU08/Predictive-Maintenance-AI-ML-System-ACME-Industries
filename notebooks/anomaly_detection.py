
import sys
sys.path.insert(0, "setup")

import os
# --- FIX: Prevent gRPC/Fork deadlocks between PySpark and MLflow on Mac ---
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from spark_session import get_spark
import pandas as pd
import numpy as np
import pickle, warnings
warnings.filterwarnings("ignore")

# ... rest of your imports (IsolationForest, StandardScaler, mlflow, etc.)import sys
sys.path.insert(0, "setup")

from spark_session import get_spark
import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, classification_report, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

spark = get_spark("acme_pm")
os.makedirs("models", exist_ok=True)
os.makedirs("eda_plots", exist_ok=True)

# ── 1. Load & Scale Data ─────────────────────────────────────────────────────
X_train_normal_raw = np.load("models/X_train_normal.npy")
X_val              = np.load("models/X_val_sc.npy")
y_val              = np.load("models/y_val.npy")
X_test             = np.load("models/X_test_sc.npy")
y_test             = np.load("models/y_test.npy")

with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# FIX: X_train_normal was previously unscaled, causing the LSTM loss to explode.
# We must apply the phase 2 scaler to it before training.
try:
    with open("models/phase2_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X_train_normal = scaler.transform(X_train_normal_raw)
    print("✅ Successfully applied phase2_scaler to training data.")
except FileNotFoundError:
    print("⚠️ phase2_scaler.pkl not found! Fitting a fallback StandardScaler.")
    scaler = StandardScaler()
    X_train_normal = scaler.fit_transform(X_train_normal_raw)

print(f"Train (normal only) : {X_train_normal.shape}")
print(f"Val                 : {X_val.shape}")
print(f"Test                : {X_test.shape}")

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/acme_pm_anomaly_detection_improved")

# ─────────────────────────────────────────────────────────────────────────────
# TRACK 1 — Isolation Forest (Production Stable)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRACK 1: Tuned Isolation Forest (Baseline & Tuning)")
print("="*60)

def tune_anomaly_threshold(model, X_val, y_val):
    scores = -model.score_samples(X_val)
    best_t, best_f1 = 0.0, 0.0
    for pct in range(50, 100):
        t     = np.percentile(scores, pct)
        preds = (scores >= t).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  Best threshold : {best_t:.4f}  (val F1: {best_f1:.4f})")
    return best_t

def evaluate_with_threshold(model, X, y_true, threshold, label):
    scores = -model.score_samples(X)
    preds  = (scores >= threshold).astype(int)
    auc    = roc_auc_score(y_true, scores)
    f1     = f1_score(y_true, preds, zero_division=0)
    prec   = precision_score(y_true, preds, zero_division=0)
    rec    = recall_score(y_true, preds, zero_division=0)
    print(f"\n── {label} ──")
    print(f"  AUC: {auc:.4f}  F1: {f1:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}")
    return {"auc": auc, "f1": f1, "precision": prec, "recall": rec, "scores": scores, "preds": preds}

# Grid search over contamination values using the original 20 features
contamination_values = [0.02, 0.034, 0.05, 0.07, 0.10]
best_if_f1, best_if_model, best_if_cont, best_if_thresh = 0, None, 0, 0

print("\nGrid search over contamination values...")
for cont in contamination_values:
    iso = IsolationForest(n_estimators=300, contamination=cont, max_samples="auto", random_state=42)
    iso.fit(X_train_normal)
    thresh = tune_anomaly_threshold(iso, X_val, y_val)
    scores = -iso.score_samples(X_val)
    preds  = (scores >= thresh).astype(int)
    f1     = f1_score(y_val, preds, zero_division=0)
    print(f"  contamination={cont:.3f}  val F1={f1:.4f}")
    if f1 > best_if_f1:
        best_if_f1     = f1
        best_if_model  = iso
        best_if_cont   = cont
        best_if_thresh = thresh

print(f"\nBest contamination: {best_if_cont}  val F1: {best_if_f1:.4f}")

with mlflow.start_run(run_name="isolation_forest_tuned"):
    mlflow.log_params({"n_estimators": 300, "contamination": best_if_cont, "tuned_threshold": best_if_thresh})
    m_val_if  = evaluate_with_threshold(best_if_model, X_val,  y_val,  best_if_thresh, "IF tuned (val)")
    m_test_if = evaluate_with_threshold(best_if_model, X_test, y_test, best_if_thresh, "IF tuned (test)")
    mlflow.log_metrics({"val_auc": m_val_if["auc"], "val_f1": m_val_if["f1"], "test_auc": m_test_if["auc"], "test_f1": m_test_if["f1"]})
    
    with open("models/isolation_forest_tuned.pkl", "wb") as f:
        pickle.dump({"model": best_if_model, "threshold": best_if_thresh}, f)
    print(f"Saved → models/isolation_forest_tuned.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# TRACK 2 — Deep LSTM Autoencoder (Experimental/Debug)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRACK 2: Deep LSTM Autoencoder (Experimental)")
print("="*60)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class DeepLSTMAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=2):
            super().__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
            self.enc_fc  = nn.Linear(hidden_dim, latent_dim)
            self.dec_fc  = nn.Linear(latent_dim, hidden_dim)
            self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
            self.bn      = nn.BatchNorm1d(latent_dim)

        def forward(self, x):
            _, (h, _) = self.encoder(x)
            h         = h[-1]
            z         = self.bn(self.enc_fc(h))
            h_dec     = self.dec_fc(z).unsqueeze(1).repeat(1, x.size(1), 1)
            out, _    = self.decoder(h_dec)
            return out

    # Using the standard 20 scaled features
    X_tr = torch.FloatTensor(X_train_normal).unsqueeze(1)
    X_vl = torch.FloatTensor(X_val).unsqueeze(1)
    X_ts = torch.FloatTensor(X_test).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_tr), batch_size=128, shuffle=True)
    input_dim    = X_train_normal.shape[1]

    with mlflow.start_run(run_name="deep_lstm_autoencoder"):
        model     = DeepLSTMAutoencoder(input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)
        criterion = nn.MSELoss()

        model.train()
        prev_loss = float("inf")
        print("Training LSTM (Loss should now be < 1.0)...")
        for epoch in range(80):
            total_loss = 0
            for (batch,) in train_loader:
                optimizer.zero_grad()
                recon = model(batch)
                loss  = criterion(recon, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:2d}/80 — loss: {avg_loss:.6f} lr: {scheduler.get_last_lr()[0]:.6f}")

        model.eval()
        with torch.no_grad():
            def recon_error(X_tensor):
                recon  = model(X_tensor)
                return ((recon - X_tensor) ** 2).mean(dim=(1, 2)).numpy()

            train_errors = recon_error(X_tr)
            val_errors   = recon_error(X_vl)
            test_errors  = recon_error(X_ts)

        best_t, best_f1_lstm = 0, 0
        for pct in range(50, 100):
            t     = np.percentile(train_errors, pct)
            preds = (val_errors >= t).astype(int)
            f1    = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1_lstm:
                best_f1_lstm, best_t = f1, t

        y_pred_test = (test_errors >= best_t).astype(int)
        test_auc = roc_auc_score(y_test, test_errors)
        test_f1  = f1_score(y_test, y_pred_test, zero_division=0)
        test_rec = recall_score(y_test, y_pred_test, zero_division=0)

        print(f"\n── Deep LSTM Autoencoder (Scaled) ──")
        print(f"  Test AUC: {test_auc:.4f}  F1: {test_f1:.4f}  Recall: {test_rec:.4f}")

        torch.save({"model_state": model.state_dict(), "threshold": best_t}, "models/deep_lstm_autoencoder.pt")

except ImportError:
    print("[ERROR] PyTorch not installed.")
    test_f1, test_rec = 0, 0

# ─────────────────────────────────────────────────────────────────────────────
# Final comparison & Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Generating comparison plots...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
models_names = ["IF Baseline (0.315)", "IF Tuned", "Deep LSTM (Scaled)"]
f1_scores    = [0.3152, m_test_if["f1"], test_f1]
colors       = ["#B4B2A9", "#4C9BE8", "#E85D30"]

axes[0].bar(models_names, f1_scores, color=colors, edgecolor="white")
axes[0].set_title("F1 score comparison (test set)")
axes[0].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("eda_plots/anomaly_improvement_comparison.png", dpi=150)
plt.close()

print(f"""
╔══════════════════════════════════════════════════════════════╗
║        Anomaly Detection — Final Resolution                  ║
╠══════════════════════════════════════════════════════════════╣
║  Production Model       : Tuned Isolation Forest             ║
║  IF Tuned F1            : {m_test_if['f1']:.4f} (Recall: {m_test_if['recall']:.4f})       ║
║  Debug LSTM F1          : {test_f1:.4f} (Recall: {test_rec if 'test_rec' in dir() else 0:.4f})       ║
║                                                              ║
║  Best model saved       → models/isolation_forest_tuned.pkl  ║
╚══════════════════════════════════════════════════════════════╝
""")