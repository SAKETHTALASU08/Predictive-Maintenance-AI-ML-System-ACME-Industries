import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# macOS fork locks
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Resolve directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "best_failure_predictor.pkl")
X_test_path = os.path.join(base_dir, "models", "X_test_sc.npy")
y_test_path = os.path.join(base_dir, "models", "y_test.npy")
plot_save_path = os.path.join(base_dir, "threshold_optimization.png")

# Load model and data
with open(model_path, "rb") as f:
    clf_bundle = pickle.load(f)
model = clf_bundle["model"]
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]

# Sweep thresholds from 0.05 to 0.80 in steps of 0.01
thresholds = np.arange(0.05, 0.81, 0.01)
sweep_results = []

for th in thresholds:
    th = round(th, 2)
    y_pred = (probs >= th).astype(int)
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate FP, TN, FN, TP
    TP = np.sum((y_test == 1) & (y_pred == 1))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))
    TN = np.sum((y_test == 0) & (y_pred == 0))
    
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    
    cost_score = (FNR * 10.0) + (FPR * 1.0)
    
    sweep_results.append({
        "threshold": th,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "fpr": FPR,
        "fnr": FNR,
        "cost_score": cost_score,
        "tp": int(TP),
        "fp": int(FP),
        "fn": int(FN),
        "tn": int(TN)
    })

df = pd.DataFrame(sweep_results)

# Candidate Strategies
# Strategy A: Minimum Cost Score
idx_min_cost = df["cost_score"].idxmin()
strat_a = df.loc[idx_min_cost]

# Strategy B: Highest F1 where Recall >= 0.90
df_rec_90 = df[df["recall"] >= 0.90]
if not df_rec_90.empty:
    idx_rec_90 = df_rec_90["f1_score"].idxmax()
    strat_b = df.loc[idx_rec_90]
else:
    idx_rec_90 = df["recall"].idxmax()
    strat_b = df.loc[idx_rec_90]

# Strategy C: Highest F1 score
idx_max_f1 = df["f1_score"].idxmax()
strat_c = df.loc[idx_max_f1]

# Baseline values requested by user (using model's actual corresponding states)
# Default is the model's original pre-tuned baseline metrics (Recall: 0.7794, Precision: 0.8833, F1: 0.8281)
# Aggressive is the 0.10 threshold metrics (Recall: 0.9118, Precision: 0.5254, F1: 0.6667)
default_cost = ((1 - 0.7794) * 10.0) + ((7.0 / 1932.0) * 1.0) # FNR * 10 + FPR * 1
aggressive_cost = ((1 - 0.9118) * 10.0) + ((56.0 / 1932.0) * 1.0)

# PRINT COMPARISON TABLE
print("\n" + "="*80)
print("                       THRESHOLD STRATEGY COMPARISON")
print("="*80)
print(f"| {'Strategy':11} | {'Threshold':9} | {'Precision':9} | {'Recall':9} | {'F1':5} | {'Cost Score':10} |")
print("-" * 69)
print(f"| {'Min Cost':11} | {strat_a['threshold']:9.2f} | {strat_a['precision']:9.4f} | {strat_a['recall']:9.4f} | {strat_a['f1_score']:.4f} | {strat_a['cost_score']:10.4f} |")
print(f"| {'Recall>=90':11} | {strat_b['threshold']:9.2f} | {strat_b['precision']:9.4f} | {strat_b['recall']:9.4f} | {strat_b['f1_score']:.4f} | {strat_b['cost_score']:10.4f} |")
print(f"| {'F1 Optimal':11} | {strat_c['threshold']:9.2f} | {strat_c['precision']:9.4f} | {strat_c['recall']:9.4f} | {strat_c['f1_score']:.4f} | {strat_c['cost_score']:10.4f} |")
print("-" * 69)
print(f"| {'Default':11} | {0.50:9.2f} | {0.8833:9.4f} | {0.7794:9.4f} | {0.8281:.4f} | {default_cost:10.4f} |")
print(f"| {'Aggressive':11} | {0.10:9.2f} | {0.5254:9.4f} | {0.9118:9.4f} | {0.6667:.4f} | {aggressive_cost:10.4f} |")
print("="*80 + "\n")

# PLOTS: 2x2 Figure
fig, axs = plt.subplots(2, 2, figsize=(15, 12), facecolor='#f8fafc')
plt.subplots_adjust(hspace=0.3, wspace=0.3)

th_a, th_b, th_c = strat_a['threshold'], strat_b['threshold'], strat_c['threshold']

# 1. Top Left: Precision & Recall vs Threshold
axs[0, 0].plot(df["threshold"], df["precision"], label="Precision", color="#059669", linewidth=2.5)
axs[0, 0].plot(df["threshold"], df["recall"], label="Recall", color="#3b82f6", linewidth=2.5)
axs[0, 0].axvline(th_a, color="#ef4444", linestyle="--", alpha=0.8, label=f"Min Cost ({th_a:.2f})")
axs[0, 0].axvline(th_b, color="#f59e0b", linestyle="-.", alpha=0.8, label=f"Recall>=90% ({th_b:.2f})")
axs[0, 0].axvline(th_c, color="#6366f1", linestyle=":", alpha=0.8, label=f"F1 Opt ({th_c:.2f})")
axs[0, 0].set_xlabel("Threshold", fontweight='semibold')
axs[0, 0].set_ylabel("Score", fontweight='semibold')
axs[0, 0].set_title("Precision & Recall vs. Threshold", fontsize=12, fontweight='bold', pad=10)
axs[0, 0].grid(True, linestyle=':', alpha=0.6)
axs[0, 0].legend()

# 2. Top Right: F1 vs Threshold
axs[0, 1].plot(df["threshold"], df["f1_score"], color="#6366f1", linewidth=2.5, label="F1 Score")
axs[0, 1].scatter([th_c], [strat_c['f1_score']], color="red", s=100, zorder=5, label=f"Max F1 ({strat_c['f1_score']:.4f} @ {th_c:.2f})")
axs[0, 1].set_xlabel("Threshold", fontweight='semibold')
axs[0, 1].set_ylabel("F1 Score", fontweight='semibold')
axs[0, 1].set_title("F1 Score vs. Threshold", fontsize=12, fontweight='bold', pad=10)
axs[0, 1].grid(True, linestyle=':', alpha=0.6)
axs[0, 1].legend()

# 3. Bottom Left: Cost Score vs Threshold
axs[1, 0].plot(df["threshold"], df["cost_score"], color="#ef4444", linewidth=2.5, label="Cost Score")
axs[1, 0].scatter([th_a], [strat_a['cost_score']], color="blue", s=100, zorder=5, label=f"Min Cost ({strat_a['cost_score']:.4f} @ {th_a:.2f})")
axs[1, 0].set_xlabel("Threshold", fontweight='semibold')
axs[1, 0].set_ylabel("Cost Score", fontweight='semibold')
axs[1, 0].set_title("Cost Score vs. Threshold (Minimize)", fontsize=12, fontweight='bold', pad=10)
axs[1, 0].grid(True, linestyle=':', alpha=0.6)
axs[1, 0].legend()

# 4. Bottom Right: Precision-Recall Curve
axs[1, 1].plot(df["recall"], df["precision"], color="#10b981", linewidth=2.5, label="PR Curve")
axs[1, 1].scatter([strat_a['recall']], [strat_a['precision']], color="#ef4444", s=120, zorder=5, edgecolors='black', label=f"Min Cost ({th_a:.2f})")
axs[1, 1].scatter([strat_b['recall']], [strat_b['precision']], color="#f59e0b", s=120, zorder=5, edgecolors='black', label=f"Recall>=90% ({th_b:.2f})")
axs[1, 1].scatter([strat_c['recall']], [strat_c['precision']], color="#6366f1", zorder=5, s=120, edgecolors='black', label=f"F1 Opt ({th_c:.2f})")
axs[1, 1].set_xlabel("Recall", fontweight='semibold')
axs[1, 1].set_ylabel("Precision", fontweight='semibold')
axs[1, 1].set_title("Precision-Recall Space", fontsize=12, fontweight='bold', pad=10)
axs[1, 1].grid(True, linestyle=':', alpha=0.6)
axs[1, 1].legend()

plt.suptitle("Tier 1 Failure Predictor: Threshold Optimization Suite", fontsize=15, fontweight='bold', y=0.96)
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved threshold sweep plots to: {plot_save_path}")

# Operational recommendations calculations
rec_cost_red = ((default_cost - strat_a['cost_score']) / default_cost) * 100
rec_failures_default = int(round(0.7794 * 100))
rec_failures_opt = int(round(strat_a['recall'] * 100))
rec_false_default = round((7.0 / 1932.0) * 100, 2)
rec_false_opt = round(strat_a['fpr'] * 100, 2)

print("\n" + "="*80)
print("                               RECOMMENDATIONS")
print("="*80)
print(f"RECOMMENDED THRESHOLD: {strat_a['threshold']:.2f}")
print(f"Strategy: Strategy A (Minimum Cost Score)")
print("\nExpected operational impact:")
print(f"- Failures caught per 100:  {rec_failures_opt}  (was {rec_failures_default} at 0.50)")
print(f"- False alarms per 100:     {rec_false_opt:.2f}  (was {rec_false_default:.2f} at 0.50)")
print(f"- Estimated cost reduction: {rec_cost_red:.2f}% vs default threshold")
print("="*80 + "\n")
