"""
ACME Industries — AI Agent Evaluation Suite (N=40)
===================================================
Evaluates the zero-shot BERT classifier on 40 hand-crafted maintenance logs.
Measures Component Extraction Accuracy and Failure Mode Classification Accuracy.

Performance metrics from this script are reported in the Streamlit dashboard
under Tab 3: Model Performance.
"""

import sys
sys.path.insert(0, "setup")

import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import pipeline

print("\n" + "="*70)
print("ACME Industries - Robust AI Agent Evaluation (N=40)")
print("="*70)

# ─── 1. GENERATE 40 VARIATIONAL GROUND-TRUTH LOGS ───────────────────────────
print("Generating expanded synthetic evaluation dataset...")

# Master lists for mapping
components = ["spindle motor", "hydraulic pump", "cooling fan", "drive belt", "bearing assembly"]
failures = ["overheating", "temperature spike", "mechanical fracture", "vibration", "voltage spike"]

# 40 hand-crafted variations to simulate realistic, messy maintenance logs
raw_data = [
    # Spindle Motor variants
    ("A severe thermal runaway event noted in the main spindle motor. Power isolated.", "spindle motor", "overheating"),
    ("Spindle motor casing is hot to the touch; cooling system failure suspected.", "spindle motor", "overheating"),
    ("High frequency oscillations and micro-shaking detected near spindle motor mount.", "spindle motor", "vibration"),
    ("Spindle motor experienced a massive power surge, tripping the local fuse.", "spindle motor", "voltage spike"),
    ("Spindle motor completely seized up after an audible crunching sound.", "spindle motor", "mechanical fracture"),
    ("Unusual heat dissipation patterns observed on the spindle housing.", "spindle motor", "temperature spike"),
    ("Spindle motor shaft wobble detected during high-speed rotation cycles.", "spindle motor", "vibration"),
    ("Electrical arc damage observed inside the spindle motor junction box.", "spindle motor", "voltage spike"),
    
    # Hydraulic Pump variants
    ("Hydraulic pump fluid lines showing severe thermal spikes under load.", "hydraulic pump", "temperature spike"),
    ("Hydraulic pump pressure dropped to zero after a violent structural snap.", "hydraulic pump", "mechanical fracture"),
    ("Pump housing is vibrating out of normal thresholds, causing pipe chatter.", "hydraulic pump", "vibration"),
    ("Main hydraulic pump circuit breaker opened due to an overvoltage condition.", "hydraulic pump", "voltage spike"),
    ("Hydraulic fluid temperature exceeded maximum safe operating limits.", "hydraulic pump", "temperature spike"),
    ("Acoustic sensors picked up a fracturing sound inside the hydraulic pump casing.", "hydraulic pump", "mechanical fracture"),
    ("Minor voltage fluctuation recorded on the hydraulic pump digital telemetry line.", "hydraulic pump", "voltage spike"),
    ("Excessive heat buildup on the hydraulic pump valve block assembly.", "hydraulic pump", "overheating"),

    # Cooling Fan variants
    ("Cooling fan blades choked with debris, causing a critical thermal lock.", "cooling fan", "overheating"),
    ("Radiator cooling fan is shaking violently due to a missing counterweight.", "cooling fan", "vibration"),
    ("Cooling fan motor winding burned out from a sudden electrical spike.", "cooling fan", "voltage spike"),
    ("One of the cooling fan composite blades suffered a clean structural snap.", "cooling fan", "mechanical fracture"),
    ("Airflow temperature downstream of the cooling fan is abnormally elevated.", "cooling fan", "temperature spike"),
    ("Cooling fan assembly loose on its mountings, rattling heavily during operation.", "cooling fan", "vibration"),
    ("Fan motor drawing excessive current; overheating warning logged.", "cooling fan", "overheating"),
    ("Cooling fan sheared completely off its drive spindle.", "cooling fan", "mechanical fracture"),

    # Drive Belt variants
    ("Drive belt snapped mid-cycle, causing an instantaneous loss of torque.", "drive belt", "mechanical fracture"),
    ("Drive belt tracking off-center, rubbing against housing and generating friction heat.", "drive belt", "overheating"),
    ("Belt tensioner jumping wildly; drive belt experiencing high-amplitude fluttering.", "drive belt", "vibration"),
    ("Frictional heat from a slipping drive belt caused local temperature values to spike.", "drive belt", "temperature spike"),
    ("Drive belt completely torn in half. Immediate replacement required.", "drive belt", "mechanical fracture"),
    ("Belt surface showing signs of glazing and extreme heat degradation.", "drive belt", "overheating"),
    ("Slipping drive belt causing erratic speed readings, triggering a transient spike.", "drive belt", "temperature spike"),
    ("Drive belt resonance causing a loud hum and heavy machine chattering.", "drive belt", "vibration"),

    # Bearing Assembly variants
    ("Bearing assembly runout values indicate severe lack of lubrication and friction heat.", "bearing assembly", "overheating"),
    ("Bearing balls flat-spotted, producing massive acoustic and physical shaking.", "bearing assembly", "vibration"),
    ("Inner race of the bearing assembly shattered under high mechanical loading.", "bearing assembly", "mechanical fracture"),
    ("Bearing housing temperature spiking rapidly during high-RPM operations.", "bearing assembly", "temperature spike"),
    ("Bearing assembly running hot, smoking slightly before automatic emergency stop.", "bearing assembly", "overheating"),
    ("Severe pitting on bearing rollers creating high vibrational harmonics.", "bearing assembly", "vibration"),
    ("Bearing sleeve suffered a catastrophic structural failure and split apart.", "bearing assembly", "mechanical fracture"),
    ("Infrared inspection reveals localized thermal anomaly inside bearing housing.", "bearing assembly", "temperature spike")
]

# Convert to DataFrame
eval_df = pd.DataFrame(raw_data, columns=["raw_log", "true_component", "true_failure"])

# ─── 2. LOAD ZERO-SHOT BERT MODEL ───────────────────────────────────────────
print("Loading HuggingFace zero-shot classifier model...")
try:
    classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
    print("  -> Success: Classifier loaded.")
except Exception as e:
    print(f"  -> ERROR: Could not load classifier: {e}")
    print("  -> Ensure 'transformers' and 'torch' are installed and the model is downloaded.")
    sys.exit(1)

# ─── 3. EXECUTE PREDICTIONS ──────────────────────────────────────────────────
print("Evaluating 40 logs through the Diagnosis Agent layer...")
pred_components = []
pred_failures = []

for idx, row in eval_df.iterrows():
    text = row["raw_log"]
    
    # Classify Component
    comp_res = classifier(text, candidate_labels=components)
    pred_components.append(comp_res['labels'][0])
    
    # Classify Failure Mode
    fail_res = classifier(text, candidate_labels=failures)
    pred_failures.append(fail_res['labels'][0])

eval_df["pred_component"] = pred_components
eval_df["pred_failure"] = pred_failures

# ─── 4. COMPUTE METRICS ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("                      FINAL EVALUATION RESULTS                        ")
print("="*70)

# Overall Accuracy Metrics
comp_acc = accuracy_score(eval_df["true_component"], eval_df["pred_component"])
fail_acc = accuracy_score(eval_df["true_failure"], eval_df["pred_failure"])

print(f"✨ Component Extraction Accuracy : {comp_acc * 100:.2f}%")
print(f"✨ Failure Mode Classification Accuracy : {fail_acc * 100:.2f}%")

# 4a. Component Classification Report
print("\n📊 1. COMPONENT NER REPORT")
print("-" * 60)
print(classification_report(eval_df["true_component"], eval_df["pred_component"], zero_division=0))

# 4b. Failure Mode Classification Report
print("\n📊 2. FAILURE MODE CLASSIFICATION REPORT")
print("-" * 60)
print(classification_report(eval_df["true_failure"], eval_df["pred_failure"], zero_division=0))

# 4c. Text-Based Confusion Matrix for Failure Modes
print("\n🧩 3. FAILURE MODE CONFUSION MATRIX")
print("-" * 60)
cm = confusion_matrix(eval_df["true_failure"], eval_df["pred_failure"], labels=failures)

# Format matrix printout cleanly
print(f"{'':<23} | " + " | ".join([f"{f[:8]:<8}" for f in failures]))
print("-" * 80)
for i, label in enumerate(failures):
    row_str = " | ".join([f"{val:<8}" for val in cm[i]])
    print(f"{label:<23} | {row_str}")

print("\n✅ Robust evaluation suite complete!")