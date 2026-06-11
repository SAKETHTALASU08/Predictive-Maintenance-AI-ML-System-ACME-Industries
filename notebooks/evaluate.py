"""
ACME Industries — Fine-Tuned AI Agent Evaluation Suite (N=100)
============================================================
Evaluates the fine-tuned DistilBERT classifiers on the locked test split (data/test_logs.csv).
Measures Component Extraction Accuracy and Failure Mode Classification Accuracy.
Compares results against the legacy zero-shot BERT baseline.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# macOS stability overrides
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("\n" + "="*70)
print("ACME Industries - Fine-Tuned NLP Agent Evaluation (N=100)")
print("="*70)

# ─── 1. DEFINE REPAIR CORPUS ───────────────────────────────────────────────
repair_corpus = pd.DataFrame({
    "doc_id": ["DOC-001", "DOC-002", "DOC-003", "DOC-004", "DOC-005"],
    "component": ["spindle motor", "hydraulic pump", "drive belt", "bearing assembly", "hydraulic pump"],
    "failure_mode": ["overheating", "temperature spike", "mechanical fracture", "vibration", "voltage spike"],
    "manual_text": [
        "To resolve severe overheating in the spindle motor, replace the motor thermal paste, verify air filter clearance, and check the ventilation shaft. Estimated downtime: 2 hours.",
        "For abnormal temperature spikes or thermal issues in the hydraulic pump, flush the coolant system and recalibrate the thermal sensors. Estimated downtime: 3 hours.",
        "A mechanical fracture in the drive belt requires a complete replacement of the Poly-V belt and recalibration of the tensioner pulley. Estimated downtime: 4 hours.",
        "Excessive vibration in the bearing assembly should be treated by applying industrial lubricant. If pitting is visible, replace the bearing kit. Estimated downtime: 1.5 hours.",
        "Voltage spikes in the hydraulic pump trigger the emergency breaker. Reset the breaker and inspect the wiring harness for arc damage. Estimated downtime: 1 hour."
    ],
    "parts_required": ["Thermal Paste, Air Filter", "Coolant Fluid", "Poly-V Belt", "Industrial Lubricant", "Wiring Harness"]
})

# ─── 2. TF-IDF RETRIEVER ──────────────────────────────────────────────────
class TfidfRetriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.corpus_matrices = self.tfidf.fit_transform(corpus['manual_text'])
        
    def get_relevant_context(self, operator_log, top_k=2):
        query_vec = self.tfidf.transform([operator_log])
        similarities = cosine_similarity(query_vec, self.corpus_matrices).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        contexts = [self.corpus.iloc[idx]['manual_text'] for idx in top_indices]
        return " ".join(contexts)

# ─── 3. LOAD TEST SPLIT AND MODEL ASSETS ──────────────────────────────────
print("Loading locked test dataset (data/test_logs.csv)...")
if not os.path.exists("data/test_logs.csv"):
    print("  -> ERROR: Test split 'data/test_logs.csv' not found. Please run generate_logs.py first.")
    sys.exit(1)
    
df_test = pd.read_csv("data/test_logs.csv")
print(f"Loaded {len(df_test)} test samples.")

print("Loading fine-tuned NLP model assets from models/...")
try:
    tokenizer_path = "./models/tier3_tokenizer"
    comp_model_path = "./models/tier3_component_classifier"
    fail_model_path = "./models/tier3_failure_classifier"
    label_maps_path = "./models/tier3_label_maps.json"
    
    with open(label_maps_path, "r") as f:
        label_maps = json.load(f)
        
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    model_component = DistilBertForSequenceClassification.from_pretrained(comp_model_path)
    model_failure = DistilBertForSequenceClassification.from_pretrained(fail_model_path)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_component.to(device)
    model_failure.to(device)
    model_component.eval()
    model_failure.eval()
    print(f"  -> Success: Loaded NLP model components onto device: {device}")
except Exception as e:
    print(f"  -> ERROR: Failed to load fine-tuned models ({e}). Ensure train_tier3_nlp.py has run successfully.")
    sys.exit(1)

# ─── 4. RUN EVALUATION INFERENCE ──────────────────────────────────────────
print("\nEvaluating fine-tuned models using RAG-enriched context...")
retriever = TfidfRetriever(repair_corpus)

pred_components = []
pred_failures = []
id2component = label_maps["id2component"]
id2failure = label_maps["id2failure"]

for idx, row in df_test.iterrows():
    log = row["log_text"]
    
    # Retrieve top_k=2 manual text context
    context = retriever.get_relevant_context(log, top_k=2)
    enriched_input = f"Operator Log: {log}\nManual Context: {context}"
    
    # Tokenize input
    inputs = tokenizer(
        enriched_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict Component
        comp_out = model_component(**inputs)
        comp_pred = int(np.argmax(F.softmax(comp_out.logits, dim=-1).cpu().numpy()[0]))
        pred_components.append(id2component[str(comp_pred)])
        
        # Predict Failure Mode
        fail_out = model_failure(**inputs)
        fail_pred = int(np.argmax(F.softmax(fail_out.logits, dim=-1).cpu().numpy()[0]))
        pred_failures.append(id2failure[str(fail_pred)])

df_test["pred_component"] = pred_components
df_test["pred_failure"] = pred_failures

# ─── 5. COMPUTE AND PRINT PERFORMANCE METRICS ──────────────────────────────
comp_acc = accuracy_score(df_test["component"], df_test["pred_component"])
fail_acc = accuracy_score(df_test["failure_mode"], df_test["pred_failure"])
comp_f1 = f1_score(df_test["component"], df_test["pred_component"], average="weighted")
fail_f1 = f1_score(df_test["failure_mode"], df_test["pred_failure"], average="weighted")

print("\n" + "="*70)
print("                      FINAL EVALUATION RESULTS                        ")
print("="*70)
print(f"✨ Component Extraction Accuracy         : {comp_acc * 100:.2f}%  (Baseline Zero-Shot: 72.50%)")
print(f"✨ Failure Mode Classification Accuracy  : {fail_acc * 100:.2f}%  (Baseline Zero-Shot: 65.00%)")
print(f"✨ Component Extraction Weighted F1      : {comp_f1:.4f}")
print(f"✨ Failure Mode Classification Weighted F1 : {fail_f1:.4f}")

# Component Classification Report
print("\n📊 1. COMPONENT NER REPORT")
print("-" * 60)
print(classification_report(df_test["component"], df_test["pred_component"], zero_division=0))

# Failure Mode Classification Report
print("\n📊 2. FAILURE MODE CLASSIFICATION REPORT")
print("-" * 60)
print(classification_report(df_test["failure_mode"], df_test["pred_failure"], zero_division=0))

# Confusion Matrix for Failure Modes
print("\n🧩 3. FAILURE MODE CONFUSION MATRIX")
print("-" * 60)
unique_failures = label_maps["failure_labels"]
cm = confusion_matrix(df_test["failure_mode"], df_test["pred_failure"], labels=unique_failures)

# Format matrix printout cleanly
print(f"{'':<25} | " + " | ".join([f"{f[:8]:<8}" for f in unique_failures]))
print("-" * 85)
for i, label in enumerate(unique_failures):
    row_str = " | ".join([f"{val:<8}" for val in cm[i]])
    print(f"{label:<25} | {row_str}")

print("\n✅ Fine-tuned evaluation suite complete!")