import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# macOS stability overrides
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("\n" + "="*70)
print("ACME Industries - Tier 3 NLP Diagnosis Verification & Robustness")
print("="*70)

# ─── 1. VERIFICATION 1: DATA LEAKAGE CHECK ─────────────────────────────────
print("\n--- Running Verification 1: Data Leakage Check ---")

# Load full dataset and splits
df_full = pd.read_csv('data/maintenance_logs.csv')
df_train = pd.read_csv('data/train_logs.csv')
df_val   = pd.read_csv('data/val_logs.csv')
df_test  = pd.read_csv('data/test_logs.csv')

# Check overlap between test and other splits
train_texts = set(df_train['log_text'].str.strip().str.lower())
val_texts   = set(df_val['log_text'].str.strip().str.lower())
test_texts  = set(df_test['log_text'].str.strip().str.lower())

train_test_overlap = test_texts.intersection(train_texts)
val_test_overlap   = test_texts.intersection(val_texts)

print("=== DATA LEAKAGE CHECK ===")
print(f"Test samples:                    {len(test_texts)}")
print(f"Overlap with train set:          {len(train_test_overlap)}")
print(f"Overlap with val set:            {len(val_test_overlap)}")

if len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
    print("✅ PASSED — No data leakage detected")
    print("   Test set was truly unseen during training")
else:
    print("❌ FAILED — Data leakage detected!")
    print("   The following test samples appeared in training:")
    for text in train_test_overlap:
        print(f"   → {text[:80]}")

# Verify split sizes
print("\n=== SPLIT SIZE CHECK ===")
print(f"Train: {len(df_train)} samples (expected ~300)")
print(f"Val:   {len(df_val)}   samples (expected ~100)")
print(f"Test:  {len(df_test)}  samples (expected ~100)")
total = len(df_train) + len(df_val) + len(df_test)
print(f"Total: {total}         samples (expected 500+)")

# Verify label distribution is consistent
print("\n=== LABEL DISTRIBUTION CHECK ===")
for split_name, split_df in [('Train', df_train), 
                               ('Val',   df_val), 
                               ('Test',  df_test)]:
    comp_dist = split_df['component'].value_counts(
        normalize=True
    ).round(2)
    fail_dist = split_df['failure_mode'].value_counts(
        normalize=True
    ).round(2)
    print(f"\n{split_name} Component Distribution:")
    print(comp_dist.to_string())
    print(f"\n{split_name} Failure Mode Distribution:")
    print(fail_dist.to_string())


# ─── 2. VERIFICATION 2: REAL MESSY OPERATOR LOGS ROBUSTNESS ────────────────
print("\n--- Running Verification 2: Messy Log Robustness Check ---")

# Define repair corpus and TF-IDF context retriever
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

tfidf_retriever = TfidfRetriever(repair_corpus)

# Load tokenizer and models
tokenizer = DistilBertTokenizerFast.from_pretrained('./models/tier3_tokenizer')
model_component = DistilBertForSequenceClassification.from_pretrained('./models/tier3_component_classifier')
model_failure = DistilBertForSequenceClassification.from_pretrained('./models/tier3_failure_classifier')

# Load label maps
with open('./models/tier3_label_maps.json', 'r') as f:
    label_maps = json.load(f)

id2component = label_maps['id2component']
id2failure   = label_maps['id2failure']

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_component.to(device)
model_failure.to(device)
model_component.eval()
model_failure.eval()

# 20 messy logs to test
messy_logs = [
    # Typos and casual language
    "thing near pump broke again",
    "mchine making wierd noise cant tell where",
    "its hot and shaking alot",
    "belt snapped i think, conveyor stopped",
    "somthing grinding in the motor area",
    
    # Vague descriptions
    "weird smell from the back unit",
    "keeps stopping every few minutes",
    "pressure gauge showing nothing",
    "spindle wont go past half speed",
    "oil everywhere near the hydraulics",
    
    # Mixed technical and casual
    "bearing assembly rattling badly at high rpm",
    "pump cant hold pressure, maybe seal is gone",
    "cnc head vibrating way more than normal",
    "drive belt looks worn, machine slipping",
    "motor housing too hot to touch",
    
    # Very short entries
    "overheating",
    "seized up",
    "leaking",
    "no power",
    "loud noise"
]

def predict_single(log_text: str) -> dict:
    # Enrich with RAG context
    context = tfidf_retriever.get_relevant_context(log_text, top_k=2)
    enriched = f"Operator Log: {log_text}\nManual Context: {context}"
    
    inputs = tokenizer(
        enriched,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding='max_length'
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        comp_logits = model_component(**inputs).logits
        fail_logits = model_failure(**inputs).logits
    
    comp_probs = torch.softmax(comp_logits, dim=-1).cpu()
    fail_probs = torch.softmax(fail_logits, dim=-1).cpu()
    
    comp_confidence = comp_probs.max().item()
    fail_confidence = fail_probs.max().item()
    
    comp_pred = comp_probs.argmax().item()
    fail_pred = fail_probs.argmax().item()
    
    return {
        'log': log_text,
        'component': id2component[str(comp_pred)],
        'component_confidence': round(comp_confidence, 3),
        'failure_mode': id2failure[str(fail_pred)],
        'failure_confidence': round(fail_confidence, 3),
        'low_confidence_flag': (
            comp_confidence < 0.60 or 
            fail_confidence < 0.60
        )
    }

# Run all 20 messy logs
print("=== MESSY LOG ROBUSTNESS TEST ===\n")
results = []
low_confidence_count = 0

for log in messy_logs:
    result = predict_single(log)
    results.append(result)
    flag = "⚠️ LOW CONF" if result['low_confidence_flag'] else "✅"
    if result['low_confidence_flag']:
        low_confidence_count += 1
    print(f"{flag} Input:      {log[:50]}")
    print(f"   Component:  {result['component']} "
          f"({result['component_confidence']:.0%})")
    print(f"   Failure:    {result['failure_mode']} "
          f"({result['failure_confidence']:.0%})")
    print()

# Summary
print("=== ROBUSTNESS SUMMARY ===")
print(f"Total messy logs tested:     20")
print(f"Low confidence predictions:  {low_confidence_count}")
print(f"High confidence predictions: {20-low_confidence_count}")
print(f"Robustness score:            "
      f"{(20-low_confidence_count)/20*100:.1f}%")

if low_confidence_count <= 4:
    print("✅ PASSED — Model is robust to messy inputs")
    print("   (80%+ predictions with high confidence)")
elif low_confidence_count <= 8:
    print("⚠️ WARNING — Model struggles with some messy inputs")
    print("   Consider adding more noisy samples to training")
else:
    print("❌ FAILED — Model is overfitted to clean text")
    print("   Must regenerate training data with more noise")

# Save full results
os.makedirs('reports', exist_ok=True)
report_path = 'reports/tier3_verification_report.json'
with open(report_path, 'w') as f:
    json.dump({
        'verification_1_data_leakage': {
            'train_test_overlap': len(train_test_overlap),
            'val_test_overlap': len(val_test_overlap),
            'status': 'PASSED' if (len(train_test_overlap) == 0 and len(val_test_overlap) == 0) else 'FAILED'
        },
        'verification_2_robustness': {
            'total_tested': 20,
            'low_confidence_count': low_confidence_count,
            'robustness_score': f"{(20-low_confidence_count)/20*100:.1f}%",
            'status': 'PASSED' if low_confidence_count <= 4 else 'FAILED',
            'predictions': results
        }
    }, f, indent=2)

print(f"\nFull report saved to: {report_path}")
print("✅ Verification process finished!")
