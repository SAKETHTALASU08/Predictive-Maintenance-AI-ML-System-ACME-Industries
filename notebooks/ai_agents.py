"""
ACME Industries — Advanced AI Diagnostic Stack (BERT + TF-IDF)
===============================================================
Standalone diagnostic pipeline with BERT zero-shot classification,
TF-IDF repair manual retrieval, and ticket generation.

For the production orchestrated version, see core_engine.py.
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline

print("\n" + "="*70)
print("ACME Industries - Advanced AI Diagnostic Stack (BERT + TF-IDF)")
print("="*70)

# ─── 1. SIMULATE INPUT DATA (Sensor Flags & Manuals) ─────────────────────────
print("1. Loading synthetic historical logs and repair manuals...")

# Simulated incoming alerts from the ML Anomaly/Prediction layer
incoming_alerts = [
    {"machine_id": "M-1042", "raw_log": "Operator reported loud grinding noise from the spindle motor. Unit automatically halted due to severe overheating."},
    {"machine_id": "M-2991", "raw_log": "Thermal sensors indicate abnormal temperature spike in the hydraulic pump. Coolant levels appear normal."},
    {"machine_id": "M-3011", "raw_log": "Sudden drop in pressure. Inspection reveals complete mechanical fracture affecting the drive belt."}
]

# Simulated Manufacturer Repair Manuals (Corpus for TF-IDF)
repair_corpus = pd.DataFrame({
    "doc_id": ["DOC-001", "DOC-002", "DOC-003", "DOC-004", "DOC-005"],
    "manual_text": [
        "To resolve severe overheating in the spindle motor, replace the motor thermal paste, verify air filter clearance, and check the ventilation shaft. Estimated downtime: 2 hours.",
        "For abnormal temperature spikes or thermal issues in the hydraulic pump, flush the coolant system and recalibrate the thermal sensors. Estimated downtime: 3 hours.",
        "A mechanical fracture in the drive belt requires a complete replacement of the Poly-V belt and recalibration of the tensioner pulley. Estimated downtime: 4 hours.",
        "Excessive vibration in the bearing assembly should be treated by applying industrial lubricant. If pitting is visible, replace the bearing kit. Estimated downtime: 1.5 hours.",
        "Voltage spikes in the hydraulic pump trigger the emergency breaker. Reset the breaker and inspect the wiring harness for arc damage. Estimated downtime: 1 hour."
    ],
    "parts_required": ["Thermal Paste, Air Filter", "Coolant Fluid", "Poly-V Belt", "Industrial Lubricant", "Wiring Harness"]
})

# ─── 2. INITIALIZE AI MODELS ─────────────────────────────────────────────────
print("2. Downloading/Loading HuggingFace NLP models (This may take a moment)...")
# Using a lightweight zero-shot classifier as a proxy for a custom-finetuned NER model
try:
    classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
    print("  -> Success: Classifier loaded.")
except Exception as e:
    print(f"  -> ERROR: Could not load classifier: {e}")
    print("  -> Ensure 'transformers' and 'torch' are installed.")
    sys.exit(1)

# ─── 3. AGENT 1: FAILURE DIAGNOSIS (BERT & LDA) ──────────────────────────────
def diagnosis_agent(logs):
    print("\n[Agent 1] Executing Failure Diagnosis (Zero-Shot BERT & LDA)...")
    diagnoses = []
    
    # 3a. Component & Failure Mode Extraction (BERT)
    component_labels = ["spindle motor", "hydraulic pump", "cooling fan", "drive belt", "bearing assembly"]
    failure_labels = ["overheating", "temperature spike", "mechanical fracture", "vibration", "voltage spike"]
    
    # 3b. Topic Modeling (LDA) on the batch to identify recurring issue clusters
    raw_texts = [alert["raw_log"] for alert in logs]
    tf_vectorizer = CountVectorizer(stop_words='english')
    tf = tf_vectorizer.fit_transform(raw_texts)
    lda = LatentDirichletAllocation(n_components=2, random_state=42).fit(tf)
    
    for alert in logs:
        text = alert["raw_log"]
        
        # Neural classification for exact component
        comp_result = classifier(text, candidate_labels=component_labels)
        predicted_comp = comp_result['labels'][0]
        
        # Neural classification for failure mode
        fail_result = classifier(text, candidate_labels=failure_labels)
        predicted_fail = fail_result['labels'][0]
        
        diagnoses.append({
            "machine_id": alert["machine_id"],
            "raw_log": text,
            "extracted_component": predicted_comp,
            "extracted_failure": predicted_fail,
            "bert_confidence": round(fail_result['scores'][0], 3)
        })
        
    return diagnoses

# ─── 4. AGENT 2: FIX RECOMMENDATION (TF-IDF Content-Based Filtering) ─────────
def recommendation_agent(diagnoses, corpus):
    print("[Agent 2] Executing Fix Recommendation (TF-IDF Cosine Similarity)...")
    
    # Fit the TF-IDF Vectorizer on our Repair Manual Corpus
    tfidf = TfidfVectorizer(stop_words='english')
    corpus_matrix = tfidf.fit_transform(corpus['manual_text'])
    
    recommendations = []
    
    for diag in diagnoses:
        # Create a search query based on the BERT diagnosis
        search_query = f"{diag['extracted_failure']} in {diag['extracted_component']}"
        query_vec = tfidf.transform([search_query])
        
        # Calculate Cosine Similarity between the diagnosis and all repair manuals
        similarities = cosine_similarity(query_vec, corpus_matrix).flatten()
        
        # Get Top-1 hit
        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]
        
        best_manual = corpus.iloc[best_match_idx]
        
        rec = {
            **diag,
            "recommended_action": best_manual['manual_text'],
            "parts_required": best_manual['parts_required'],
            "match_confidence": round(confidence, 3)
        }
        recommendations.append(rec)
        
    return recommendations

# ─── 5. AGENT 3: TICKET GENERATION (Slot-Fill) ───────────────────────────────
def ticket_generation_agent(recommendations):
    print("[Agent 3] Generating Structured Maintenance Tickets...\n")
    tickets = []
    
    for rec in recommendations:
        ticket = {
            "Ticket_ID": f"TKT-{np.random.randint(10000, 99999)}",
            "Status": "CRITICAL DISPATCH",
            "Machine": rec["machine_id"],
            "AI_Diagnosis": {
                "Component": rec["extracted_component"].title(),
                "Failure_Mode": rec["extracted_failure"].title(),
                "Model_Confidence": f"{rec['bert_confidence']*100}%"
            },
            "AI_Recommendation": {
                "Action": rec["recommended_action"],
                "Parts_Required": rec["parts_required"],
                "Manual_Match_Score": f"{rec['match_confidence']*100}%"
            },
            "Original_Telemetry_Log": rec["raw_log"]
        }
        tickets.append(ticket)
        
    return tickets

# ─── 6. EXECUTE THE PIPELINE ─────────────────────────────────────────────────
diagnosed_alerts = diagnosis_agent(incoming_alerts)
actionable_recommendations = recommendation_agent(diagnosed_alerts, repair_corpus)
final_tickets = ticket_generation_agent(actionable_recommendations)

# Output results
for tkt in final_tickets:
    print(json.dumps(tkt, indent=4))
    print("-" * 70)

print("✅ Advanced AI Pipeline Complete!")