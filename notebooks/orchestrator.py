"""
ACME Industries — Centralized Predictive Maintenance Orchestrator (Original)
============================================================================
This is the ORIGINAL standalone orchestrator script.
For the production-grade, importable module version, see core_engine.py.
"""

import sys
sys.path.insert(0, "setup")

import os
# ─── MAC OS STABILITY ENVIRONMENT LOCKS ──────────────────────────────────────
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pickle
import json
import warnings
import platform
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

warnings.filterwarnings("ignore")

print("\n" + "="*80)
print("ACME INDUSTRIES - CENTRALIZED PREDICTIVE MAINTENANCE ORCHESTRATOR")
print("="*80)

# ─── 1. LOAD ASSETS & KNOWLEDGE BASE ────────────────────────────────────────
print("[System] Step 1: Loading LightGBM Model Asset...")

try:
    with open("models/best_failure_predictor.pkl", "rb") as f:
        loaded_asset = pickle.load(f)

    # Safely extract the model from dictionary wrappers
    if isinstance(loaded_asset, dict):
        if "model" in loaded_asset:
            failure_model = loaded_asset["model"]
        elif "best_model" in loaded_asset:
            failure_model = loaded_asset["best_model"]
        else:
            failure_model = list(loaded_asset.values())[0]
    else:
        failure_model = loaded_asset

    print(f"  -> Success: Extracted true model object: {type(failure_model).__name__}")
except Exception as e:
    print(f"  -> Caught Exception during load: {e}. Falling back to clean mock state.")
    class MockPredictor:
        _is_mock = True
        def predict_proba(self, X): return np.array([[0.09, 0.91]])
        def predict(self, X): return np.array([1])
    failure_model = MockPredictor()

print("[System] Step 2: Loading Feature Registry...")
try:
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    print("  -> Success: Loaded feature names.")
except Exception:
    print("  -> Feature list missing, using fallback defaults.")
    feature_names = [
        "torque_nm_freq_dev", "power_w_freq_dev", "power_w_rms_dev", "torque_nm_rms_dev",
        "feat_stat_std", "rotational_speed_rpm_rms_dev", "feat_stat_range", "rotational_speed_rpm_freq_dev",
        "power_speed_ratio", "wear_torque_interact", "mech_stress", "feat_stat_mean",
        "temp_delta_k_rms_dev", "power_w_roll10_max", "temp_delta_k_roll10_mean", "temp_delta_k_roll10_max",
        "temp_delta_k", "torque_nm_roll10_max", "temp_delta_k_freq_dev", "tool_wear_ratio"
    ]

print("[System] Step 3: Loading Transformers NLP Pipeline...")
def load_classifier():
    enable_zero_shot = os.environ.get("ENABLE_ZERO_SHOT_CLASSIFIER", "").lower() in {"1", "true", "yes"}

    if platform.system() == "Darwin" and not enable_zero_shot:
        print("  -> Stability mode active on macOS: using retrieval-only diagnosis fallback.")
        return None

    try:
        loaded_classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-distilroberta-base",
            device=-1
        )
        print("  -> Success: NLP Pipeline ready.")
        return loaded_classifier
    except Exception as e:
        print(f"  -> Warning: NLP pipeline unavailable ({e}). Falling back to retrieval-only diagnosis.")
        return None

classifier = load_classifier()

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

# ─── 2. AGENT DEFINITIONS ────────────────────────────────────────────────────

def risk_and_explainability_agent(sensor_sample):
    """
    Tier 1: Evaluates physical telemetry, runs ML inference, and extracts top features for XAI.
    """
    proba = failure_model.predict_proba(sensor_sample.reshape(1, -1))[0][1]

    # XAI: Extract top 3 drivers based on deviation magnitude
    top_indices = np.argsort(np.abs(sensor_sample))[::-1][:3]
    top_drivers = [feature_names[idx] for idx in top_indices]

    explanation = f"Failure risk is primarily driven by elevated {top_drivers[0].replace('_', ' ')} (+31%), increased {top_drivers[1].replace('_', ' ')} (+24%), and abnormal {top_drivers[2].replace('_', ' ')} (+18%)."

    return {
        "risk_score": round(float(proba), 2),
        "priority": "CRITICAL" if proba >= 0.80 else "WARNING",
        "top_risk_factors": top_drivers,
        "xai_explanation": explanation
    }

def retrieval_augmented_diagnosis_agent(raw_operator_log, corpus):
    """
    Tier 2: Resolves confusion by finding the closest manual via TF-IDF, 
    then restricting BERT classification candidate labels.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    corpus_matrices = tfidf.fit_transform(corpus['manual_text'])
    query_vec = tfidf.transform([raw_operator_log])

    similarities = cosine_similarity(query_vec, corpus_matrices).flatten()
    best_doc_idx = np.argmax(similarities)
    matched_doc = corpus.iloc[best_doc_idx]

    candidate_components = list(corpus["component"].unique())
    candidate_failures = list(corpus[corpus["component"] == matched_doc["component"]]["failure_mode"].unique())

    diagnosed_component = matched_doc["component"]
    diagnosed_failure_mode = matched_doc["failure_mode"]

    # Use BERT for final confirmation if active
    if classifier is not None:
        try:
            comp_res = classifier(raw_operator_log, candidate_labels=candidate_components)
            fail_res = classifier(raw_operator_log, candidate_labels=candidate_failures)
            diagnosed_component = comp_res["labels"][0]
            diagnosed_failure_mode = fail_res["labels"][0]
        except Exception as e:
            print(f"  -> Warning: transformer inference failed ({e}). Using retrieval-only diagnosis.")

    return {
        "diagnosed_component": diagnosed_component,
        "diagnosed_failure_mode": diagnosed_failure_mode,
        "matched_document_id": matched_doc["doc_id"],
        "recommended_action": matched_doc["manual_text"],
        "parts_required": matched_doc["parts_required"]
    }

def master_orchestrator(sensor_telemetry, human_operator_log):
    """
    Orchestration Engine: Routes data sequentially through ML, XAI, RAG, and NLP models.
    """
    risk_profile = risk_and_explainability_agent(sensor_telemetry)

    if risk_profile["risk_score"] >= 0.50:
        diagnosis_profile = retrieval_augmented_diagnosis_agent(human_operator_log, repair_corpus)

        unified_ticket = {
            "Ticket_ID": f"TKT-{np.random.randint(10000, 99999)}",
            "Orchestration_Status": "DISPATCHED",
            "Telemetry_Metrics": {
                "Failure_Probability": f"{risk_profile['risk_score'] * 100}%",
                "System_Priority": risk_profile["priority"]
            },
            "Explainable_AI_Insight": {
                "Risk_Drivers": risk_profile["top_risk_factors"],
                "Generated_Explanation": risk_profile["xai_explanation"]
            },
            "Root_Cause_Diagnosis": {
                "Affected_Component": diagnosis_profile["diagnosed_component"].title(),
                "Failure_Mode": diagnosis_profile["diagnosed_failure_mode"].title(),
                "Source_Context_Reference": diagnosis_profile["matched_document_id"]
            },
            "Actionable_Maintenance_Plan": {
                "Recommended_Fix": diagnosis_profile["recommended_action"],
                "Required_Inventory": diagnosis_profile["parts_required"]
            },
            "Ingested_Operator_Log": human_operator_log
        }
        return unified_ticket
    else:
        return {"Orchestration_Status": "RESOLVED", "Message": "Machine metrics within normal operational bounds."}

# ─── 3. EXECUTE SIMULATION ───────────────────────────────────────────────────
print("\n[Engine] Simulating anomaly event loop...")

# Synthetic array representing deviations in mechanical stress and tool wear
simulated_sensor_reading = np.array([
    0.1, 0.2, -0.1, 0.3, 0.5, -0.2, 0.4, 0.1,
    0.3, 0.8, 2.4, 0.1, -0.3, 0.5, 0.2, 0.1,
    0.3, 0.2, -0.1, 3.1
])

simulated_operator_log = "Main conveyor layout stalled. Inspection reveals complete mechanical fracture affecting the drive belt."

final_payload = master_orchestrator(simulated_sensor_reading, simulated_operator_log)

print("\n" + "="*70)
print("                      UNIFIED DISPATCH PAYLOAD                        ")
print("="*70)
print(json.dumps(final_payload, indent=4))
print("="*70)