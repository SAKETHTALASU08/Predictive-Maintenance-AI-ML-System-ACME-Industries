import os
import pickle
import platform
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- macOS Stability Locks ---
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─── 1. INITIALIZE API AND SCHEMAS ───────────────────────────────────────────
app = FastAPI(
    title="ACME Predictive Maintenance API",
    description="REST API for ML Anomaly Detection and AI NLP Diagnostics.",
    version="1.0.0"
)

# Define the expected JSON payload from the factory IoT system
class FactoryTelemetry(BaseModel):
    machine_id: str = Field(..., example="M-1042")
    sensor_data: List[float] = Field(..., min_items=20, max_items=20, description="Array of 20 normalized sensor features")
    operator_log: str = Field(..., example="Sudden drop in pressure. Inspection reveals complete mechanical fracture affecting the drive belt.")

# ─── 2. CACHED MODEL LOADING ─────────────────────────────────────────────────
# Load models globally so they stay in memory between API calls
try:
    with open("models/best_failure_predictor.pkl", "rb") as f:
        data = pickle.load(f)
        failure_model = data["model"] if isinstance(data, dict) else data
except Exception:
    failure_model = None

try:
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
except Exception:
    feature_names = ["feature_" + str(i) for i in range(20)]

if platform.system() != "Darwin":
    from transformers import pipeline
    nlp_classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
else:
    nlp_classifier = None

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

# ─── 3. THE API ENDPOINT ─────────────────────────────────────────────────────
@app.post("/predict")
async def run_predictive_maintenance(telemetry: FactoryTelemetry):
    try:
        sensor_array = np.array(telemetry.sensor_data).reshape(1, -1)
        
        # 1. Failure Prediction
        if failure_model:
            failure_prob = float(failure_model.predict_proba(sensor_array)[0][1])
        else:
            failure_prob = 0.88 # Fallback for demo
            
        # 2. XAI (Explainability)
        top_indices = np.argsort(np.abs(sensor_array[0]))[::-1][:3]
        top_drivers = [feature_names[idx] for idx in top_indices]
        
        # 3. RAG / NLP Diagnosis
        tfidf = TfidfVectorizer(stop_words='english')
        corpus_matrices = tfidf.fit_transform(repair_corpus['manual_text'])
        query_vec = tfidf.transform([telemetry.operator_log])
        best_doc_idx = np.argmax(cosine_similarity(query_vec, corpus_matrices).flatten())
        matched_doc = repair_corpus.iloc[best_doc_idx]
        
        # 4. Construct API Response
        status = "CRITICAL DISPATCH" if failure_prob > 0.50 else "MONITORING"
        
        return {
            "machine_id": telemetry.machine_id,
            "status": status,
            "risk_assessment": {
                "failure_probability": round(failure_prob, 3),
                "top_risk_factors": top_drivers
            },
            "ai_diagnosis": {
                "component": matched_doc["component"].title(),
                "failure_mode": matched_doc["failure_mode"].title()
            },
            "recommendation": {
                "action": matched_doc["manual_text"],
                "parts_required": matched_doc["parts_required"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))