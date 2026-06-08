"""
ACME Industries — FastAPI Predictive Maintenance REST API
=========================================================
High-performance REST API serving the ML/AI inference pipeline.

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /           — API info
    GET  /health     — System readiness check
    POST /predict    — Full inference pipeline
    GET  /models/performance — Live model metrics
"""

import os
import sys

# ─── macOS Stability Locks (before any heavy imports) ─────────────────────────
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

# Import the globally-cached core engine
import core_engine

# ══════════════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ACME Industries Predictive Maintenance API",
    description=(
        "Enterprise REST API for ML-powered failure prediction, anomaly detection, "
        "and AI-driven root cause diagnosis. Powered by LightGBM, Isolation Forest, "
        "and HuggingFace BERT Zero-Shot Classification."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC V2 SCHEMAS (strict — no deprecated V1 syntax)
# ══════════════════════════════════════════════════════════════════════════════

class SensorPayload(BaseModel):
    """Input schema for the /predict endpoint with raw sensor values."""
    torque_nm: float = Field(..., ge=0.0, description="Torque in Nm.")
    spindle_speed_rpm: float = Field(..., ge=0.0, description="Spindle speed in RPM.")
    tool_wear_min: float = Field(..., ge=0.0, le=255.0, description="Tool wear in minutes (0-255).")
    rotational_speed_rpm: float = Field(..., ge=0.0, description="Rotational speed in RPM.")
    power_w: float = Field(..., ge=0.0, description="Power in Watts.")
    voltage_v: float = Field(..., ge=0.0, description="Voltage in Volts.")
    current_a: float = Field(..., ge=0.0, description="Current in Amperes.")
    vibration_mm_s: float = Field(..., ge=0.0, description="Vibration amplitude in mm/s.")
    temperature_k: float = Field(..., ge=0.0, description="Temperature in Kelvin.")
    operator_log: str = Field(..., description="Free-text operator maintenance log.")
    machine_id: Optional[str] = Field(default="M-0000", description="Unique machine identifier.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "torque_nm": 40.0,
                    "spindle_speed_rpm": 1500.0,
                    "tool_wear_min": 25.0,
                    "rotational_speed_rpm": 1500.0,
                    "power_w": 6280.0,
                    "voltage_v": 220.0,
                    "current_a": 28.5,
                    "vibration_mm_s": 1.5,
                    "temperature_k": 300.0,
                    "operator_log": "Main conveyor layout stalled. Inspection reveals complete mechanical fracture affecting the drive belt.",
                    "machine_id": "M-1042"
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    status: str
    failure_model_loaded: bool
    anomaly_model_loaded: bool
    nlp_model_loaded: bool
    feature_count: int
    platform: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "failure_model_loaded": True,
                    "anomaly_model_loaded": True,
                    "nlp_model_loaded": False,
                    "feature_count": 20,
                    "platform": "Darwin"
                }
            ]
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["System"])
async def root():
    """API information and available endpoints."""
    return {
        "application": "ACME Industries Predictive Maintenance API",
        "version": "2.0.0",
        "endpoints": {
            "POST /predict": "Run full ML/AI inference pipeline on sensor data",
            "GET /health": "Check system readiness and model load status",
            "GET /models/performance": "View live model performance metrics",
            "GET /docs": "Interactive Swagger UI documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System readiness check — reports which models are loaded."""
    import platform
    return HealthResponse(
        status="healthy" if core_engine.FAILURE_MODEL_LOADED else "degraded",
        failure_model_loaded=core_engine.FAILURE_MODEL_LOADED,
        anomaly_model_loaded=core_engine.ANOMALY_MODEL_LOADED,
        nlp_model_loaded=core_engine.NLP_MODEL_LOADED,
        feature_count=len(core_engine.feature_names),
        platform=platform.system()
    )


@app.post("/predict", tags=["Inference"])
async def predict(payload: SensorPayload):
    """
    Full Predictive Maintenance Inference Pipeline.

    Accepts 20 normalized sensor features + an operator log.
    Returns a unified dispatch ticket with:
    - Anomaly Detection status
    - Failure probability and risk priority
    - Explainable AI top risk factors
    - Root cause diagnosis (BERT + TF-IDF)
    - Actionable maintenance plan with parts list
    """
    try:
        # Convert Pydantic payload to dictionary of raw sensors
        raw_sensors = {
            "torque_nm": payload.torque_nm,
            "spindle_speed_rpm": payload.spindle_speed_rpm,
            "tool_wear_min": payload.tool_wear_min,
            "rotational_speed_rpm": payload.rotational_speed_rpm,
            "power_w": payload.power_w,
            "voltage_v": payload.voltage_v,
            "current_a": payload.current_a,
            "vibration_mm_s": payload.vibration_mm_s,
            "temperature_k": payload.temperature_k
        }

        # Run the orchestrator
        ticket = core_engine.master_orchestrator(
            sensor_telemetry=raw_sensors,
            human_operator_log=payload.operator_log,
            machine_id=payload.machine_id
        )

        # Append API contract keys at the top-level
        prob_raw = ticket.get("Telemetry_Metrics", {}).get("Failure_Probability_Raw", 0.0)
        prediction = "Failure" if prob_raw >= 0.50 else "No Failure"
        confidence = float(prob_raw)
        
        failure_type = None
        if prediction == "Failure":
            failure_type = ticket.get("Root_Cause_Diagnosis", {}).get("Failure_Mode", "UNKNOWN")

        ticket["prediction"] = prediction
        ticket["confidence"] = confidence
        ticket["failure_type"] = failure_type

        return ticket


    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference pipeline error: {str(e)}"
        )


@app.get("/models/performance", tags=["Model Metrics"])
async def model_performance():
    """
    Live performance metrics for all model tiers.

    Computes AUC-ROC, F1, Precision, and Recall using saved test splits.
    """
    try:
        return core_engine.get_model_performance()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error computing model metrics: {str(e)}"
        )


# ── Startup Event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    lgbm_status = "✅ LOADED" if core_engine.FAILURE_MODEL_LOADED else "⚠️ MOCK"
    iso_status = "✅ LOADED" if core_engine.ANOMALY_MODEL_LOADED else "⚠️ MOCK"
    nlp_status = "✅ LOADED" if core_engine.NLP_MODEL_LOADED else "⚠️ DISABLED"
    feat_count = len(core_engine.feature_names)
    print("\n" + "=" * 70)
    print("  ACME INDUSTRIES — Predictive Maintenance API v2.0")
    print("=" * 70)
    print(f"  LightGBM Failure Predictor : {lgbm_status}")
    print(f"  Isolation Forest Detector  : {iso_status}")
    print(f"  NLP Classifier (BERT)      : {nlp_status}")
    print(f"  Feature Count              : {feat_count}")
    print("=" * 70)
    print("  API ready at http://0.0.0.0:8000")
    print("  Docs at http://0.0.0.0:8000/docs")
    print("=" * 70 + "\n")
