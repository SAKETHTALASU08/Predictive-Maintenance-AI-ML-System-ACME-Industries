# ACME Industries — Production Deployment Guide

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│                  (streamlit_app.py :8501)                    │
│                                                             │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ Sensor Input  │  │  Live Diagnostics │  │ Model Perf  │  │
│  │   Sliders     │  │  Risk + Anomaly   │  │  Metrics    │  │
│  │ Operator Log  │  │  XAI + Diagnosis  │  │  Dashboard  │  │
│  └──────┬───────┘  └──────────────────┘  └──────────────┘  │
│         │ HTTP POST /predict                                │
├─────────┼───────────────────────────────────────────────────┤
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────┐       │
│  │           FASTAPI BACKEND (api.py :8000)          │       │
│  │                                                    │       │
│  │  POST /predict    — Full inference pipeline        │       │
│  │  GET  /health     — System readiness               │       │
│  │  GET  /models/performance — Live metrics           │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │ imports                            │
│  ┌──────────────────────▼───────────────────────────┐       │
│  │        CORE ENGINE (core_engine.py)                │       │
│  │                                                    │       │
│  │  Tier 1:  LightGBM Failure Predictor               │       │
│  │  Tier 1b: Isolation Forest Anomaly Detector       │       │
│  │  Tier 2:  BERT + TF-IDF RAG Diagnosis              │       │
│  │  Tier 3:  XAI Explainability                       │       │
│  └──────────────────────────────────────────────────┘       │
│                                                             │
│  models/                                                    │
│  ├── best_failure_predictor.pkl                             │
│  ├── isolation_forest_tuned.pkl                             │
│  ├── feature_names.pkl                                      │
│  ├── X_test_sc.npy / y_test.npy                            │
│  └── ...                                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start FastAPI Backend
```bash
cd notebooks
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Verify: Open http://localhost:8000/docs for interactive Swagger UI.

### Step 3: Start Streamlit Frontend
```bash
# In a new terminal
cd notebooks
streamlit run streamlit_app.py --server.port 8501
```

Verify: Open http://localhost:8501 for the dashboard.

---

## API Quick Test

```bash
# Health Check
curl http://localhost:8000/health

# Run Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "M-1042",
    "sensor_data": [0.1, 0.2, -0.1, 0.3, 0.5, -0.2, 0.4, 0.1, 0.3, 0.8, 2.4, 0.1, -0.3, 0.5, 0.2, 0.1, 0.3, 0.2, -0.1, 3.1],
    "operator_log": "Sudden drop in pressure. Inspection reveals complete mechanical fracture affecting the drive belt."
  }'

# Model Performance
curl http://localhost:8000/models/performance
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ZERO_SHOT_CLASSIFIER` | `false` | Set to `1` or `true` to enable BERT NLP on macOS |
| `API_BASE_URL` | `http://localhost:8000` | FastAPI backend URL for Streamlit |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | Prevents Intel MKL segfaults on macOS |

---

## File Structure

```
Predictive_Analysis/
├── requirements.txt              ← Production dependencies
├── README_DEPLOYMENT.md          ← This file
├── models/                       ← Trained model artifacts
│   ├── best_failure_predictor.pkl
│   ├── isolation_forest_tuned.pkl
│   ├── feature_names.pkl
│   ├── X_test_sc.npy / y_test.npy
│   └── ...
├── notebooks/
│   ├── core_engine.py            ← [NEW] Refactored inference engine
│   ├── api.py                    ← [NEW] FastAPI REST backend
│   ├── streamlit_app.py          ← [NEW] Streamlit dashboard
│   ├── orchestrator.py           ← [ORIGINAL] Preserved
│   ├── ai_agents.py              ← [ORIGINAL] Preserved
│   ├── evaluate.py               ← [ORIGINAL] NLP evaluation suite
│   ├── failure_prediction.py     ← [ORIGINAL] Model training
│   ├── anomaly_detection.py      ← [ORIGINAL] Anomaly model training
│   ├── feature_engineering.py    ← [ORIGINAL] Feature pipeline
│   ├── pre_processing.py         ← [ORIGINAL] Data prep + SMOTE
│   ├── data_ingestion_cleaning.py← [ORIGINAL] UCI data ingestion
│   └── eda.py                    ← [ORIGINAL] Exploratory analysis
└── eda_plots/                    ← Generated visualizations
```

---

## Model Performance Summary

Performance metrics are available live via:
- **API**: `GET http://localhost:8000/models/performance`
- **Dashboard**: Tab 3 "Model Performance" in the Streamlit app

### Tier 1: LightGBM Failure Predictor
- **Task**: Binary classification (Normal vs Failure)
- **Training**: SMOTE-balanced (30% minority), 300 estimators, max_depth=6
- **Threshold**: Optimized on validation F1
- **Metrics**: AUC-ROC, F1, Precision, Recall (computed live from test split)

### Tier 2: Isolation Forest Anomaly Detector
- **Task**: Unsupervised anomaly detection
- **Training**: Normal-only samples, grid-searched contamination
- **Threshold**: Optimized on validation F1
- **Metrics**: AUC-ROC, F1, Precision, Recall (computed live from test split)

### Tier 3: NLP Diagnosis Agent
- **Model**: `cross-encoder/nli-distilroberta-base` (Zero-Shot)
- **Fallback**: TF-IDF cosine similarity (always available)
- **Evaluated**: 40-sample synthetic test suite (evaluate.py)
- **Metrics**: Component Accuracy ~72.5%, Failure Mode Accuracy ~65.0%
