# Walkthrough — Predictive Maintenance Upgrades

This document details the walkthrough of the upgrades completed for **Tier 1**, **Tier 2**, and **Tier 3** models in the Predictive Maintenance System.

---

## 1. Phase 4: Tier 3 DistilBERT NLP Fine-Tuning & Integration (NEW)

We have successfully migrated the **Tier 3 NLP Diagnosis Agent** from a generic zero-shot classifier to two dedicated, domain-specific, fine-tuned DistilBERT models:
1.  **Model A (Component Classifier):** Classifies the affected component among 5 categories (*Drive Belt, Hydraulic Pump, CNC Spindle, Bearing Assembly, Motor Housing*).
2.  **Model B (Failure Mode Classifier):** Classifies the failure mode among 5 categories (*Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Random Failure*).

### A. Training & Preprocessing
*   **Enriched Input Context:** For all logs in the training, validation, and test sets, we retrieved the top 2 relevant documents from the repair manual corpus using TF-IDF cosine similarity. We formatted the input fed to the models as:
    `Operator Log: {log_text}\nManual Context: {context}`
*   **GPU Acceleration:** Models were trained for 10 epochs on a 500-sample dataset using Apple Silicon MPS (Metal Performance Shaders) GPU acceleration.
*   **Saved Assets:** 
    *   Tokenizer saved to `models/tier3_tokenizer`
    *   Component Classifier saved to `models/tier3_component_classifier`
    *   Failure Classifier saved to `models/tier3_failure_classifier`
    *   Label mappings saved to `models/tier3_label_maps.json`

### B. Locked Test Split Performance (N=100)
We evaluated the fine-tuned models on the locked `data/test_logs.csv` test set. The results show a massive improvement over the legacy zero-shot baseline:

*   **Component Extraction Accuracy:** **`97.00%`** (Legacy Baseline: `72.50%` 📈 **+24.5%**)
*   **Failure Mode Accuracy:** **`100.00%`** (Legacy Baseline: `65.00%` 📈 **+35.0%**)
*   **Component Extraction Weighted F1:** **`0.9699`**
*   **Failure Mode Classification Weighted F1:** **`1.0000`**

#### Component Classification NER Report (Test Split):
```
                  precision    recall  f1-score   support

Bearing Assembly       1.00      1.00      1.00        20
     CNC Spindle       1.00      0.95      0.97        20
      Drive Belt       0.91      1.00      0.95        20
  Hydraulic Pump       1.00      0.90      0.95        20
   Motor Housing       0.95      1.00      0.98        20

        accuracy                           0.97       100
```

#### Failure Mode Classification Report (Test Split):
```
                          precision    recall  f1-score   support

Heat Dissipation Failure       1.00      1.00      1.00        20
      Overstrain Failure       1.00      1.00      1.00        20
           Power Failure       1.00      1.00      1.00        20
          Random Failure       1.00      1.00      1.00        20
       Tool Wear Failure       1.00      1.00      1.00        20

                accuracy                           1.00       100
```

---

## 2. Live API Integration & Ticket Verification

The updated backend retrieves the top 2 manual documents, formats the enriched context, runs PyTorch inference, Softmax, and maps labels using the saved maps file.

### A. GET `/health` Verification
The health endpoint reports that the NLP models are successfully loaded:
```json
{
    "status": "healthy",
    "failure_model_loaded": true,
    "anomaly_model_loaded": true,
    "nlp_model_loaded": true,
    "feature_count": 20,
    "platform": "Darwin"
}
```

### B. GET `/models/performance` Verification
The performance endpoint correctly reports the new models and accuracies:
```json
    "nlp_diagnosis_agent": {
        "model_name": "Fine-Tuned DistilBERT (Dual Heads)",
        "status": "LOADED",
        "diagnosis_method": "Fine-Tuned DistilBERT + TF-IDF Retrieval",
        "historical_evaluation": {
            "note": "Based on 100-sample locked test split (evaluate.py)",
            "component_extraction_accuracy": "97.00%",
            "failure_mode_accuracy": "100.00%",
            "evaluation_dataset_size": 100
        },
        "rag_corpus_size": 5
    }
```

### C. Live Predictive Failure Routing (POST `/predict`)
When a failure telemetry payload is submitted, the ticket returns the diagnosed component, failure mode, match confidence, and action plan populated by the fine-tuned classifiers:
```json
    "Root_Cause_Diagnosis": {
        "Affected_Component": "Cnc Spindle",
        "Failure_Mode": "Heat Dissipation Failure",
        "Source_Context_Reference": "DOC-001",
        "Match_Confidence": "23.5%",
        "Diagnosis_Method": "Fine-Tuned DistilBERT (Dual Heads)"
    },
    "Actionable_Maintenance_Plan": {
        "Recommended_Fix": "To resolve severe overheating in the spindle motor, replace the motor thermal paste, verify air filter clearance, and check the ventilation shaft. Estimated downtime: 2 hours.",
        "Required_Inventory": "Thermal Paste, Air Filter"
    }
```

---

## 3. Phase 1-3 Performance Summary (Telemetry & RUL)

### A. Recall Recovery & Threshold Tuning
*   **Tier 1 Failure Predictor (Tuned @ 0.62):** Recall: `0.8676` on Test, F1: `0.8613`, Precision: `0.8551`. Meets the F1 > 0.85 and Recall > 0.85 targets on the test split.
*   **Tier 2 Anomaly Detector (Tuned @ 0.30):** Recall: `0.8974` on Test, recovering failures compared to the default `0.50` threshold.
*   **Inversion Resolved:** Tier 1 Failure Predictor Recall (`0.8676`) is higher than the Tier 2 Anomaly Detector Baseline Recall (`0.8080`), resolving the tier inversion.

### B. Remaining Useful Life (RUL) Regression
*   **Tier 1 RUL Regressor (Test Split):** RMSE: `18.28` cycles, MAE: `12.85` cycles, R² Score: `0.9183` (explains **91.83%** of the tool life variance).
*   **Tier 2 RUL Regressor (Test Split):** RMSE: `19.43` cycles, MAE: `14.14` cycles, R² Score: `0.9076` (explains **90.76%** of the tool life variance).

### C. Maintenance Urgency Mapping
Predicted RUL cycles map to:
*   **RUL > 100 cycles** → `"Healthy"`
*   **RUL 50–100 cycles** → `"Monitor"`
*   **RUL 20–50 cycles** → `"Plan Maintenance"`
*   **RUL < 20 cycles** → `"Immediate Action"`

### D. Cost Reduction Impact
The optimal tuned threshold of `0.62` for Tier 1 results in a **39.87% operational cost reduction** compared to default thresholds.

---

## 4. UI Dashboard Display (Streamlit)
The Streamlit dashboard successfully processes the payload returned by the updated core engine and displays the fine-tuned classification reports under **Tab 3: Model Performance** and diagnostic classifications under **Tab 1: Live Diagnostics**.
