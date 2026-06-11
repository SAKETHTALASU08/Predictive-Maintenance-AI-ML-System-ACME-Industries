# Fine-Tuned DistilBERT Transition for Tier 3 NLP Diagnosis Agent

This plan outlines the replacement of the generic zero-shot classifier (`cross-encoder/nli-distilroberta-base`) in Tier 3 with two domain-specific, fine-tuned DistilBERT models. This transition aims to improve component extraction accuracy from ~72.5% to 88%+, and failure mode classification accuracy from ~65.0% to 82%+ using a generated training dataset of 500+ maintenance logs.

---

## User Review Required

> [!IMPORTANT]
> 1. **Model Architecture Transition:** We are replacing the generic Zero-Shot classification approach with two dedicated `DistilBertForSequenceClassification` models (Model A for Components, Model B for Failure Modes).
> 2. **Inference Input Enrichment:** The RAG TF-IDF retrieval context from the 5 OEM manual documents will be prepended to the operator log during training and inference to provide richer signal to the classifiers.
> 3. **FastAPI Contract Integrity:** The schema and JSON payload structure returned by the backend REST API (`/predict` and `/models/performance`) will remain unchanged, preserving compatibility with the Streamlit frontend.

---

## Open Questions

> [!NOTE]
> No major blocking open questions. We will use the local Apple Silicon MPS (Metal Performance Shaders) GPU acceleration to speed up the DistilBERT training loops.

---

## Proposed Changes

### Data Layer

#### [NEW] [generate_logs.py](file:///Users/sakethtalasu/Downloads/Predictive_Analysis/notebooks/generate_logs.py)
* Create a script to generate `data/maintenance_logs.csv` containing exactly 500 samples (5 components × 5 failure modes × 20 variations).
* Variations will cover standard technical descriptions, colloquial terms, operator typos, terse logs, and multi-symptom descriptions.
* Perform stratified splitting to create:
  * Training Split: 300 samples (60%)
  * Validation Split: 100 samples (20%)
  * Test Split: 100 samples (20%)
* Save the split datasets (`train_logs.csv`, `val_logs.csv`, `test_logs.csv`) under `data/` for auditability and lock the test split immediately.

---

### Model Training & Tuning

#### [NEW] [train_tier3_nlp.py](file:///Users/sakethtalasu/Downloads/Predictive_Analysis/notebooks/train_tier3_nlp.py)
* Implement PyTorch/HuggingFace dataset loader and tokenize inputs using DistilBERT tokenizer with a max length of 256 to support the combined log and TF-IDF retrieved context.
* Load and pre-retrieve manual context via TF-IDF for all logs before tokenization so the model learns from the enriched representation:
  `enriched_input = f"Operator Log: {log}\nManual Context: {context}"`
* Train two separate models using HuggingFace `Trainer`:
  * **Model A:** Component Classifier (5 output labels: CNC Spindle, Hydraulic Pump, Drive Belt, Bearing Assembly, Motor Housing).
  * **Model B:** Failure Mode Classifier (5 output labels: Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Random Failure).
* Configure training args: 10 epochs, batch size 16, learning rate 2e-5, weight decay 0.01, evaluating and saving the best model based on weighted F1.
* Save the models, tokenizer, and label mappings under `models/` as requested:
  * `models/tier3_component_classifier/`
  * `models/tier3_failure_classifier/`
  * `models/tier3_tokenizer/`
  * `models/tier3_label_maps.json`

---

### Backend & Evaluation Integration

#### [MODIFY] [core_engine.py](file:///Users/sakethtalasu/Downloads/Predictive_Analysis/notebooks/core_engine.py)
* Replace the HuggingFace zero-shot pipeline loading block with a custom loader for the fine-tuned DistilBERT models, tokenizer, and label mapping JSON.
* Update `retrieval_augmented_diagnosis_agent` to:
  1. Retrieve relevant manual context via TF-IDF (`top_k=2`) from the `repair_corpus`.
  2. Format input as: `Operator Log: {operator_log}\nManual Context: {context}`.
  3. Run predictions using both classifiers and calculate Softmax probabilities.
  4. Extract diagnosed component, failure mode, confidence metrics, and return them using the existing dict structure to preserve downstream orchestration.

#### [MODIFY] [evaluate.py](file:///Users/sakethtalasu/Downloads/Predictive_Analysis/notebooks/evaluate.py)
* Rewrite `evaluate.py` to:
  1. Load the fine-tuned models from `./models/tier3_*`.
  2. Load the locked test split `data/test_logs.csv` from disk.
  3. Perform predictions using the RAG-enriched pipeline.
  4. Calculate and display overall accuracy, classification reports (Precision, Recall, F1), and confusion matrices for both tasks.
  5. Print a baseline comparison comparison against zero-shot metrics (Component: 72.5%, Failure: 65.0%).

---

## Verification Plan

### Automated Tests
* Run dataset generation script:
  `python notebooks/generate_logs.py`
* Run the training script:
  `python notebooks/train_tier3_nlp.py`
* Run the rewritten evaluation script:
  `python notebooks/evaluate.py`
* Assert that:
  * Component Accuracy >= 88%
  * Failure Mode Accuracy >= 82%
  * Weighted F1 >= 0.82
* Test the API performance endpoint and predict endpoint using curl:
  `curl -s http://127.0.0.1:8000/models/performance | python -m json.tool`
  `curl -s -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"machine_type": "M", "torque_nm": 40.0, "spindle_speed_rpm": 1500.0, "tool_wear_min": 25.0, "rotational_speed_rpm": 1500.0, "power_w": 6280.0, "voltage_v": 220.0, "current_a": 28.5, "vibration_mm_s": 1.5, "temperature_k": 300.0, "operator_log": "spindle running extremely hot, smoke coming from cnc spindle housing.", "machine_id": "M-1042"}' | python -m json.tool`

### Manual Verification
* Access the Streamlit dashboard performance tab to verify the updated metrics are displayed live.
