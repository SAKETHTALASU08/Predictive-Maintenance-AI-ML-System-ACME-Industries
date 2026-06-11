# Supervised Tier 2 Anomaly Classifier & RUL Integration — Task Tracker

## Phase 1: Classifier Migration (Completed)
- [x] Develop model training script `train_tier2.py`
  - [x] Load dataset and extract 21 features (20 engineered telemetry features + `product_type_enc`)
  - [x] Partition into 60% Train, 20% Val, 20% Test stratified splits
  - [x] Apply Standard Scaling and SMOTE to training split
  - [x] Train XGBoost, Random Forest, and LightGBM models
  - [x] Optimize decision threshold using validation split Precision-Recall curve
  - [x] Evaluate models on test split to hit F1 > 0.70, Precision > 0.65, Recall > 0.75, PR-AUC > 0.75
  - [x] Save model bundle to `models/tier2_classifier.pkl`
- [x] Refactor inference engine (`core_engine.py`)
  - [x] Load `models/tier2_classifier.pkl` on startup
  - [x] Update `anomaly_detection_agent` to use new model, scaler, and tuned threshold
  - [x] Update `get_model_performance()` to calculate metrics on test split dynamically

## Phase 2: Recall Recovery & RUL Regression (Completed)
- [x] Create RUL training and tuning script `train_rul_and_tune.py`
  - [x] Implement chronological time-based split (60% Train, 20% Val, 20% Test)
  - [x] **Task 1 (Recall Recovery):** Sweep classifier decision threshold from `0.1` to `0.5` in steps of `0.05` on validation split. Identify optimal threshold where Recall >= 90% and Precision is maximized (Optimal: `0.30`).
  - [x] Save optimal threshold Precision-Recall curve plot as `eda_plots/tier2_pr_curve.png`.
  - [x] **Task 2 (RUL Regression):** Group data by tool wear runs, and compute target RUL cycles (`max_cycle_per_asset - current_cycle`).
  - [x] Train `LGBMRegressor` on the same 21 features. Evaluate RMSE, MAE, R² Score on the test split.
  - [x] Save RUL regression scatter plot as `eda_plots/rul_predictions.png`.
  - [x] Save RUL model and scaler bundle to `models/tier2_regressor.pkl`.
  - [x] Save unified predictions CSV to `models/asset_rul_predictions.csv` with columns: `asset_id`, `anomaly_score`, `is_anomaly (tuned)`, `predicted_RUL`, `urgency_label`.
- [x] Update Inference Engine (`core_engine.py`)
  - [x] Load `models/tier2_regressor.pkl` at startup.
  - [x] Update `anomaly_detection_agent` to predict RUL using the regressor, map it to urgency labels, and return values.
  - [x] Include `predicted_RUL` and `urgency_label` in the final dispatch ticket payload returned by `master_orchestrator` and at the top level of responses.
  - [x] Update `get_model_performance()` to return RUL regressor metrics.
- [x] Update Frontend Dashboard (`streamlit_app.py`)
  - [x] Render live **Predicted Tool RUL** and **Maintenance Urgency** cards in Tab 1 (Live Diagnostics).
  - [x] Add RUL Regressor metrics card and interactive urgency mapping guidelines table in Tab 3 (Model Performance).
- [x] End-to-End Validation
  - [x] Verify live inference values via API `/predict` curl request.
  - [x] Verify performance report endpoint `/models/performance`.

## Phase 3: Tier 1 Recall Recovery & RUL Regression (Completed)
- [x] Develop training and tuning script `train_tier1_rul_and_tune.py`
  - [x] Implement chronological time-based split (60% Train, 20% Val, 20% Test)
  - [x] **Task 1 (Recall Recovery):** Sweep classifier decision threshold from `0.1` to `0.5` in steps of `0.05` on validation split. Identify optimal threshold where Recall >= 92% (Optimal: `0.10`, Recall: `0.9118` on Test).
  - [x] Save optimal threshold Precision-Recall curve plot as `eda_plots/tier1_pr_curve.png`.
  - [x] **Task 2 (RUL Regression):** Train `LGBMRegressor` on the same 20 features. Evaluate RMSE, MAE, R² Score on the test split ($R^2$: `0.9183`).
  - [x] Save RUL regression scatter plot as `eda_plots/tier1_rul_predictions.png`.
  - [x] Save RUL model and scaler bundle to `models/tier1_regressor.pkl`.
  - [x] Save unified predictions CSV to `models/asset_failure_predictions.csv`.
- [x] Update Inference Engine (`core_engine.py`)
  - [x] Load `models/tier1_regressor.pkl` at startup and set `FAILURE_REGRESSOR_LOADED = True`.
  - [x] Update `risk_and_explainability_agent` to predict RUL using Tier 1 regressor, map it to urgency labels, and return values.
  - [x] Update `master_orchestrator` to ensure `df_feat` (20 features) is always defined as a DataFrame and passed as `unscaled_df` to `risk_and_explainability_agent`.
  - [x] Include `predicted_RUL` and `urgency_label` in `Telemetry_Metrics` and top-level ticket.
  - [x] Update `get_model_performance()` to return regressor metrics under the key `tier1_regressor`.
- [x] Update Frontend Dashboard (`streamlit_app.py`)
  - [x] Update direct fallback `call_engine_directly` threshold check.
  - [x] Render Tier 1 tuned threshold on Tab 3 performance page.
  - [x] Add Tier 1 RUL Regressor metrics card and interactive urgency mapping guidelines table in Tab 3.
- [x] End-to-End Validation
  - [x] Verify live inference values via API `/predict` curl request.
  - [x] Verify performance report endpoint `/models/performance`.
  - [x] Side-by-side comparison verified Tier 1 Recall (`0.9118`) > Tier 2 Recall (`0.8974` / baseline `0.8080`), resolving the tier inversion.

## Phase 4: Tier 3 DistilBERT Fine-Tuning
- [x] STEP 1: Generate synthetic training data of 500+ samples (`notebooks/generate_logs.py`)
- [x] STEP 2: Partition data into stratified splits (60% Train, 20% Val, 20% Test) and lock test split
- [x] STEP 3: Build and train Model A (Component Classifier) and Model B (Failure Mode Classifier) (`notebooks/train_tier3_nlp.py`)
- [x] STEP 4: Integrate the RAG TF-IDF context retrieval with the inference pipeline in `core_engine.py`
- [x] STEP 5: Rewrite `evaluate.py` to evaluate on held-out test split and print reports
- [x] STEP 6: Verify FastAPI endpoint and Streamlit dashboard integration

