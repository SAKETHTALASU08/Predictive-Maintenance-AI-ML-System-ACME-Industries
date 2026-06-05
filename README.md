# Predictive Maintenance AI/ML System — ACME Industries

An end-to-end predictive maintenance pipeline for industrial CNC welding robots, laser cutting systems, and robotic arms. Built on **Databricks Connect + VS Code**, this system combines classical ML anomaly detection, supervised failure prediction, and an NLP-powered AI diagnostic agent stack to minimize unplanned downtime across manufacturing facilities.

---

## Architecture

```
Sensor Data (AI4I 2020 UCI)
        │
        ▼
┌─────────────────────┐
│   Phase 1: Data     │  Ingestion · Cleaning · Feature Engineering · EDA
│   (Databricks Delta)│  10,000 rows · 100 engineered features · Delta tables
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Phase 2: ML       │  Anomaly Detection + Failure Prediction
│   Models            │  Isolation Forest · LSTM Autoencoder · XGBoost · LightGBM
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Phase 3: AI       │  Diagnosis Agent · Fix Recommendation · Ticket Generation
│   Agents            │  BERT zero-shot · TF-IDF RAG · LDA topic modeling
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Orchestrator      │  Unified pipeline: sensor → risk score → diagnosis → ticket
└─────────────────────┘
```

---

## Results

| Module | Model | Metric | Score |
|---|---|---|---|
| Anomaly Detection | Isolation Forest (tuned) | AUC-ROC | 0.89 |
| Anomaly Detection | LSTM Autoencoder | AUC-ROC | 0.88 |
| Failure Prediction | LightGBM ← best | AUC-ROC | 0.982 |
| Failure Prediction | LightGBM | F1 Score | 0.828 |
| Failure Prediction | XGBoost | AUC-ROC | 0.987 |
| Diagnosis Agent | BERT zero-shot | Component Accuracy | ~80% |

---

## Tech Stack

- **Platform:** Databricks (Runtime 14.3 LTS ML) + Databricks Connect + VS Code
- **Data layer:** Apache Spark · Delta Lake
- **ML:** scikit-learn · XGBoost · LightGBM · PyTorch (LSTM Autoencoder)
- **NLP:** HuggingFace Transformers (cross-encoder/nli-distilroberta-base) · TF-IDF · LDA
- **Experiment tracking:** MLflow (Databricks-managed)
- **Imbalance handling:** SMOTE (97:3 class ratio)
- **Language:** Python 3.10

---

## Project Structure

```
Predictive_Analysis/
├── setup/
│   ├── spark_session.py           # Databricks Connect session factory
│   ├── setup_databricks_connect.py
│   └── requirements.txt
├── notebooks/
│   ├── setup.py                   # Create acme_pm database
│   ├── data_ingestion_cleaning.py # Download AI4I 2020, clean, save Delta
│   ├── feature_engineering.py     # 100 engineered features
│   ├── eda.py                     # EDA plots + correlation analysis
│   ├── pre_processing.py          # SMOTE, train/val/test splits
│   ├── anomaly_detection.py       # Isolation Forest + LSTM Autoencoder
│   ├── failure_prediction.py      # XGBoost + LightGBM
│   ├── ai_agents.py               # Wired AI diagnostic agent stack
│   ├── evaluate.py                # BERT agent evaluation (40 logs)
│   └── orchestrator.py            # End-to-end unified pipeline
├── models/                        # Saved model artifacts (git-ignored)
├── eda_plots/                     # Generated visualizations (git-ignored)
├── .env.template                  # Credentials template
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/<your-username>/predictive-maintenance-acme.git
cd predictive-maintenance-acme
python3.10 -m venv .venv && source .venv/bin/activate

# 2. Fill in credentials
cp setup/.env.template .env
# Edit .env with your DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_CLUSTER_ID

# 3. Install dependencies
pip install -r setup/requirements.txt

# 4. Run setup
python setup/setup_databricks_connect.py

# 5. Run the full pipeline in order
python notebooks/setup.py
python notebooks/data_ingestion_cleaning.py
python notebooks/feature_engineering.py
python notebooks/eda.py
python notebooks/pre_processing.py
python notebooks/anomaly_detection.py
python notebooks/failure_prediction.py
python notebooks/ai_agents.py
python notebooks/evaluate.py
python notebooks/orchestrator.py
```

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository  
10,000 rows · 14 raw features · 5 failure modes (TWF, HDF, PWF, OSF, RNF)  
Auto-downloaded via `ucimlrepo` — no manual download needed.

---

## Key Engineering Decisions

- **SMOTE** applied only to training set to fix 97:3 class imbalance
- **`power_w` dropped** from features (0.98 correlation with `torque_nm`)
- **Threshold tuned** on validation set (0.82 vs default 0.5) for LightGBM
- **Isolation Forest** used as production anomaly model (AUC ceiling ~0.89 for this dataset)
- **BERT zero-shot** used for NER without domain fine-tuning (evaluate.py measures accuracy)
- **RAG-style TF-IDF** retrieval narrows BERT candidate labels to improve diagnosis precision

---

## MLflow Experiments

| Experiment | Path |
|---|---|
| Anomaly detection | `/acme_pm_anomaly_detection` |
| Failure prediction | `/acme_pm_failure_prediction` |

View in Databricks UI → Experiments tab.

---

## License

MIT License — free to use for educational and research purposes.
