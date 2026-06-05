# ACME Industries — Predictive Maintenance (Phase 1)
## Databricks Connect + VS Code Setup

---

## Folder structure

```
phase1_databricks/
├── .env                      ← your credentials (never commit)
├── .env.template             ← copy this to .env
├── .gitignore
├── README.md
├── setup/
│   ├── requirements.txt
│   ├── setup_databricks_connect.py   ← run once
│   └── spark_session.py              ← shared session helper
└── notebooks/
    ├── 00_setup.py
    ├── 01_data_ingestion_cleaning.py
    ├── 02_feature_engineering.py
    └── 03_eda.py
```

---

## One-time setup (5 steps)

### Step 1 — Open in VS Code
```bash
cd phase1_databricks
code .
```

### Step 2 — Create virtual environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 3 — Fill in credentials
```bash
cp setup/.env.template .env
# Fill in DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_CLUSTER_ID
```

| Value | Where to find it |
|---|---|
| `DATABRICKS_HOST` | Databricks UI → top-left URL bar |
| `DATABRICKS_TOKEN` | User icon → Settings → Access Tokens → Generate |
| `DATABRICKS_CLUSTER_ID` | Compute → your cluster → Configuration → Tags → ClusterId |

### Step 4 — Run setup script
```bash
python setup/setup_databricks_connect.py
```

### Step 5 — Install VS Code Databricks extension
Search **"Databricks"** by Databricks Inc. in Extensions panel.

---

## Running the notebooks

```bash
python notebooks/00_setup.py
python notebooks/01_data_ingestion_cleaning.py
python notebooks/02_feature_engineering.py
python notebooks/03_eda.py
```

Or import the `.py` files directly into Databricks UI (File → Import).

---

## Cluster requirements

| Setting | Value |
|---|---|
| Databricks Runtime | 14.3 LTS ML |
| Node type | Standard_DS3_v2 (14 GB, 4 cores) |
| databricks-connect | 14.3.* |# Predictive-Maintenance-AI-ML-System-ACME-Industries
