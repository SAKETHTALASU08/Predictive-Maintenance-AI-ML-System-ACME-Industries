import os
from pathlib import Path

def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

def get_spark(app_name="acme_pm", mode="auto"):
    _load_env()
    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.getOrCreate()
    print(f"[spark_session] Connected to Databricks")
    return spark
