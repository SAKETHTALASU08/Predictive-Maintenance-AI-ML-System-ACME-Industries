import sys
sys.path.insert(0, "setup")
from spark_session import get_spark

spark = get_spark("acme_pm")

spark.sql("CREATE DATABASE IF NOT EXISTS acme_pm")
print("✅ Database created → acme_pm")

dbs = spark.sql("SHOW DATABASES").toPandas()
print("\nAvailable databases:")
print(dbs.to_string())
