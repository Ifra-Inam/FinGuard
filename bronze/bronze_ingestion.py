import dlt
from pyspark.sql import functions as F

BASE_PATH = "/Volumes/finguard/landing/raw_files"

def ingest_to_bronze(dataset_name):

    source_path = f"{BASE_PATH}/{dataset_name}"

    df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("header", "true")
        .load("source_path")
    )

    return (
        df 
        .withColumn("_ingestion_timestamp", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_release_id", F.regexp_extract(F.col("_metadata.file_path"), r"/(day_\d+)_"), 1)
    )

@dlt.table(name="transactions", table_properties={"quality": "bronze"})
def bronze_transactions():
    return ingest_to_bronze("transactions")

@dlt.table(name="customers", table_properties={"quality": "bronze"})
def bronze_customers():
    return ingest_to_bronze("customers")

@dlt.table(name="fraud_investigations", table_properties={"quality": "bronze"})
def bronze_fraud_investigations():
    return ingest_to_bronze("fraud_investigations")