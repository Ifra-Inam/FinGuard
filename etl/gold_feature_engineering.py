from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

catalog = "finguard"
silver_schema = "silver"
gold_schema = "gold"

# Merchant categories typically associated with elevated fraud risk
RISKY_MERCHANT_CATEGORIES = ("crypto_exchange", "gambling", "money_transfer")

# GOLD FRAUD FEATURES
@dp.materialized_view(
    name=f"{catalog}.{gold_schema}.fraud_features",
    comment="Transactions with fraud labels and historically accurate features for model training and evaluation",
    table_properties={"quality": "gold"}
)

def gold_fraud_features():
    transactions = spark.read.table(f"{catalog}.{silver_schema}.transactions").drop("_rescued_data", "_ingestion_timestamp", "_source_file", "_release_id")

    customers = spark.read.table(f"{catalog}.{silver_schema}.customers").select(
        "customer_id",
        "account_open_date",
        "customer_age",
        "account_type",
        "home_country",
        "risk_segment",
        "average_monthly_spending",
        "average_transaction_amount",
        "typical_transaction_country",
        "account_status",
    )

    # One label per transaction: any confirmed investigation is fraudulent
    labels = spark.read.table(f"{catalog}.{silver_schema}.fraud_investigations").groupBy("transaction_id").agg(F.max("is_fraud").alias("is_fraud"))

    # These windows only use information available before the transaction timestamp
    ts_long = F.col("transaction_timestamp").cast("long") # Converts to Unix timestamp
    w_history = Window.partitionBy("customer_id").orderBy("transaction_timestamp") # Orders each customer's transactions chronologically
    w_1h = Window.partitionBy("customer_id").orderBy(ts_long).rangeBetween(-3600, -1) # 1 hour = 3600 seconds
    w_24h = Window.partitionBy("customer_id").orderBy(ts_long).rangeBetween(-86400, -1) # 24 hours = 86400 seconds
    w_7d = Window.partitionBy("customer_id").orderBy(ts_long).rangeBetween(-604800, -1) # 7 days = 604800 seconds
    w_7d_currency = Window.partitionBy("customer_id", "currency").orderBy(ts_long).rangeBetween(-604800, -1)
    w_merchant = Window.partitionBy("customer_id", "merchant_id").orderBy("transaction_timestamp")

    enriched_transactions = transactions.select(
        "*",
        F.hour("transaction_timestamp").alias("hour_of_day"),
        F.dayofweek("transaction_timestamp").alias("day_of_week"),
        F.dayofweek("transaction_timestamp").isin(1, 7).alias("is_weekend"),

        (ts_long - F.lag(ts_long).over(w_history)).alias("time_since_prev_transaction_seconds"),

        F.count(F.lit(1)).over(w_1h).alias("transaction_count_1h"),
        F.count(F.lit(1)).over(w_24h).alias("transaction_count_24h"),
        F.count(F.lit(1)).over(w_7d).alias("transaction_count_7d"),

        F.size(F.collect_set("merchant_id").over(w_7d)).alias("distinct_merchants_7d"),
        F.size(F.collect_set("transaction_country").over(w_7d)).alias("distinct_countries_7d"),

        F.avg("transaction_amount").over(w_7d_currency).alias("avg_amount_7d"),

        F.lag("transaction_timestamp").over(w_merchant).isNull().alias("is_new_merchant_for_customer"),

        F.col("merchant_category").isin(*RISKY_MERCHANT_CATEGORIES).alias("risky_merchant_category")
    )

    return (
        enriched_transactions
        .join(customers, "customer_id", "left")
        .select(
            "*",
            F.datediff("transaction_timestamp", "account_open_date").alias("account_age_days"),
            F.when(F.col("average_transaction_amount") > 0, F.col("transaction_amount") / F.col("average_transaction_amount")).alias("amount_vs_customer_avg_ratio"),
            F.when(F.col("avg_amount_7d") > 0, F.col("transaction_amount") / F.col("avg_amount_7d")).alias("amount_vs_7d_avg_ratio"),
            (F.col("transaction_country") != F.col("typical_transaction_country")).alias("is_country_atypical")
        )
        .join(labels, "transaction_id", "left")
    )