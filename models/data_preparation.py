from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder

df = spark.read.table("finguard.gold.fraud_features")

df = df.filter(F.col("is_fraud").isNotNull())

# Boolean conversion
boolean_features = ["is_international", "is_weekend", "is_new_merchant_for_customer", "risky_merchant_category", "is_country_atypical", "is_fraud"]

for feature in boolean_features:
    df = df.withColumn(feature, F.col(feature).cast("integer"))

# Ordinal encoding
df = df.withColumn(
    "risk_segment_encoded",
    F.when(F.col("risk_segment") == "low", 0)
    .when(F.col("risk_segment") == "medium", 1)
    .when(F.col("risk_segment") == "high", 2)
    .otherwise(None)
)

# Chronological train-test split
w = Window.orderBy(F.col("transaction_timestamp").asc())
df = df.withColumn("_time_percentile", F.percent_rank().over(w)) # temporary column that stores the relative position of each row in the ordered dataset

train_df = df.filter("_time_percentile <= 0.75").drop("_time_percentile")
test_df = df.filter("_time_percentile > 0.75").drop("_time_percentile")

# Numerical imputation 
numeric_features = ["transaction_amount", "hour_of_day", "day_of_week", "time_since_prev_transaction_seconds", "transaction_count_1h", "transaction_count_24h", "transaction_count_7d", "distinct_merchants_7d", "distinct_countries_7d", "avg_amount_7d", "customer_age", "average_monthly_spending", "average_transaction_amount", "account_age_days", "amount_vs_customer_avg_ratio", "amount_vs_7d_avg_ratio"]

numeric_imputer = Imputer(inputCols=numeric_features, outputCols=numeric_features, strategy="median")

numeric_imputer_model = numeric_imputer.fit(train_df)

train_df = numeric_imputer_model.transform(train_df)
test_df = numeric_imputer_model.transform(test_df)

# One-hot encoding
one_hot_features = ["account_type", "account_status", "merchant_category", "transaction_type", "payment_method", "device_type", "channel", "currency", "transaction_country", "home_country", "typical_transaction_country"]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep") for c in one_hot_features]
encoder = OneHotEncoder(inputCols=[f"{c}_index" for c in one_hot_features], outputCols=[f"{c}_encoded" for c in one_hot_features], handleInvalid="keep")

pipeline = Pipeline(stages=indexers + [encoder])

one_hot_encoding_model = pipeline.fit(train_df)

train_df = one_hot_encoding_model.transform(train_df)
test_df = one_hot_encoding_model.transform(test_df)
