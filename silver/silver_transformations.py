import dlt 
from pyspark.sql import functions as F

catalog = "finguard"
bronze_schema = "bronze"
silver_schema = "silver"

COUNTRY_MAP = {
    "US": "US", "USA": "US", "UNITED STATES": "US",
    "GB": "GB", "UK": "GB", "GBR": "GB", "GREAT BRITAIN": "GB", "UNITED KINGDOM": "GB",
    "AU": "AU", "AUS": "AU", "AUSTRALIA": "AU",
    "BR": "BR", "BRA": "BR", "BRAZIL": "BR",
    "CA": "CA", "CAN": "CA", "CANADA": "CA",
    "DE": "DE", "DEU": "DE", "GERMANY": "DE",
    "ES": "ES", "ESP": "ES", "SPAIN": "ES",
    "FR": "FR", "FRA": "FR", "FRANCE": "FR",
    "IN": "IN", "IND": "IN", "INDIA": "IN",
    "IT": "IT", "ITA": "IT", "ITALY": "IT",
    "JP": "JP", "JPN": "JP", "JAPAN": "JP",
    "MX": "MX", "MEX": "MX", "MEXICO": "MX"
}

# Normalizes countries to ISO-2 codes; unmappable values become NULL and are dropped
def normalize_country(col_name):
    mapping = F.create_map([F.lit(x) for kv in COUNTRY_MAP.items() for x in kv])
    return mapping[F.upper(F.trim(F.regexp_replace(F.col(col_name), "\\.", "")))]

CURRENCY_MAP = {
    "USD": "USD", "$": "USD", "US$": "USD", "US DOLLAR": "USD",
    "EUR": "EUR", "€": "EUR", "EURO": "EUR",
    "GBP": "GBP", "£": "GBP", "POUND STERLING": "GBP",
    "JPY": "JPY", "¥": "JPY", "JAPANESE YEN": "JPY",
    "AUD": "AUD", "A$": "AUD", "AUSTRALIAN DOLLAR": "AUD",
    "BRL": "BRL", "R$": "BRL", "BRAZILIAN REAL": "BRL",
    "CAD": "CAD", "CA$": "CAD", "CANADIAN DOLLAR": "CAD",
    "INR": "INR", "₹": "INR", "INDIAN RUPEE": "INR",
    "MXN": "MXN", "MX$": "MXN", "MEXICAN PESO": "MXN"
}

# Normalizes currency symbols and names to ISO-4217 codes
def normalize_currency(col_name): 
    mapping = F.create_map([F.lit(x) for kv in CURRENCY_MAP.items() for x in kv])
    return mapping[F.upper(F.trim(F.col(col_name)))]

# Valid values for transactions categorical columns after trim and lowercase normalization
VALID_MERCHANT_CATEGORIES = ("atm_cash", "clothing", "crypto_exchange", "electronics", "entertainment",
                             "fuel", "gambling", "groceries", "healthcare", "home_improvement",
                             "insurance", "luxury_goods", "money_transfer", "restaurants",
                             "subscriptions", "telecom", "travel", "utilities")
VALID_TRANSACTION_TYPES = ("bill_payment", "cash_advance", "purchase", "refund", "transfer", "withdrawal")
VALID_PAYMENT_METHODS = ("bank_transfer", "buy_now_pay_later", "credit_card", "debit_card", "digital_wallet", "direct_debit")
VALID_DEVICE_TYPES = ("atm", "desktop", "mobile", "other", "pos_terminal", "tablet")
VALID_CHANNELS = ("atm", "in_store", "mobile_banking", "online", "phone")

# Valid values for fraud_investigations categorical columns after trim and lowercase normalization
VALID_FRAUD_CATEGORIES = ("account_takeover", "card_testing", "identity_theft", "phishing_scam", "stolen_card")
VALID_INVESTIGATION_STATUSES = ("confirmed", "dismissed")
VALID_LABEL_SOURCES = ("auto_clear", "bank_review", "chargeback", "customer_report")

# Converts a tuple of values into a SQL IN list that can be used in data quality expectations
def sql_in_list(values):
    return ", ".join(f"'{v}'" for v in values)

# Trims whitespace, replaces internal spaces with underscores, converts empty strings to NULL, and lowercases values
def clean_text(col_name):
    return F.lower(F.nullif(F.regexp_replace(F.trim(F.col(col_name)), "\\s+", "_"), F.lit("")))

# ------ CUSTOMERS ------

# SILVER CUSTOMERS
@dlt.table(
    name=f"{catalog}.{silver_schema}.customers",
    comment="Cleaned customers: normalized countries, categories, and casing. Validated by quality expectations.",
    table_properties={"quality": "silver"}
)

# Invalid primary keys stop the pipeline instead of polluting silver
@dlt.expect_or_fail("valid_customer_id", "customer_id RLIKE '^CUST[0-9]+$'")
@dlt.expect_or_fail("valid_account_id", "account_id RLIKE '^ACC[0-9]+$'")

# Records violating any of these checks are dropped and logged in pipeline metrics
@dlt.expect_all_or_drop({
    "customer_age_is_present": "customer_age IS NOT NULL AND customer_age BETWEEN 18 AND 100",
    "home_country_is_valid_iso2": "home_country IS NOT NULL",
    "typical_transaction_country_is_valid_iso2": "typical_transaction_country IS NOT NULL",
    "risk_segment_is_valid": "risk_segment IN ('low', 'medium', 'high')",
    "account_type_is_valid": "account_type IN ('checking', 'credit', 'savings', 'premium')",
    "account_status_is_valid": "account_status IN ('active', 'closed', 'suspended')",
    "monthly_spending_is_non_negative": "average_monthly_spending >= 0",
    "average_transaction_amount_is_positive": "average_transaction_amount > 0",
    "account_open_date_is_not_in_future": "account_open_date IS NOT NULL AND account_open_date <= current_date()"
})

def silver_customers():
    return (
        spark.read.table(f"{catalog}.{bronze_schema}.customers")
        .withColumn("account_type", F.lower(F.trim("account_type")))
        .withColumn("risk_segment", F.lower(F.trim("risk_segment")))
        .withColumn("account_status", F.lower(F.trim("account_status")))
        .withColumn("home_country", normalize_country("home_country"))
        .withColumn("typical_transaction_country", normalize_country("typical_transaction_country"))
    )

# ------ TRANSACTIONS ------

# SILVER TRANSACTIONS
@dlt.table(
    name=f"{catalog}.{silver_schema}.transactions",
    comment="Cleaned transactions: deduplicated, normalized currency/country/categories/boolean flags. Validated by quality expectations.",
    table_properties={"quality": "silver"}
)

# Invalid keys stop the pipeline instead of polluting silver
@dlt.expect_or_fail("valid_transaction_id", "transaction_id RLIKE '^TXN[0-9]{9}$'")
@dlt.expect_or_fail("valid_customer_id", "customer_id RLIKE '^CUST[0-9]+$'")
@dlt.expect_or_fail("valid_account_id", "account_id RLIKE '^ACC[0-9]+$'")
@dlt.expect_or_fail("valid_merchant_id", "merchant_id RLIKE '^MER[0-9]+$'")

# Records violating any of these checks are dropped and logged in pipeline metrics
# Missing descriptor values are kept as 'unknown'
@dlt.expect_all_or_drop({
    "amount_is_present_and_positive": "transaction_amount IS NOT NULL AND transaction_amount > 0",
    "timestamp_is_present_and_not_in_future": "transaction_timestamp IS NOT NULL AND transaction_timestamp <= current_timestamp()",
    "currency_is_valid_iso4217": "currency IS NOT NULL",
    "transaction_country_is_valid_iso2": "transaction_country IS NOT NULL",
    "is_international_is_valid_boolean": "is_international IS NOT NULL",
    "transaction_type_is_valid": f"transaction_type IN ({sql_in_list(VALID_TRANSACTION_TYPES)})",
    "merchant_category_is_valid": f"merchant_category IN ({sql_in_list(VALID_MERCHANT_CATEGORIES)}, 'unknown')",
    "payment_method_is_valid": f"payment_method IN ({sql_in_list(VALID_PAYMENT_METHODS)}, 'unknown')",
    "device_type_is_valid": f"device_type IN ({sql_in_list(VALID_DEVICE_TYPES)}, 'unknown')",
    "channel_is_valid": f"channel IN ({sql_in_list(VALID_CHANNELS)}, 'unknown')"
})

def silver_transactions():
    return (
        spark.read.table(f"{catalog}.{bronze_schema}.transactions")
        .dropDuplicates(["transaction_id"])
        .withColumn("merchant_category", F.coalesce(clean_text("merchant_category"), F.lit("unknown")))
        .withColumn("transaction_type", clean_text("transaction_type"))
        .withColumn("payment_method", F.coalesce(clean_text("payment_method"), F.lit("unknown")))
        .withColumn("device_type", F.coalesce(clean_text("device_type"), F.lit("unknown")))
        .withColumn("channel", F.coalesce(clean_text("channel"), F.lit("unknown")))
        .withColumn("currency", normalize_currency("currency"))
        .withColumn("transaction_country", normalize_country("transaction_country"))
        .withColumn("is_international", 
                    F.when(F.upper(F.trim(F.col("is_international"))).isin("TRUE", "1"), F.lit(True))
                    .when(F.upper(F.trim(F.col("is_international"))).isin("FALSE", "0"), F.lit(False)),
        )
    )

# ------ FRAUD INVESTIGATIONS ------

# SILVER FRAUD TRANSACTIONS
@dlt.table(
    name=f"{catalog}.{silver_schema}.fraud_investigations",
    comment="Cleaned fraud investegations: boolean fraud flag, 'none' for dismissed categories. Validated by quality expectations.",
    table_properties={"quality": "silver"}
)

# Invalid keys stop the pipeline instead of polluting silver
@dlt.expect_or_fail("valid_investigation_id", "investigaition_id RLIKE '^INV[0-9]{9}$'")
@dlt.expect_or_fail("valid_transaction_id", "transaction_id RLIKE '^TXN[0-9]{9}$'")

# Records violating any of these checks are dropped and logged in pipeline metrics
@dlt.expect_all_or_drop({
    "fraud_label_is_valid": "is_fraud IS NOT NULL",
    "investigation_status_is_valid": f"investigation_status IN ({sql_in_list(VALID_INVESTIGATION_STATUSES)})",
    "label_source_is_valid": f"label_source IN ({sql_in_list(VALID_LABEL_SOURCES)})",
    "fraud_category_is_valid": f"fraud_category IN ({sql_in_list(VALID_FRAUD_CATEGORIES)}, 'none')",
    "investigation_timestamp_is_present_and_not_in_future": "investigation_timestamp IS NOT NULL AND investigation_timestamp <= current_timestamp()",
    "label_timestamp_is_present_and_after_investigation": "label_timestamp IS NOT NULL AND label_timestamp >= investigation_timestamp",
    "fraud_label_matches_status": "(is_fraud AND investigation_status = 'confirmed' AND fraud_category != 'none') OR (NOT is_fraud AND investigation_status = 'dismissed' AND fraud_category = 'none')"
})

def silver_fraud_investigations():
    return (
        spark.read.table(f"{catalog}.{bronze_schema}.fraud_investigations")
        .dropDuplicates(["investigation_id"])
        .withColumn("investigation_status", clean_text("investigation_status"))
        .withColumn("fraud_category", F.coalesce(clean_text("fraud_category"), F.lit("none")))
        .withColumn("label_source", clean_text("label_source"))
        .withColumn("is_fraud", F.when(F.col("is_fraud") == 1, F.lit(True)).when(F.col("is_fraud") == 0, F.lit(False)))
    )