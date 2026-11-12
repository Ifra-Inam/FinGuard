# Landing Layer

## Overview

The landing layer is the entry point for the FinGuard transaction data pipeline. It stores raw transaction, customer, and fraud investigation datasets before they are processed and converted into Bronze Delta tables.

Raw files are stored in a Unity Catalog managed Volume. The original CSV files are preserved without applying transformations.

## Storage Location

**Unity Catalog Volume:**

```text
/Volumes/finguard/landing/raw_files/
```

## Directory Structure

```text
raw_files/
├── initial/
│   ├── transactions/
│   ├── customers/
│   └── fraud_investigations/
│
└── scheduled_releases/
    ├── day_3/
    │   ├── transactions/
    │   └── customers/
    ├── day_5/
    │   └── fraud_investigations/
    ├── day_6/
    │   └── transactions/
    └── day_8/
        └── fraud_investigations/
```

## Datasets

| Dataset                              | Description                                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| Initial Transactions                 | Initial transaction batches containing known fraud labels, used for initial model training. |
| Initial Customer Profiles            | Customer and account information associated with the initial transaction records.           |
| Initial Fraud Investigation Outcomes | Investigation results containing confirmed fraud labels and investigation details.          |
| Incoming Transactions                | New transaction batches without known fraud labels at ingestion time.                       |
| New Customer Profiles                | Newly added or updated customer information.                                                |
| Fraud Investigation Outcomes         | Delayed investigation results associated with previously processed transactions.            |

## Data Arrival Simulation

The project simulates a banking transaction lifecycle through scheduled data releases:

| Release | Incoming data                                                              |
| ------- | -------------------------------------------------------------------------- |
| Day 0   | Initial transaction batches, customer profiles, and investigation outcomes |
| Day 3   | New transaction batches and customer profiles                              |
| Day 5   | Delayed fraud investigation outcomes                                       |
| Day 6   | Additional transaction batches                                             |
| Day 8   | Additional fraud investigation outcomes                                    |

## Data Handling

- Raw CSV files are preserved in their original form.
- Incoming transactions do not contain actual fraud labels at prediction time.
- Fraud investigation outcomes are stored separately to simulate delayed ground-truth labels.
- Each incoming release is organized by arrival date to support incremental ingestion.
- Raw files are processed into Bronze Delta tables by the ingestion pipeline.
