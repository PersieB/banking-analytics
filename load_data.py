"""
Load synthetic banking data into Supabase PostgreSQL.

This script demonstrates a simple ETL process:

    CSV files
        ↓
    pandas
        ↓
    PostgreSQL / Supabase

The script reads the generated customer and transaction datasets,
performs basic validation, and loads them into the relational
database created for the project.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is not set. "
        "Add it to your .env file."
    )


# ------------------------------------------------------------
# Create database connection
# ------------------------------------------------------------

engine = create_engine(DATABASE_URL)


# ------------------------------------------------------------
# Read CSV files
# ------------------------------------------------------------

customers = pd.read_csv("customers.csv")
transactions = pd.read_csv("transactions.csv")


print("Loaded CSV files:")
print(f"Customers: {len(customers):,}")
print(f"Transactions: {len(transactions):,}")


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

print("\nValidating data...")

if customers["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer IDs found.")

if transactions["transaction_id"].duplicated().any():
    raise ValueError("Duplicate transaction IDs found.")

if not transactions["customer_id"].isin(
    customers["customer_id"]
).all():
    raise ValueError(
        "Some transactions reference customers "
        "that do not exist."
    )

if (transactions["amount"] <= 0).any():
    raise ValueError(
        "Transactions contain non-positive amounts."
    )


print("Validation passed.")


# ------------------------------------------------------------
# Load customers
# ------------------------------------------------------------

print("\nLoading customers...")

customers.to_sql(
    "customers",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=500,
)

print("Customers loaded.")


# ------------------------------------------------------------
# Load transactions
# ------------------------------------------------------------

print("\nLoading transactions...")

transactions.to_sql(
    "transactions",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=1_000,
)

print("Transactions loaded.")


# ------------------------------------------------------------
# Verify database contents
# ------------------------------------------------------------

with engine.connect() as connection:

    customer_count = connection.execute(
        text("SELECT COUNT(*) FROM customers")
    ).scalar()

    transaction_count = connection.execute(
        text("SELECT COUNT(*) FROM transactions")
    ).scalar()


print("\nDatabase verification")
print("---------------------")
print(f"Customers in database: {customer_count:,}")
print(f"Transactions in database: {transaction_count:,}")

print("\nData loading complete.")