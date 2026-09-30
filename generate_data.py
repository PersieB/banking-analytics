"""
Synthetic Banking Transaction Data Generator
=============================================

This script creates a completely synthetic banking dataset for a
customer transaction analytics project.

The project is designed to demonstrate a simple data science workflow:

    Synthetic data
          ↓
    Database / SQL
          ↓
    Customer analytics
          ↓
    Feature engineering
          ↓
    Predictive modelling

The dataset contains two main entities:

1. Customers
   - Fictional customer identity
   - Fictional home planet
   - Customer segment
   - Account tenure
   - Behavioural profile

2. Transactions
   - Transaction date
   - Transaction amount
   - Banking channel
   - Banking product
   - Customer who made the transaction

The customer names and planets are intentionally fictional and
space-inspired. No real customer information is used.

IMPORTANT DESIGN DECISION
-------------------------
Customers are assigned one of four behavioural profiles:

    stable      → relatively consistent transaction activity
    growing     → transaction activity increases over time
    declining   → transaction activity decreases over time
    occasional  → relatively low and irregular activity

These profiles are not included in the transaction data as a feature
that the future ML model will use. Instead, they influence how the
synthetic transaction history is generated.

This gives us realistic behavioural patterns that we can later use
to create features such as:

    - transactions in the last 30 days
    - transactions in the previous 30 days
    - days since last transaction
    - average transaction value
    - number of channels used

Those features will eventually be used to demonstrate a simple
customer inactivity prediction model.

Outputs
-------
customers.csv
transactions.csv
"""


# ============================================================
# Imports
# ============================================================

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# Reproducibility
# ============================================================
#
# Setting random seeds means that running this script multiple times
# produces the same synthetic dataset.
#
# This is useful for a portfolio project because:
#
#   1. We can reproduce our results.
#   2. Someone reviewing the project can generate the same data.
#   3. Model results will remain consistent between runs.
#

random.seed(42)
np.random.seed(42)


# ============================================================
# Configuration
# ============================================================

# Number of fictional customers to generate.
NUM_CUSTOMERS = 2_000

# The actual number of transactions will be determined by the
# behavioural assumptions below rather than forcing exactly this
# number of rows.

# One year of transaction history.
START_DATE = datetime(2025, 10, 1)
END_DATE = datetime(2026, 9, 30)


# ============================================================
# Banking dimensions
# ============================================================

# Fictional planets used as customer locations.
#
# These are purely decorative and are not intended to represent
# real geographic information.
PLANETS = [
    "Kepler-442b",
    "Proxima",
    "Titan",
    "Europa",
    "Elysium",
]


# Customer segments.
#
# The segments allow us to introduce realistic differences in
# transaction values and behaviour.
SEGMENTS = [
    "Standard",
    "Premium",
    "Student",
    "Business",
]


# Banking channels through which transactions can occur.
CHANNELS = [
    "Mobile",
    "Online",
    "ATM",
    "Branch",
]


# Banking products associated with transactions.
PRODUCTS = [
    "Savings",
    "Current",
    "Payments",
    "Investment",
    "Loan",
]


# ============================================================
# Space-inspired customer names
# ============================================================
#
# These are deliberately unusual fictional names so that the
# dataset is clearly synthetic.
#

FIRST_NAMES = [
    "Nova",
    "Orion",
    "Lyra",
    "Vega",
    "Atlas",
    "Luna",
    "Cosmo",
    "Astra",
    "Sol",
    "Nebula",
    "Phoenix",
    "Zenith",
    "Elara",
    "Stella",
    "Cygnus",
    "Draco",
    "Aria",
    "Rigel",
    "Altair",
    "Andromeda",
    "Io",
    "Calypso",
    "Titan",
    "Aurora",
    "Quasar",
]


LAST_NAMES = [
    "Quill",
    "Vale",
    "Vex",
    "Rune",
    "Flux",
    "Orbit",
    "Ray",
    "Echo",
    "Void",
    "Star",
    "Comet",
    "Drift",
    "Pulse",
    "Cosmos",
    "Vector",
    "Zen",
    "Nexus",
    "Halo",
    "Solace",
    "Meteor",
    "Astro",
    "Spectra",
    "Lumen",
    "Gravity",
]


# ============================================================
# Behavioural profiles
# ============================================================
#
# These profiles are the foundation of the synthetic behaviour.
#
# We don't want every customer to behave identically because that
# would make the later analytics and ML model unrealistic.
#
# "stable":
#     Activity remains relatively consistent throughout the year.
#
# "growing":
#     Activity starts relatively low and gradually increases.
#
# "declining":
#     Activity starts relatively high and gradually decreases.
#
# "occasional":
#     Activity remains low and irregular.
#

BEHAVIOURS = [
    "stable",
    "growing",
    "declining",
    "occasional",
]


# The probability of assigning each behavioural profile.
#
# This is deliberately weighted rather than evenly distributed.
# Most customers are stable, while smaller groups show growing,
# declining, or occasional behaviour.
BEHAVIOUR_WEIGHTS = [
    45,
    20,
    20,
    15,
]


# Approximate monthly transaction rates before applying the
# customer segment multiplier.
#
# These values are not meant to represent a real bank.
# They simply provide sensible differences between behaviours.
BASE_MONTHLY_RATES = {
    "stable": 5.0,
    "growing": 3.0,
    "declining": 7.0,
    "occasional": 1.5,
}


# ============================================================
# Behavioural trends
# ============================================================
#
# Each behaviour has a starting activity multiplier and an ending
# activity multiplier.
#
# Example:
#
# growing:
#     starts at 0.6 × its base rate
#     ends at 1.5 × its base rate
#
# declining:
#     starts at 1.5 × its base rate
#     ends at 0.4 × its base rate
#
# This creates an actual temporal pattern in the transaction data.
#

BEHAVIOUR_TRENDS = {
    "stable": (1.0, 1.0),
    "growing": (0.6, 1.5),
    "declining": (1.5, 0.4),
    "occasional": (0.8, 0.8),
}


# ============================================================
# Customer segment configuration
# ============================================================
#
# These weights determine how frequently customers are assigned
# to each segment.
#

SEGMENT_WEIGHTS = {
    "Standard": 55,
    "Premium": 15,
    "Student": 15,
    "Business": 15,
}


# Segment multipliers influence transaction frequency.
#
# For example, Business customers are more likely to generate
# transactions than Student customers in this synthetic dataset.
#

SEGMENT_ACTIVITY_MULTIPLIERS = {
    "Standard": 1.0,
    "Premium": 1.4,
    "Student": 0.8,
    "Business": 1.6,
}


# Typical transaction amounts by customer segment.
#
# We use these as a starting point and introduce random variation
# later so that transactions do not all have the same value.
#

BASE_TRANSACTION_AMOUNTS = {
    "Standard": 250,
    "Premium": 700,
    "Student": 100,
    "Business": 1200,
}


# ============================================================
# Generate customers
# ============================================================

customers = []


for i in range(1, NUM_CUSTOMERS + 1):

    customer_id = f"C{i:05d}"

    # Create a fictional space-inspired name.
    customer_name = (
        f"{random.choice(FIRST_NAMES)} "
        f"{random.choice(LAST_NAMES)}"
    )

    # Assign a fictional home planet.
    home_planet = random.choice(PLANETS)

    # Assign a customer segment using the configured weights.
    segment = random.choices(
        list(SEGMENT_WEIGHTS.keys()),
        weights=list(SEGMENT_WEIGHTS.values()),
        k=1,
    )[0]

    # Account tenure is expressed in months.
    # This will later allow us to explore relationships between
    # customer tenure and transaction behaviour.
    tenure_months = random.randint(3, 84)

    # Assign the customer's behavioural profile.
    #
    # IMPORTANT:
    # This column is useful during data generation, but we will
    # NOT give it to the ML model later. Otherwise the model would
    # simply receive the answer instead of learning from behaviour.
    behaviour = random.choices(
        BEHAVIOURS,
        weights=BEHAVIOUR_WEIGHTS,
        k=1,
    )[0]

    customers.append(
        {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "home_planet": home_planet,
            "segment": segment,
            "tenure_months": tenure_months,
            "behaviour": behaviour,
        }
    )


# Convert the list of dictionaries into a DataFrame.
customers_df = pd.DataFrame(customers)


# ============================================================
# Generate transactions
# ============================================================
#
# Instead of generating completely random transactions, we generate
# them customer-by-customer and month-by-month.
#
# This is important because it gives us a temporal structure.
#
# Later we can ask questions such as:
#
#   "How many transactions did this customer make recently?"
#
#   "Is their activity increasing or decreasing?"
#
#   "How long has it been since their last transaction?"
#
# Those questions form the basis of our customer activity model.
#

transactions = []


# Calculate the total number of days in our data period.
date_range_days = (END_DATE - START_DATE).days


# Loop through every customer.
for customer in customers:

    customer_id = customer["customer_id"]
    behaviour = customer["behaviour"]
    segment = customer["segment"]


    # --------------------------------------------------------
    # Determine behavioural trend
    # --------------------------------------------------------
    #
    # Retrieve the starting and ending activity levels associated
    # with the customer's behavioural profile.
    #

    start_multiplier, end_multiplier = BEHAVIOUR_TRENDS[behaviour]

    base_rate = BASE_MONTHLY_RATES[behaviour]


    # --------------------------------------------------------
    # Generate activity for each month
    # --------------------------------------------------------
    #
    # We divide the year into 12 monthly periods.
    #
    # For every month we calculate how far through the year we are.
    #
    # progress = 0
    #     beginning of the period
    #
    # progress = 1
    #     end of the period
    #
    # This allows us to gradually increase or decrease activity.
    #

    for month in range(12):

        progress = month / 11


        # ----------------------------------------------------
        # Calculate the behavioural trend
        # ----------------------------------------------------
        #
        # Linear interpolation between the starting and ending
        # activity levels.
        #
        # Example for a growing customer:
        #
        # Month 1  → approximately 0.6 × base rate
        # Month 6  → approximately 1.0 × base rate
        # Month 12 → approximately 1.5 × base rate
        #

        trend_multiplier = (
            start_multiplier
            + (end_multiplier - start_multiplier) * progress
        )


        # Calculate the expected monthly activity.
        monthly_rate = base_rate * trend_multiplier


        # ----------------------------------------------------
        # Customer-specific variation
        # ----------------------------------------------------
        #
        # Two customers with the same behavioural profile should
        # not behave identically.
        #
        # For example, two "stable" customers may have different
        # levels of activity.
        #

        customer_factor = np.random.uniform(0.7, 1.3)


        # Combine behavioural activity with customer segment.
        expected_transactions = (
            monthly_rate
            * customer_factor
            * SEGMENT_ACTIVITY_MULTIPLIERS[segment]
        )


        # ----------------------------------------------------
        # Generate actual transaction count
        # ----------------------------------------------------
        #
        # We use a Poisson distribution because transaction counts
        # are discrete events.
        #
        # The expected value is our estimated monthly transaction
        # activity, while the actual number varies naturally.
        #

        transaction_count = np.random.poisson(
            max(expected_transactions, 0.1)
        )


        # ----------------------------------------------------
        # Generate individual transactions
        # ----------------------------------------------------

        for _ in range(transaction_count):

            # Calculate the approximate start of the current month.
            #
            # We use 30-day periods for simplicity because this is
            # synthetic data and exact calendar-month handling is
            # unnecessary for this demonstration.
            month_start = START_DATE + timedelta(
                days=month * 30
            )


            # Calculate the approximate end of the current month.
            month_end = START_DATE + timedelta(
                days=min(
                    (month + 1) * 30 - 1,
                    date_range_days,
                )
            )


            # Don't create transactions beyond our configured
            # date range.
            if month_start > END_DATE:
                continue


            if month_end > END_DATE:
                month_end = END_DATE


            # Number of days available within this period.
            days_in_period = (month_end - month_start).days


            # Pick a random day within the month.
            transaction_date = month_start + timedelta(
                days=random.randint(
                    0,
                    max(days_in_period, 0),
                )
            )


            # ------------------------------------------------
            # Generate transaction amount
            # ------------------------------------------------
            #
            # Start with a segment-specific amount.
            #
            # Then use a log-normal distribution to introduce
            # realistic variation. This means most transactions
            # are relatively moderate, while a smaller number
            # can be substantially larger.
            #

            base_amount = BASE_TRANSACTION_AMOUNTS[segment]

            amount = (
                base_amount
                * np.random.lognormal(
                    mean=0,
                    sigma=0.65,
                )
            )


            # Prevent extremely small or extremely large values
            # from dominating the synthetic dataset.
            amount = round(
                max(
                    10,
                    min(amount, 25_000),
                ),
                2,
            )


            # ------------------------------------------------
            # Select transaction channel
            # ------------------------------------------------
            #
            # Mobile and Online are more common in this synthetic
            # dataset, while Branch transactions are less frequent.
            #

            channel = random.choices(
                CHANNELS,
                weights=[45, 25, 20, 10],
                k=1,
            )[0]


            # ------------------------------------------------
            # Select banking product
            # ------------------------------------------------

            product = random.choices(
                PRODUCTS,
                weights=[35, 25, 20, 10, 10],
                k=1,
            )[0]


            # Add the transaction to our list.
            transactions.append(
                {
                    "transaction_id": (
                        f"T{len(transactions) + 1:06d}"
                    ),
                    "customer_id": customer_id,
                    "transaction_date": transaction_date.date(),
                    "amount": amount,
                    "channel": channel,
                    "product": product,
                }
            )


# Convert transactions into a DataFrame.
transactions_df = pd.DataFrame(transactions)


# ============================================================
# Sort transactions
# ============================================================
#
# Sorting makes the resulting CSV easier to inspect and also gives
# us a sensible chronological order for later analysis.
#

transactions_df = transactions_df.sort_values(
    ["transaction_date", "customer_id"]
).reset_index(drop=True)


# ============================================================
# Save datasets
# ============================================================

customers_df.to_csv(
    "customers.csv",
    index=False,
)

transactions_df.to_csv(
    "transactions.csv",
    index=False,
)


# ============================================================
# Dataset summary
# ============================================================
#
# Print useful information so we can immediately verify that the
# generated dataset looks sensible.
#

print("\nCustomer dataset")
print("----------------")
print(f"Customers: {len(customers_df):,}")


print("\nTransaction dataset")
print("-------------------")
print(f"Transactions: {len(transactions_df):,}")


print("\nDate range:")
print(
    transactions_df["transaction_date"].min(),
    "to",
    transactions_df["transaction_date"].max(),
)


print("\nCustomer segments:")
print(
    customers_df["segment"].value_counts()
)


print("\nBehaviour profiles:")
print(
    customers_df["behaviour"].value_counts()
)


print("\nTransaction channels:")
print(
    transactions_df["channel"].value_counts()
)


print("\nProducts:")
print(
    transactions_df["product"].value_counts()
)


print("\nAverage transaction value:")
print(
    f"{transactions_df['amount'].mean():,.2f}"
)


print("\nTotal transaction value:")
print(
    f"{transactions_df['amount'].sum():,.2f}"
)


print("\nSample customers:")
print(
    customers_df.head()
)


print("\nSample transactions:")
print(
    transactions_df.head()
)


# ---------------------------------------------------------
# VALIDATE BEHAVIOUR PATTERNS
# ---------------------------------------------------------
# At this stage, both customers and transactions are Python
# lists of dictionaries. We convert them to DataFrames so
# we can perform the analysis.

customers_df = pd.DataFrame(customers)
transactions_df = pd.DataFrame(transactions)

transactions_df["transaction_date"] = pd.to_datetime(
    transactions_df["transaction_date"]
)

# Split the year into an early period and a late period.
early_period = transactions_df[
    transactions_df["transaction_date"] < "2026-01-01"
]

late_period = transactions_df[
    transactions_df["transaction_date"] >= "2026-07-01"
]

# Count transactions per customer in each period.
early_counts = early_period.groupby("customer_id").size()
late_counts = late_period.groupby("customer_id").size()

# Create a table containing each customer's behaviour.
behaviour_check = customers_df[
    ["customer_id", "behaviour"]
].copy()

# Add each customer's transaction count for the two periods.
behaviour_check["early_transactions"] = (
    behaviour_check["customer_id"]
    .map(early_counts)
    .fillna(0)
)

behaviour_check["late_transactions"] = (
    behaviour_check["customer_id"]
    .map(late_counts)
    .fillna(0)
)

# Calculate the average activity for each behaviour.
validation = (
    behaviour_check
    .groupby("behaviour")[
        ["early_transactions", "late_transactions"]
    ]
    .mean()
)

print("\nBehaviour validation:")
print(validation.round(2))