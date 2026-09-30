
"""
Customer inactivity prediction model.

Prediction framework:

    M-2 + M-1 + M
            ↓
    Predict activity in M+1

Example:

    May + June + July
            ↓
    Predict August


MODEL DEVELOPMENT APPROACH
--------------------------

We preserve the time order of the data.

Training:
    December 2025 → June 2026

Testing:
    July 2026 → August 2026

The model is therefore tested on later months that it did
not see during training.

After evaluating the model, we will train a final version
using all historical labelled snapshots and use the September
snapshot to predict October inactivity.


IMPORTANT
---------

The target is:

    inactive_next_month

where:

    0 = customer transacted in the following month
    1 = customer did not transact in the following month

Because inactivity is relatively rare, accuracy alone is
not sufficient for evaluating the model.

We therefore inspect:

    - Accuracy
    - ROC-AUC
    - Precision
    - Recall
    - F1
    - Confusion matrix
    - Predicted probability distribution

The probability analysis helps us understand whether the
model is ranking higher-risk customers correctly even when
the default 0.50 classification threshold is too conservative.
"""


# ============================================================
# IMPORTS
# ============================================================

import os

import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. DATABASE CONNECTION
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is not set. "
        "Add it to your .env file."
    )

engine = create_engine(DATABASE_URL)


# ============================================================
# 2. LOAD HISTORICAL CUSTOMER SNAPSHOTS
# ============================================================

query = """
SELECT
    customer_id,
    snapshot_month,
    transactions_two_months_ago,
    transactions_last_month,
    transactions_current_month,
    average_transaction_value,
    number_of_channels,
    days_since_last_transaction,
    inactive_next_month
FROM customer_activity_snapshots
ORDER BY snapshot_month, customer_id;
"""

df = pd.read_sql(query, engine)


# PostgreSQL may return timezone-aware timestamps.
#
# We only care about the calendar month, so we remove the
# timezone information and work with a consistent datetime type.

df["snapshot_month"] = (
    pd.to_datetime(
        df["snapshot_month"],
        utc=True,
    )
    .dt.tz_localize(None)
)


print("Historical data loaded.")
print(f"Rows: {len(df):,}")
print(f"Customers: {df['customer_id'].nunique():,}")


print("\nSnapshot months:")
print(
    df["snapshot_month"]
    .drop_duplicates()
    .sort_values()
    .dt.strftime("%Y-%m")
    .to_list()
)


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "transactions_two_months_ago",
    "transactions_last_month",
    "transactions_current_month",
    "average_transaction_value",
    "number_of_channels",
    "days_since_last_transaction",
]

target = "inactive_next_month"


# ============================================================
# 4. PREPARE MODELLING DATA
# ============================================================

model_data = df[
    [
        "customer_id",
        "snapshot_month",
        *features,
        target,
    ]
].copy()


# A missing recency value means that no previous transaction
# was available for calculating the number of days since the
# customer's last transaction.
#
# We represent this explicitly as -1 rather than removing
# those customers.

model_data["days_since_last_transaction"] = (
    model_data["days_since_last_transaction"]
    .fillna(-1)
)


# Remove rows with any other missing modelling values.

model_data = model_data.dropna(
    subset=features + [target]
)


print("\nModelling dataset")
print("-----------------")
print(f"Rows: {len(model_data):,}")


# ============================================================
# 5. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

# June is the final month used for training.
#
# Training:
#     December 2025 → June 2026
#
# Testing:
#     July 2026 → August 2026
#
# snapshot_month uses the first day of each month to represent
# the calendar month.

training_end = pd.Timestamp("2026-06-01")


training_data = model_data[
    model_data["snapshot_month"] <= training_end
].copy()


test_data = model_data[
    model_data["snapshot_month"] > training_end
].copy()


print("\nTemporal evaluation split")
print("-------------------------")

print(
    "Training:",
    training_data["snapshot_month"]
    .min()
    .strftime("%Y-%m"),
    "to",
    training_data["snapshot_month"]
    .max()
    .strftime("%Y-%m"),
)

print(
    "Testing:",
    test_data["snapshot_month"]
    .min()
    .strftime("%Y-%m"),
    "to",
    test_data["snapshot_month"]
    .max()
    .strftime("%Y-%m"),
)

print(
    f"Training rows: {len(training_data):,}"
)

print(
    f"Testing rows: {len(test_data):,}"
)


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X_train = training_data[features]
y_train = training_data[target]

X_test = test_data[features]
y_test = test_data[target]


# ============================================================
# 7. CHECK TARGET DISTRIBUTION
# ============================================================

print("\nTraining target distribution")
print("----------------------------")

print(
    y_train.value_counts()
)


print("\nTest target distribution")
print("-------------------------")

print(
    y_test.value_counts()
)


# ============================================================
# 8. BUILD THE MODEL
# ============================================================

# Logistic Regression is intentionally being used as our
# first baseline model.
#
# It is:
#
#     - simple
#     - interpretable
#     - fast
#     - suitable for probability estimates
#
# StandardScaler puts the numerical features on comparable
# scales before the logistic regression model is trained.

model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# 9. TRAIN THE MODEL
# ============================================================

model.fit(
    X_train,
    y_train,
)


# ============================================================
# 10. GENERATE PREDICTIONS
# ============================================================

# Standard classification prediction.
#
# Logistic Regression normally uses:
#
#     probability >= 0.50 → class 1
#     probability < 0.50  → class 0

y_pred = model.predict(X_test)


# Probability that each customer will be inactive next month.

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 11. BASIC MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

roc_auc = roc_auc_score(
    y_test,
    y_probability,
)


print("\nModel evaluation")
print("----------------")

print(
    f"Accuracy: {accuracy:.3f}"
)

print(
    f"ROC-AUC:  {roc_auc:.3f}"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification report")
print("---------------------")

print(
    classification_report(
        y_test,
        y_pred,
        digits=3,
        zero_division=0,
    )
)


# zero_division=0 prevents scikit-learn from printing warnings
# when a class has no predicted observations.
#
# We still report the underlying problem rather than hiding it.


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\nConfusion matrix")
print("----------------")

print(
    confusion_matrix(
        y_test,
        y_pred,
    )
)


# ============================================================
# 14. INSPECT PREDICTED PROBABILITIES
# ============================================================
#
# This is the important new section.
#
# We want to understand whether the model is producing useful
# risk scores even though its default 0.50 threshold produced
# zero inactive predictions.
#


print("\nPredicted inactivity probabilities")
print("----------------------------------")

print(
    f"Minimum: {y_probability.min():.4f}"
)

print(
    f"Maximum: {y_probability.max():.4f}"
)

print(
    f"Mean:    {y_probability.mean():.4f}"
)

print(
    f"Median:  {pd.Series(y_probability).median():.4f}"
)


# ============================================================
# 15. SHOW PROBABILITY DISTRIBUTION
# ============================================================
#
# Instead of looking at every probability individually,
# group customers into risk bands.
#
# These are NOT final business risk categories.
# They are simply a way to inspect the model output.

probability_df = pd.DataFrame(
    {
        "actual": y_test.values,
        "predicted_probability": y_probability,
    }
)


probability_df["risk_band"] = pd.cut(
    probability_df["predicted_probability"],
    bins=[
        -0.001,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        1.00,
    ],
    labels=[
        "0-5%",
        "5-10%",
        "10-20%",
        "20-30%",
        "30-50%",
        "50%+",
    ],
)


risk_summary = (
    probability_df
    .groupby(
        "risk_band",
        observed=False,
    )
    .agg(
        customers=("actual", "size"),
        actual_inactive=("actual", "sum"),
        average_predicted_risk=(
            "predicted_probability",
            "mean",
        ),
    )
    .reset_index()
)


# Calculate the actual inactivity rate within each band.

risk_summary["actual_inactivity_rate"] = (
    risk_summary["actual_inactive"]
    / risk_summary["customers"]
)


print("\nRisk-band analysis")
print("------------------")

print(
    risk_summary.to_string(
        index=False,
        formatters={
            "average_predicted_risk": "{:.3f}".format,
            "actual_inactivity_rate": "{:.3f}".format,
        },
    )
)


# ============================================================
# 16. SHOW HIGHEST-RISK CUSTOMERS
# ============================================================
#
# This lets us see what the model is actually producing.
#
# We are not claiming these customers are definitely going
# inactive. These are simply the customers receiving the
# highest predicted probabilities in the test period.

high_risk = test_data[
    [
        "customer_id",
        "snapshot_month",
        *features,
    ]
].copy()


high_risk["predicted_inactivity_probability"] = (
    y_probability
)


high_risk["actual_inactive_next_month"] = (
    y_test.values
)


high_risk = high_risk.sort_values(
    "predicted_inactivity_probability",
    ascending=False,
)


print("\nTop 20 predicted-risk observations")
print("----------------------------------")

print(
    high_risk[
        [
            "customer_id",
            "snapshot_month",
            "predicted_inactivity_probability",
            "actual_inactive_next_month",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

# ============================================================
# 17. THRESHOLD ANALYSIS
# ============================================================
#
# Logistic Regression normally classifies a customer as
# inactive when the predicted probability is >= 0.50.
#
# Our model does not produce probabilities above 0.50,
# so that default threshold predicts zero inactive customers.
#
# We therefore evaluate several lower thresholds.
#
# These thresholds are being evaluated on the held-out
# test period. We are not choosing one arbitrarily.
# ============================================================

from sklearn.metrics import precision_score, recall_score, f1_score


print("\nThreshold analysis")
print("------------------")

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:

    threshold_predictions = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0,
    )

    predicted_inactive = (
        threshold_predictions == 1
    ).sum()

    print(
        f"Threshold: {threshold:.2f} | "
        f"Predicted inactive: {predicted_inactive:,} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )