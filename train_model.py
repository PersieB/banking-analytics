import os
import joblib
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# 1. Database connection
# --------------------------------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env")

engine = create_engine(DATABASE_URL)


# --------------------------------------------------
# 2. Load historical modelling data
# --------------------------------------------------

historical_query = """
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

df = pd.read_sql(historical_query, engine)

print(f"Historical rows: {len(df):,}")


# --------------------------------------------------
# 3. Prepare data
# --------------------------------------------------

df["snapshot_month"] = (
    pd.to_datetime(df["snapshot_month"], utc=True)
    .dt.tz_localize(None)
)

features = [
    "transactions_two_months_ago",
    "transactions_last_month",
    "transactions_current_month",
    "average_transaction_value",
    "number_of_channels",
    "days_since_last_transaction",
]

target = "inactive_next_month"

X = df[features].copy()
y = df[target].copy()

# -1 means the customer had no previous transaction
# available at that snapshot.
X["days_since_last_transaction"] = (
    X["days_since_last_transaction"].fillna(-1)
)


# --------------------------------------------------
# 4. Temporal train/test split
# --------------------------------------------------

train = df["snapshot_month"] <= pd.Timestamp("2026-06-01")
test = df["snapshot_month"] >= pd.Timestamp("2026-07-01")

X_train = X.loc[train]
y_train = y.loc[train]

X_test = X.loc[test]
y_test = y.loc[test]

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")


# --------------------------------------------------
# 5. Build model
# --------------------------------------------------

def create_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),
            ),
        ]
    )


# --------------------------------------------------
# 6. Evaluate historical model
# --------------------------------------------------

evaluation_model = create_model()

evaluation_model.fit(X_train, y_train)

test_probabilities = evaluation_model.predict_proba(X_test)[:, 1]

default_predictions = (
    test_probabilities >= 0.50
).astype(int)

print("\nHistorical evaluation")
print("---------------------")

print(
    f"Accuracy: "
    f"{accuracy_score(y_test, default_predictions):.3f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, test_probabilities):.3f}"
)


# --------------------------------------------------
# 7. Test classification thresholds
# --------------------------------------------------

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]

print("\nThreshold analysis")
print("------------------")

threshold_results = []

for threshold in thresholds:

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    print(
        f"{threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )


# --------------------------------------------------
# 8. Select project threshold
# --------------------------------------------------

threshold_results_df = pd.DataFrame(threshold_results)

best_threshold = threshold_results_df.loc[
    threshold_results_df["f1"].idxmax(),
    "threshold"
]

print(
    f"\nSelected threshold: {best_threshold:.2f}"
)


# --------------------------------------------------
# 9. Train final model using all historical data
# --------------------------------------------------

final_model = create_model()

final_model.fit(X, y)

print(
    f"\nFinal model trained on "
    f"{len(X):,} historical snapshots."
)


# --------------------------------------------------
# 10. Load current customer snapshot
# --------------------------------------------------

current_query = """
SELECT
    customer_id,
    customer_name,
    segment,
    current_month,
    transactions_two_months_ago,
    transactions_last_month,
    transactions_current_month,
    average_transaction_value,
    number_of_channels,
    days_since_last_transaction
FROM current_customer_activity
ORDER BY customer_id;
"""

current_df = pd.read_sql(
    current_query,
    engine
)

print(
    f"Current customer rows: "
    f"{len(current_df):,}"
)


# --------------------------------------------------
# 11. Prepare current features
# --------------------------------------------------

X_current = current_df[features].copy()

X_current["days_since_last_transaction"] = (
    X_current["days_since_last_transaction"].fillna(-1)
)


# --------------------------------------------------
# 12. Predict October inactivity
# --------------------------------------------------

current_probabilities = final_model.predict_proba(
    X_current
)[:, 1]

current_df["inactivity_probability"] = (
    current_probabilities
)

current_df["predicted_inactive"] = (
    current_df["inactivity_probability"]
    >= best_threshold
).astype(int)


# --------------------------------------------------
# 13. Create simple risk categories
# --------------------------------------------------

current_df["risk_category"] = pd.cut(
    current_df["inactivity_probability"],
    bins=[-float("inf"), 0.05, 0.15, float("inf")],
    labels=["Low", "Medium", "High"],
    right=False
)


# --------------------------------------------------
# 14. Save predictions
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

prediction_file = (
    "outputs/october_activity_predictions.csv"
)

current_df.to_csv(
    prediction_file,
    index=False
)

joblib.dump(
    final_model,
    "models/inactivity_model.joblib"
)


# --------------------------------------------------
# 15. Summary
# --------------------------------------------------

print("\nPrediction summary")
print("-------------------")

print(
    f"Customers: "
    f"{len(current_df):,}"
)

print(
    f"Predicted inactive: "
    f"{current_df['predicted_inactive'].sum():,}"
)

print(
    f"Predicted active: "
    f"{(current_df['predicted_inactive'] == 0).sum():,}"
)

print("\nRisk categories:")
print(
    current_df["risk_category"]
    .value_counts()
    .sort_index()
)

print(
    f"\nPredictions saved to: "
    f"{prediction_file}"
)

print(
    "Model saved to: "
    "models/inactivity_model.joblib"
)