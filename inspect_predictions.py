import pandas as pd

df = pd.read_csv(
    "outputs/october_activity_predictions.csv"
)

print("\nPrediction distribution")
print("-----------------------")
print(
    df["risk_category"]
    .value_counts()
    .sort_index()
)

print("\nTop 20 highest-risk customers")
print("------------------------------")

top_risk = df.sort_values(
    "inactivity_probability",
    ascending=False
)[
    [
        "customer_id",
        "customer_name",
        "segment",
        "transactions_two_months_ago",
        "transactions_last_month",
        "transactions_current_month",
        "days_since_last_transaction",
        "inactivity_probability",
        "risk_category",
    ]
].head(20)

print(top_risk.to_string(index=False))

print("\nRisk by segment")
print("---------------")

risk_by_segment = (
    df.groupby("segment")
    .agg(
        customers=("customer_id", "count"),
        predicted_inactive=("predicted_inactive", "sum"),
        average_risk=("inactivity_probability", "mean"),
    )
    .sort_values(
        "predicted_inactive",
        ascending=False
    )
)

print(risk_by_segment)

print("\nAverage risk")
print("------------")
print(
    f"{df['inactivity_probability'].mean():.3f}"
)