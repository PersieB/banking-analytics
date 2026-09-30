import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Banking Metrics Explorer",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# 2. Database connection
# --------------------------------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    st.error("DATABASE_URL is not set in the .env file.")
    st.stop()

engine = create_engine(DATABASE_URL)


# --------------------------------------------------
# 3. Data loading
# --------------------------------------------------

@st.cache_data
def load_overall_metrics():

    query = """
    SELECT
        COUNT(*) AS total_transactions,
        COUNT(DISTINCT customer_id) AS active_customers,
        ROUND(SUM(amount), 2) AS total_transaction_value,
        ROUND(AVG(amount), 2) AS average_transaction_value
    FROM transactions;
    """

    return pd.read_sql(query, engine)


@st.cache_data
def load_monthly_metrics():

    query = """
    SELECT
        DATE_TRUNC('month', transaction_date)::date AS month,
        COUNT(*) AS transaction_count,
        ROUND(SUM(amount), 2) AS transaction_value,
        ROUND(AVG(amount), 2) AS average_transaction_value
    FROM transactions
    GROUP BY DATE_TRUNC('month', transaction_date)
    ORDER BY month;
    """

    return pd.read_sql(query, engine)


@st.cache_data
def load_segment_metrics():

    query = """
    SELECT
        c.segment,
        COUNT(DISTINCT c.customer_id) AS customers,
        COUNT(t.transaction_id) AS transactions,
        ROUND(SUM(t.amount), 2) AS transaction_value,
        ROUND(AVG(t.amount), 2) AS average_transaction_value
    FROM customers c
    LEFT JOIN transactions t
        ON c.customer_id = t.customer_id
    GROUP BY c.segment
    ORDER BY transaction_value DESC;
    """

    return pd.read_sql(query, engine)


@st.cache_data
def load_channel_metrics():

    query = """
    SELECT
        channel,
        COUNT(*) AS transactions,
        ROUND(SUM(amount), 2) AS transaction_value,
        ROUND(AVG(amount), 2) AS average_transaction_value
    FROM transactions
    GROUP BY channel
    ORDER BY transaction_value DESC;
    """

    return pd.read_sql(query, engine)


@st.cache_data
def load_product_metrics():

    query = """
    SELECT
        product,
        COUNT(*) AS transactions,
        ROUND(SUM(amount), 2) AS transaction_value,
        ROUND(AVG(amount), 2) AS average_transaction_value
    FROM transactions
    GROUP BY product
    ORDER BY transaction_value DESC;
    """

    return pd.read_sql(query, engine)


@st.cache_data
def load_predictions():

    path = "outputs/october_activity_predictions.csv"

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


# --------------------------------------------------
# 4. Load data
# --------------------------------------------------

try:

    overall = load_overall_metrics().iloc[0]
    monthly = load_monthly_metrics()
    segment = load_segment_metrics()
    channel = load_channel_metrics()
    product = load_product_metrics()
    predictions = load_predictions()

except Exception as e:

    st.error(f"Unable to load dashboard data: {e}")
    st.stop()


# --------------------------------------------------
# 5. Header
# --------------------------------------------------

st.title("Banking Metrics Explorer")

st.write(
    "A synthetic banking analytics dashboard exploring "
    "transaction activity, customer behaviour and "
    "predicted inactivity risk."
)

st.caption(
    "Synthetic data for demonstration purposes. "
    "This dashboard does not use proprietary bank data."
)


# --------------------------------------------------
# 6. Key metrics
# --------------------------------------------------

st.subheader("Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers",
    f"{overall['active_customers']:,.0f}"
)

col2.metric(
    "Transactions",
    f"{overall['total_transactions']:,.0f}"
)

col3.metric(
    "Transaction Value",
    f"{overall['total_transaction_value']:,.0f}"
)

col4.metric(
    "Average Transaction",
    f"{overall['average_transaction_value']:,.2f}"
)


# --------------------------------------------------
# 7. Monthly transaction trend
# --------------------------------------------------

st.subheader("Monthly Transaction Activity")

monthly["month"] = pd.to_datetime(monthly["month"])

chart_data = monthly.set_index("month")[
    ["transaction_count"]
]

st.line_chart(chart_data)

st.dataframe(
    monthly,
    width="stretch",
    hide_index=True,
)


# --------------------------------------------------
# 8. Segment analysis
# --------------------------------------------------

st.subheader("Customer Segment Analysis")

col1, col2 = st.columns(2)

with col1:

    st.write("Transaction Value by Segment")

    segment_chart = segment.set_index(
        "segment"
    )[["transaction_value"]]

    st.bar_chart(segment_chart)


with col2:

    st.write("Customers by Segment")

    customer_chart = segment.set_index(
        "segment"
    )[["customers"]]

    st.bar_chart(customer_chart)


st.dataframe(
    segment,
    width="stretch",
    hide_index=True,
)


# --------------------------------------------------
# 9. Channel and product analysis
# --------------------------------------------------

st.subheader("Transaction Analysis")

col1, col2 = st.columns(2)

with col1:

    st.write("Transaction Value by Channel")

    channel_chart = channel.set_index(
        "channel"
    )[["transaction_value"]]

    st.bar_chart(channel_chart)

    st.dataframe(
        channel,
        width="stretch",
        hide_index=True,
    )


with col2:

    st.write("Transaction Value by Product")

    product_chart = product.set_index(
        "product"
    )[["transaction_value"]]

    st.bar_chart(product_chart)

    st.dataframe(
        product,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# 10. Inactivity prediction
# --------------------------------------------------

st.subheader("October Inactivity Risk")

if predictions.empty:

    st.warning(
        "Prediction file not found. "
        "Run train_model.py first."
    )

else:

    current_month = pd.to_datetime(
        predictions["current_month"]
    ).max()

    st.write(
        f"Using customer activity from "
        f"the current observed month "
        f"({current_month.strftime('%B %Y')}) "
        f"and the two preceding months to estimate "
        f"October inactivity risk."
    )

    col1, col2, col3 = st.columns(3)

    total_customers = len(predictions)

    predicted_inactive = int(
        predictions["predicted_inactive"].sum()
    )

    high_risk = int(
        (
            predictions["risk_category"]
            == "High"
        ).sum()
    )

    col1.metric(
        "Customers Analysed",
        f"{total_customers:,}"
    )

    col2.metric(
        "Predicted Inactive",
        f"{predicted_inactive:,}"
    )

    col3.metric(
        "High Risk",
        f"{high_risk:,}"
    )


    # --------------------------------------------------
    # Risk distribution
    # --------------------------------------------------

    st.write("Risk Distribution")

    risk_counts = (
        predictions["risk_category"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High"]
        )
        .fillna(0)
    )

    st.bar_chart(risk_counts)


    # --------------------------------------------------
    # Filters
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        selected_segment = st.selectbox(
            "Segment",
            ["All"]
            + sorted(
                predictions["segment"]
                .dropna()
                .unique()
                .tolist()
            ),
        )

    with col2:

        selected_risk = st.selectbox(
            "Risk Category",
            ["All", "Low", "Medium", "High"],
        )


    filtered = predictions.copy()

    if selected_segment != "All":

        filtered = filtered[
            filtered["segment"]
            == selected_segment
        ]

    if selected_risk != "All":

        filtered = filtered[
            filtered["risk_category"]
            == selected_risk
        ]


    # --------------------------------------------------
    # High-risk customer table
    # --------------------------------------------------

    st.write("Customers with Highest Predicted Risk")

    display_columns = [
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

    filtered = filtered.sort_values(
        "inactivity_probability",
        ascending=False,
    )

    display_df = filtered[
        display_columns
    ].copy()

    display_df[
        "inactivity_probability"
    ] = (
        display_df[
            "inactivity_probability"
        ] * 100
    ).round(1)

    display_df = display_df.rename(
        columns={
            "customer_id": "Customer ID",
            "customer_name": "Customer",
            "segment": "Segment",
            "transactions_two_months_ago":
                "2 Months Ago",
            "transactions_last_month":
                "Last Month",
            "transactions_current_month":
                "Current Month",
            "days_since_last_transaction":
                "Days Since Last Transaction",
            "inactivity_probability":
                "Inactivity Probability (%)",
            "risk_category":
                "Risk",
        }
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# 11. Methodology
# --------------------------------------------------

with st.expander("Methodology"):

    st.write(
        """
        The inactivity model uses a monthly customer snapshot.

        For each historical snapshot, transaction activity from
        the current month and the two preceding calendar months
        is used to predict whether the customer will have no
        transactions in the following month.

        A logistic regression model was evaluated using a
        temporal split. Historical snapshots from December 2025
        through June 2026 were used for training, while July and
        August 2026 were held out for evaluation.

        After evaluation, the final model was trained on all
        18,000 labelled historical customer-month observations.

        The classification threshold of 0.15 was selected from
        the held-out evaluation results because it produced the
        highest F1 score among the tested thresholds.

        The current prediction uses July, August and September
        2026 activity to estimate October inactivity risk.

        The data used in this project is synthetic and is intended
        for demonstration of data engineering, SQL analytics and
        predictive modelling workflows.
        """
    )