# Banking Analytics Dashboard

A financial data analytics project focused on **customer transactions, product activity, customer behaviour, and inactivity risk**.

**Live Dashboard:** https://mock-banking-analytics.streamlit.app/
**GitHub:** https://github.com/PersieB/banking-analytics

> **Note:** This project uses synthetic banking data created for portfolio demonstration purposes. It does not use Ecobank or any other bank's proprietary customer data.

## Business Question

**Which customers appear to be at risk of becoming inactive based on their recent transaction behaviour?**

The project explores how customer transaction data can be used to monitor business metrics, understand customer behaviour and trends, and identify potential signals that could support customer retention.

## Data & Workflow

The synthetic dataset was designed around common banking transaction information rather than completely random records.

| Data               | Why it was included                                              |
| ------------------ | ---------------------------------------------------------------- |
| Customer segment   | Compare activity across customer groups                          |
| Transaction date   | Analyse activity and trends over time                            |
| Transaction amount | Measure transaction value and average transaction size           |
| Channel            | Understand Mobile, Online, ATM and Branch usage                  |
| Product            | Compare Savings, Current, Payments, Investment and Loan activity |

The dataset contains **2,000 customers and 118,324 transactions** covering approximately one year.

The workflow was:

```text
Excel / Tabular Data
        ↓
Python Data Validation
        ↓
ETL Pipeline
        ↓
PostgreSQL Database
        ↓
SQL Analysis
        ↓
Customer Activity Features
        ↓
Predictive Modelling
        ↓
Streamlit Dashboard
```

Data quality checks were performed before loading the data, including duplicate ID checks, customer-reference validation and transaction amount validation.

## Customer & Financial Analysis

The dashboard looks at:

* Overall transaction volume and value
* Monthly transaction trends
* Customer segment activity
* Product activity
* Transaction channel usage
* Customer-level activity
* Potential inactivity risk

### Portfolio snapshot

| Metric                                       |  Result |
| -------------------------------------------- | ------: |
| Customers                                    |   2,000 |
| Transactions                                 | 118,324 |
| Transaction value                            |   75.9M |
| Average transaction                          |  641.81 |
| Customers classified as high inactivity risk |     236 |

The latest available transaction data runs through **September 25, 2026**, so September is treated as the current observed month rather than a completed calendar month.

## Customer Inactivity Model

The prediction problem was designed around a simple business question:

> **Can recent customer activity provide an early signal that a customer may become inactive?**

For each customer, the model uses the **current month and the two preceding months** to predict activity in the following month.

```text
July + August + September
             ↓
       Predict October
```

The model uses:

* Recent transaction frequency
* Average transaction value
* Number of transaction channels used
* Days since the most recent transaction

A customer is considered inactive for the target month when they record **zero transactions during that month**.

Historical customer-month observations were used to evaluate the model using a time-based split, keeping future observations separate from training data.

| Evaluation              |    Result |
| ----------------------- | --------: |
| Historical observations |    18,000 |
| Training observations   |    14,000 |
| Test observations       |     4,000 |
| ROC-AUC                 | **0.824** |
| Accuracy                | **93.6%** |

A **15% probability threshold** was selected from historical test results for the dashboard's high-risk group. This is a project-specific threshold and would need to be validated against real banking outcomes before operational use.

## Why This Matters in Banking

The analysis connects transaction data to practical questions around customer and product activity:

**Transactions → Customer Behaviour → Trends → Risk Signal → Potential Business Action**

In a real banking environment, this type of analysis could be combined with additional information such as account balances, product holdings, customer tenure, service interactions and campaign history to better understand customers showing reduced activity.

## Skills Demonstrated

* Quantitative analysis and data mining
* Customer and financial transaction analysis
* Python and SQL
* Excel/tabular data handling
* Data profiling and quality checks
* ETL and data pipelines
* Relational database analysis
* Monitoring business and product metrics
* Customer behaviour and trend analysis
* Predictive/statistical modelling
* Data visualisation and presentation
* Translating a business question into an analytical solution

## Technology

**Python · Excel · SQL · PostgreSQL/Supabase · Pandas · Scikit-learn · Streamlit**

## Limitations

This is a synthetic portfolio project, so the results should not be interpreted as real banking performance.

A production implementation would require real transaction data, additional customer and product variables, model validation and monitoring, appropriate governance, and calibration against actual business outcomes.

## Project Structure

```text
banking-analytics/
├── app.py
├── generate_data.py
├── load_data.py
├── train_model.py
├── customers.csv
├── transactions.csv
├── models/
├── outputs/
└── README.md
```
