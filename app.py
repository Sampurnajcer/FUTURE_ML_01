import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ====================================
# CONSTANTS
# ====================================

ORDER_DATE_COL = 'Order Date'
SALES_COL = 'Sales'

# ====================================
# PAGE TITLE
# ====================================

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("Sales Forecasting Dashboard")
st.write("Machine Learning Project - Future Interns")

# ====================================
# LOAD DATASET
# ====================================

try:
    # CSV FILE
    df = pd.read_csv("superstore.csv", encoding='latin1')

except FileNotFoundError:
    st.error(
        "Dataset file not found. Make sure 'superstore.csv' is in the same folder as app.py"
    )
    st.stop()

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ====================================
# SHOW DATASET
# ====================================

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ====================================
# DATA CLEANING
# ====================================

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert Order Date column to datetime
df[ORDER_DATE_COL] = pd.to_datetime(
    df[ORDER_DATE_COL],
    errors='coerce'
)

# Remove missing values
df = df.dropna()

# ====================================
# FEATURE ENGINEERING
# ====================================

df['Year'] = df[ORDER_DATE_COL].dt.year
df['Month'] = df[ORDER_DATE_COL].dt.month
df['Day'] = df[ORDER_DATE_COL].dt.day

# ====================================
# MONTHLY SALES ANALYSIS
# ====================================

monthly_sales = (
    df.groupby('Month')[SALES_COL]
    .sum()
    .sort_index()
)

# ====================================
# SALES TREND GRAPH
# ====================================

st.subheader("Monthly Sales Trend")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker='o',
    linewidth=2,
    color='blue'
)

ax.set_xlabel("Month")
ax.set_ylabel("Total Sales")
ax.set_title("Monthly Sales Trend")
ax.grid(True)

st.pyplot(fig)

# ====================================
# MACHINE LEARNING MODEL
# ====================================

X = df[['Year', 'Month', 'Day']]
y = df[SALES_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ====================================
# MODEL EVALUATION
# ====================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

st.subheader("Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")

# ====================================
# ACTUAL VS PREDICTED GRAPH
# ====================================

st.subheader("Actual vs Predicted Sales")

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    y_test.values[:100],
    label='Actual Sales',
    color='green'
)

ax2.plot(
    y_pred[:100],
    label='Predicted Sales',
    color='red'
)

ax2.set_xlabel("Samples")
ax2.set_ylabel("Sales")
ax2.set_title("Actual vs Predicted Sales")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# ====================================
# FUTURE SALES PREDICTION
# ====================================

st.subheader("Future Sales Prediction")

future_year = st.number_input(
    "Enter Year",
    min_value=2024,
    max_value=2100,
    value=2026
)

future_month = st.slider(
    "Select Month",
    1,
    12,
    12
)

future_day = st.slider(
    "Select Day",
    1,
    31,
    15
)

future_data = pd.DataFrame({
    'Year': [future_year],
    'Month': [future_month],
    'Day': [future_day]
})

future_prediction = model.predict(future_data)

st.success(
    f"Predicted Sales: {future_prediction[0]:.2f}"
)